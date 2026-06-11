def _rank_two_clocks(X_tr: np.ndarray, y_tr: np.ndarray, pool: list[int], rising_only: bool) -> list[int]:
    """Two clocks: full training fold vs its most recent 25% (fold rows are
    time-ordered). 'dawn' (rising_only) ranks features whose recent-clock corr
    EXCEEDS their full-clock corr first -- a measured bet on the new regime.
    'both_clocks' ranks by the weaker of the two clocks (robust intersection)."""
    cut = int(0.75 * len(X_tr))
    c_full = np.abs(corr_vector(X_tr[:, pool], y_tr))
    if len(X_tr) - cut >= 200:
        c_rec = np.abs(corr_vector(X_tr[cut:][:, pool], y_tr[cut:]))
    else:
        c_rec = c_full
    if rising_only:
        rising = c_rec > c_full
        score = np.where(rising, c_rec, c_rec - 1.0)   # rising first, then rest by recent corr
    else:
        score = np.minimum(c_full, c_rec)
    return [pool[i] for i in np.argsort(-score)]


def _pair_op(op: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if op == 0:
        return a * b
    if op == 1:
        return a - b
    if op == 2:
        return a + b
    return a / (np.abs(b) + 1.0)


def build_viewport(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
                   cols: list[str], rng: np.random.Generator) -> tuple[list[int], Callable[[np.ndarray], np.ndarray]]:
    ranked = _ranked_for(spec, X_tr, y_tr, seg_tr, cols)
    idx = ranked[: spec.k]

    Z_tr = X_tr[:, idx]
    mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(idx)) if spec.transform == "signed_hadamard" else None
    proj = (rng.normal(size=(len(idx), min(spec.proj_dim, len(idx)))).astype(np.float32)
            / math.sqrt(min(spec.proj_dim, len(idx)))) if spec.transform == "rand_proj" else None

    # v18 random_fourier (FABRIC EXPANSION): lift features into a random
    # trigonometric basis so a downstream linear/ridge fit approximates an
    # RBF-KERNEL ridge -- curved feature-space capacity in the winning family.
    # Bandwidth = median pairwise distance heuristic (target-free).
    rff_W = rff_b = rff_mu = rff_sd = None
    if spec.transform == "random_fourier" and len(idx) >= 2:
        rff_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        rff_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        sub = ((Z_tr[:: max(1, len(Z_tr) // 4000)] - rff_mu) / rff_sd).astype(np.float32)
        gamma = 1.0 / max(1e-6, float(np.median(np.var(sub, axis=0))) * len(idx))
        D = min(256, max(32, 4 * len(idx)))
        rff_W = (rng.normal(size=(len(idx), D)).astype(np.float32) * math.sqrt(2.0 * gamma))
        rff_b = (rng.uniform(0, 2 * math.pi, size=D)).astype(np.float32)

    comps = None
    if spec.transform in ("pca", "pca_aug"):
        sub = Z_tr[:: max(1, len(Z_tr) // 20_000)] - mu
        try:
            _, _, vt = np.linalg.svd(sub, full_matrices=False)
            n_comp = min(spec.proj_dim if spec.transform == "pca" else 8, vt.shape[0])
            comps = vt[:n_comp].astype(np.float32)
        except Exception:
            comps = None

    q_lo = q_hi = None
    q_levels = {"quantize8": 255, "quantize4": 15, "quantize2": 3}.get(spec.transform, 15)
    if spec.transform in ("quantize8", "quantize4", "quantize2", "dual_exposure"):
        q_lo = np.percentile(Z_tr, 0.5, axis=0, keepdims=True).astype(np.float32)
        q_hi = np.percentile(Z_tr, 99.5, axis=0, keepdims=True).astype(np.float32)

    dual_grids = None
    if spec.transform == "dual_exposure":
        qs = np.linspace(0.0, 1.0, 32)[1:-1]            # 30 interior quantiles ~ 5 bits
        dual_grids = np.quantile(Z_tr, qs, axis=0).astype(np.float32)   # (30, k)

    # v11 lateral line (fish near-field flow): each feature's deviation from
    # the local consensus of its most-correlated neighbors. Built on the
    # training fold only (feature-feature corr, no y) -> fold-honest.
    lat_mu = lat_sd = lat_nbr = None
    if spec.transform == "lateral_line" and len(idx) >= 3:
        lat_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        lat_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        sub = ((Z_tr[:: max(1, len(Z_tr) // 20_000)] - lat_mu) / lat_sd)
        C = np.abs(np.nan_to_num(np.corrcoef(sub.T), nan=0.0))
        np.fill_diagonal(C, -1.0)
        nn = min(CFG.LATERAL_NEIGHBORS, len(idx) - 1)
        lat_nbr = np.argsort(-C, axis=1)[:, :nn].astype(np.int32)   # (k, nn) neighbor cols

    # v13 prism: train-fold spectral band edges (3-band piecewise light)
    prism_lo = prism_hi = None
    if spec.transform == "prism":
        prism_lo = np.quantile(Z_tr, 0.33, axis=0, keepdims=True).astype(np.float32)
        prism_hi = np.quantile(Z_tr, 0.66, axis=0, keepdims=True).astype(np.float32)

    # v13 moire: viewport self-dispersion interference stats
    moire_mu = moire_sd = None
    if spec.transform == "moire":
        moire_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        moire_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    # v8 folding: global (fold_abs) and regional (fold_pairs) space symmetry
    fold_mu = fold_sd = None
    fold_pair_idx: list[tuple[int, int]] = []
    if spec.transform in ("fold_abs", "fold_pairs"):
        fold_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        fold_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        if spec.transform == "fold_pairs" and len(idx) >= 4:
            sub = (Z_tr[:: max(1, len(Z_tr) // 24_000)] - fold_mu) / fold_sd
            C = np.corrcoef(sub.T)
            C = np.nan_to_num(C, nan=0.0)
            order = np.dstack(np.unravel_index(np.argsort(C, axis=None), C.shape))[0]
            used: set[int] = set()
            for i, j in order:                      # most anti-correlated first
                i, j = int(i), int(j)
                if i >= j or i in used or j in used or C[i, j] >= -0.05:
                    continue
                fold_pair_idx.append((i, j))
                used.update((i, j))
                if len(fold_pair_idx) >= CFG.FOLD_PAIRS:
                    break

    rank_grids = None
    if spec.transform == "rank":
        qs = np.linspace(0.0, 1.0, 66)[1:-1]            # 64 interior quantiles ~ 6 bits
        rank_grids = np.quantile(Z_tr, qs, axis=0).astype(np.float32)   # (64, k)

    sign_med = None
    if spec.transform == "sign_only":
        sign_med = np.median(Z_tr, axis=0, keepdims=True).astype(np.float32)

    fov = None
    if spec.transform == "foveated" and len(idx) > 8:
        nf = 8
        npr = min(64, len(idx) - nf)
        peri = slice(nf, nf + npr)
        back = slice(nf + npr, len(idx))
        mu_p = Z_tr[:, peri].mean(axis=0, keepdims=True).astype(np.float32)
        comps_p = None
        try:
            subp = Z_tr[:: max(1, len(Z_tr) // 20_000), peri] - mu_p
            _, _, vtp = np.linalg.svd(subp, full_matrices=False)
            comps_p = vtp[: min(4, vtp.shape[0])].astype(np.float32)
        except Exception:
            comps_p = None
        p_lo = p_hi = None
        if comps_p is not None:
            P_tr = (Z_tr[:, peri] - mu_p) @ comps_p.T
            p_lo = np.percentile(P_tr, 0.5, axis=0, keepdims=True).astype(np.float32)
            p_hi = np.percentile(P_tr, 99.5, axis=0, keepdims=True).astype(np.float32)
        back_w = back_mu = back_sd = None
        if back.stop > back.start:
            bw = corr_vector(Z_tr[:, back], y_tr)
            s_abs = float(np.sum(np.abs(bw)))
            if s_abs > 1e-12:
                back_w = (bw / s_abs).astype(np.float32)
                back_mu = Z_tr[:, back].mean(axis=0).astype(np.float32)
                back_sd = (Z_tr[:, back].std(axis=0) + 1e-6).astype(np.float32)
        fov = {"nf": nf, "peri": peri, "back": back, "mu_p": mu_p, "comps": comps_p,
               "p_lo": p_lo, "p_hi": p_hi, "bw": back_w, "bmu": back_mu, "bsd": back_sd}

    pair_m, pair_mu, pair_sd, pair_keep = 0, None, None, []
    if spec.transform == "pair_aug" and len(idx) >= 2:
        pair_m = min(CFG.PAIR_BASE, len(idx))
        pair_mu = Z_tr[:, :pair_m].mean(0).astype(np.float32)
        pair_sd = (Z_tr[:, :pair_m].std(0) + 1e-6).astype(np.float32)
        step = max(1, len(Z_tr) // 24_000)
        Bz = (Z_tr[::step, :pair_m] - pair_mu) / pair_sd
        ys = y_tr[::step]
        feats, ops = [], []
        for i in range(pair_m):
            for j in range(i + 1, pair_m):
                for op in range(4):
                    feats.append(_pair_op(op, Bz[:, i], Bz[:, j]))
                    ops.append((op, i, j))
        if feats:
            F = np.stack(feats, axis=1).astype(np.float32)
            sc = np.abs(corr_vector(F, ys))
            pair_keep = [ops[t] for t in np.argsort(-sc)[: CFG.PAIR_KEEP]]

    def transform(Z: np.ndarray) -> np.ndarray:
        if spec.transform == "signed_hadamard" and Z.shape[1] >= 2:
            return fwht(Z * signs)
        if spec.transform == "rand_proj" and proj is not None:
            return Z @ proj
        if spec.transform == "pca" and comps is not None:
            return (Z - mu) @ comps.T
        if spec.transform == "pca_aug" and comps is not None:
            return np.concatenate([Z, (Z - mu) @ comps.T], axis=1).astype(np.float32)
        if spec.transform in ("quantize8", "quantize4", "quantize2") and q_lo is not None:
            span = np.maximum(q_hi - q_lo, 1e-6)
            code = np.clip(np.round((Z - q_lo) / span * q_levels), 0, q_levels)
            return (q_lo + code / float(q_levels) * span).astype(np.float32)
        if spec.transform == "doppler":
            # the motion sense: levels + causal first differences. First row's
            # delta is zero; rows after fold gaps carry slightly stale deltas
            # (a handful per fold) -- measured approximation, documented.
            D = np.diff(Z, axis=0, prepend=Z[:1])
            return np.concatenate([Z, D], axis=1).astype(np.float32)
        if spec.transform == "prism" and prism_lo is not None:
            # refraction: the same light split into three spectral bands --
            # piecewise-linear sight for linear skills (train-fold quantiles)
            return np.concatenate([Z, Z * (Z <= prism_lo), Z * (Z >= prism_hi)],
                                  axis=1).astype(np.float32)
        if spec.transform == "moire" and moire_mu is not None:
            # interference: each column times the viewport's OWN local
            # agitation -- regime-conditional slopes with no gate model
            Zz = (Z - moire_mu) / moire_sd
            agitation = np.abs(Zz).mean(axis=1, keepdims=True)
            return np.concatenate([Zz, Zz * agitation], axis=1).astype(np.float32)
        if spec.transform == "tide":
            # the slow swell subtracted (causal EMA); rows after fold gaps
            # carry slightly stale tide -- same documented caveat as doppler
            ema = pd.DataFrame(Z).ewm(span=CFG.TIDE_SPAN, adjust=False).mean().to_numpy(np.float32)
            return (Z - ema).astype(np.float32)
        if spec.transform == "fractal":
            # v16 TREES/FRACTALS: the same signal at three resolutions, a
            # self-similar multiresolution pyramid (level + two coarse-grained
            # causal scales). Mandelbrot's self-similarity made a viewport.
            zdf = pd.DataFrame(Z)
            s1 = zdf.ewm(span=8, adjust=False).mean().to_numpy(np.float32)
            s2 = zdf.ewm(span=32, adjust=False).mean().to_numpy(np.float32)
            return np.concatenate([Z, s1, s2], axis=1).astype(np.float32)
        if spec.transform == "reaction_diffusion":
            # v16 SPOTS/STRIPES (Turing): activator (short-range excitation)
            # MINUS inhibitor (long-range diffusion) -- the band-pass morphogen
            # channel that makes standing-wave patterns. Distinct from tide
            # (high-pass): this keeps the MID band where patterns live.
            zdf = pd.DataFrame(Z)
            act = zdf.ewm(span=4, adjust=False).mean().to_numpy(np.float32)
            inh = zdf.ewm(span=32, adjust=False).mean().to_numpy(np.float32)
            return np.concatenate([Z, (act - inh)], axis=1).astype(np.float32)
        if spec.transform == "random_fourier" and rff_W is not None:
            # the fabric expansion: sqrt(2/D) cos(Wz + b), a Monte-Carlo RBF
            # feature map -- ridge on these = approximate kernel ridge
            Zz = (Z - rff_mu) / rff_sd
            return (math.sqrt(2.0 / rff_W.shape[1])
                    * np.cos(Zz @ rff_W + rff_b)).astype(np.float32)
        if spec.transform == "curvature":
            # acceleration: the SECOND causal difference (the dual of doppler's
            # velocity) -- where the world's motion is itself changing
            d1 = np.diff(Z, axis=0, prepend=Z[:1])
            d2 = np.diff(d1, axis=0, prepend=d1[:1])
            return np.concatenate([Z, d2], axis=1).astype(np.float32)
        if spec.transform == "lorentz_boost":
            # v19 RELATIVITY: a moving-observer mix of level x and velocity v.
            # beta = clipped row volatility (target-free, from the gauge idea);
            # calm rows (beta~0) see mostly level (-> degrades to doppler),
            # storm rows see velocity-boosted level. x'=g(x-bv), v'=g(v-bx).
            v = np.diff(Z, axis=0, prepend=Z[:1])
            beta = np.clip(np.abs(v).mean(axis=1, keepdims=True), 0.0, 0.99) * 0.6
            g = 1.0 / np.sqrt(1.0 - beta ** 2 + 1e-6)
            xp = g * (Z - beta * v)
            vp = g * (v - beta * Z)
            return np.concatenate([xp, vp], axis=1).astype(np.float32)
        if spec.transform == "lateral_line" and lat_nbr is not None:
            # near-field flow: levels + each feature's divergence from the
            # local consensus of its correlated neighbors (the eddy it sits in)
            Zz = (Z - lat_mu) / lat_sd
            consensus = Zz[:, lat_nbr].mean(axis=2)        # (n, k) neighbor mean
            return np.concatenate([Zz, Zz - consensus], axis=1).astype(np.float32)
        if spec.transform == "dual_exposure" and dual_grids is not None and q_lo is not None:
            # two eyes on the same features: rank (order) + quantize4 (magnitude)
            r = np.empty(Z.shape, dtype=np.float32)
            for j in range(Z.shape[1]):
                r[:, j] = np.searchsorted(dual_grids[:, j], Z[:, j])
            r /= np.float32(dual_grids.shape[0] + 1)
            span = np.maximum(q_hi - q_lo, 1e-6)
            code = np.clip(np.round((Z - q_lo) / span * 15), 0, 15)
            q = (q_lo + code / 15.0 * span).astype(np.float32)
            return np.concatenate([r, q], axis=1).astype(np.float32)
        if spec.transform == "fold_abs" and fold_mu is not None:
            # global folding: reflect every feature about its mean plane --
            # the EVEN-response detector (terrain that rises on both sides)
            return np.abs((Z - fold_mu) / fold_sd).astype(np.float32)
        if spec.transform == "fold_pairs" and fold_mu is not None:
            Zz = (Z - fold_mu) / fold_sd
            if fold_pair_idx:
                folded = np.stack([(Zz[:, i] - Zz[:, j]) * 0.5 for i, j in fold_pair_idx], axis=1)
                ridge_ = np.stack([np.abs(Zz[:, i] + Zz[:, j]) * 0.5 for i, j in fold_pair_idx], axis=1)
                return np.concatenate([Zz, folded, ridge_], axis=1).astype(np.float32)
            return Zz.astype(np.float32)
        if spec.transform == "rank" and rank_grids is not None:
            out = np.empty(Z.shape, dtype=np.float32)
            for j in range(Z.shape[1]):
                out[:, j] = np.searchsorted(rank_grids[:, j], Z[:, j])
            return out / np.float32(rank_grids.shape[0] + 1)
        if spec.transform == "sign_only" and sign_med is not None:
            return np.sign(Z - sign_med).astype(np.float32)
        if spec.transform == "foveated" and fov is not None:
            parts = [Z[:, : fov["nf"]]]
            if fov["comps"] is not None:
                P = (Z[:, fov["peri"]] - fov["mu_p"]) @ fov["comps"].T
                if fov["p_lo"] is not None:
                    span = np.maximum(fov["p_hi"] - fov["p_lo"], 1e-6)
                    code = np.clip(np.round((P - fov["p_lo"]) / span * 255), 0, 255)
                    P = fov["p_lo"] + code / 255.0 * span
                parts.append(P.astype(np.float32))
            if fov["bw"] is not None:
                bcol = ((Z[:, fov["back"]] - fov["bmu"]) / fov["bsd"]) @ fov["bw"]
                parts.append(bcol.reshape(-1, 1).astype(np.float32))
            return np.concatenate(parts, axis=1).astype(np.float32)
        if spec.transform == "pair_aug" and pair_keep:
            Zb = (Z[:, :pair_m] - pair_mu) / pair_sd
            extra = np.stack([_pair_op(op, Zb[:, i], Zb[:, j]) for op, i, j in pair_keep], axis=1)
            return np.concatenate([Z, extra], axis=1).astype(np.float32)
        return Z

    return idx, transform


