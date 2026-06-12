BEACONS: "BeaconField | None" = None

# (fold_signature, family[, k]) -> ranked feature indices. The signature hashes
# strided y CONTENT, so shuffled-label null runs rebuild their own rankings.
_RANK_CACHE: dict[tuple, list[int]] = {}


def _fold_sig(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple:
    step = max(1, len(X_tr) // 4096)
    yh = hashlib.sha1(np.ascontiguousarray(y_tr[::step]).tobytes()).hexdigest()[:12]
    return (X_tr.shape[0], X_tr.shape[1], round(float(X_tr[::step, 0].sum()), 4), yh)


def _family_pool(family: str, cols: list[str]) -> list[int]:
    if family == "anon":
        idx = [i for i, c in enumerate(cols) if c.startswith("X")]
    elif family == "market":
        idx = [i for i, c in enumerate(cols) if not c.startswith("X")]
    elif family == "beacon":
        # v15: fish in the dropped items' field channels (+ a few strong reals
        # so a beacon viewport is never starved when no field channel helps)
        idx = [i for i, c in enumerate(cols) if c.startswith("beacon_")]
        if not idx:
            idx = list(range(len(cols)))
    elif family == "lastN":
        xs = [i for i, c in enumerate(cols) if c.startswith("X")] or list(range(len(cols)))
        idx = xs[-CFG.LASTN_BLOCK:]
    elif family in ("head", "mid", "tail"):
        # v22: contiguous feature-ORDER blocks (the host appended its strongest
        # features last -> the trailing block often carries the cleanest alpha).
        # General, not DRW-specific: head/mid/tail are all tested; measurement
        # picks. Falls through to corr-ranking WITHIN the block.
        xs = [i for i, c in enumerate(cols) if c.startswith("X")] or list(range(len(cols)))
        b = min(CFG.POSITIONAL_BLOCK, len(xs))
        if family == "head":
            idx = xs[:b]
        elif family == "tail":
            idx = xs[-b:]
        else:
            s = max(0, (len(xs) - b) // 2)
            idx = xs[s:s + b]
    else:
        idx = list(range(len(cols)))
    return idx or list(range(len(cols)))


def _compass_rankvec(ordered: list[int], cand: list[int]) -> np.ndarray:
    """v11 compass helper: map an ordered feature list to a rank-position
    vector aligned to `cand` (position 0 = best). Lower = more-agreed-upon."""
    pos = {c: r for r, c in enumerate(ordered)}
    return np.array([pos.get(c, len(cand)) for c in cand], np.float64)


def _rank_stable(X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray, pool: list[int]) -> list[int]:
    segs = np.unique(seg_tr)
    if len(segs) < 2:
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        return [pool[i] for i in np.argsort(-c)]
    per = []
    for s in segs:
        m = seg_tr == s
        per.append(corr_vector(X_tr[m][:, pool], y_tr[m]) if m.sum() >= 50 else np.zeros(len(pool)))
    P = np.vstack(per)
    score = np.abs(P.mean(axis=0)) - P.std(axis=0)
    return [pool[i] for i in np.argsort(-score)]


def _rank_decor(X_tr: np.ndarray, ranked: list[int], k: int) -> list[int]:
    cand = ranked[: min(len(ranked), max(2 * k, 64), 320)]
    sub = X_tr[:: max(1, len(X_tr) // 20_000)][:, cand]
    sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-6)
    C = np.abs(np.corrcoef(sub, rowvar=False))
    keep: list[int] = []
    kept_pos: list[int] = []
    for i in range(len(cand)):
        if all(C[i, j] < 0.90 for j in kept_pos):
            keep.append(cand[i])
            kept_pos.append(i)
        if len(keep) >= k:
            break
    return keep or cand[:k]


def _rank_medoid(X_tr: np.ndarray, y_tr: np.ndarray, pool: list[int], threshold: float) -> list[int]:
    """DRW 1st-place style: leader-cluster features at |corr|>=threshold over
    the top-|target-corr| candidates, keep each cluster's medoid."""
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    order = np.argsort(-c)
    cand = [pool[i] for i in order[: min(CFG.MEDOID_POOL, len(pool))]]
    if len(cand) < 3:
        return cand
    sub = X_tr[:: max(1, len(X_tr) // 12_000)][:, cand]
    sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-6)
    C = np.abs(np.corrcoef(sub, rowvar=False))
    clusters: list[list[int]] = []
    for i in range(len(cand)):
        placed = False
        for members in clusters:
            if C[i, members[0]] >= threshold:
                members.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    medoid_pos: list[int] = []
    for members in clusters:
        if len(members) == 1:
            medoid_pos.append(members[0])
        else:
            sums = C[np.ix_(members, members)].sum(axis=1)
            medoid_pos.append(members[int(np.argmax(sums))])
    medoid_pos.sort()
    return [cand[p] for p in medoid_pos]


# ---------------------------------------------------------------------------
# RANKERS registry: family -> _rank_<family>(spec, X_tr, y_tr, seg_tr, pool,
# sig) -> ranked pool indices. Families with no entry corr-rank (the default).
# Adding a family = one function + one registry row (+ FAMILIES tuple entry).
# ---------------------------------------------------------------------------

def _rank_by_corr(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                  seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_stable_family(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                        seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    return _rank_stable(X_tr, y_tr, seg_tr, pool)


def _rank_medoid_family(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                        seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    return _rank_medoid(X_tr, y_tr, pool, CFG.MEDOID_THRESHOLD)


def _rank_clocks_family(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                        seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    return _rank_two_clocks(X_tr, y_tr, pool, rising_only=(spec.family == "dawn"))


def _rank_terrain(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                  seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    if ATLAS is not None:
        return ATLAS.f_rank(X_tr, pool)     # fully target-free ranking
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_weather(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                  seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # robust intersection across weather bands: rank by the WEAKEST
    # per-state |corr| -- alpha that survives storms (v9)
    if GAUGE is not None:
        wth = GAUGE.assign(X_tr)
        worst = np.full(len(pool), np.inf, np.float64)
        seen = 0
        for s in np.unique(wth):
            msk = wth == s
            if msk.sum() < 300:
                continue
            worst = np.minimum(worst, np.abs(corr_vector(X_tr[msk][:, pool], y_tr[msk])))
            seen += 1
        if seen >= 2:
            return [pool[i] for i in np.argsort(-worst)]
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        return [pool[i] for i in np.argsort(-c)]
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_pressure(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                   seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v20 MICROSTRUCTURE: the order-book twin of 'weather' -- rank features
    # by their WEAKEST |corr| across order-book pressure states (alpha that
    # survives both slack and stressed books). Same robust-intersection
    # idea, on flow/depth pressure instead of generic dispersion.
    if PRESSURE is not None:
        prs = PRESSURE.assign(X_tr)
        worst = np.full(len(pool), np.inf, np.float64)
        seen = 0
        for s in np.unique(prs):
            msk = prs == s
            if msk.sum() < 300:
                continue
            worst = np.minimum(worst, np.abs(corr_vector(X_tr[msk][:, pool], y_tr[msk])))
            seen += 1
        return ([pool[i] for i in np.argsort(-worst)] if seen >= 2
                else [pool[i] for i in np.argsort(-np.abs(corr_vector(X_tr[:, pool], y_tr)))])
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_mycelium(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                   seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # stigmergy: follow the pheromone other explorers' PROMOTED lessons
    # deposited on these columns; |corr| as scent while the net is young
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    if MYCELIUM:
        scent = np.array([MYCELIUM.get(int(i), 0.0) for i in pool], np.float64)
        if CFG.MYCELIUM_SATURATE:
            # v23: sqrt-saturate the pheromone (sublinear -- a rich column
            # cannot run away) and let |corr| genuinely co-rank, both
            # normalized to [0,1]. Breaks the v12/v19 monoculture feedback
            # loop (scent -> reuse -> more scent) while keeping mycelium a
            # useful family. corr alone still leads when the net is young.
            s_n = np.sqrt(np.maximum(scent, 0.0))
            s_n = s_n / (s_n.max() + 1e-12)
            c_n = c / (c.max() + 1e-12)
            score = 0.65 * s_n + 0.35 * c_n
        else:
            score = scent + 0.01 * c              # legacy: corr only breaks ties
    else:
        score = c
    return [pool[i] for i in np.argsort(-score)]


def _rank_shadow(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                 seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # negative space: HIGH variance, LOW |corr| -- the big quiet regions
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    v = X_tr[:: max(1, len(X_tr) // 20_000)][:, pool].var(axis=0)
    loud = v >= np.quantile(v, CFG.SHADOW_VAR_Q)
    quiet = np.where(loud, -c, -np.inf)           # among loud, quietest first
    return [pool[i] for i in np.argsort(-quiet)]


def _rank_periphery(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                    seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v10: motion in the corner of the eye -- |corr| shift between the
    # early 75% and late 25% of the fold, discounted where the mycelium
    # is already thick (everyone is fixating there anyway)
    cut = max(100, int(0.75 * len(y_tr)))
    c_early = np.abs(corr_vector(X_tr[:cut][:, pool], y_tr[:cut]))
    c_late = (np.abs(corr_vector(X_tr[cut:][:, pool], y_tr[cut:]))
              if len(y_tr) - cut >= 100 else c_early)
    shift = np.abs(c_late - c_early)
    pher = np.array([MYCELIUM.get(int(i), 0.0) for i in pool], np.float64)
    return [pool[i] for i in np.argsort(-(shift / (1.0 + 10.0 * pher)))]


def _rank_springs(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                  seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v13 persistence wells: slow geology, not fast weather -- rank by
    # lag-1 self-autocorrelation x |corr| (a spring that flows today
    # flowed yesterday too); both measured on the training fold only
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    sub = X_tr[:: max(1, len(X_tr) // 20_000)][:, pool]
    if len(sub) >= 200:
        a, b = sub[:-1], sub[1:]
        az = (a - a.mean(0)) / (a.std(0) + 1e-9)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-9)
        ac1 = np.clip((az * bz).mean(0), 0.0, None)
        return [pool[i] for i in np.argsort(-(ac1 * c))]
    return [pool[i] for i in np.argsort(-c)]


def _rank_watershed(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                    seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v13 valley specialists: how much the BEST single-terrain |corr|
    # exceeds the pooled |corr| -- the expert of ONE valley (the exact
    # complement of 'weather', which demands all-band robustness)
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    if ATLAS is not None:
        t_ids = ATLAS.assign(X_tr)
        best_t = np.zeros(len(pool), np.float64)
        seen_t = 0
        for t in np.unique(t_ids):
            m = t_ids == t
            if m.sum() < 300:
                continue
            best_t = np.maximum(best_t, np.abs(corr_vector(X_tr[m][:, pool], y_tr[m])))
            seen_t += 1
        return ([pool[i] for i in np.argsort(-(best_t - c))] if seen_t >= 2
                else [pool[i] for i in np.argsort(-c)])
    return [pool[i] for i in np.argsort(-c)]


def _rank_echo(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
               seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v13: columns still ringing with YESTERDAY'S outcome -- ranked by
    # |corr(x_t, y_{t-1})| on the training fold. The model still maps
    # x -> y; only the RANKING listens backward (no test-time y needed).
    if len(y_tr) > 300:
        c = np.abs(corr_vector(X_tr[1:][:, pool], y_tr[:-1]))
    else:
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_fault(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v16 CRACKS: rank features by the largest DISCONTINUITY in their
    # per-segment corr between ADJACENT time segments -- where the
    # feature's relationship to y FRACTURES at a regime boundary. The
    # complement of 'stable'/'springs': fault surfaces the breaks so a
    # regime-aware skill (recency, caravan, weather) can model them.
    segs = np.unique(seg_tr)
    if len(segs) >= 3:
        per = []
        for s in segs:
            m = seg_tr == s
            per.append(corr_vector(X_tr[m][:, pool], y_tr[m]) if m.sum() >= 50
                       else np.zeros(len(pool)))
        P = np.vstack(per)
        frac = np.abs(np.diff(P, axis=0)).max(axis=0)     # biggest adjacent jump
        return [pool[i] for i in np.argsort(-frac)]
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_invariant(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                    seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v19 INVARIANT FEATURE COURT (causal robustness): not "which features
    # correlate most" but "which keep their relationship to y across MANY
    # ENVIRONMENTS" -- time segments AND target-free terrain/weather states.
    # score = mean|corr| - lambda * std(corr across worlds). Distinct from
    # 'stable' (temporal only): invariance survives environment partitions,
    # the features least likely to break under regime shift.
    worlds = []
    for s in np.unique(seg_tr):
        m = seg_tr == s
        if m.sum() >= 80:
            worlds.append(corr_vector(X_tr[m][:, pool], y_tr[m]))
    for organ in (ATLAS, GAUGE):
        if organ is not None:
            try:
                ids = organ.assign(X_tr)
                for s in np.unique(ids):
                    m = ids == s
                    if m.sum() >= 80:
                        worlds.append(corr_vector(X_tr[m][:, pool], y_tr[m]))
            except Exception:
                pass
    if len(worlds) >= 3:
        W = np.vstack(worlds)
        score = np.abs(W.mean(axis=0)) - np.std(W, axis=0)   # mean signal minus cross-world instability
        return [pool[i] for i in np.argsort(-score)]
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_stabsel(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                  seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v24 STABILITY SELECTION (Meinshausen-Buhlmann): rank features by how
    # OFTEN they survive an L1 (Lasso) fit across bootstrap subsamples of
    # the training fold -- finite-sample false-discovery control, the
    # principled "which of 800 features are real". Pre-screen to the top
    # STABSEL_POOL by |corr| (fold-honest), stabilize among those, append
    # the rest. Cached per (fold, family) so it is paid once, not per lesson.
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    order0 = np.argsort(-c)
    ntop = min(CFG.STABSEL_POOL, len(pool))
    cand = [pool[i] for i in order0[:ntop]]
    rest = [pool[i] for i in order0[ntop:]]
    try:
        step = max(1, len(X_tr) // 20000)
        Xc = X_tr[::step][:, cand].astype(np.float64)
        yc = y_tr[::step].astype(np.float64)
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
        yc = yc - yc.mean()
        freq = np.zeros(len(cand), np.float64)
        rng_s = np.random.default_rng(stable_seed(CFG.SEED, "stabsel", len(cand), len(Xc)))
        nB = max(2, int(CFG.STABSEL_BOOT))
        half = max(2, len(cand) // 2)
        for _ in range(nB):
            bi = rng_s.integers(0, len(Xc), len(Xc))
            jc = rng_s.choice(len(cand), size=half, replace=False)
            try:
                m = Lasso(alpha=CFG.STABSEL_ALPHA, max_iter=300)
                m.fit(Xc[bi][:, jc], yc[bi])
                freq[jc[np.abs(m.coef_) > 1e-8]] += 1.0
            except Exception:
                pass
        freq /= nB
        return [cand[i] for i in np.argsort(-freq, kind="stable")] + rest
    except Exception:
        return [pool[i] for i in order0]


def _rank_irm(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
              seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v24 INVARIANT-RISK (IRM-flavoured) feature selection: keep features
    # whose univariate SLOPE to y is INVARIANT across ENVIRONMENTS (time
    # segments + target-free terrain/weather states). score = |mean_e b_e|
    # - std_e(b_e) - penalty*signflip*|mean_e b_e|. The slope twin of
    # 'invariant' (which uses |corr|), explicitly punishing sign flips --
    # the causally-stable signal least likely to break under regime shift.
    envs = []
    for s in np.unique(seg_tr):
        m = seg_tr == s
        if m.sum() >= 80:
            envs.append(m)
    for organ in (ATLAS, GAUGE):
        if organ is not None:
            try:
                ids = organ.assign(X_tr)
                for s in np.unique(ids):
                    m = ids == s
                    if m.sum() >= 80:
                        envs.append(m)
            except Exception:
                pass
    if len(envs) >= 3:
        slopes = []
        for m in envs:
            Xe = X_tr[m][:, pool].astype(np.float64)
            ye = y_tr[m].astype(np.float64)
            Xz = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-9)
            slopes.append((Xz * (ye - ye.mean())[:, None]).mean(0))   # univariate slope/feature
        S = np.vstack(slopes)
        mean_b = np.abs(S.mean(axis=0))
        std_b = S.std(axis=0)
        flip = np.mean(np.sign(S) != np.sign(S.mean(axis=0))[None, :], axis=0)
        score = mean_b - std_b - CFG.IRM_SIGNFLIP_PENALTY * flip * mean_b
        return [pool[i] for i in np.argsort(-score)]
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    return [pool[i] for i in np.argsort(-c)]


def _rank_phyllotaxis(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                      seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v16 SPIRALS: sunflower-seed packing maximizes coverage. Order
    # features by |corr|, then SELECT by a golden-ratio low-discrepancy
    # stride so the chosen viewport spreads across the corr spectrum
    # (strong + medium together) -- deterministic decorrelation by optimal
    # spacing, the phyllotaxis dual of greedy 'decor'.
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    by_corr = [pool[i] for i in np.argsort(-c)]
    nN = len(by_corr)
    phi = 0.6180339887498949
    order, seen = [], set()
    for i in range(nN):
        p = int((i * phi % 1.0) * nN)
        while p in seen:
            p = (p + 1) % nN
        seen.add(p)
        order.append(by_corr[p])
    return order


def _rank_compass(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                  seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v11 bird multi-sensor navigation: rank by AGREEMENT across three
    # target-free frames -- magnetic (terrain separation), sun (time
    # stability across segments), star (weather-band robustness). The
    # features all compasses point at are true north. Restricted to the
    # strongest |corr| candidates so the three rank-agreements are cheap.
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    cand = [pool[i] for i in np.argsort(-c)[: min(CFG.COMPASS_POOL, len(pool))]]
    ranks = []
    # sun compass: stability across time segments (mean - std of per-seg corr)
    ranks.append(_compass_rankvec(_rank_stable(X_tr, y_tr, seg_tr, cand), cand))
    # magnetic compass: separation between terrains (atlas f_rank), target-free
    if ATLAS is not None:
        ranks.append(_compass_rankvec(ATLAS.f_rank(X_tr, cand), cand))
    # star compass: weakest-weather-band robustness, target-free bands
    if GAUGE is not None:
        wth = GAUGE.assign(X_tr)
        worst = np.full(len(cand), np.inf, np.float64)
        seen = 0
        for s in np.unique(wth):
            m = wth == s
            if m.sum() < 300:
                continue
            worst = np.minimum(worst, np.abs(corr_vector(X_tr[m][:, cand], y_tr[m])))
            seen += 1
        if seen >= 2:
            ranks.append(_compass_rankvec([cand[i] for i in np.argsort(-worst)], cand))
    consensus = np.mean(np.vstack(ranks), axis=0) if ranks else c[np.argsort(-c)][: len(cand)]
    return [cand[i] for i in np.argsort(consensus)] + [p for p in pool if p not in set(cand)]


def _rank_sign_stability(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                         seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    # v28 SIGN-STABILITY gate (the published 4th-place private-LB recipe): a
    # feature whose corr-to-y SIGN flips across time segments is regime noise
    # however large its pooled |corr| -- the cheapest measurable form of "this
    # alpha does not survive worlds". Stable-sign features rank first (by
    # pooled |corr|), sign-flippers are demoted to the hard back (demoted by
    # measurement, never pruned). INPUT-space hardener; adds zero capacity.
    c_pool = corr_vector(X_tr[:, pool], y_tr)
    per = []
    for s in np.unique(seg_tr):
        m = seg_tr == s
        if m.sum() >= 50:
            per.append(corr_vector(X_tr[m][:, pool], y_tr[m]))
    order = np.argsort(-np.abs(c_pool))
    if len(per) < 3:
        return [pool[i] for i in order]
    P = np.vstack(per)
    flip = np.mean(np.sign(P) != np.sign(c_pool)[None, :], axis=0)
    stable = [pool[i] for i in order if flip[i] <= CFG.SIGNSTAB_MAX_FLIP]
    flippy = [pool[i] for i in order if flip[i] > CFG.SIGNSTAB_MAX_FLIP]
    return stable + flippy


def _rank_decor_family(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray,
                       seg_tr: np.ndarray, pool: list[int], sig: tuple) -> list[int]:
    base_key = (sig, "top")
    base = _RANK_CACHE.get(base_key)
    if base is None:
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        base = [pool[i] for i in np.argsort(-c)]
        _RANK_CACHE[base_key] = base
    return _rank_decor(X_tr, base, spec.k)


RANKERS: dict[str, Callable[..., list[int]]] = {
    "stable": _rank_stable_family,
    "medoid": _rank_medoid_family,
    "dawn": _rank_clocks_family,
    "both_clocks": _rank_clocks_family,
    "terrain": _rank_terrain,
    "weather": _rank_weather,
    "pressure": _rank_pressure,
    "mycelium": _rank_mycelium,
    "shadow": _rank_shadow,
    "periphery": _rank_periphery,
    "springs": _rank_springs,
    "watershed": _rank_watershed,
    "echo": _rank_echo,
    "fault": _rank_fault,
    "invariant": _rank_invariant,
    "stabsel": _rank_stabsel,
    "irm": _rank_irm,
    "sign_stability": _rank_sign_stability,
    "phyllotaxis": _rank_phyllotaxis,
    "compass": _rank_compass,
    "decor": _rank_decor_family,
}


def _ranked_for(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
                cols: list[str]) -> list[int]:
    sig = _fold_sig(X_tr, y_tr)
    cache_key = (sig, spec.family) if spec.family != "decor" else (sig, "decor", spec.k)
    hit = _RANK_CACHE.get(cache_key)
    if hit is not None:
        return hit
    pool = _family_pool(spec.family, cols)
    ranked = RANKERS.get(spec.family, _rank_by_corr)(spec, X_tr, y_tr, seg_tr, pool, sig)
    if spec.family in ("top", "anon", "stable", "lastN", "dawn", "both_clocks",
                       "springs", "watershed", "echo"):
        # v14 repellent stigmergy: poisoned columns (predator kills + trap
        # mirages deposited red pheromone here) get a SOFT positional penalty
        # -- explorers are warned away from bad ground, not just drawn to good
        if RED_MYCELIUM:
            mx = max(RED_MYCELIUM.values()) + 1e-9
            span = len(ranked)
            pos = {col: r for r, col in enumerate(ranked)}
            ranked = sorted(ranked, key=lambda col: pos[col]
                            + CFG.RED_PHEROMONE_W * span * RED_MYCELIUM.get(int(col), 0.0) / mx)
        # v10 threat-first: mirage features go to the HARD back of corr-driven
        # rankings (stable partition -- order within each group preserved)
        if TRAPS:
            ranked = [i for i in ranked if i not in TRAPS] + [i for i in ranked if i in TRAPS]
    _RANK_CACHE[cache_key] = ranked
    return ranked


