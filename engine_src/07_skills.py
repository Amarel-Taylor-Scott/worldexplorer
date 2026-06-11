# ----------------------------------------------------------------------------
# 5. Skills
# ----------------------------------------------------------------------------

def _torch_pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def _fit_torch_mlp(Zs: np.ndarray, yz: np.ndarray, cfg: HarnessConfig, seed: int,
                   linear: bool = False, lr: float = 1e-3) -> dict[str, Any]:
    """GPU-schedule net: round-robin device placement, DRW 1st-place loss
    (1-w)*MSE + w*(1 - Pearson). Deterministic per seed up to CUDA kernel
    nondeterminism (epsilon -- the seed_var probe measures it anyway).
    v17: linear=True builds a single Linear layer -- the WINNING loss on the
    WINNING model class (ridge-family). The MLP fails because it is nonlinear,
    not because the loss is wrong; this pairs the 1st-place loss with linearity."""
    dev = next_device()
    torch.manual_seed(seed)
    d_in = Zs.shape[1]
    if linear:
        net = torch.nn.Linear(d_in, 1).to(dev)        # v17: pure linear-Pearson
    else:
        layers: list[Any] = []
        drop = float(getattr(cfg, "MLP_DROPOUT", 0.0))
        for h in cfg.MLP_HIDDEN:
            layers += [torch.nn.Linear(d_in, h), torch.nn.ReLU()]
            if drop > 0:
                layers.append(torch.nn.Dropout(drop))   # v13: forget a little, on purpose
            d_in = h
        layers.append(torch.nn.Linear(d_in, 1))
        net = torch.nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    # v13 CAUSAL EARLY STOPPING: measured 0 promotions / overfit 11-56x in
    # EVERY real run -- the net trained to the bitter end. The last 20% of
    # this fit's rows becomes a time-ordered validation tail; training stops
    # when the tail's Pearson stops improving; best-epoch weights restored.
    n_all = len(yz)
    cut = int(0.8 * n_all)
    use_es = (n_all - cut) >= 256
    Xt = torch.from_numpy(np.ascontiguousarray(Zs, dtype=np.float32)).to(dev)
    yt = torch.from_numpy(np.ascontiguousarray(yz, dtype=np.float32)).to(dev)
    g = torch.Generator().manual_seed(seed)
    n = cut if use_es else n_all
    bs, w = max(256, int(cfg.MLP_BATCH)), float(cfg.MLP_PEARSON_LOSS_W)
    best_val, best_state, since = -1e9, None, 0
    net.train()
    for _ in range(cfg.MLP_MAX_ITER):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(dev)
            if len(idx) < 32:
                continue
            pred = net(Xt[idx]).squeeze(-1)
            yb = yt[idx]
            loss = (1.0 - w) * torch.mean((pred - yb) ** 2) + w * (1.0 - _torch_pearson(pred, yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if use_es:
            net.eval()
            with torch.no_grad():
                val = float(_torch_pearson(net(Xt[cut:]).squeeze(-1), yt[cut:]).item())
            net.train()
            if val > best_val + 1e-5:
                best_val, since = val, 0
                best_state = {k_: v_.detach().clone() for k_, v_ in net.state_dict().items()}
            else:
                since += 1
                if since >= int(getattr(cfg, "MLP_PATIENCE", 8)):
                    break
    if use_es and best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    del Xt, yt                       # release the big device tensors eagerly
    free_gpu_mem()
    return {"net": net, "device": dev}


def _predict_torch_mlp(pack: dict[str, Any], Zs: np.ndarray) -> np.ndarray:
    net, dev = pack["net"], pack["device"]
    outs = []
    with torch.no_grad():
        for i in range(0, len(Zs), 262_144):
            t = torch.from_numpy(np.ascontiguousarray(Zs[i:i + 262_144], dtype=np.float32)).to(dev)
            outs.append(net(t).squeeze(-1).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32) if outs else np.zeros(0, np.float32)


def _fit_gpu_swarm(Z: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
                   cfg: HarnessConfig, seed: int) -> dict[str, Any]:
    """v17 GPU RIDGE SWARM -- use the idle T4s for what actually WINS. Every
    champion in every real run is ridge-family; a single ridge is a microsecond
    CPU op, so the harness ran a handful. This fits SWARM_SCOUTS ridge models
    on random column-subsets via on-device linear algebra (torch when present,
    numpy fallback), verifies each on the inner time split, and keeps the
    survivors -- scout_lattice (the v8 champion family) at swarm scale. The
    GPU path is fully guarded: ANY device error falls back to a numpy full
    ridge, so an OOM on the 11h run degrades gracefully and never crashes
    (a crash = lost submission = the catastrophe the shipping reserve guards)."""
    n, d = Z.shape
    mu = Z.mean(0, keepdims=True).astype(np.float32)
    sd = (Z.std(0, keepdims=True) + 1e-6).astype(np.float32)
    Zs = ((Z - mu) / sd).astype(np.float32)
    in_tr, in_va = _inner_time_split(seg_tr, seed)
    rng = np.random.default_rng(seed)
    s = max(2, d // 3 if d >= 6 else d)
    M, alpha = int(cfg.SWARM_SCOUTS), float(cfg.SWARM_ALPHA)
    tr_rows, va_rows = np.where(in_tr)[0], np.where(in_va)[0]
    yv = y_tr[in_va] if in_va.sum() >= 50 else None
    scouts: list[tuple[np.ndarray, np.ndarray, float]] = []
    try:
        if HAVE_TORCH:
            dev = next_device()                       # 'cpu' when no CUDA -- still vectorized
            Zt = torch.from_numpy(np.ascontiguousarray(Zs)).to(dev)
            yt = torch.from_numpy((y_tr - y_tr.mean()).astype(np.float32)).to(dev)
            Im = torch.eye(s, device=dev)
            for m in range(M):
                rows = tr_rows[m % 3::3] if len(tr_rows) >= 300 else tr_rows
                if len(rows) < 100:
                    continue
                sub = np.sort(rng.choice(d, size=s, replace=False))
                subt = torch.from_numpy(sub.astype(np.int64)).to(dev)
                rowt = torch.from_numpy(rows.astype(np.int64)).to(dev)
                Xs = Zt.index_select(0, rowt).index_select(1, subt)
                ys = yt.index_select(0, rowt)
                G = Xs.transpose(0, 1) @ Xs + alpha * Im
                b = Xs.transpose(0, 1) @ ys
                beta = torch.linalg.solve(G, b).detach().cpu().numpy().astype(np.float32)
                v = pearson(yv, Zs[va_rows][:, sub] @ beta) if yv is not None else 1.0
                if v >= cfg.SCOUT_ACCEPT:
                    scouts.append((sub, beta, max(float(v), 1e-3)))
            del Zt, yt
            free_gpu_mem()
        else:
            for m in range(M):
                rows = tr_rows[m % 3::3] if len(tr_rows) >= 300 else tr_rows
                if len(rows) < 100:
                    continue
                sub = np.sort(rng.choice(d, size=s, replace=False))
                Xs = Zs[rows][:, sub].astype(np.float64)
                ys = (y_tr[rows] - y_tr[rows].mean()).astype(np.float64)
                beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(s), Xs.T @ ys).astype(np.float32)
                v = pearson(yv, Zs[va_rows][:, sub] @ beta) if yv is not None else 1.0
                if v >= cfg.SCOUT_ACCEPT:
                    scouts.append((sub, beta, max(float(v), 1e-3)))
    except Exception:
        scouts = []                                   # any device failure -> fallback below
    fb = None
    if not scouts:                                    # nothing survived (or GPU failed): full ridge
        Zd = Zs.astype(np.float64)
        try:
            fb = np.linalg.solve(Zd.T @ Zd + alpha * np.eye(d),
                                 Zd.T @ (y_tr - y_tr.mean())).astype(np.float32)
        except Exception:
            fb = np.zeros(d, np.float32)
    return {"mu": mu, "sd": sd, "scouts": scouts, "fb": fb}


def _predict_gpu_swarm(state: dict[str, Any], Z: np.ndarray) -> np.ndarray:
    Zs = ((Z - state["mu"]) / state["sd"]).astype(np.float64)
    if state["scouts"]:
        tot = sum(w for _, _, w in state["scouts"])
        out = np.zeros(len(Z), np.float64)
        for sub, beta, w in state["scouts"]:
            p = Zs[:, sub] @ beta.astype(np.float64)
            out += (w / tot) * (p - p.mean()) / (p.std() + 1e-9)
        return out.astype(np.float32)
    return (Zs @ state["fb"].astype(np.float64)).astype(np.float32)


SKILL_REGISTRY: dict[str, dict[str, Any]] = {
    "single_factor":   {"kind": "uni",      "cost": 1, "stage": 0, "needs_identity": True,  "label": "naive_single_association"},
    "bin_association": {"kind": "binmean",  "cost": 1, "stage": 0, "needs_identity": True,  "label": "naive_bin_association"},
    "majority_vote":   {"kind": "vote",     "cost": 1, "stage": 0, "needs_identity": False, "label": "unweighted_sign_committee"},
    "theil_sen":       {"kind": "tsen",     "cost": 1, "stage": 0, "needs_identity": True,  "label": "median_slope_single_factor"},
    "recency_linear":  {"kind": "recency",  "cost": 2, "stage": 1, "needs_identity": False, "label": "modeling_by_temporal_analogy"},
    "local_interp":    {"kind": "knn",      "cost": 4, "stage": 1, "needs_identity": False, "label": "modeling_by_interpolation"},  # never promoted in any real run
    "linear_assoc":    {"kind": "ridge",    "cost": 2, "stage": 2, "needs_identity": False, "label": "linear_association"},
    "bagged_linear":   {"kind": "ridgebag", "cost": 2, "stage": 2, "needs_identity": False, "label": "strided_subset_bagging"},
    "residual_ladder": {"kind": "ladder",   "cost": 2, "stage": 2, "needs_identity": False, "label": "greedy_residual_extrapolation"},
    "nonlinear_assoc": {"kind": "hgb",      "cost": 4, "stage": 3, "needs_identity": False, "label": "nonlinear_association"},
    "mlp_assoc":       {"kind": "mlp",      "cost": 5, "stage": 3, "needs_identity": False, "label": "small_mlp_association"},
    # v8 topography skills -- sub-models of the space itself
    "codebook":        {"kind": "codebook", "cost": 2, "stage": 2, "needs_identity": False, "label": "vector_quantized_space_lut"},
    "terrain_router":  {"kind": "router",   "cost": 3, "stage": 2, "needs_identity": False, "label": "per_terrain_expert_moe"},
    "steepness_gate":  {"kind": "steep",    "cost": 3, "stage": 2, "needs_identity": False, "label": "magnitude_gated_direction"},
    "scout_lattice":   {"kind": "scout",    "cost": 3, "stage": 2, "needs_identity": False, "label": "speculative_path_lattice"},
    # v9 ecology skills -- models that MOVE through the world
    "relay_caravan":   {"kind": "relay",    "cost": 3, "stage": 2, "needs_identity": False, "label": "drifting_coefficient_caravan"},
    "swell_rider":     {"kind": "swell",    "cost": 2, "stage": 2, "needs_identity": False, "label": "smoothed_label_swell"},
    # v22: plain ordinary least squares (NO shrinkage) -- the model class the
    # 11th-place 0.111 solution used on a small trailing feature block. General:
    # gated by the same overfit-ratio door (huge/noisy viewports get culled),
    # so it only ships where an unregularized fit genuinely generalizes.
    "linear_ols":      {"kind": "ols",      "cost": 2, "stage": 2, "needs_identity": False, "label": "plain_ordinary_least_squares"},
    # v23 simplicity levers -- robust + sparse linear (push toward the 0.111
    # recipe; gated by the same overfit door, judged by the robust selector)
    "huber_linear":    {"kind": "huber",    "cost": 2, "stage": 2, "needs_identity": False, "label": "huber_robust_linear"},
    "elastic_net":     {"kind": "elastic",  "cost": 3, "stage": 2, "needs_identity": False, "label": "elasticnet_sparse_linear"},
    # v24 levers -- recency-weighted + adversarial covariate-shift reweighted linear
    "recency_weighted":{"kind": "recw",     "cost": 2, "stage": 2, "needs_identity": False, "label": "exp_recency_weighted_ridge"},
    "shift_linear":    {"kind": "shiftlin", "cost": 3, "stage": 2, "needs_identity": False, "label": "covariate_shift_reweighted_ridge"},
    # v25 DRW plateau lever -- the published 11th-place 0.111 recipe as a skill
    "greedy_ols":      {"kind": "greedyols", "cost": 3, "stage": 2, "needs_identity": False, "label": "greedy_suffix_selection_ols"},
    # v13 sensorium skills -- new gaits over the same honest ground
    "terrace":         {"kind": "terrace",  "cost": 2, "stage": 2, "needs_identity": False, "label": "score_terraced_calibration"},
    "rapids":          {"kind": "rapids",   "cost": 3, "stage": 2, "needs_identity": False, "label": "residual_gated_two_stage"},
    # v17 silicon skills -- use the idle T4s for the WINNING (ridge) family
    "linear_pearson":  {"kind": "linpearson", "cost": 3, "stage": 2, "needs_identity": False, "label": "gpu_linear_first_place_loss"},
    "gpu_ridge_swarm": {"kind": "gpuswarm",   "cost": 3, "stage": 2, "needs_identity": False, "label": "gpu_batched_ridge_swarm"},
    # v19 cognitive skills -- weird learning machines, ridge-safe readouts
    "echo_state_ridge": {"kind": "reservoir", "cost": 3, "stage": 2, "needs_identity": False, "label": "reservoir_echo_state_ridge"},
    # v27 -- linear diversity LEARNED FROM the top public DRW solutions (studied
    # via Kaggle API, methodology only). All low-complexity + overfit-resistant,
    # so the v27 governor favours them where capacity decays; judged by the same
    # doors + robust selector. PLS = supervised SVD-regression (taylorsamarel's
    # zoo); bayes_ridge/ard = sparse Bayesian linear (the pasted ARD+XGB recipe).
    "pls":            {"kind": "pls",        "cost": 2, "stage": 2, "needs_identity": False, "label": "partial_least_squares_supervised_svd"},
    "bayes_ridge":    {"kind": "bayesridge", "cost": 2, "stage": 2, "needs_identity": False, "label": "bayesian_ridge_evidence"},
    "ard_linear":     {"kind": "ard",        "cost": 3, "stage": 2, "needs_identity": False, "label": "ard_sparse_bayesian_linear"},
}
if GBDT_BACKEND is not None:
    SKILL_REGISTRY["gbdt_lib"] = {"kind": "gbdt", "cost": 4, "stage": 3, "needs_identity": False,
                                  "label": f"gbdt_{GBDT_BACKEND}"}


def lesson_lane(skill: str) -> str:
    """Heterogeneous lane: 'gpu' for skills whose fit actually runs on CUDA,
    'cpu' for everything else (and for everything when no CUDA is present).
    v8: torch MLP ONLY -- the xgboost-cuda lane caused the fatal thrust abort."""
    if N_GPUS > 0 and SKILL_REGISTRY[skill]["kind"] in ("mlp", "linpearson", "gpuswarm"):
        return "gpu"        # v17: the linear-Pearson net and the ridge swarm run on CUDA too
    return "cpu"


def is_stochastic(skill: str, spec: ViewportSpec) -> bool:
    return (spec.transform in ("rand_proj", "signed_hadamard")
            or skill in ("nonlinear_assoc", "mlp_assoc", "gbdt_lib", "codebook", "scout_lattice",
                         "linear_pearson", "gpu_ridge_swarm", "echo_state_ridge"))


def _inner_time_split(seg_tr: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    segs = np.unique(seg_tr)
    if len(segs) >= 2:
        val = seg_tr == segs.max()
        if 50 <= val.sum() <= 0.6 * len(seg_tr):
            return ~val, val
    rng = np.random.default_rng(seed)
    val = rng.random(len(seg_tr)) < 0.25
    return ~val, val


def fit_skill(skill: str, spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
              cols: list[str], rng: np.random.Generator, cfg: HarnessConfig, seed: int) -> dict[str, Any]:
    kind = SKILL_REGISTRY[skill]["kind"]
    idx, transform = build_viewport(spec, X_tr, y_tr, seg_tr, cols, rng)
    state: dict[str, Any] = {"kind": kind, "skill": skill, "idx": idx, "tf": transform}
    Z = transform(X_tr[:, idx])
    in_tr, in_va = _inner_time_split(seg_tr, seed)

    if kind == "uni":
        best_j, best_sc = 0, -1e9
        for j in range(min(5, Z.shape[1])):
            x = Z[:, j].astype(np.float64)
            var = float(np.var(x[in_tr]))
            if var <= 1e-12:
                continue
            m0 = float(np.mean(x[in_tr]))
            slope = float(np.mean((x[in_tr] - m0) * (y_tr[in_tr] - y_tr[in_tr].mean())) / var)
            sc = pearson(y_tr[in_va], slope * (x[in_va] - m0)) if in_va.sum() >= 50 else abs(slope) * math.sqrt(var)
            if sc > best_sc:
                best_j, best_sc = j, sc
        x = Z[:, best_j].astype(np.float64)
        var = float(np.var(x)) + 1e-12
        m0 = float(np.mean(x))
        slope = float(np.mean((x - m0) * (y_tr - y_tr.mean())) / var)
        state.update(j=best_j, slope=slope, mu=m0)
        return state

    if kind == "binmean":
        best_j, best_sc = 0, -1e9
        for j in range(min(3, Z.shape[1])):
            x = Z[:, j]
            edges = np.unique(np.quantile(x[in_tr], np.linspace(0, 1, 17)[1:-1])).astype(np.float32)
            codes = np.searchsorted(edges, x[in_tr], side="right")
            means = np.full(len(edges) + 1, float(np.mean(y_tr[in_tr])), np.float32)
            for b in range(len(edges) + 1):
                m = codes == b
                if m.any():
                    means[b] = float(np.mean(y_tr[in_tr][m]))
            if in_va.sum() >= 50:
                pv = means[np.clip(np.searchsorted(edges, x[in_va], side="right"), 0, len(means) - 1)]
                sc = pearson(y_tr[in_va], pv)
            else:
                sc = float(np.std(means))
            if sc > best_sc:
                best_j, best_sc = j, sc
        x = Z[:, best_j]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 17)[1:-1])).astype(np.float32)
        codes = np.searchsorted(edges, x, side="right")
        means = np.full(len(edges) + 1, float(np.mean(y_tr)), np.float32)
        for b in range(len(edges) + 1):
            m = codes == b
            if m.any():
                means[b] = float(np.mean(y_tr[m]))
        state.update(j=best_j, edges=edges, means=means)
        return state

    if kind == "vote":
        # The purest primitive: every feature casts sign(fold_corr)*sign(x-median),
        # prediction is the unweighted mean vote. No fitted weights at all --
        # equal-weighting under estimation noise (the 1/N result).
        med = np.median(Z, axis=0).astype(np.float32)
        c = corr_vector(Z, y_tr)
        signs = np.sign(c).astype(np.float32)
        signs[signs == 0] = 1.0
        state.update(med=med, signs=signs)
        return state

    if kind == "tsen":
        def ts_slope(xv: np.ndarray, yv: np.ndarray) -> float:
            m = min(4000, max(200, len(xv)))
            a = rng.integers(0, len(xv), m)
            b = rng.integers(0, len(xv), m)
            dx = xv[a] - xv[b]
            ok = np.abs(dx) > 1e-9
            if ok.sum() < 50:
                return 0.0
            return float(np.median((yv[a] - yv[b])[ok] / dx[ok]))

        best_j, best_sc = 0, -1e9
        for j in range(min(5, Z.shape[1])):
            x = Z[:, j].astype(np.float64)
            sl = ts_slope(x[in_tr], y_tr[in_tr].astype(np.float64))
            m0 = float(np.median(x[in_tr]))
            sc = pearson(y_tr[in_va], sl * (x[in_va] - m0)) if in_va.sum() >= 50 else abs(sl)
            if sc > best_sc:
                best_j, best_sc = j, sc
        x = Z[:, best_j].astype(np.float64)
        state.update(j=best_j, slope=ts_slope(x, y_tr.astype(np.float64)), mu=float(np.median(x)))
        return state

    if kind == "recency":
        order_pos = np.arange(len(y_tr))
        best_f, best_sc = 1.0, -1e9
        for f in (0.15, 0.30, 0.50, 1.00):
            cut = order_pos >= int((1 - f) * len(y_tr))
            fit_mask = cut & in_tr
            if fit_mask.sum() < 200:
                continue
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            model.fit(Z[fit_mask], y_tr[fit_mask])
            sc = pearson(y_tr[in_va], model.predict(Z[in_va])) if in_va.sum() >= 50 else 0.0
            if sc > best_sc:
                best_f, best_sc = f, sc
        cut = order_pos >= int((1 - best_f) * len(y_tr))
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        model.fit(Z[cut], y_tr[cut])
        state.update(model=model, fraction=best_f)
        return state

    if kind == "knn":
        d = min(Z.shape[1], 24)
        Zk = Z[:, :d]
        bank = np.arange(len(Zk))[:: max(1, len(Zk) // cfg.KNN_BANK)]
        scaler = StandardScaler().fit(Zk[bank])
        Zb = scaler.transform(Zk[bank])
        best_k, best_sc = 50, -1e9
        iv = np.where(in_va)[0][:: max(1, int(in_va.sum()) // 6000)]
        for k in (15, 50, 150):
            kk = min(k, max(2, len(bank) - 1))
            m = KNeighborsRegressor(n_neighbors=kk, weights="distance", n_jobs=-1)
            m.fit(Zb, y_tr[bank])
            sc = pearson(y_tr[iv], m.predict(scaler.transform(Zk[iv]))) if len(iv) >= 50 else 0.0
            if sc > best_sc:
                best_k, best_sc = kk, sc
        model = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)
        model.fit(Zb, y_tr[bank])
        state.update(model=model, scaler=scaler, d=d)
        return state

    if kind == "ridge":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 10)))
        model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "ols":
        # v22: plain OLS, NO shrinkage (the 11th-place 0.111 model class).
        # General -- the overfit-ratio door culls it on noisy/huge viewports, so
        # it only survives where an unregularized fit genuinely generalizes
        # (clean, small feature sets like the trailing positional block).
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "huber":
        # v23 HUBER robust linear: L2 on the bulk, L1 on the tails, so the
        # fat-tailed crypto outliers the row-influence court already flags
        # cannot dominate the fit. Robust-OLS for heavy-tailed labels; degrades
        # to RidgeCV if the IRLS solve fails to converge.
        try:
            model = make_pipeline(StandardScaler(),
                                  HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=300))
            model.fit(Z, y_tr)
        except Exception:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "elastic":
        # v23 ELASTICNET: L1+L2 with self-tuned l1_ratio + alpha, so feature
        # selection happens INSIDE the fit (sparse -- keep the few real signals,
        # zero the noise). The principled "which of 800 features are real"
        # lever, complementary to corr-ranking; l1_ratio grid spans ridge<->lasso.
        try:
            model = make_pipeline(StandardScaler(),
                                  ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99], n_alphas=20,
                                               cv=3, max_iter=2000, random_state=seed, n_jobs=-1))
            model.fit(Z, y_tr)
        except Exception:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "pls":
        # v27 (taylorsamarel's zoo): Partial Least Squares -- SUPERVISED SVD
        # regression. Projects features onto components MAXIMALLY covariant with y
        # (the supervised dual of unsupervised PCA), then linear. Signal-dense +
        # overfit-resistant on the collinear crypto-microstructure feature set.
        nc = int(min(16, max(2, Z.shape[1] - 1)))
        try:
            model = make_pipeline(StandardScaler(), PLSRegression(n_components=nc, scale=False))
            model.fit(Z, y_tr)
        except Exception:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "bayesridge":
        # v27: Bayesian ridge -- ridge whose L2 strength is learned by EVIDENCE
        # maximization (no CV grid). Overfit-resistant, deterministic, cheap.
        try:
            model = make_pipeline(StandardScaler(), BayesianRidge())
            model.fit(Z, y_tr)
        except Exception:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "ard":
        # v27 (the pasted top ARD+XGB recipe): ARD = sparse BAYESIAN linear with
        # Automatic Relevance Determination -- a per-feature precision prunes
        # irrelevant features (built-in selection, overfit-resistant). ARD is
        # O(k^3) per EM step, so fit on a ROW SUBSAMPLE (the recipe also fit on a
        # data slice); predict uses the full model. Degrades to BayesianRidge/Ridge.
        rows = np.arange(len(y_tr))[:: max(1, len(y_tr) // 50000)]
        try:
            model = make_pipeline(StandardScaler(), ARDRegression())
            model.fit(Z[rows], y_tr[rows])
        except Exception:
            try:
                model = make_pipeline(StandardScaler(), BayesianRidge())
                model.fit(Z[rows], y_tr[rows])
            except Exception:
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
                model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "recw":
        # v24: continuous exponential RECENCY-weighted ridge -- weight each row
        # by exp(-lam*age), age = fraction back from the newest training row.
        # The continuous form of recency_linear's hard cutoff; lam (decay) is
        # chosen on the inner time split. lam=0 -> plain ridge (a no-op).
        n = len(y_tr)
        age = ((n - 1 - np.arange(n)) / max(1, n - 1)).astype(np.float64)
        best_lam, best_sc, best_model = 0.0, -1e9, None
        for lam in (0.0, 1.0, 3.0, 8.0):
            wt = np.exp(-lam * age)
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            swk = m.steps[-1][0] + "__sample_weight"
            if in_va.sum() >= 50:
                m.fit(Z[in_tr], y_tr[in_tr], **{swk: wt[in_tr]})
                sc = pearson(y_tr[in_va], m.predict(Z[in_va]))
            else:
                m.fit(Z, y_tr, **{swk: wt})
                sc = 0.0
            if sc > best_sc or best_model is None:
                best_lam, best_sc, best_model = lam, sc, m
        wt = np.exp(-best_lam * age)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        model.fit(Z, y_tr, **{model.steps[-1][0] + "__sample_weight": wt})
        state.update(model=model, recency_lam=float(best_lam))
        return state

    if kind == "shiftlin":
        # v24 ADVERSARIAL covariate-shift reweighting: a classifier learns to
        # tell EARLY from LATE rows of this fold by their FEATURES alone
        # (target-free -- the label is row position, not y). Each training row
        # is weighted by its odds of looking LATE (a density ratio toward the
        # future the test set lives in), so rows resembling the recent regime
        # count more. Distinct from recency: weights by resemblance, not position.
        n = len(y_tr)
        wt = np.ones(n, np.float64)
        try:
            late = (np.arange(n) >= int(0.6 * n)).astype(np.int32)
            rows = np.arange(n)[:: max(1, n // 40000)]
            if len(np.unique(late[rows])) > 1:
                clf = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                                                     learning_rate=0.08, random_state=seed)
                clf.fit(Z[rows], late[rows])
                p = np.clip(clf.predict_proba(Z)[:, 1], 1e-4, 1 - 1e-4)
                wt = np.clip(p / (1.0 - p), 1.0 / cfg.SHIFT_CLIP, cfg.SHIFT_CLIP)
                wt = wt / wt.mean()
        except Exception:
            wt = np.ones(n, np.float64)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        model.fit(Z, y_tr, **{model.steps[-1][0] + "__sample_weight": wt})
        state["model"] = model
        return state

    if kind == "greedyols":
        # v25 11th-place PLATEAU LEVER: greedy SUFFIX feature selection (walk
        # the viewport features in order, keep each only if it improves the
        # inner out-of-sample corr) + plain OLS. The published 0.111 recipe --
        # "small trailing block + OLS + walk-forward feature selection" -- as a
        # skill: general (any family's order), strongest on the `tail` block.
        # Plain LinearRegression (no shrinkage); the overfit door + robust
        # selector keep it honest, so it ships only where it genuinely wins.
        d = Z.shape[1]
        cap = min(d, 48)                        # candidate features the scan considers
        tr_rows = np.where(in_tr)[0]
        va_rows = np.where(in_va)[0]
        # the SELECTION scan runs on a SMALL row sample (the feature ranking is
        # cheap + stable); the FINAL OLS below fits on ALL rows. This keeps
        # greedy_ols affordable even when the robust/forensic layer refits it
        # across many partitions -- the cost guard that makes it run-ready.
        if len(tr_rows) > 6000:
            tr_rows = tr_rows[:: max(1, len(tr_rows) // 6000)]
        if len(va_rows) > 4000:
            va_rows = va_rows[:: max(1, len(va_rows) // 4000)]
        Ztr, ytr_s = Z[tr_rows][:, :cap], y_tr[tr_rows]
        Zva, yva_s = Z[va_rows][:, :cap], y_tr[va_rows]
        selected, best = [], -1e9
        if len(va_rows) >= 50:
            for j in range(cap):
                cand = selected + [j]
                m = make_pipeline(StandardScaler(), LinearRegression())
                m.fit(Ztr[:, cand], ytr_s)
                sc = pearson(yva_s, m.predict(Zva[:, cand]))
                if sc > best + 1e-4:           # keep the feature only if it helps OOS
                    best, selected = sc, cand
        if not selected:
            selected = list(range(min(d, 30)))
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(Z[:, selected], y_tr)         # final fit uses ALL rows + selected cols
        state.update(model=model, sel=selected, n_selected=len(selected))
        return state

    if kind == "linpearson":
        # v17: GPU linear layer trained with the DRW 1st-place Pearson loss
        # (the WINNING loss on the WINNING model class). Reuses the v13 causal
        # early-stop machinery; falls back to RidgeCV when torch is absent.
        rows = np.arange(len(y_tr))[:: max(1, len(y_tr) // cfg.MLP_MAX_ROWS)]
        if HAVE_TORCH and len(rows) >= 300:
            scaler = StandardScaler().fit(Z[rows])
            Zsc = scaler.transform(Z[rows])
            ymu, ysd = float(np.mean(y_tr[rows])), float(np.std(y_tr[rows])) + 1e-9
            yz = (y_tr[rows] - ymu) / ysd
            state.update(model=_fit_torch_mlp(np.asarray(Zsc, np.float32), np.asarray(yz, np.float32),
                                              cfg, seed, linear=True, lr=1e-2),
                         scaler=scaler, torch_net=True)
            return state
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 10)))
        model.fit(Z, y_tr)
        state.update(model=model, torch_net=False)
        return state

    if kind == "gpuswarm":
        state.update(_fit_gpu_swarm(Z, y_tr, seg_tr, cfg, seed))
        return state

    if kind == "reservoir":
        # v19 RESERVOIR COMPUTING (echo state network): a random recurrent
        # machine with a LINEAR ridge readout -- cheap temporal memory for a
        # time-ordered noisy system. The reservoir weights are random+frozen
        # (never trained); only the ridge readout is fit, so it cannot overfit
        # like an RNN. Hidden state h_t = (1-a)h_{t-1} + a*tanh(W_in z_t + W h_{t-1}).
        rng_r = np.random.default_rng(seed)
        d = Z.shape[1]
        res = int(min(cfg.RESERVOIR_SIZE, 128))
        a = float(cfg.RESERVOIR_LEAK)
        mu0 = Z.mean(axis=0, keepdims=True).astype(np.float32)
        sd0 = (Z.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        W_in = (rng_r.normal(size=(d, res)).astype(np.float32) / math.sqrt(d))
        W = rng_r.normal(size=(res, res)).astype(np.float32)
        # scale recurrent matrix to the target spectral radius (echo-state property)
        try:
            sr = float(np.max(np.abs(np.linalg.eigvals(W))))
            W = (W * (cfg.RESERVOIR_SPECTRAL / (sr + 1e-9))).astype(np.float32)
        except Exception:
            W = (W * 0.1).astype(np.float32)

        def run_reservoir(Zin: np.ndarray) -> np.ndarray:
            Zs = ((Zin - mu0) / sd0).astype(np.float32)
            h = np.zeros(res, np.float32)
            H = np.empty((len(Zs), res), np.float32)
            drive = Zs @ W_in
            for t in range(len(Zs)):
                h = (1.0 - a) * h + a * np.tanh(drive[t] + h @ W)
                H[t] = h
            return H

        H = run_reservoir(Z)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        ridge.fit(H, y_tr)
        state.update(model=ridge, W_in=W_in, W=W, res_mu=mu0, res_sd=sd0,
                     leak=a, res_size=res)
        return state

    if kind == "ridgebag":
        B = cfg.BAG_SUBSETS
        models = []
        for b in range(B):
            rows = np.arange(b, len(y_tr), B)
            if len(rows) < 100:
                continue
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            m.fit(Z[rows], y_tr[rows])
            models.append(m)
        if not models:
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
            m.fit(Z, y_tr)
            models = [m]
        state.update(models=models)
        return state

    if kind == "ladder":
        mu0 = Z.mean(axis=0, keepdims=True)
        sd0 = Z.std(axis=0, keepdims=True) + 1e-6
        Zs = (Z - mu0) / sd0

        def run_path(rows: np.ndarray, val_rows: np.ndarray | None) -> tuple[np.ndarray, int]:
            beta = np.zeros(Zs.shape[1], np.float64)
            pred = np.full(int(rows.sum()), float(np.mean(y_tr[rows])), np.float64)
            best_L, best_sc = 0, -1e9
            Xr, yr = Zs[rows].astype(np.float64), y_tr[rows].astype(np.float64)
            Xv = Zs[val_rows].astype(np.float64) if val_rows is not None else None
            for L in range(1, cfg.LADDER_MAX_ROUNDS + 1):
                r = yr - pred
                cov = Xr.T @ r
                j = int(np.argmax(np.abs(cov)))
                denom = float(Xr[:, j] @ Xr[:, j]) + 1e-9
                step = cfg.LADDER_SHRINK * float(cov[j]) / denom
                beta[j] += step
                pred += step * Xr[:, j]
                if Xv is not None:
                    sc = pearson(y_tr[val_rows], Xv @ beta)
                    if sc > best_sc:
                        best_sc, best_L = sc, L
                    if L - best_L > 15:
                        break
            return beta, (best_L if Xv is not None and best_L > 0 else cfg.LADDER_MAX_ROUNDS // 2)

        _, L_star = run_path(in_tr, in_va if in_va.sum() >= 50 else None)
        beta = np.zeros(Zs.shape[1], np.float64)
        pred = np.full(len(y_tr), float(np.mean(y_tr)), np.float64)
        Xall, yall = Zs.astype(np.float64), y_tr.astype(np.float64)
        for _ in range(L_star):
            r = yall - pred
            cov = Xall.T @ r
            j = int(np.argmax(np.abs(cov)))
            denom = float(Xall[:, j] @ Xall[:, j]) + 1e-9
            step = cfg.LADDER_SHRINK * float(cov[j]) / denom
            beta[j] += step
            pred += step * Xall[:, j]
        state.update(beta=beta.astype(np.float32), mu=mu0, sd=sd0, rounds=int(L_star))
        return state

    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=cfg.HGB_ITERS, learning_rate=cfg.HGB_LR,
            max_leaf_nodes=cfg.HGB_LEAVES, min_samples_leaf=cfg.HGB_MIN_LEAF,
            l2_regularization=cfg.HGB_L2, random_state=seed, early_stopping=False)
        model.fit(Z, y_tr)
        state["model"] = model
        return state

    if kind == "mlp":
        rows = np.arange(len(y_tr))[:: max(1, len(y_tr) // cfg.MLP_MAX_ROWS)]
        scaler = StandardScaler().fit(Z[rows])
        Zs = scaler.transform(Z[rows])
        ymu = float(np.mean(y_tr[rows]))
        ysd = float(np.std(y_tr[rows])) + 1e-9
        yz = (y_tr[rows] - ymu) / ysd
        if HAVE_TORCH:
            state.update(model=_fit_torch_mlp(np.asarray(Zs, np.float32),
                                              np.asarray(yz, np.float32), cfg, seed),
                         scaler=scaler, torch_mlp=True)
            return state
        model = MLPRegressor(hidden_layer_sizes=cfg.MLP_HIDDEN, activation="relu", solver="adam",
                             alpha=1e-4, batch_size=min(1024, int(cfg.MLP_BATCH)), learning_rate_init=1e-3,
                             max_iter=cfg.MLP_MAX_ITER, shuffle=True, random_state=seed,
                             tol=1e-5, early_stopping=True, validation_fraction=0.2,
                             n_iter_no_change=max(3, int(getattr(cfg, "MLP_PATIENCE", 8)) // 2))
        model.fit(Zs, yz)
        state.update(model=model, scaler=scaler)
        return state

    if kind == "codebook":
        # Vector quantization of SPACE itself: bin_association generalized to
        # k dimensions. Prediction is a pure lookup table -- per-centroid mean
        # of y, shrunk toward the global mean by CODEBOOK_SHRINK virtual rows.
        mu0 = Z.mean(axis=0, keepdims=True).astype(np.float32)
        sd0 = (Z.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        Zs = (Z - mu0) / sd0
        rows = np.arange(len(Zs))[:: max(1, len(Zs) // 40_000)]
        nc = int(min(cfg.CODEBOOK_SIZE, max(8, len(rows) // 50)))
        km = MiniBatchKMeans(n_clusters=nc, random_state=seed, n_init=3,
                             batch_size=4096).fit(Zs[rows])
        code = km.predict(Zs)
        gmean = float(np.mean(y_tr))
        table = np.full(nc, gmean, np.float64)
        for c in range(nc):
            m = code == c
            n_c = int(m.sum())
            if n_c:
                table[c] = (float(y_tr[m].sum()) + cfg.CODEBOOK_SHRINK * gmean) / (n_c + cfg.CODEBOOK_SHRINK)
        state.update(km=km, table=table.astype(np.float32), mu0=mu0, sd0=sd0)
        return state

    if kind == "router":
        # Regime mixture-of-experts: per-terrain ridge experts shrunk toward a
        # global ridge by terrain population. Falls back to the global model
        # when the atlas is absent (degrades to linear_assoc, never breaks).
        global_m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        global_m.fit(Z, y_tr)
        experts: dict[int, tuple[Any, int]] = {}
        if ATLAS is not None:
            t_ids = ATLAS.assign(X_tr)
            for c in np.unique(t_ids):
                m = t_ids == c
                n_c = int(m.sum())
                if n_c >= cfg.TERRAIN_EXPERT_MIN:
                    e = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 6)))
                    e.fit(Z[m], y_tr[m])
                    experts[int(c)] = (e, n_c)
        state.update(model=global_m, experts=experts, shrink_n=float(cfg.TERRAIN_EXPERT_SHRINK))
        return state

    if kind == "steep":
        # Two sub-models: WHERE the trail points (direction ridge on y) and
        # HOW STEEP the ground is (magnitude ridge on |y|). The bet is scaled
        # by predicted steepness -- bigger where the mountain face is steep.
        dir_m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        dir_m.fit(Z, y_tr)
        steep_m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        steep_m.fit(Z, np.abs(y_tr))
        sp = np.asarray(steep_m.predict(Z), np.float64)
        state.update(model=dir_m, steep=steep_m,
                     smu=float(np.mean(sp)), ssd=float(np.std(sp)) + 1e-9)
        return state

    if kind == "scout":
        # Speculative path lattice: M cheap scouts each draft a path through a
        # random third of the viewport on a third of the rows; the inner-split
        # verifier accepts scouts whose drafted path holds (corr >= SCOUT_ACCEPT);
        # accepted scouts merge corr-weighted. No scout survives -> fall back to
        # the single large viewport (full-Z ridge). Small view when stable,
        # large view when fragile.
        fallback = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        fallback.fit(Z, y_tr)
        scouts: list[tuple[np.ndarray, Any, float]] = []
        if in_va.sum() >= 50 and Z.shape[1] >= 4:
            d = Z.shape[1]
            tr_rows = np.where(in_tr)[0]
            for s_i in range(cfg.SCOUT_COUNT):
                sub = np.sort(rng.choice(d, size=max(2, d // 3), replace=False))
                rows = tr_rows[s_i % 3 :: 3]
                if len(rows) < 100:
                    continue
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 6)))
                m.fit(Z[rows][:, sub], y_tr[rows])
                v = pearson(y_tr[in_va], m.predict(Z[in_va][:, sub]))
                if v >= cfg.SCOUT_ACCEPT:
                    scouts.append((sub, m, max(float(v), 1e-3)))
        state.update(model=fallback, scouts=scouts)
        return state

    if kind == "relay":
        # The caravan: walk the fold segment by segment; each block's ridge
        # coefficients are shrunk toward the previous block's position
        # (random walk over coefficients -- the measured enemy here is regime
        # decay, so the model itself drifts). Predict from the FINAL position.
        mu0 = Z.mean(axis=0, keepdims=True).astype(np.float64)
        sd0 = (Z.std(axis=0, keepdims=True) + 1e-6).astype(np.float64)
        Zs = (Z - mu0) / sd0

        def caravan(rows_mask: np.ndarray, tau_mult: float) -> np.ndarray:
            beta = np.zeros(Zs.shape[1], np.float64)
            first = True
            for s in np.unique(seg_tr[rows_mask]):
                m = rows_mask & (seg_tr == s)
                n_b = int(m.sum())
                if n_b < 50:
                    continue
                Xb, yb = Zs[m], y_tr[m].astype(np.float64)
                lam = 0.5 * n_b
                tau = 0.0 if first else tau_mult * lam
                A = Xb.T @ Xb + (lam + tau) * np.eye(Zs.shape[1])
                b = Xb.T @ yb + tau * beta
                beta = np.linalg.solve(A, b)
                first = False
            return beta

        best_tau, best_sc = cfg.RELAY_TAUS[0], -1e9
        if in_va.sum() >= 50:
            for tm in cfg.RELAY_TAUS:
                bt = caravan(in_tr, tm)
                sc = pearson(y_tr[in_va], Zs[in_va] @ bt)
                if sc > best_sc:
                    best_tau, best_sc = tm, sc
        beta = caravan(np.ones(len(y_tr), bool), best_tau)
        state.update(beta=beta.astype(np.float32), mu=mu0.astype(np.float32),
                     sd=sd0.astype(np.float32), tau=float(best_tau))
        return state

    if kind == "swell":
        # Ride the swell underneath the chop: fit the EMA-smoothed label,
        # span chosen on the inner split scored against the RAW label.
        # span=1 IS the raw label, so the skill degrades to plain ridge.
        best_span, best_sc, best_model = 1, -1e9, None
        for span in cfg.SWELL_SPANS:
            y_s = (y_tr if span <= 1 else
                   pd.Series(y_tr).ewm(span=span, adjust=False).mean().to_numpy(np.float32))
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 6)))
            if in_va.sum() >= 50:
                m.fit(Z[in_tr], y_s[in_tr])
                sc = pearson(y_tr[in_va], m.predict(Z[in_va]))
            else:
                m.fit(Z, y_s)
                sc = 0.0
            if sc > best_sc or best_model is None:
                best_span, best_sc, best_model = span, sc, m
        y_s = (y_tr if best_span <= 1 else
               pd.Series(y_tr).ewm(span=best_span, adjust=False).mean().to_numpy(np.float32))
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 6)))
        model.fit(Z, y_s)
        state.update(model=model, span=int(best_span))
        return state

    if kind == "terrace":
        # v13 terraced fields: the ridge score is quantile-cut into steps,
        # each step's shrunken target mean forms a LUT; prediction is half
        # slope, half steps -- the codebook idea applied to the model's OWN
        # 1-D score (a fold-honest piecewise calibration).
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        model.fit(Z, y_tr)
        s = np.asarray(model.predict(Z), np.float64)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, cfg.TERRACE_BINS + 1)[1:-1]))
        codes = np.clip(np.searchsorted(edges, s, side="right"), 0, len(edges))
        gmean = float(np.mean(y_tr))
        lut = np.full(len(edges) + 1, gmean, np.float64)
        for b in range(len(edges) + 1):
            m = codes == b
            if m.any():
                lut[b] = (float(y_tr[m].sum()) + cfg.TERRACE_SHRINK * gmean) / (m.sum() + cfg.TERRACE_SHRINK)
        steps_tr = lut[codes]
        state.update(model=model, edges=edges.astype(np.float64), lut=lut.astype(np.float32),
                     smu=float(np.mean(s)), ssd=float(np.std(s)) + 1e-9,
                     lmu=float(np.mean(steps_tr)), lsd=float(np.std(steps_tr)) + 1e-9)
        return state

    if kind == "rapids":
        # v13 two-stage water: ridge the river, find where it runs roughest
        # (top-|residual| rows), fit a second ridge THERE on the residual,
        # and re-enter at a weight chosen on the inner split (0 is allowed:
        # a calm river keeps one stage).
        m1 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 8)))
        m1.fit(Z, y_tr)
        p1 = np.asarray(m1.predict(Z), np.float64)
        r = y_tr.astype(np.float64) - p1
        rough = np.abs(r) >= np.quantile(np.abs(r), 1.0 - cfg.RAPIDS_TOP_FRAC)
        m2, w2 = None, 0.0
        if rough.sum() >= 200:
            m2 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 6)))
            m2.fit(Z[rough], r[rough])
            if in_va.sum() >= 50:
                p1v = np.asarray(m1.predict(Z[in_va]), np.float64)
                p2v = np.asarray(m2.predict(Z[in_va]), np.float64)
                best_sc = -1e9
                for wc in cfg.RAPIDS_WEIGHTS:
                    sc = pearson(y_tr[in_va], p1v + wc * p2v)
                    if sc > best_sc:
                        best_sc, w2 = sc, float(wc)
        state.update(model=m1, m2=m2, w2=w2)
        return state

    if kind == "gbdt":
        if GBDT_BACKEND == "lightgbm":
            model = LGBMRegressor(n_estimators=cfg.GBDT_ESTIMATORS, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=5.0, min_child_samples=200,
                                  random_state=seed, n_jobs=-1, verbose=-1)
            model.fit(Z, y_tr)
            state["model"] = model
            return state
        # CPU ONLY by design: xgboost device=cuda in a lane thread caused an
        # uncatchable thrust abort on the 2026-06 T4x2 run -- never again.
        model = XGBRegressor(n_estimators=cfg.GBDT_ESTIMATORS, learning_rate=0.05,
                             max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=5.0, min_child_weight=200,
                             tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0)
        model.fit(Z, y_tr)
        state["model"] = model
        return state

    raise ValueError(f"unknown skill kind {kind}")


def _predict_core(state: dict[str, Any], Z: np.ndarray, X_raw: np.ndarray | None = None) -> np.ndarray:
    kind = state["kind"]
    if kind == "codebook":
        code = state["km"].predict((Z - state["mu0"]) / state["sd0"])
        return state["table"][code].astype(np.float32)
    if kind == "router":
        base = np.asarray(state["model"].predict(Z), np.float64)
        if state["experts"] and X_raw is not None and ATLAS is not None:
            t_ids = ATLAS.assign(X_raw)
            for c, (e, n_c) in state["experts"].items():
                m = t_ids == c
                if m.any():
                    wgt = n_c / (n_c + state["shrink_n"])
                    base[m] = (1.0 - wgt) * base[m] + wgt * np.asarray(e.predict(Z[m]), np.float64)
        return base.astype(np.float32)
    if kind == "steep":
        d = np.asarray(state["model"].predict(Z), np.float64)
        s = (np.asarray(state["steep"].predict(Z), np.float64) - state["smu"]) / state["ssd"]
        return (d * (1.0 + 0.5 * np.clip(s, -1.5, 1.5))).astype(np.float32)
    if kind == "scout":
        if state["scouts"]:
            tot = sum(w0 for _, _, w0 in state["scouts"])
            out = np.zeros(len(Z), np.float64)
            for sub, m, w0 in state["scouts"]:
                p = np.asarray(m.predict(Z[:, sub]), np.float64)
                out += (w0 / tot) * (p - p.mean()) / (p.std() + 1e-9)
            return out.astype(np.float32)
        return np.asarray(state["model"].predict(Z), np.float32)
    if kind == "relay":
        Zs = (Z - state["mu"]) / state["sd"]
        return (Zs.astype(np.float64) @ state["beta"].astype(np.float64)).astype(np.float32)
    if kind == "greedyols":
        return np.asarray(state["model"].predict(Z[:, state["sel"]]), np.float32)
    if kind == "linpearson":
        if state.get("torch_net"):
            Zsc = state["scaler"].transform(Z)
            return _predict_torch_mlp(state["model"], np.asarray(Zsc, np.float32))
        return np.asarray(state["model"].predict(Z), np.float32)
    if kind == "gpuswarm":
        return _predict_gpu_swarm(state, Z)
    if kind == "reservoir":
        Zs = ((Z - state["res_mu"]) / state["res_sd"]).astype(np.float32)
        res, a, W_in, W = state["res_size"], state["leak"], state["W_in"], state["W"]
        h = np.zeros(res, np.float32)
        H = np.empty((len(Zs), res), np.float32)
        drive = Zs @ W_in
        for t in range(len(Zs)):
            h = (1.0 - a) * h + a * np.tanh(drive[t] + h @ W)
            H[t] = h
        return np.asarray(state["model"].predict(H), np.float32)
    if kind == "terrace":
        s = np.asarray(state["model"].predict(Z), np.float64)
        steps = state["lut"][np.clip(np.searchsorted(state["edges"], s, side="right"),
                                     0, len(state["lut"]) - 1)].astype(np.float64)
        return (0.5 * (s - state["smu"]) / state["ssd"]
                + 0.5 * (steps - state["lmu"]) / state["lsd"]).astype(np.float32)
    if kind == "rapids":
        p = np.asarray(state["model"].predict(Z), np.float64)
        if state["m2"] is not None and state["w2"] > 0:
            p = p + state["w2"] * np.asarray(state["m2"].predict(Z), np.float64)
        return p.astype(np.float32)
    if kind == "uni":
        return (state["slope"] * (Z[:, state["j"]].astype(np.float64) - state["mu"])).astype(np.float32)
    if kind == "binmean":
        codes = np.searchsorted(state["edges"], Z[:, state["j"]], side="right")
        return state["means"][np.clip(codes, 0, len(state["means"]) - 1)].astype(np.float32)
    if kind == "vote":
        votes = np.sign(Z - state["med"]) * state["signs"]
        return votes.mean(axis=1).astype(np.float32)
    if kind == "tsen":
        return (state["slope"] * (Z[:, state["j"]].astype(np.float64) - state["mu"])).astype(np.float32)
    if kind == "knn":
        return state["model"].predict(state["scaler"].transform(Z[:, : state["d"]])).astype(np.float32)
    if kind == "ladder":
        Zs = (Z - state["mu"]) / state["sd"]
        return (Zs.astype(np.float64) @ state["beta"].astype(np.float64)).astype(np.float32)
    if kind == "ridgebag":
        return np.mean([m.predict(Z) for m in state["models"]], axis=0).astype(np.float32)
    if kind == "mlp":
        Zs = state["scaler"].transform(Z)
        if state.get("torch_mlp"):
            return _predict_torch_mlp(state["model"], np.asarray(Zs, np.float32))
        return state["model"].predict(Zs).astype(np.float32)
    if kind == "pls":                                  # v27: PLS predict is 2-D -> flatten
        return np.asarray(state["model"].predict(Z), np.float32).ravel()
    return np.asarray(state["model"].predict(Z), np.float32)


def predict_skill(state: dict[str, Any], X_any: np.ndarray) -> np.ndarray:
    Z = state["tf"](X_any[:, state["idx"]])
    return _predict_core(state, Z, X_any)


def stability_probe(state: dict[str, Any], X_va: np.ndarray, base_pred: np.ndarray,
                    rng: np.random.Generator, cfg: HarnessConfig) -> float:
    Z = state["tf"](X_va[:, state["idx"]]).copy()
    col_sd = Z.std(axis=0, keepdims=True) + 1e-6
    Zn = Z + rng.normal(0, cfg.STABILITY_NOISE, Z.shape).astype(np.float32) * col_sd
    p = _predict_core(state, Zn, X_va)    # terrain ids stay honest (unperturbed X)
    scale = float(np.std(base_pred)) + 1e-9
    return float(np.sqrt(np.mean((p - base_pred) ** 2)) / scale)


