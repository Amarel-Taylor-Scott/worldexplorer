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


