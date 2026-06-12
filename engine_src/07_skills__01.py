class FitCtx(NamedTuple):
    """Shared fit-time context for the _FIT registry. Z, y_tr, the inner
    split and `state` travel as positional args into every branch; the raw
    fold, fold segments, column names, the lesson rng/cfg/seed and the
    viewport spec ride here for the branches that need them."""
    spec: ViewportSpec
    X_tr: np.ndarray
    seg_tr: np.ndarray
    cols: list[str]
    rng: np.random.Generator
    cfg: HarnessConfig
    seed: int


def _fit_uni(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_binmean(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                 state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_vote(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
              state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    # The purest primitive: every feature casts sign(fold_corr)*sign(x-median),
    # prediction is the unweighted mean vote. No fitted weights at all --
    # equal-weighting under estimation noise (the 1/N result).
    med = np.median(Z, axis=0).astype(np.float32)
    c = corr_vector(Z, y_tr)
    signs = np.sign(c).astype(np.float32)
    signs[signs == 0] = 1.0
    state.update(med=med, signs=signs)
    return state


def _fit_tsen(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
              state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    rng = ctx.rng

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


def _fit_recency(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                 state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_knn(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg = ctx.cfg
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


def _fit_ridge(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
               state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 10)))
    model.fit(Z, y_tr)
    state["model"] = model
    return state


def _fit_ols(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    # v22: plain OLS, NO shrinkage (the 11th-place 0.111 model class).
    # General -- the overfit-ratio door culls it on noisy/huge viewports, so
    # it only survives where an unregularized fit genuinely generalizes
    # (clean, small feature sets like the trailing positional block).
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(Z, y_tr)
    state["model"] = model
    return state


def _fit_huber(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
               state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_elastic(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                 state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    seed = ctx.seed
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


def _fit_pls(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_bayesridge(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                    state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_ard(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_recw(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
              state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_shiftlin(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                  state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
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


def _fit_greedyols(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                   state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_linpearson(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                    state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
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


def _fit_gpuswarm(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                  state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    state.update(_fit_gpu_swarm(Z, y_tr, ctx.seg_tr, ctx.cfg, ctx.seed))
    return state


def _fit_reservoir(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                   state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
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


def _fit_ridgebag(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                  state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg = ctx.cfg
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


def _fit_ladder(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg = ctx.cfg
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


def _fit_hgb(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
    model = HistGradientBoostingRegressor(
        max_iter=cfg.HGB_ITERS, learning_rate=cfg.HGB_LR,
        max_leaf_nodes=cfg.HGB_LEAVES, min_samples_leaf=cfg.HGB_MIN_LEAF,
        l2_regularization=cfg.HGB_L2, random_state=seed, early_stopping=False)
    model.fit(Z, y_tr)
    state["model"] = model
    return state


def _fit_mlp(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
             state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
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


def _fit_codebook(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                  state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
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


def _fit_router(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    X_tr, cfg = ctx.X_tr, ctx.cfg
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


def _fit_steep(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
               state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
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


def _fit_scout(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
               state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, rng = ctx.cfg, ctx.rng
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


def _fit_relay(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
               state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    seg_tr, cfg = ctx.seg_tr, ctx.cfg
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


def _fit_swell(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
               state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg = ctx.cfg
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


def _fit_terrace(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                 state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg = ctx.cfg
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


def _fit_rapids(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
                state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg = ctx.cfg
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


def _fit_gbdt(Z: np.ndarray, y_tr: np.ndarray, in_tr: np.ndarray, in_va: np.ndarray,
              state: dict[str, Any], ctx: FitCtx) -> dict[str, Any]:
    cfg, seed = ctx.cfg, ctx.seed
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


# The fit-side dispatch registry. Adding a skill = one _fit_<kind> function +
# one row here + one SKILL_REGISTRY entry (and a _PREDICT row only when the
# predict side is not the default state["model"].predict).
_FIT: dict[str, Callable[..., dict[str, Any]]] = {
    "uni": _fit_uni,
    "binmean": _fit_binmean,
    "vote": _fit_vote,
    "tsen": _fit_tsen,
    "recency": _fit_recency,
    "knn": _fit_knn,
    "ridge": _fit_ridge,
    "ols": _fit_ols,
    "huber": _fit_huber,
    "elastic": _fit_elastic,
    "pls": _fit_pls,
    "bayesridge": _fit_bayesridge,
    "ard": _fit_ard,
    "recw": _fit_recw,
    "shiftlin": _fit_shiftlin,
    "greedyols": _fit_greedyols,
    "linpearson": _fit_linpearson,
    "gpuswarm": _fit_gpuswarm,
    "reservoir": _fit_reservoir,
    "ridgebag": _fit_ridgebag,
    "ladder": _fit_ladder,
    "hgb": _fit_hgb,
    "mlp": _fit_mlp,
    "codebook": _fit_codebook,
    "router": _fit_router,
    "steep": _fit_steep,
    "scout": _fit_scout,
    "relay": _fit_relay,
    "swell": _fit_swell,
    "terrace": _fit_terrace,
    "rapids": _fit_rapids,
    "gbdt": _fit_gbdt,
}


def fit_skill(skill: str, spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
              cols: list[str], rng: np.random.Generator, cfg: HarnessConfig, seed: int) -> dict[str, Any]:
    kind = SKILL_REGISTRY[skill]["kind"]
    idx, transform = build_viewport(spec, X_tr, y_tr, seg_tr, cols, rng)
    state: dict[str, Any] = {"kind": kind, "skill": skill, "idx": idx, "tf": transform}
    Z = transform(X_tr[:, idx])
    in_tr, in_va = _inner_time_split(seg_tr, seed)
    fit = _FIT.get(kind)
    if fit is None:
        raise ValueError(f"unknown skill kind {kind}")
    return fit(Z, y_tr, in_tr, in_va, state,
               FitCtx(spec=spec, X_tr=X_tr, seg_tr=seg_tr, cols=cols, rng=rng, cfg=cfg, seed=seed))


