# ----------------------------------------------------------------------------
# 10b. v21 FORENSIC REGIME-SCIENCE LAYER -- the full forensic stack, integrated
#      and SELF-TUNING. Nothing is deferred: every diagnostic is computed and
#      written, every candidate shipping config and every repair ACTION is
#      MEASURED on the forward slice (the CV->forward gap, this dataset's real
#      lever), and the harness OVERRIDES its own shipping choice ONLY when a
#      regime-aware config strictly beats the incumbent out-of-sample -- a
#      strict no-op otherwise. Rejected candidates/actions are kept and reported
#      with their measured deltas (an omitted mechanism is a blind spot; a
#      measured-and-rejected one is a learned data point -- "leaving things out
#      doesn't even allow us to learn"). Born from the v12 monoculture
#      regression: a single-family blend that decayed out-of-regime, invisible
#      to every in-working-region door. The cure: MEASURE input-space
#      concentration + worst-world behaviour and let measurement pick the blend.
#
#      Emits: regime_change_passports.csv, cv_train_gap_passports.csv,
#      partial_dynamics_tensor.csv, support_lattice_cells.csv,
#      support_sheaf_consistency.csv, row_influence_court.csv,
#      distributional_voi.csv, expert_dispatch_log.csv,
#      backward_feedback_actions.csv, mistake_memory_bank.csv,
#      forensic_selection_decision.json.
# ----------------------------------------------------------------------------

def _fz(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    s = float(a.std())
    return (a - a.mean()) / (s + 1e-12) if s > 1e-12 else np.zeros(len(a), np.float64)


def _ridge_coef(Z: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    Z = np.asarray(Z, np.float64)
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    yc = np.asarray(y, np.float64) - float(np.mean(y))
    try:
        return np.linalg.solve(Zc.T @ Zc + lam * np.eye(Zc.shape[1]), Zc.T @ yc)
    except Exception:
        return np.zeros(Zc.shape[1], np.float64)


def _feature_clusters(Xs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Target-free: cluster FEATURES (features as samples) by their behaviour
    across a strided row sample. Used for partial-dynamics + diversity + 'why'."""
    F = np.ascontiguousarray(Xs.T)
    F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-9)
    kk = int(min(k, max(2, F.shape[0] // 4)))
    try:
        return MiniBatchKMeans(kk, random_state=seed, n_init=3, batch_size=512).fit_predict(F).astype(np.int32)
    except Exception:
        return np.zeros(F.shape[0], np.int32)


def _forensic_worlds(seg, terr, wth, hab, min_rows):
    out = []
    for tag, ids in (("time", seg), ("terrain", terr), ("weather", wth), ("habitat", hab)):
        if ids is None:
            continue
        for v in np.unique(ids):
            m = ids == v
            if int(m.sum()) >= min_rows:
                out.append((f"{tag}{int(v)}", m))
    return out


def _world_floor(y, pred, worlds, q):
    vals = [(nm, pearson(y[m], pred[m])) for nm, m in worlds]
    if not vals:
        return 0.0, 0.0, ""
    cs = np.array([v for _, v in vals], np.float64)
    return float(cs.min()), float(np.quantile(cs, q)), vals[int(np.argmin(cs))][0]


def _jac(a, b) -> float:
    a, b = set(a), set(b)
    return (len(a & b) / len(a | b)) if (a and b) else 0.0


def _regime_passports(Xs, ys, segs, cols, fclust, cfg) -> pd.DataFrame:
    """Where/when/why a block boundary shifted: per-block ridge-coef drift +
    feature-graph edge shift + prediction-direction drift, voted; complete
    (>=3 detectors) vs partial; 'why' = which feature-cluster moved most."""
    c = np.abs(corr_vector(Xs, ys))
    top = list(np.argsort(-c)[:60])
    blocks = sorted(np.unique(segs).tolist())
    probe = np.arange(len(Xs))[:: max(1, len(Xs) // 3000)]
    betas, graphs, preds = {}, {}, {}
    for b in blocks:
        m = segs == b
        if int(m.sum()) < 200:
            continue
        Z = Xs[m][:, top]
        beta = _ridge_coef(Z, ys[m])
        betas[b] = beta
        mu, sd = Xs[probe][:, top].mean(0), Xs[probe][:, top].std(0) + 1e-9
        preds[b] = ((Xs[probe][:, top] - mu) / sd) @ beta
        sub = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        graphs[b] = (np.abs(np.nan_to_num(np.corrcoef(sub.T))) > 0.5)
    rows, have = [], [b for b in blocks if b in betas]
    for i in range(1, len(have)):
        a, b = have[i - 1], have[i]
        ba, bb = betas[a], betas[b]
        cos = float(ba @ bb / (np.linalg.norm(ba) * np.linalg.norm(bb) + 1e-12))
        coef_drift = 1.0 - cos
        sign_flip = float(np.mean(np.sign(ba) != np.sign(bb)))
        iu = np.triu_indices_from(graphs[a], k=1)
        ea, eb = graphs[a][iu], graphs[b][iu]
        graph_shift = 1.0 - float((ea & eb).sum()) / max(1.0, float((ea | eb).sum()))
        pred_drift = 1.0 - abs(pearson(preds[a], preds[b]))
        votes = int(coef_drift > 0.5) + int(graph_shift > 0.4) + int(pred_drift > 0.4) + int(sign_flip > 0.35)
        dmag = np.abs(bb - ba)
        cl = fclust[np.array(top)]
        why = {int(cv): float(dmag[cl == cv].sum()) for cv in np.unique(cl)}
        wt = max(why, key=why.get) if why else -1
        rows.append({"boundary": f"seg{a}->seg{b}", "coef_drift": round(coef_drift, 4),
                     "sign_flip_rate": round(sign_flip, 4), "feature_graph_shift": round(graph_shift, 4),
                     "pred_direction_drift": round(pred_drift, 4), "detector_votes": votes,
                     "change": "complete" if votes >= 3 else ("partial" if votes >= 1 else "stable"),
                     "why_top_feature_cluster": wt, "why_cluster_coef_mass": round(why.get(wt, 0.0), 4)})
    return pd.DataFrame(rows)


def _partial_dynamics(Xs, ys, segs, fclust, cfg) -> pd.DataFrame:
    """D[time-block x feature-cluster]: per-cluster ridge prediction's corr in
    each block. Surfaces partial regime change -- e.g. cluster 7 dies only late."""
    rows = []
    for cv in np.unique(fclust):
        ccols = list(np.where(fclust == cv)[0])
        if len(ccols) < 2:
            continue
        beta = _ridge_coef(Xs[:, ccols], ys)
        mu, sd = Xs[:, ccols].mean(0), Xs[:, ccols].std(0) + 1e-9
        pred = ((Xs[:, ccols] - mu) / sd) @ beta
        for b in np.unique(segs):
            m = segs == b
            if int(m.sum()) >= 50:
                rows.append({"feature_cluster": int(cv), "time_block": int(b),
                             "n_cols": len(ccols), "rows": int(m.sum()),
                             "corr": round(pearson(ys[m], pred[m]), 4)})
    return pd.DataFrame(rows)


def _support_sheaf(Xs, ys, terr_s, cols, cfg) -> pd.DataFrame:
    """Sheaf defect: fit a local ridge per terrain (top features), apply it to
    OTHER terrains; high own-vs-foreign corr gap = incompatible local truths
    (where regime-split / caveat experts should go)."""
    c = np.abs(corr_vector(Xs, ys))
    top = list(np.argsort(-c)[:40])
    locals_ = {}
    for t in np.unique(terr_s):
        m = terr_s == t
        if int(m.sum()) >= 300:
            locals_[int(t)] = _ridge_coef(Xs[m][:, top], ys[m])
    mu, sd = Xs[:, top].mean(0), Xs[:, top].std(0) + 1e-9
    Zt = (Xs[:, top] - mu) / sd
    rows = []
    ts = sorted(locals_)
    for a in ts:
        ma = terr_s == a
        own = pearson(ys[ma], Zt[ma] @ locals_[a])
        for b in ts:
            if b == a:
                continue
            foreign = pearson(ys[ma], Zt[ma] @ locals_[b])
            rows.append({"home_terrain": a, "applied_model_from": b,
                         "own_corr": round(own, 4), "foreign_corr": round(foreign, 4),
                         "sheaf_defect": round(own - foreign, 4)})
    return pd.DataFrame(rows).sort_values("sheaf_defect", ascending=False) if rows else pd.DataFrame()


def _row_court(blend_oof, y, terr, wth, cfg) -> pd.DataFrame:
    """Bad-row forensics: studentized residual of the incumbent blend + how far
    each suspect's label sits from its (terrain,weather) cell mean. Top suspects
    only -- whether quarantining them HELPS is measured as an action below."""
    o = _fz(blend_oof)
    slope = float(np.mean(o * (np.asarray(y, np.float64) - np.mean(y))))
    resid = np.asarray(y, np.float64) - slope * o
    rz = np.abs(resid) / (resid.std() + 1e-12)
    cut = np.quantile(rz, cfg.FORENSIC_INFLUENCE_Q)
    cellmean = np.zeros(len(y), np.float64)
    key = (terr.astype(np.int64) * 97 + wth.astype(np.int64)) if (terr is not None and wth is not None) else np.zeros(len(y), np.int64)
    for k in np.unique(key):
        m = key == k
        cellmean[m] = float(np.mean(y[m]))
    cell_dev = np.abs(np.asarray(y, np.float64) - cellmean) / (np.std(y) + 1e-9)
    susp = np.where(rz >= cut)[0]
    order = susp[np.argsort(-rz[susp])][:200]
    return pd.DataFrame([{"row": int(i), "studentized_resid": round(float(rz[i]), 3),
                          "cell_dev": round(float(cell_dev[i]), 3),
                          "bad_data_score": round(float(rz[i] * (0.5 + 0.5 * min(cell_dev[i], 3.0))), 3)}
                         for i in order])


def adversarial_validation_report(X, cols, cfg):
    """v24 ADVERSARIAL VALIDATION (target-free): can a classifier tell EARLY
    from LATE working rows by features alone? AUC ~ 0.5 = stable covariates;
    >> 0.5 = strong covariate shift -- the makeup the model trains on differs
    from the makeup it is tested on, this dataset's measured enemy. Reports the
    AUC + the most-shifted features (standardized mean difference). Gates
    nothing; a drift map that motivates the shift_linear / recency_weighted skills."""
    n = len(X)
    cut = int(0.75 * n)
    if cut < 200 or n - cut < 200:
        return pd.DataFrame(), 0.5
    lab = (np.arange(n) >= cut).astype(np.int32)
    step = max(1, n // 40000)
    Xs, ls = X[::step], lab[::step]
    auc = 0.5
    try:
        from sklearn.metrics import roc_auc_score
        ntr = int(0.7 * len(Xs))
        if len(np.unique(ls[:ntr])) > 1 and len(np.unique(ls[ntr:])) > 1:
            clf = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31,
                                                 learning_rate=0.06, random_state=cfg.SEED)
            clf.fit(Xs[:ntr], ls[:ntr])
            auc = float(roc_auc_score(ls[ntr:], clf.predict_proba(Xs[ntr:])[:, 1]))
    except Exception:
        auc = 0.5
    e_mu, l_mu = X[:cut].mean(0), X[cut:].mean(0)
    smd = np.abs(l_mu - e_mu) / (X.std(0) + 1e-9)
    order = np.argsort(-smd)[:40]
    df = pd.DataFrame([{"feature": cols[int(j)] if int(j) < len(cols) else str(j),
                        "std_mean_diff": round(float(smd[j]), 4),
                        "mean_early": round(float(e_mu[j]), 5),
                        "mean_late": round(float(l_mu[j]), 5)} for j in order])
    return df, auc


# v27 RUNTIME COMPLEXITY-GENERALIZATION GOVERNOR ------------------------------
# Shipping-time state: the harness measures this dataset's decay~complexity
# slope and writes {"lambda","beta","complexity":{member:C}} here; the robust
# selector below subtracts lambda*complexity from each candidate's robust score,
# so the SHIPPED config is chosen at the complexity the data rewards. Empty =>
# lambda 0 => byte-identical to v25 (the equivalence gate still passes off).
GOVERNOR: dict = {}

# v27 CROSS-RUN LEARNING LEDGER -- the self-improvement loop. Each run distills
# OUT-OF-SAMPLE-grounded learnings (governor beta, per-family/skill generalization
# track record, survivors, decayers, data profile) into world_cairn.json["ledger"];
# the NEXT run loads them here and uses them as PRIORS -- shrunk by evidence count
# so one noisy run cannot dominate, and carrying ANTI-priors (decayers) so the loop
# self-corrects instead of compounding a monoculture. Empty => no prior => no-op.
LEDGER_PRIOR: dict = {}


def _ledger_decay_stats(lessons, attr):
    """Per-(family|skill) mean OUT-OF-PERIOD decay (oof_corr - wf_corr) + mean oof,
    with evidence counts -- the generalization track record this run measured."""
    agg = {}
    for l in lessons:
        if l.oof_corr <= 0.02 or not np.isfinite(l.wf_corr):
            continue
        k = getattr(l, attr)
        a = agg.setdefault(k, {"sd": 0.0, "so": 0.0, "n": 0})
        a["sd"] += float(l.oof_corr - l.wf_corr); a["so"] += float(l.oof_corr); a["n"] += 1
    return {k: {"mean_decay": round(a["sd"] / a["n"], 5), "mean_oof": round(a["so"] / a["n"], 5),
                "n": a["n"]} for k, a in agg.items() if a["n"] > 0}


