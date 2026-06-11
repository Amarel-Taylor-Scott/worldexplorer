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


def _ledger_merge(prior, new):
    """Running-mean merge of decay-stat dicts by evidence count (Bayesian shrinkage
    -- a new run UPDATES the accumulated track record, never overwrites it)."""
    out = {k: dict(v) for k, v in (prior or {}).items()}
    for k, nv in (new or {}).items():
        if k in out:
            pn, nn = int(out[k].get("n", 0)), int(nv.get("n", 0))
            tot = pn + nn
            if tot > 0:
                for m in ("mean_decay", "mean_oof"):
                    out[k][m] = round((pn * float(out[k].get(m, 0.0)) + nn * float(nv.get(m, 0.0))) / tot, 5)
                out[k]["n"] = tot
        else:
            out[k] = dict(nv)
    return out

# capacity tier per skill (linear/primitive low -> gpu/mlp/gbdt high) and per
# transform (identity/sign low -> engineered/kernel high): how much in-sample-
# fitting CAPACITY a lesson spends. Used only to REGRESS decay on capacity;
# the dataset, not this table, decides whether capacity is penalized.
_GOV_CAPACITY = {
    "single_factor": 0.0, "bin_association": 0.0, "majority_vote": 0.0, "theil_sen": 0.05,
    "linear_ols": 0.10, "greedy_ols": 0.18, "swell_rider": 0.20, "recency_linear": 0.20,
    "recency_weighted": 0.20, "linear_assoc": 0.22, "huber_linear": 0.22, "elastic_net": 0.26,
    "bayes_ridge": 0.18, "ard_linear": 0.20, "pls": 0.30,   # v27 linear-diversity skills (low-complexity)
    "shift_linear": 0.30, "relay_caravan": 0.35, "bagged_linear": 0.40, "residual_ladder": 0.42,
    "codebook": 0.45, "terrace": 0.45, "rapids": 0.50, "scout_lattice": 0.55, "terrain_router": 0.60,
    "echo_state_ridge": 0.60, "linear_pearson": 0.70, "steepness_gate": 0.70, "local_interp": 0.70,
    "gpu_ridge_swarm": 0.80, "nonlinear_assoc": 0.92, "mlp_assoc": 1.0, "gbdt_lib": 1.0,
}
_GOV_TRANSFORM = {
    "identity": 0.0, "sign_only": 0.0, "rank": 0.10, "quantize2": 0.10, "quantize4": 0.15,
    "quantize8": 0.20, "fold_abs": 0.40, "pca": 0.40, "lateral_line": 0.40, "moire": 0.40,
    "dual_exposure": 0.30, "doppler": 0.30, "tide": 0.30, "curvature": 0.30, "lorentz_boost": 0.50,
    "prism": 0.50, "fractal": 0.50, "reaction_diffusion": 0.50, "rand_proj": 0.50,
    "pair_aug": 0.60, "pca_aug": 0.60, "fold_pairs": 0.60, "signed_hadamard": 0.60,
    "foveated": 0.60, "random_fourier": 0.70,
}


def member_complexity(l, cfg) -> float:
    """A lesson's in-sample-fitting CAPACITY in [0,1]: model tier, viewport
    width k, transform richness, and the measured overfit ratio. The governor
    regresses out-of-period DECAY (oof_corr - wf_corr) on THIS to learn whether
    the dataset rewards or punishes capacity -- the relationship is measured,
    never assumed."""
    cap = _GOV_CAPACITY.get(l.skill, 0.5)
    kfrac = min(1.0, float(l.k) / max(1.0, float(cfg.K_MAX)))
    tr = _GOV_TRANSFORM.get(l.transform, 0.30)
    of = min(1.0, max(0.0, (float(l.overfit_ratio) - 1.0) / max(1e-6, float(cfg.MAX_OVERFIT_RATIO) - 1.0)))
    return float(0.40 * cap + 0.30 * kfrac + 0.15 * tr + 0.15 * of)


def _gov_config_complexity(weights: dict) -> float:
    """Weight-averaged complexity of a candidate shipping config (0 when the
    governor is inactive -- the cmap is empty)."""
    cmap = GOVERNOR.get("complexity", {}) if GOVERNOR else {}
    if not cmap:
        return 0.0
    tot = sum(weights.values()) or 1.0
    return float(sum((wt / tot) * float(cmap.get(nm, 0.0)) for nm, wt in weights.items()))


def shipping_court(final_weights, members, member_lessons, yp, segp, terr_p, wth_p, cfg):
    """v27 ANTI-OVERFIT SHIPPING COURT -- a cheap, OUT-OF-SAMPLE-grounded selection
    hardener distilled from the regime-criticality / overfit-gravity-well /
    CV-reality-distortion / prediction-crowding ideas. Per shipped member it
    measures decay (oof-wf), the worst-world FLOOR, the CV reality-distortion
    index (oof - floor: wins only in friendly worlds), governor complexity, and
    its loading on the dominant prediction axis (condensation). A member whose
    path WIDTH does not clear its local ESCAPE VELOCITY (and whose world floor is
    non-positive) is DOWN-WEIGHTED; high regime CRITICALITY (residual auto-
    correlation) shrinks the whole blend toward equal-weight (robustness). Adds
    NO capacity -- only raises the bar where overfit risk is measured. Conservative
    => near-no-op on a healthy blend. Writes shipping_court_report.csv + regime_criticality.json."""
    if not getattr(cfg, "SHIPPING_COURT", True) or len(final_weights) < 2:
        return dict(final_weights)
    try:
        nms = list(final_weights)
        worlds = [("seg", segp)]
        if terr_p is not None:
            worlds.append(("terrain", terr_p))
        if wth_p is not None:
            worlds.append(("weather", wth_p))
        gcmap = GOVERNOR.get("complexity", {}) if GOVERNOR else {}
        M = np.vstack([np.asarray(members[nm], np.float64) for nm in nms])
        try:
            C = np.nan_to_num(np.corrcoef(M), nan=0.0)
            evals, evecs = np.linalg.eigh(C)
            load = np.abs(evecs[:, -1]); load = load / (load.sum() + 1e-12)
            cond_share = float(evals[-1] / (float(evals.sum()) + 1e-12))
            crowd = {nm: float(load[i]) for i, nm in enumerate(nms)}
        except Exception:
            cond_share, crowd = 0.0, {nm: 1.0 / len(nms) for nm in nms}
        eqw = 1.0 / len(nms)
        rows, penalty = [], {}
        for nm in nms:
            l = member_lessons.get(nm)
            oof = float(l.oof_corr) if l is not None else pearson(yp, members[nm])
            wf = float(l.wf_corr) if (l is not None and np.isfinite(l.wf_corr)) else oof
            decay = oof - wf
            floors = [pearson(yp[ids == s], members[nm][ids == s])
                      for _, ids in worlds for s in np.unique(ids)
                      if int((ids == s).sum()) >= cfg.FORENSIC_MIN_WORLD_ROWS]
            wfloor = float(min(floors)) if floors else oof
            rdi = oof - wfloor
            cplx = float(gcmap.get(nm, 0.30))
            width = float(l.width) if (l is not None and np.isfinite(l.width)) else 0.0
            crd = float(crowd.get(nm, eqw))
            escape = (cfg.COURT_BASE + cfg.COURT_W_DECAY * max(0.0, decay)
                      + cfg.COURT_W_RDI * max(0.0, rdi) + cfg.COURT_W_CPLX * cplx
                      + cfg.COURT_W_CROWD * max(0.0, crd - eqw))
            fails = (width < escape) and (wfloor <= 0.0)
            penalty[nm] = cfg.COURT_PENALTY if fails else 1.0
            rows.append({"member": nm, "oof": round(oof, 4), "wf": round(wf, 4),
                         "decay": round(decay, 4), "world_floor": round(wfloor, 4),
                         "rdi": round(rdi, 4), "complexity": round(cplx, 3),
                         "crowd_load": round(crd, 3), "width": round(width, 4),
                         "escape_velocity": round(escape, 4),
                         "verdict": "down_weighted" if fails else "ok"})
        blend = sum(final_weights[nm] * members[nm] for nm in nms)
        o = (blend - blend.mean()) / (blend.std() + 1e-12)
        sl = float(np.mean(o * (yp - yp.mean())))
        resid = np.asarray(yp, np.float64) - sl * o
        r_ac1 = abs(pearson(resid[:-1], resid[1:])) if len(resid) > 1000 else 0.0
        disagree = float(np.mean(M.std(axis=0)))
        criticality = float(r_ac1)
        out = {nm: final_weights[nm] * penalty[nm] for nm in nms}
        tot = sum(out.values()) or 1.0
        out = {nm: v / tot for nm, v in out.items()}
        crit_hi = criticality >= cfg.COURT_CRIT_HI
        if crit_hi:
            out = {nm: (1.0 - cfg.COURT_CRIT_SHRINK) * out[nm] + cfg.COURT_CRIT_SHRINK * eqw for nm in nms}
        n_dw = sum(1 for r in rows if r["verdict"] == "down_weighted")
        write_csv(pd.DataFrame(rows).sort_values("rdi", ascending=False), "shipping_court_report.csv")
        write_json({"criticality": round(criticality, 5), "residual_autocorr_lag1": round(r_ac1, 5),
                    "condensation_share": round(cond_share, 4), "member_disagreement": round(disagree, 4),
                    "criticality_high": bool(crit_hi), "members_down_weighted": int(n_dw),
                    "note": "regime slowing-down + anti-overfit selection hardener; high criticality "
                            "shrinks the blend toward equal-weight (robustness). Out-of-sample-grounded."},
                   "regime_criticality.json")
        log("shipping_court", down_weighted=n_dw, condensation=round(cond_share, 3),
            criticality=round(criticality, 4), crit_high=crit_hi,
            note="anti-overfit selection hardener (out-of-sample-grounded, conservative)")
        return out
    except Exception as e:
        log("shipping_court_failed_noop", err=str(e)[:120])
        return dict(final_weights)


def robust_oos_select(cand_weights, members, member_lessons, spec_lookup,
                      X_full, y_full, seg_full, n_work, cols, cfg):
    """v22 SPINE: choose the shipping config by stability across MANY structure-
    aware train/test partitions of different makeup, not one slice. Partitions:
    time geometries (expanding walk-forward / sliding-recent / REVERSED) AND
    explorer-discovered structure (leave-one-terrain-out, leave-one-weather-out)
    -- "does the signal survive a regime it was never trained on?". Each gets a
    block-bootstrap lower-tail corr; the robust score is mean - std ACROSS
    partitions (stable everywhere, not lucky once). Ties -> HEDGE (blend). Fully
    wrapped: any failure -> no-op (keep the incumbent)."""
    out = {"best": "incumbent", "weights": cand_weights.get("incumbent", {}),
           "hedged": False, "scores": [], "partitions": 0}
    try:
        stride = max(1, n_work // cfg.ROBUST_SAMPLE_ROWS)
        rows = np.arange(0, n_work, stride)
        Xw, yw, segw = X_full[rows], y_full[rows], seg_full[rows]
        terr = ATLAS.assign(Xw) if ATLAS is not None else np.zeros(len(Xw), np.int32)
        wth = GAUGE.assign(Xw) if GAUGE is not None else np.zeros(len(Xw), np.int32)
        segs = sorted(np.unique(segw).tolist())
        if len(segs) < 5:
            return out
        nseg = len(segs)
        parts = []
        for i in range(nseg // 2, nseg):                              # expanding walk-forward
            parts.append((f"wf{i}", np.isin(segw, segs[:i]), segw == segs[i]))
        for i in range(2, nseg):                                      # sliding-recent (last 3 segs)
            parts.append((f"slide{i}", np.isin(segw, segs[max(0, i - 3):i]), segw == segs[i]))
        for i in range(0, nseg // 2):                                 # REVERSED: train late, test early
            parts.append((f"rev{i}", np.isin(segw, segs[i + 1:]), segw == segs[i]))
        for t in np.unique(terr):                                     # leave-one-terrain-out
            parts.append((f"terr{int(t)}", terr != t, terr == t))
        for s in np.unique(wth):                                      # leave-one-weather-out
            parts.append((f"wx{int(s)}", wth != s, wth == s))
        # v24 COMBINATORIAL PURGED CV (Lopez de Prado): test on COMBINATIONS of
        # segment groups with the ADJACENT groups PURGED from train -- many
        # statistically-distinct backtest paths, not one. Capped for cost.
        try:
            import itertools
            segarr = np.array(segs)
            groups = [g for g in np.array_split(segarr, min(cfg.CPCV_GROUPS, nseg)) if len(g)]
            for ci, combo in enumerate(itertools.combinations(
                    range(len(groups)), min(cfg.CPCV_TEST_GROUPS, len(groups)))):
                if ci >= cfg.CPCV_MAX_PATHS:
                    break
                test_segs = np.concatenate([groups[g] for g in combo])
                purge_segs: list = []
                for g in combo:
                    if g > 0:
                        purge_segs += groups[g - 1].tolist()
                    if g < len(groups) - 1:
                        purge_segs += groups[g + 1].tolist()
                te = np.isin(segw, test_segs)
                excl = (np.concatenate([test_segs, np.array(purge_segs, dtype=test_segs.dtype)])
                        if purge_segs else test_segs)
                parts.append((f"cpcv{ci}", ~np.isin(segw, excl), te))
        except Exception:
            pass
        parts = [(nm, tr, te) for nm, tr, te in parts
                 if int(tr.sum()) >= cfg.ROBUST_MIN_TRAIN and int(te.sum()) >= cfg.ROBUST_MIN_TEST]
        if len(parts) < 3:
            return out

        mem_union = []
        for w in cand_weights.values():
            for nm in w:
                if nm not in mem_union and nm in member_lessons:
                    mem_union.append(nm)
        mem_union = mem_union[: cfg.ROBUST_MAX_MEMBERS]

        cache: dict = {}
        for nm in mem_union:
            l = member_lessons[nm]; spec = spec_lookup[l.key]
            for pidx, (pn, tr, te) in enumerate(parts):
                try:
                    st = fit_skill(l.skill, spec, Xw[tr], yw[tr], segw[tr], cols,
                                   np.random.default_rng(l.seed), cfg, l.seed)
                    p = predict_skill(st, Xw[te]).astype(np.float64)
                    cache[(nm, pidx)] = (p - p.mean()) / (p.std() + 1e-12)
                except Exception:
                    cache[(nm, pidx)] = None

        rng = np.random.default_rng(stable_seed(cfg.SEED, "robust_oos"))

        def cand_score(w):
            per = []
            for pidx, (pn, tr, te) in enumerate(parts):
                ok = [nm for nm in w if cache.get((nm, pidx)) is not None]
                if not ok:
                    continue
                tot = sum(w[nm] for nm in ok) or 1.0
                blend = sum((w[nm] / tot) * cache[(nm, pidx)] for nm in ok)
                yte = yw[te]
                seg_te = segw[te]; useg = np.unique(seg_te)
                if len(useg) >= 3:
                    sidx = [np.where(seg_te == s)[0] for s in useg]
                    boot = []                                    # block-bootstrap over segment blocks
                    for _ in range(cfg.ROBUST_BOOT):
                        pick = rng.integers(0, len(useg), len(useg))
                        bi = np.concatenate([sidx[int(p)] for p in pick])
                        boot.append(score_metric(yte[bi], blend[bi]))
                    per.append(float(np.quantile(boot, cfg.ROBUST_BOOT_Q)))
                else:
                    per.append(score_metric(yte, blend))
            if not per:
                return -1e9, 0.0, 0.0
            arr = np.array(per, np.float64)
            return float(arr.mean() - arr.std()), float(arr.mean()), float(arr.std())

        # v27 governor: penalize each candidate's robust score by lambda*complexity
        # (lambda set by the MEASURED decay~complexity slope; 0 => unchanged, so
        # round(rs - 0*cplx, 5) == round(rs, 5) exactly and the gate still passes).
        lam = float(GOVERNOR.get("lambda", 0.0)) if GOVERNOR else 0.0
        scored = []
        for name, w in cand_weights.items():
            rs, mu, sd = cand_score(w)
            cplx = _gov_config_complexity(w)
            scored.append({"config": name, "robust_score": round(rs - lam * cplx, 5),
                           "robust_raw": round(rs, 5), "complexity": round(cplx, 4),
                           "mean_partition": round(mu, 5), "std_partition": round(sd, 5)})
        scored.sort(key=lambda r: -r["robust_score"])
        out["scores"] = scored; out["partitions"] = len(parts)
        out["best"] = scored[0]["config"]; out["weights"] = cand_weights[scored[0]["config"]]
        if len(scored) >= 2 and scored[0]["robust_score"] - scored[1]["robust_score"] <= cfg.ROBUST_HEDGE_BAND:
            a, b = cand_weights[scored[0]["config"]], cand_weights[scored[1]["config"]]
            ta, tb = (sum(a.values()) or 1.0), (sum(b.values()) or 1.0)
            hedge = {nm: 0.5 * a.get(nm, 0.0) / ta + 0.5 * b.get(nm, 0.0) / tb for nm in set(a) | set(b)}
            hs, _, _ = cand_score(hedge)
            hs_adj = hs - lam * _gov_config_complexity(hedge)
            if hs_adj >= scored[0]["robust_score"] - 1e-9:
                out["best"] = f"hedge({scored[0]['config']}+{scored[1]['config']})"
                out["weights"] = hedge; out["hedged"] = True
                scored.insert(0, {"config": out["best"], "robust_score": round(hs_adj, 5),
                                  "robust_raw": round(hs, 5),
                                  "complexity": round(_gov_config_complexity(hedge), 4),
                                  "mean_partition": None, "std_partition": None})
        return out
    except Exception as e:
        log("robust_oos_failed_noop", err=str(e)[:120])
        return out


def forensic_regime_science(members, member_lessons, spec_lookup, w0, is_median0, weather_states0,
                            incumbent_fwd_corr, fwd_parts, yp, segp, terr_p, wth_p, volp,
                            X_full, y_full, seg_full, n_work, cols, past, future, cfg):
    """Orchestrator. Writes the full forensic suite, MEASURES candidate shipping
    configs + repair actions on the forward slice, and returns a possibly-
    overridden shipping decision. Fully wrapped: any failure -> no-op (ship the
    incumbent), so the forensic layer can never break the run."""
    dec = {"override": False, "weights": w0, "is_median": is_median0,
           "weather_states": weather_states0, "winner": "incumbent",
           "fwd_parts": fwd_parts, "forward_blend_corr": incumbent_fwd_corr}
    if not cfg.FORENSIC_ENABLED or not members:
        return dec
    try:
        global FCLUST, HABITAT
        stride = max(1, n_work // 40_000)
        Xs = X_full[:n_work][::stride]
        ys = y_full[:n_work][::stride]
        segs = seg_full[:n_work][::stride]
        FCLUST = _feature_clusters(Xs, cfg.FORENSIC_FEATURE_CLUSTERS, cfg.SEED)
        terr_s = ATLAS.assign(Xs) if ATLAS is not None else np.zeros(len(Xs), np.int32)
        hab_p = None
        if volp is not None and np.isfinite(volp).any():
            ed = np.quantile(volp, [1 / 3, 2 / 3])
            hab_p = np.searchsorted(ed, volp).astype(np.int32)
        HABITAT = hab_p
        worlds = _forensic_worlds(segp, terr_p, wth_p, hab_p, cfg.FORENSIC_MIN_WORLD_ROWS)

        # complete forward parts for EVERY member (reuse what v20 already refit)
        fwd_all = dict(fwd_parts)
        for nm in members:
            if nm in fwd_all:
                continue
            try:
                l = member_lessons[nm]
                st = fit_skill(l.skill, spec_lookup[l.key], X_full[past], y_full[past],
                               seg_full[past], cols, np.random.default_rng(l.seed), cfg, l.seed)
                fwd_all[nm] = _fz(predict_skill(st, X_full[future]))
            except Exception:
                fwd_all[nm] = np.zeros(len(future), np.float64)

        # ---- per-member gap passports + distributional VOI + mistake memory ----
        yf = y_full[future]
        gap_rows, voi_rows, mistake_rows = [], [], []
        for nm in members:
            l = member_lessons[nm]
            wmin, wq10, worst = _world_floor(yp, members[nm], worlds, cfg.FORENSIC_WORLD_Q)
            fwd_c = pearson(yf, fwd_all[nm]) if nm in fwd_all else float("nan")
            gap = l.fit_corr - l.oof_corr
            decay = l.oof_corr - fwd_c if np.isfinite(fwd_c) else float("nan")
            null_c = l.null_corr if np.isfinite(l.null_corr) else 0.0
            if l.oof_corr <= 0.01:
                cause = "no_signal"
            elif gap > 0.6 * max(l.fit_corr, 1e-6):
                cause = "model_overfit"
            elif wmin < 0.4 * max(l.oof_corr, 1e-6):
                cause = "regime_fragile"
            elif np.isfinite(decay) and decay > 0.5 * max(l.oof_corr, 1e-6):
                cause = "forward_decay"
            else:
                cause = "honest"
            gap_rows.append({"member": nm, "family": l.family, "transform": l.transform, "k": l.k,
                             "n_cols": len(l.used_cols), "fit_corr": round(l.fit_corr, 4),
                             "oof_corr": round(l.oof_corr, 4), "gap": round(gap, 4),
                             "forward_corr": round(fwd_c, 4) if np.isfinite(fwd_c) else None,
                             "decay": round(decay, 4) if np.isfinite(decay) else None,
                             "world_min": round(wmin, 4), "world_q10": round(wq10, 4),
                             "worst_world": worst, "gap_cause": cause})
            voi_rows.append({"member": nm, "mean_world": round(l.oof_corr, 4),
                             "q10_world": round(wq10, 4), "worst_world": round(wmin, 4),
                             "null_adjusted_q10": round(wq10 - max(0.0, null_c), 4),
                             "voi_lcb": round(wq10 - max(0.0, null_c), 4),
                             "admit_exploration": bool(wq10 - max(0.0, null_c) > 0)})
            if np.isfinite(decay) and decay > 0.5 * max(l.oof_corr, 1e-6) and l.oof_corr > 0.02:
                mistake_rows.append({"member": nm, "family": l.family, "transform": l.transform,
                                     "oof_corr": round(l.oof_corr, 4), "forward_corr": round(fwd_c, 4),
                                     "decay": round(decay, 4),
                                     "archetype": f"{l.family}|{l.transform}",
                                     "lesson": "looked strong in-regime, decayed on the forward slice"})
        write_csv(pd.DataFrame(gap_rows).sort_values("oof_corr", ascending=False), "cv_train_gap_passports.csv")
        write_csv(pd.DataFrame(voi_rows).sort_values("voi_lcb", ascending=False), "distributional_voi.csv")
        if mistake_rows:
            write_csv(pd.DataFrame(mistake_rows).sort_values("decay", ascending=False), "mistake_memory_bank.csv")

        # ---- regime cartography + partial dynamics + sheaf ---------------------
        for fn, name in ((lambda: _regime_passports(Xs, ys, segs, cols, FCLUST, cfg), "regime_change_passports.csv"),
                         (lambda: _partial_dynamics(Xs, ys, segs, FCLUST, cfg), "partial_dynamics_tensor.csv"),
                         (lambda: _support_sheaf(Xs, ys, terr_s, cols, cfg), "support_sheaf_consistency.csv")):
            try:
                df = fn()
                if not df.empty:
                    write_csv(df, name)
            except Exception as e:
                log("forensic_diag_skipped", artifact=name, err=str(e)[:80])

        # ---- support-lattice (terrain x weather) of the incumbent blend --------
        blend_oof = apply_weights_rows(members, w0, is_median0, weather_states0, wth_p).astype(np.float64)
        if terr_p is not None and wth_p is not None:
            lat = [{"terrain": int(t), "weather": int(s), "rows": int(((terr_p == t) & (wth_p == s)).sum()),
                    "corr": round(pearson(yp[(terr_p == t) & (wth_p == s)], blend_oof[(terr_p == t) & (wth_p == s)]), 4)}
                   for t in np.unique(terr_p) for s in np.unique(wth_p)
                   if ((terr_p == t) & (wth_p == s)).sum() >= cfg.FORENSIC_MIN_WORLD_ROWS]
            if lat:
                write_csv(pd.DataFrame(lat).sort_values("corr"), "support_lattice_cells.csv")
        try:
            write_csv(_row_court(blend_oof, yp, terr_p, wth_p, cfg), "row_influence_court.csv")
        except Exception as e:
            log("forensic_diag_skipped", artifact="row_influence_court.csv", err=str(e)[:80])

        # ---- CANDIDATE BAKE-OFF on the forward slice (self-tuning) -------------
        def fwd_corr(weights):
            blend = sum(weights[nm] * fwd_all[nm] for nm in weights if nm in fwd_all)
            return pearson(yf, blend)

        # worst-world + input-diversity reselection (the measured v12 fix)
        wq10 = {r["member"]: r["world_q10"] for r in gap_rows}
        ranked = sorted((nm for nm in members if member_lessons[nm].oof_corr > 0),
                        key=lambda nm: -wq10.get(nm, -1))
        k_target = min(len(members), max(len(w0), 6), cfg.MAX_MEMBERS)
        chosen = []
        for nm in ranked:
            cs = set(member_lessons[nm].used_cols)
            if any(_jac(cs, member_lessons[s].used_cols) > cfg.FORENSIC_DIVERSITY_JACCARD for s in chosen):
                continue
            chosen.append(nm)
            if len(chosen) >= k_target:
                break
        ww_weights = {nm: 1.0 / len(chosen) for nm in chosen} if chosen else dict(w0)
        ww_corr = fwd_corr(ww_weights)

        cand_rows = [{"config": "incumbent", "members": len(w0),
                      "forward_corr": round(incumbent_fwd_corr, 5), "applied": True},
                     {"config": "worst_world_diversity", "members": len(ww_weights),
                      "forward_corr": round(ww_corr, 5),
                      "applied": bool(ww_corr > incumbent_fwd_corr + cfg.FORENSIC_MARGIN)}]

        # ---- repair ACTIONS measured on forward (reported; applied only if a
        #      clean composable winner). row-quarantine + regime-split. ----------
        action_rows = []
        if cfg.FORENSIC_ACTIONS and len(w0) >= 1:
            try:
                nm0 = max(w0, key=lambda nm: member_lessons[nm].oof_corr)
                l0 = member_lessons[nm0]
                st0 = fit_skill(l0.skill, spec_lookup[l0.key], X_full[past], y_full[past],
                                seg_full[past], cols, np.random.default_rng(l0.seed), cfg, l0.seed)
                pred_past = predict_skill(st0, X_full[past])
                o = _fz(pred_past)
                sl = float(np.mean(o * (y_full[past] - np.mean(y_full[past]))))
                rzz = np.abs(y_full[past] - sl * o)
                keep_past = past[rzz < np.quantile(rzz, cfg.FORENSIC_INFLUENCE_Q)]
                qparts = {}
                for nm in w0:
                    l = member_lessons[nm]
                    st = fit_skill(l.skill, spec_lookup[l.key], X_full[keep_past], y_full[keep_past],
                                   seg_full[keep_past], cols, np.random.default_rng(l.seed), cfg, l.seed)
                    qparts[nm] = _fz(predict_skill(st, X_full[future]))
                q_corr = pearson(yf, sum(w0[nm] * qparts[nm] for nm in w0))
                action_rows.append({"action": "row_quarantine", "trigger": "top_residual_rows",
                                    "forward_corr": round(q_corr, 5),
                                    "delta_vs_incumbent": round(q_corr - incumbent_fwd_corr, 5),
                                    "status": "measured_only"})
            except Exception as e:
                log("forensic_action_skipped", action="row_quarantine", err=str(e)[:80])
            try:
                nm0 = max(w0, key=lambda nm: member_lessons[nm].oof_corr)
                l0 = member_lessons[nm0]
                tp = ATLAS.assign(X_full[past]) if ATLAS is not None else None
                tf = ATLAS.assign(X_full[future]) if ATLAS is not None else None
                if tp is not None:
                    split_pred = np.zeros(len(future), np.float64)
                    glob = fit_skill(l0.skill, spec_lookup[l0.key], X_full[past], y_full[past],
                                     seg_full[past], cols, np.random.default_rng(l0.seed), cfg, l0.seed)
                    split_pred[:] = predict_skill(glob, X_full[future])
                    for t in np.unique(tp):
                        mp = past[tp == t]
                        mf = np.where(tf == t)[0]
                        if len(mp) >= 500 and len(mf) >= 20:
                            ste = fit_skill(l0.skill, spec_lookup[l0.key], X_full[mp], y_full[mp],
                                            seg_full[mp], cols, np.random.default_rng(l0.seed + 1), cfg, l0.seed + 1)
                            split_pred[mf] = predict_skill(ste, X_full[future[mf]])
                    s_corr = pearson(yf, _fz(split_pred))
                    base_corr = pearson(yf, fwd_all.get(nm0, np.zeros(len(future))))
                    action_rows.append({"action": "regime_split", "trigger": f"per_terrain_experts:{nm0}",
                                        "forward_corr": round(s_corr, 5),
                                        "delta_vs_incumbent": round(s_corr - base_corr, 5),
                                        "status": "measured_only"})
            except Exception as e:
                log("forensic_action_skipped", action="regime_split", err=str(e)[:80])
        if action_rows:
            write_csv(pd.DataFrame(action_rows), "expert_dispatch_log.csv")
            write_csv(pd.DataFrame([{"action_id": i, "source": "forensic_self_tuner",
                                     "action_type": r["action"], "expected_effect": r["delta_vs_incumbent"],
                                     "status": r["status"]} for i, r in enumerate(action_rows)]),
                      "backward_feedback_actions.csv")

        # ---- v22 ROBUST MULTI-PARTITION SELECTION (the spine) -----------------
        # The forward slice is ONE noisy sample. v22 instead scores each candidate
        # shipping config across many structure-aware train/test partitions and
        # overrides only if a config ROBUSTLY beats the incumbent. No-op-safe.
        write_csv(pd.DataFrame(cand_rows), "forensic_self_tuning.csv")

        def _norm(wd):
            t = sum(wd.values()) or 1.0
            return {k: v / t for k, v in wd.items()}

        cand_weights = {"incumbent": _norm(dict(w0)),
                        "worst_world_diversity": ww_weights,
                        "equal_all": {nm: 1.0 / len(members) for nm in members}}
        # v23 diverse_families: one best member (by world floor) per DISTINCT
        # viewport family, equal-weighted -- a maximally-cross-family config so
        # the robust selector ALWAYS has a diverse alternative on the menu (the
        # structural opposite of the v19 0.0749 single-family blend).
        by_fam: dict[str, str] = {}
        for nm in members:
            fam = member_lessons[nm].family
            if fam not in by_fam or wq10.get(nm, -1) > wq10.get(by_fam[fam], -1):
                by_fam[fam] = nm
        if len(by_fam) >= 2:
            cand_weights["diverse_families"] = {nm: 1.0 / len(by_fam) for nm in by_fam.values()}
        # the 11th-place shape: one simple/robust/sparse linear OR positional-
        # order member, shipped ALONE (fixes the v22 bug that matched the KIND
        # "ols" against the skill NAME and so never fired on the skill axis).
        pos = [nm for nm in members
               if member_lessons[nm].skill in ("linear_ols", "huber_linear", "elastic_net")
               or member_lessons[nm].family in ("tail", "head", "mid")]
        if pos:
            cand_weights["positional_ols_single"] = {max(pos, key=lambda nm: wq10.get(nm, -1)): 1.0}
        # v27 governor: a low-COMPLEXITY anchor config (the best-generalizing
        # simple members, equal-weighted) so the selector has a genuinely simple
        # alternative to pull toward WHEN the measured decay~complexity slope says
        # capacity overfits here. On a capacity-friendly dataset lambda~0 and this
        # just competes on robust score like any other candidate (no-op-safe).
        gov_cmap = GOVERNOR.get("complexity", {}) if GOVERNOR else {}
        if gov_cmap:
            simple = sorted((nm for nm in members if member_lessons[nm].oof_corr > 0),
                            key=lambda nm: gov_cmap.get(nm, 1.0))[:max(cfg.MIN_BLEND_MEMBERS, 6)]
            if len(simple) >= 2:
                cand_weights["complexity_anchor"] = {nm: 1.0 / len(simple) for nm in simple}

        if cfg.ROBUST_OOS_SELECT:
            robust = robust_oos_select(cand_weights, members, member_lessons, spec_lookup,
                                       X_full, y_full, seg_full, n_work, cols, cfg)
        else:
            robust = {"best": "incumbent", "weights": cand_weights["incumbent"],
                      "hedged": False, "scores": [], "partitions": 0}
        if robust["scores"]:
            write_csv(pd.DataFrame(robust["scores"]), "robust_oos_selection.csv")
        chosen_name, chosen_w = robust["best"], robust["weights"]
        chosen_fwd = fwd_corr(chosen_w)
        inc_rs = next((r["robust_score"] for r in robust["scores"] if r["config"] == "incumbent"), None)
        best_rs = robust["scores"][0]["robust_score"] if robust["scores"] else None
        # v23 SELECTION DEFLATION: we compared len(cand_weights) configs across
        # the partitions, so the winner's edge is upward-biased by max-selection.
        # Raise the override bar by ROBUST_DEFLATE*log(#configs) -- ship an
        # override only if it survives the multiplicity it was found under.
        n_cand = max(2, len(cand_weights))
        override_margin = cfg.FORENSIC_MARGIN + cfg.ROBUST_DEFLATE * math.log(n_cand)
        override = (chosen_name != "incumbent" and best_rs is not None and inc_rs is not None
                    and best_rs > inc_rs + override_margin)
        if override:
            dec.update(override=True, weights=chosen_w, is_median=False, weather_states=None,
                       winner=f"robust:{chosen_name}", fwd_parts=fwd_all, forward_blend_corr=chosen_fwd)
            log("ROBUST_OVERRIDE", winner=chosen_name, partitions=robust["partitions"],
                hedged=robust["hedged"], incumbent_robust=inc_rs, chosen_robust=best_rs,
                deflated_margin=round(override_margin, 5), forward=round(chosen_fwd, 5),
                note="config robustly beat the incumbent across many train/test partitions")
        else:
            dec["fwd_parts"] = fwd_all
            log("robust_no_override", best=chosen_name, partitions=robust["partitions"],
                incumbent_robust=inc_rs, chosen_robust=best_rs,
                deflated_margin=round(override_margin, 5),
                note="incumbent kept; no config robustly beat it across the partitions")
        write_json({"override": dec["override"], "winner": dec["winner"],
                    "robust_partitions": robust["partitions"], "hedged": robust["hedged"],
                    "override_margin": round(override_margin, 5),
                    "robust_scores": robust["scores"],
                    "incumbent_forward_corr": round(incumbent_fwd_corr, 5),
                    "chosen_forward_corr": round(chosen_fwd, 5),
                    "worst_world_forward_corr": round(ww_corr, 5),
                    "forensic_margin": cfg.FORENSIC_MARGIN,
                    "candidates_forward": cand_rows, "actions_measured": action_rows,
                    "note": "selected across many structure-aware train/test partitions with block-"
                            "bootstrap; ties hedged; no-op unless a config ROBUSTLY beats the incumbent"},
                   "forensic_selection_decision.json")
        return dec
    except Exception as e:
        log("forensic_layer_failed_noop", err=str(e)[:120])
        return dec


