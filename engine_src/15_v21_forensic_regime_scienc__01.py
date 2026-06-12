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
        if getattr(cfg, "ROBUST_INTERIOR", True):
            # v28 INTERIOR-BLOCK CV (4th-place geometry): train on the OLDEST +
            # NEWEST segments, validate on the bracketed MIDDLE -- punishes
            # configs that only fit the recent regime (winning here means the
            # signal holds across the whole gap the train brackets).
            for j, (lo, hi) in enumerate(((nseg // 3, (2 * nseg) // 3),
                                          (nseg // 4, (3 * nseg) // 4))):
                mid = segs[lo:hi]
                if mid and lo > 0 and hi < nseg:
                    parts.append((f"interior{j}", ~np.isin(segw, mid), np.isin(segw, mid)))
        if getattr(cfg, "TESTLIKE_PARTITIONS", False) and TESTLIKE is not None:
            # v31 TEST-LIKENESS partitions (IDEAS_ZOO B1): validate on the working
            # rows that LOOK most like the test distribution (X-only sensor) --
            # the court now contains worlds shaped like the world being predicted.
            tl = TESTLIKE[rows]
            for j, frac in enumerate(cfg.TESTLIKE_FRACS):
                thr = np.quantile(tl, 1.0 - float(frac))
                te_m = tl >= thr
                parts.append((f"testlike{j}", ~te_m, te_m))
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


