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
        # v33 WIDE-CONFIGURATION GRID (user-directed): grid_* preference
        # configs -- wide / agreeing / stable channel compositions vs the
        # sharp greedy control, all judged by the same court below.
        if getattr(cfg, "CONFIG_GRID", False):
            try:
                grid_cands = config_grid_candidates(members, member_lessons, cfg)
                cand_weights.update(grid_cands)
                if grid_cands:
                    log("config_grid", configs=len(grid_cands),
                        names="|".join(grid_cands),
                        note="preference grid over the measured pool; the robust court arbitrates")
            except Exception as e:
                log("config_grid_skipped", err=str(e)[:80])

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


