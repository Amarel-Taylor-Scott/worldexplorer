# ----------------------------------------------------------------------------
# 9. Predator persona (falsification engine; executes + extends the null tax)
# ----------------------------------------------------------------------------

class PredatorEngine:
    """Immune-system analog: a persona whose budget is spent trying to KILL
    promoted lessons rather than find new ones. Five attacks per target:

      sub-period : worst 3-consecutive-segment mean corr from the stored OOF
                   (free). A lesson clearly harmful for a sustained stretch
                   is killed ('dead_subperiod').
      terrain    : v8, free. Worst populated-terrain corr from the stored OOF
                   against the target-free atlas. A lesson clearly harmful
                   inside one whole valley is killed ('dead_terrain') -- the
                   time attack's spatial twin.
      weather    : v9, free. Worst populated weather-band corr against the
                   target-free gauge ('dead_weather') -- a lesson that drowns
                   whenever it storms never ships.
      null probe : one full CV on within-segment-shuffled labels (cost 1).
                   The pooled |null| distribution sets the multiplicity tax
                   q = quantile(NULL_Q); deflated_corr = oof_corr - q <= 0
                   kills ('failed_null_tax') -- this IS the v4 null tax.
      perturb    : shrunken-viewport draft, k -> 0.8k (cost 1). A promoted
                   lesson whose draft collapses to non-positive width under a
                   mild viewport perturbation is killed ('fragile_viewport').

    Survivors keep their promotion; kills become decision='predator_killed'
    and never reach the member pool."""

    def __init__(self, cfg: HarnessConfig, library: SharedLibrary,
                 spec_lookup: dict[str, ViewportSpec], budget: int | None = None,
                 exclude: set[str] | None = None) -> None:
        self.cfg = cfg
        self.library = library
        self.spec_lookup = spec_lookup
        self.budget = cfg.PREDATOR_BUDGET if budget is None else int(budget)
        self.exclude = exclude or set()   # keys already attacked in an earlier raid

    def run(self, X: np.ndarray, y: np.ndarray, seg: np.ndarray, cols: list[str],
            embargo: int) -> pd.DataFrame:
        cfg = self.cfg
        targets = sorted((l for l in self.library.promoted() if l.key not in self.exclude),
                         key=lambda l: -l.oof_corr)[: cfg.PREDATOR_MAX_TARGETS]
        if not targets:
            log("predator_skipped", reason="no promotions to attack")
            return pd.DataFrame()
        rng = np.random.default_rng(stable_seed(cfg.SEED, "predator"))
        y_null = y.copy()
        for s in np.unique(seg):
            m = np.where(seg == s)[0]
            y_null[m] = y_null[m[rng.permutation(len(m))]]
        segs = np.unique(seg)
        terr = ATLAS.assign(X) if ATLAS is not None else None
        wth = GAUGE.assign(X) if GAUGE is not None else None
        bcn = BEACONS.assign(X) if BEACONS is not None else None     # v15 beacon basins
        if META is not None and META.enabled:
            # v12 metabolism: with hours in the larder every target affords
            # the FULL attack suite (null + perturb); raids stop on the clock
            self.budget = max(self.budget, 2 * len(targets))
        log("predator_start", targets=len(targets), budget=self.budget,
            terrain_attack=terr is not None, weather_attack=wth is not None,
            beacon_attack=bcn is not None)

        rows, null_corrs = [], []
        for l in targets:
            if META is not None and not META.allow("predator"):
                log("predator_out_of_time", attacked=len(rows),
                    remaining_targets=len(targets) - len(rows))
                break
            spec = self.spec_lookup[l.key]

            # attack 1 (free): worst sustained sub-period from the stored OOF
            per = [pearson(y[seg == s], l.oof[seg == s]) for s in segs]
            win = min(3, len(per))
            worst3 = float(min(np.mean(per[i:i + win]) for i in range(len(per) - win + 1)))
            l.worst3_corr = worst3

            # attack 1b (free, v13): the trail's most recent ground -- regime
            # decay is this dataset's measured enemy, so the predator hunts it
            recent = float(np.mean(per[-min(2, len(per)):]))
            l.recent_corr = recent

            # attack 2 (free, v8): worst populated-terrain corr -- spatial twin
            t_min = float("nan")
            if terr is not None:
                tc = [pearson(y[terr == t], l.oof[terr == t])
                      for t in np.unique(terr) if (terr == t).sum() >= cfg.TERRAIN_MIN_ROWS]
                if tc:
                    t_min = float(min(tc))
            l.terrain_min_corr = t_min

            # attack 2b (free, v9): worst populated weather-band corr
            w_min = float("nan")
            if wth is not None:
                wc = [pearson(y[wth == s], l.oof[wth == s])
                      for s in np.unique(wth) if (wth == s).sum() >= cfg.WEATHER_MIN_ROWS]
                if wc:
                    w_min = float(min(wc))
            l.weather_min_corr = w_min

            # attack 2c (free, v15): worst populated BEACON-BASIN corr -- a
            # trail that dies at a dropped landmark (unique typology) is fragile
            b_min = float("nan")
            if bcn is not None:
                bc = [pearson(y[bcn == b], l.oof[bcn == b])
                      for b in np.unique(bcn) if (bcn == b).sum() >= cfg.TERRAIN_MIN_ROWS]
                if bc:
                    b_min = float(min(bc))
            l.beacon_min_corr = b_min

            # attack 3 (cost 1): null probe on shuffled labels, full pipeline
            nc = float("nan")
            if self.budget >= 1:
                oof_n = np.zeros(len(y_null), np.float32)
                rng_l = np.random.default_rng(stable_seed(cfg.SEED, "predator_null", l.key))
                for tr, va in purged_segment_splits(seg, cfg.N_SPLITS, embargo):
                    state = fit_skill(l.skill, spec, X[tr], y_null[tr], seg[tr], cols, rng_l, cfg, l.seed)
                    oof_n[va] = predict_skill(state, X[va])
                nc = abs(pearson(y_null, oof_n))
                null_corrs.append(nc)
                self.budget -= 1
            l.null_corr = nc

            # attack 4 (cost 1): shrunken-viewport perturbation draft.
            # v12 PERTURB_ALL: v9+v11 measured this attack NEVER firing -- the
            # cost gate excluded every cost-2/3 champion family. Now every
            # cost class affords it (a draft of a cheap skill is cheap too).
            pw = float("nan")
            if self.budget >= 1 and (cfg.PERTURB_ALL
                                     or SKILL_REGISTRY[l.skill]["cost"] >= cfg.DRAFT_MIN_COST):
                k2 = max(cfg.K_MIN, int(spec.k * 0.8))
                spec2 = ViewportSpec(name=f"{spec.family}{k2}_{spec.transform}", family=spec.family,
                                     k=k2, transform=spec.transform, proj_dim=spec.proj_dim)
                d = run_draft(l.skill, spec2, X, y, seg, cols, cfg, embargo,
                              stable_seed(cfg.SEED, "predator_perturb", l.key))
                pw = d["draft_width"]
                self.budget -= 1
            l.perturb_width = pw

            # attack 5 (cost 1, v18): the PALINDROME -- refit on TIME-REVERSED
            # training rows and score on the ordinary order. Structural
            # relationships survive reversal; momentum/autocorrelation mirages
            # invert or collapse. A free falsification axis nobody uses. Kills
            # only on a clear INVERSION (reversed corr strongly negative while
            # the trail is positive) -- conservative so it can't wrongly cull.
            palin = float("nan")
            if self.budget >= 1 and l.oof_corr > 0.02:
                try:
                    rev = slice(None, None, -1)
                    spec_p = self.spec_lookup[l.key]
                    d = run_draft(l.skill, spec_p, X[rev], y[rev], seg[rev], cols, cfg, embargo,
                                  stable_seed(cfg.SEED, "predator_palindrome", l.key))
                    palin = d["draft_corr"]
                except Exception:
                    palin = float("nan")
                self.budget -= 1
            l.palindrome_corr = palin

            rows.append({"key": l.key, "oof_corr": l.oof_corr, "wf_corr": l.wf_corr,
                         "worst3_seg_corr": worst3, "recent2_corr": recent,
                         "terrain_min_corr": t_min, "weather_min_corr": w_min,
                         "beacon_min_corr": b_min, "palindrome_corr": palin,
                         "null_abs_corr": nc, "perturb_width": pw})
            log("predator_attack", key=l.key, worst3=round(worst3, 4),
                terrain_min=round(t_min, 4) if np.isfinite(t_min) else None,
                weather_min=round(w_min, 4) if np.isfinite(w_min) else None,
                beacon_min=round(b_min, 4) if np.isfinite(b_min) else None,
                null=round(nc, 4) if np.isfinite(nc) else None,
                palindrome=round(palin, 4) if np.isfinite(palin) else None,
                perturb_w=round(pw, 4) if np.isfinite(pw) else None, budget=self.budget)

        q = float(np.quantile(null_corrs, cfg.NULL_Q)) if null_corrs else 0.0
        kills = 0
        for l, row in zip(targets, rows):
            reasons = []
            if np.isfinite(l.null_corr):
                l.deflated_corr = l.oof_corr - q
                if l.deflated_corr <= 0:
                    reasons.append("failed_null_tax")
            if l.worst3_corr < cfg.PREDATOR_WORST3_FLOOR:
                reasons.append("dead_subperiod")
            if np.isfinite(l.recent_corr) and l.recent_corr < cfg.PREDATOR_RECENT_FLOOR:
                reasons.append("fading_trail")      # v13: dies on the newest ground
            if np.isfinite(l.terrain_min_corr) and l.terrain_min_corr < cfg.PREDATOR_TERRAIN_FLOOR:
                reasons.append("dead_terrain")
            if np.isfinite(l.weather_min_corr) and l.weather_min_corr < cfg.PREDATOR_WEATHER_FLOOR:
                reasons.append("dead_weather")
            if np.isfinite(l.beacon_min_corr) and l.beacon_min_corr < cfg.PREDATOR_BEACON_FLOOR:
                reasons.append("dead_beacon")     # v15: dies at a dropped landmark
            if (np.isfinite(l.palindrome_corr) and l.oof_corr > 0.02
                    and l.palindrome_corr < cfg.PREDATOR_PALINDROME_FLOOR):
                reasons.append("time_inverted")   # v18: flips sign under time reversal (momentum mirage)
            if np.isfinite(l.perturb_width) and l.perturb_width <= 0 and l.width > 0.01:
                reasons.append("fragile_viewport")
            row["null_q"] = q
            row["deflated_corr"] = l.deflated_corr
            row["verdict"] = ("killed:" + "|".join(reasons)) if reasons else "survived"
            l.predator_verdict = row["verdict"]
            if reasons:
                l.decision = "predator_killed"
                l.reason = (l.reason + "|" if l.reason else "") + "|".join(reasons)
                kills += 1
                # v10 venom memory: one-trial fear on the killed motif
                TABOO[f"{l.skill}|{l.transform}"] = TABOO.get(f"{l.skill}|{l.transform}", 0.0) + 1.0
                TABOO[f"{l.skill}|{l.family}"] = TABOO.get(f"{l.skill}|{l.family}", 0.0) + 1.0
                # v14 repellent stigmergy: lay RED pheromone on the columns the
                # killed trail used -- future explorers smell the poison ground
                for ci in l.used_cols:
                    RED_MYCELIUM[ci] = RED_MYCELIUM.get(ci, 0.0) + 1.0
        log("predator_done", targets=len(targets), null_q=round(q, 5), killed=kills,
            budget_left=self.budget)
        return pd.DataFrame(rows)


