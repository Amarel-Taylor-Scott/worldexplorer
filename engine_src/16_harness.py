class ExplorerHarness:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        np.random.seed(cfg.SEED)

    def run(self) -> dict[str, Any]:
        global ATLAS, GAUGE
        cfg = self.cfg
        MYCELIUM.clear()                 # fresh pheromone network every run
        TRAPS.clear()                    # fresh threat map (v10)
        TABOO.clear()                    # fresh venom memory (v10)
        SURVEY.clear()                   # fresh satellite map (v10)
        QUORUM.clear()                   # fresh colony consensus (v11)
        DANCES.clear()                   # fresh waggle floor (v11)
        GENE_POOL.clear()                # fresh plasmid pool (v11)
        RED_MYCELIUM.clear()             # fresh repellent channel (v14)
        GOVERNOR.clear()                 # v27: fresh runtime complexity-generalization governor
        LEDGER_PRIOR.clear()             # v27: fresh cross-run learning ledger (repopulated from prior cairn below)
        global FCLUST, HABITAT
        FCLUST = None; HABITAT = None; QUARANTINE.clear()   # fresh forensic sensors (v21)
        SEEDBANK.clear()                 # v14: refilled below from a prior run's cairn, if any

        # ---- v14 SEED BANK: germinate a prior run's measured losers ----------
        # Cross-run temporal biodiversity: warm starts carried only WINNERS, so
        # diversity a later regime shift would reward was forgotten. Load the
        # previous cairn's seed bank (best non-champion genomes) EARLY so
        # evolution's generation-0 warm start can re-measure them.
        try:
            seed_paths = [Path(p) for p in cfg.CAIRN_PATHS]
            try:
                seed_paths += list(Path("/kaggle/input").glob("*/world_cairn.json"))
            except Exception:
                pass
            for pth in seed_paths:
                if pth.exists():
                    prev = json.loads(pth.read_text())
                    for wk in (prev.get("seed_bank") or [])[: cfg.SEED_GERMINATE]:
                        g = parse_genome_key(wk)
                        if g is not None:
                            SEEDBANK.append(g)
                    # v27 self-improvement: load the cross-run learning ledger as priors
                    if cfg.SELF_IMPROVE and prev.get("ledger"):
                        LEDGER_PRIOR.clear(); LEDGER_PRIOR.update(prev["ledger"])
                        for wk in (LEDGER_PRIOR.get("survivors") or [])[: cfg.SEED_GERMINATE]:
                            gs = parse_genome_key(wk)
                            if gs is not None and all(s.key != gs.key for s in SEEDBANK):
                                SEEDBANK.append(gs)
                        for mk in (LEDGER_PRIOR.get("decayers") or [])[: cfg.LEDGER_MAX_DECAYERS]:
                            TABOO[mk] = TABOO.get(mk, 0.0) + 1.0     # cross-run anti-prior: avoid decayed motifs
                        gp = LEDGER_PRIOR.get("governor") or {}
                        log("learning_ledger_loaded", governor_runs=int(gp.get("count", 0)),
                            prior_beta=round(float(gp.get("beta", 0.0)), 4),
                            families=len(LEDGER_PRIOR.get("family_decay") or {}),
                            survivors=len(LEDGER_PRIOR.get("survivors") or []),
                            decayers=len(LEDGER_PRIOR.get("decayers") or []))
                    if SEEDBANK:
                        log("seedbank_loaded", count=len(SEEDBANK), from_cairn=str(pth),
                            keys="|".join(g.key for g in SEEDBANK))
                    break
        except Exception:
            pass

        def circadian(frac_target: float, tag: str) -> None:
            """v10 governor: the body has a clock. If the elapsed fraction of
            RUN_DEADLINE_MIN exceeds this phase's scheduled fraction, shed
            cost gracefully -- the organism always makes camp before dark."""
            if not cfg.RUN_DEADLINE_MIN or cfg.RUN_DEADLINE_MIN <= 0:
                return
            frac = (time.monotonic() - RUN_START) / 60.0 / cfg.RUN_DEADLINE_MIN
            if frac <= frac_target:
                return
            over = frac / max(frac_target, 1e-9)
            cuts = []
            if cfg.SEED_REPS_STOCHASTIC > 0:
                cfg.SEED_REPS_STOCHASTIC = 0
                cuts.append("seed_reps->0")
            new_gen = max(1, int(cfg.EVOLUTION_MAX_GENERATIONS / over))
            if new_gen < cfg.EVOLUTION_MAX_GENERATIONS:
                cfg.EVOLUTION_MAX_GENERATIONS = new_gen
                cuts.append(f"generations->{new_gen}")
            if cfg.DREAM_REPLAYS > 30:
                cfg.DREAM_REPLAYS = max(30, cfg.DREAM_REPLAYS // 2)
                cuts.append(f"dream_replays->{cfg.DREAM_REPLAYS}")
            if cfg.DIVE_BUDGET > 2:
                cfg.DIVE_BUDGET = max(2, cfg.DIVE_BUDGET // 2)
                cuts.append(f"dive_budget->{cfg.DIVE_BUDGET}")
            if cfg.PREDATOR_MAX_TARGETS > 4:
                cfg.PREDATOR_MAX_TARGETS = max(4, cfg.PREDATOR_MAX_TARGETS // 2)
                cuts.append(f"predator_targets->{cfg.PREDATOR_MAX_TARGETS}")
            self._circadian_cuts += cuts
            log("CIRCADIAN_CUT", tag=tag, elapsed_frac=round(frac, 3),
                target_frac=frac_target, cuts="|".join(cuts) if cuts else "none_left")

        self._circadian_cuts: list[str] = []

        # ---- v12 METABOLISM: arm the energy ledger ---------------------------
        global META
        META = Metabolism(cfg)
        if META.enabled and not cfg.RUN_DEADLINE_MIN:
            # the circadian governor becomes the metabolism's late-running
            # backstop: shed-cost cuts engage just past the metabolic budget
            cfg.RUN_DEADLINE_MIN = round(cfg.TIME_BUDGET_MIN * 1.04, 1)
            log("circadian_backstop_armed", run_deadline_min=cfg.RUN_DEADLINE_MIN,
                note="sheds cost only if shipping runs past the metabolic plan")

        def phase_frac(phase: str, fallback: float) -> float:
            """When the metabolism is armed, the circadian backstop's phase
            targets follow the metabolic plan instead of the fixed v10 map."""
            if META.enabled and cfg.RUN_DEADLINE_MIN and phase in META.deadline:
                return min(1.0, META.deadline[phase] / cfg.RUN_DEADLINE_MIN + 0.05)
            return fallback

        log("run_start", out=str(OUT), seed=cfg.SEED,
            time_budget_min=cfg.TIME_BUDGET_MIN, explorers=cfg.N_EXPLORERS,
            lesson_budget=cfg.LESSON_BUDGET, evolution_budget=cfg.EVOLUTION_BUDGET,
            gbdt_backend=str(GBDT_BACKEND), nnls=HAVE_NNLS,
            torch=HAVE_TORCH, gpus=N_GPUS)

        # hardware profile: same objective, the expensive skills reshape per device
        if N_GPUS > 0:
            cfg.MLP_HIDDEN = cfg.GPU_MLP_HIDDEN
            cfg.MLP_MAX_ITER = cfg.GPU_MLP_MAX_ITER
            cfg.MLP_MAX_ROWS = cfg.GPU_MLP_MAX_ROWS
            cfg.MLP_BATCH = cfg.GPU_MLP_BATCH
            cfg.GBDT_ESTIMATORS = cfg.GPU_GBDT_ESTIMATORS
            log("hardware_profile", schedule="gpu", gpus=N_GPUS,
                names="|".join(_gpu_names()),
                mlp_hidden=str(cfg.MLP_HIDDEN), mlp_rows=cfg.MLP_MAX_ROWS,
                mlp_iters=cfg.MLP_MAX_ITER, gbdt_estimators=cfg.GBDT_ESTIMATORS,
                two_gpu_concurrency=N_GPUS >= 2)
        else:
            log("hardware_profile", schedule="cpu",
                note="identical_v5_behavior" if not HAVE_TORCH else "torch_cpu_mlp_with_pearson_loss")
        hetero = bool(cfg.HETERO_PAIRING) and N_GPUS > 0
        if hetero:
            log("hetero_lanes_enabled",
                note="gpu-lane and cpu-lane lessons run simultaneously (phase 1 + evolution)")
        personas = [{k: v for k, v in t.items() if not isinstance(v, dict)} for t in EXPLORER_TRAITS]
        personas.append({"name": "predator_skeptic", "metaheuristic": "immune_system_falsifier",
                         "curiosity": 0.0, "caution": 1.0, "sociality": 1.0})
        write_csv(pd.DataFrame(personas), "explorer_personas.csv")

        # ---- data ----------------------------------------------------------
        root = find_data_root()
        if root is not None:
            train, test = load_competition(root)
            data_source = "competition"
        else:
            if not cfg.ALLOW_SYNTHETIC_FALLBACK:
                raise SystemExit("DRW data not found and synthetic fallback disabled.")
            print("=" * 78)
            print("!! NO COMPETITION DATA FOUND -> RUNNING ON A LABELED SYNTHETIC SMOKE FIXTURE")
            print("!! Attach the DRW dataset for real results. All artifacts are tagged.")
            print("=" * 78)
            train, test = make_synthetic(cfg.SYN_ROWS, cfg.SYN_ANON, cfg.SEED)
            data_source = "SYNTHETIC_SMOKE"
        log("data_loaded", source=data_source, train_rows=len(train), test_rows=len(test))

        eng_tr = add_market_features(train)
        eng_te = add_market_features(test)
        if not eng_tr.empty:
            train = pd.concat([eng_tr, train], axis=1)
            test = pd.concat([eng_te, test], axis=1)
        del eng_tr, eng_te
        gc.collect()

        cols = [c for c in train.columns
                if c not in ("label", "timestamp") and pd.api.types.is_numeric_dtype(train[c])]
        cols = [c for c in cols if c in test.columns]
        y_full = train["label"].to_numpy(np.float32)
        n = len(train)
        seg_full = (np.arange(n) * cfg.N_SEGMENTS // n).astype(np.int32)
        medians = train[cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        vol_full = train["volume"].to_numpy(np.float32) if "volume" in train.columns else None

        log("building_matrices", features=len(cols))
        X_full = build_matrix(train, cols, medians)
        X_test = build_matrix(test, cols, medians)
        del train, test
        gc.collect()
        mem_status("post_data_matrices")

        # ---- v26 DATA PROFILE + GEOMETRY SEAM (generalization) ---------------
        # Detect target type + time-order; under cfg.METRIC/GEOMETRY="auto" this
        # picks the active metric and CV geometry. DEFAULTS (pearson/temporal)
        # reproduce DRW exactly. For NON-temporal data, ONE row-permutation turns
        # all positional CV (segments, walk-forward, sealed/forward tails) into
        # honest RANDOM CV -- so the same harness runs on any tabular dataset.
        global PROFILE
        _prof = profile_data(X_full, y_full, cols, cfg)
        PROFILE = resolve_profile(_prof, cfg)
        write_json({**_prof, "active_metric": PROFILE["metric"],
                    "active_temporal": PROFILE["temporal"],
                    "metric_cfg": cfg.METRIC, "geometry_cfg": cfg.GEOMETRY}, "data_profile.json")
        log("data_profile", target_kind=PROFILE["target_kind"], metric=PROFILE["metric"],
            temporal=PROFILE["temporal"], feature_autocorr=_prof["feature_autocorr"],
            note="defaults (pearson/temporal) reproduce v25; 'auto' adapts to any dataset")
        if not PROFILE["temporal"]:
            _perm = np.random.default_rng(cfg.SEED + 777).permutation(n)
            X_full = X_full[_perm]; y_full = y_full[_perm]
            if vol_full is not None:
                vol_full = vol_full[_perm]
            log("geometry_randomized",
                note="non-temporal data: rows permuted so positional CV (segments/WF/sealed) becomes random CV")

        # ---- SEALED HOLDOUT quarantine ---------------------------------------
        # The final SEALED_FRACTION of rows is invisible to every decision below.
        seal_cut = int((1 - cfg.SEALED_FRACTION) * n)
        n_work = seal_cut
        sealed_idx = np.arange(seal_cut, n)
        log("sealed_holdout_quarantined", sealed_rows=len(sealed_idx), working_rows=n_work,
            note="evaluated exactly once after the blend is frozen; never gated on")

        # ---- probe subsample from the WORKING region only --------------------
        if n_work > cfg.PROBE_MAX_ROWS:
            keep = np.unique(np.linspace(0, n_work - 1, cfg.PROBE_MAX_ROWS).round().astype(int))
        else:
            keep = np.arange(n_work)
        Xp, yp, segp = X_full[keep], y_full[keep], seg_full[keep]
        volp = vol_full[keep] if vol_full is not None else None
        embargo_p = max(1, int(cfg.EMBARGO_ROWS * len(keep) / max(1, n_work)))
        log("probe_ready", probe_rows=len(keep), segments=int(len(np.unique(segp))),
            embargo_rows=embargo_p, splits=cfg.N_SPLITS, wf_folds=cfg.WF_FOLDS)

        # ---- TERRAIN ATLAS: target-free map of the space (working X only) -----
        # y never touches the atlas, so its ids are leak-free everywhere; the
        # 'terrain' family, terrain_router skill, predator terrain attack and
        # all per-terrain reports read from this one map.
        ATLAS = TerrainAtlas(cfg.TERRAIN_CLUSTERS, cfg.SEED).fit(Xp, cols, cfg.TERRAIN_FIT_ROWS)
        terr_p = ATLAS.assign(Xp)
        t_pop = {int(t): int((terr_p == t).sum()) for t in np.unique(terr_p)}
        alt_p = ATLAS.altitude(Xp)
        write_csv(pd.DataFrame([{"terrain": t, "rows": npop,
                                 "frac": npop / len(terr_p),
                                 "mean_altitude": float(np.mean(alt_p[terr_p == t])),
                                 "max_altitude": float(np.max(alt_p[terr_p == t]))}
                                for t, npop in sorted(t_pop.items())]),
                  "terrain_atlas_report.csv")
        log("terrain_atlas_built", clusters=len(t_pop), populations=str(t_pop),
            note="target_free__leakfree_everywhere")

        # ---- WEATHER GAUGE (v9): row-local volatility states, target-free ----
        GAUGE = WeatherGauge(cfg.WEATHER_STATES).fit(Xp, cols)
        wth_p = GAUGE.assign(Xp)
        w_pop = {int(s): int((wth_p == s).sum()) for s in np.unique(wth_p)}
        log("weather_gauge_built", states=len(w_pop), populations=str(w_pop),
            note="row_local__order_free__target_free")

        # ---- ADVERSARIAL VALIDATION (v24): early-vs-late covariate-shift map --
        try:
            adv_df, adv_auc = adversarial_validation_report(Xp, cols, cfg)
            if not adv_df.empty:
                write_csv(adv_df, "adversarial_validation.csv")
            log("adversarial_validation", auc=round(adv_auc, 4),
                top_shift=str(adv_df["feature"].iloc[0]) if not adv_df.empty else None,
                note="early-vs-late drift; AUC~0.5 stable, >>0.5 strong shift; target-free, gates nothing")
        except Exception as e:
            log("adversarial_validation_skipped", err=str(e)[:80])

        # ---- BEACON FIELD (v15): drop items at unique typologies -------------
        # Items placed at rare-terrain + novelty-altitude coordinates (all
        # target-free, from the atlas) emit a radial RBF field; the field
        # becomes NEW FEATURE CHANNELS appended to the matrix, so every
        # explorer can see and bend around the landmarks. Leak-free everywhere.
        global BEACONS
        if cfg.BEACON_DROP and ATLAS is not None:
            BEACONS = BeaconField(ATLAS, cfg).fit(Xp)
            fld_full = BEACONS.field(X_full)
            fld_test = BEACONS.field(X_test)
            beacon_names = [f"beacon_{i}_{BEACONS.kinds[i]}" for i in range(fld_full.shape[1])]
            X_full = np.ascontiguousarray(np.hstack([X_full, fld_full]).astype(np.float32))
            X_test = np.ascontiguousarray(np.hstack([X_test, fld_test]).astype(np.float32))
            cols = cols + beacon_names
            Xp = X_full[keep]            # the probe now carries the field channels too
            bf_p = BEACONS.field(Xp)
            b_assign = BEACONS.assign(Xp)
            b_pop = {int(b): int((b_assign == b).sum()) for b in np.unique(b_assign)}
            write_csv(pd.DataFrame([{"beacon": i, "kind": BEACONS.kinds[i],
                                     "sigma": round(float(BEACONS.sigmas[i]), 4),
                                     "basin_rows": b_pop.get(i, 0),
                                     "mean_activation": round(float(bf_p[:, i].mean()), 4),
                                     "corr_to_y": round(pearson(yp, bf_p[:, i]), 4)}
                                    for i in range(len(BEACONS.kinds))]),
                      "beacon_atlas_report.csv")
            log("beacons_dropped", n=len(BEACONS.kinds),
                rare=sum(k == "rare" for k in BEACONS.kinds),
                novelty=sum(k == "novelty" for k in BEACONS.kinds),
                features=len(cols), note="field channels appended; target_free__leakfree")
        else:
            BEACONS = None

        # ---- SYMMETRY FIELD: even-vs-odd response of the strongest features ---
        sym_df = symmetry_field_report(Xp, yp, cols)
        write_csv(sym_df, "symmetry_field_report.csv")
        # v18 label archaeology (train-side forensics on the anonymized target)
        try:
            arch_df = label_archaeology(Xp, yp, segp, cols)
            write_csv(arch_df, "label_archaeology.csv")
            if not arch_df.empty:
                bestrow = arch_df.iloc[arch_df["best_lagged_corr"].abs().argmax()]
                log("label_archaeology", best_lag=int(bestrow["lag"]),
                    best_feature=str(bestrow["best_feature"]),
                    best_lagged_corr=round(float(bestrow["best_lagged_corr"]), 4),
                    note="is y a feature's future value? train-side map, gates nothing")
        except Exception as e:
            log("label_archaeology_skipped", err=str(e)[:80])
        n_even = int((sym_df["even_excess"] > 0).sum()) if not sym_df.empty else 0
        log("symmetry_field", features=len(sym_df), even_dominant=n_even,
            note="motivates fold_abs/fold_pairs viewports (report, not a gate)")

        # ---- TRAP MAP (v10): fear scans before hunger ------------------------
        traps, trap_df = build_trap_map(Xp, yp, segp, cfg)
        TRAPS.update(traps)
        for ci in traps:                 # v14: mirages seed the repellent channel
            RED_MYCELIUM[ci] = RED_MYCELIUM.get(ci, 0.0) + 1.0
        write_csv(trap_df, "trap_map.csv")
        log("trap_map", scanned=len(trap_df), mirages=len(TRAPS),
            note="mirage features demoted in corr-driven rankings")

        # ---- SATELLITE SURVEY (v10): orbit every family before walking -------
        sat_cfg = replace(cfg, DRAFT_ROWS=max(2_000, len(keep) // cfg.SAT_STRIDE),
                          DRAFT_FOLDS=cfg.SAT_FOLDS)
        sat_rows = []
        k_sat = min(len(cols), max(8, cfg.BIT_BUDGET // TRANSFORM_BITS["quantize2"]))
        for fam in FAMILIES:
            spec_s = ViewportSpec(name=f"{fam}{k_sat}_quantize2", family=fam, k=k_sat,
                                  transform="quantize2", proj_dim=16)
            d = run_draft("majority_vote", spec_s, Xp, yp, segp, cols, sat_cfg,
                          embargo_p, stable_seed(cfg.SEED, "satellite", fam))
            SURVEY[fam] = max(0.0, float(d["draft_corr"]))
            sat_rows.append({"family": fam, "survey_corr": d["draft_corr"],
                             "survey_width": d["draft_width"], "k": k_sat})
        write_csv(pd.DataFrame(sat_rows).sort_values("survey_corr", ascending=False),
                  "survey_map.csv")
        log("satellite_survey", families=len(SURVEY),
            best=max(SURVEY, key=SURVEY.get), note="feeds bandit family priors")

        # ---- SENSORY THRESHOLD (v10): how quiet a signal can we hear? --------
        jnd = jnd_probe(Xp, yp, segp, cols, cfg, embargo_p)
        write_json(jnd, "sensory_threshold.json")
        log("sensory_threshold", jnd=jnd["jnd"],
            grid="|".join(str(s) for s in cfg.JND_STRENGTHS), note="calibration only")
        gc.collect()
        mem_status("post_atlas")

        # ---- PHASE 1: developmental explorers (with draft culling) -----------
        library = SharedLibrary()
        gate = DraftGate(cfg)
        spec_lookup: dict[str, ViewportSpec] = {}
        journal_rows, growth_rows = [], []

        def attempt_lesson(ex_name: str, stage_name: str, skill: str, spec: ViewportSpec,
                           oofs_snap: dict[str, np.ndarray]) -> tuple:
            """Draft-gate + full lesson, with NO shared-state mutation beyond the
            draft gate's append (GIL-atomic) -- safe to run in a lane thread.
            Library/journal/budget updates happen in the main thread after join."""
            key = f"{skill}|{spec.name}"
            d_width = float("nan")
            if SKILL_REGISTRY[skill]["cost"] >= cfg.DRAFT_MIN_COST:
                d = run_draft(skill, spec, Xp, yp, segp, cols, cfg, embargo_p,
                              stable_seed(cfg.SEED, "draft", key))
                d_width = d["draft_width"]
                if not gate.passes(d_width):
                    return ("culled", skill, spec, d)
                car = run_car(skill, spec, Xp, yp, segp, cols, cfg, embargo_p,
                              stable_seed(cfg.SEED, "car", key))
                if car["draft_width"] <= 0:           # v10: stalled on the road
                    car["car_stalled"] = 1.0
                    return ("culled", skill, spec, car)
            seed = stable_seed(cfg.SEED, key, library.runs.get(key, 0))
            lesson = run_lesson(ex_name, stage_name, skill, spec, Xp, yp, segp, cols,
                                cfg, embargo_p, seed, oofs_snap, draft_width=d_width)
            return ("done", skill, spec, lesson)

        # ---- PANORAMA (v10): the 360-degree freeze-and-orient scan -----------
        if cfg.PANORAMA_FIRST:
            k_pan = min(len(cols), max(8, cfg.BIT_BUDGET // TRANSFORM_BITS["sign_only"]))
            spec_pan = ViewportSpec(name=f"top{k_pan}_sign_only", family="top", k=k_pan,
                                    transform="sign_only", proj_dim=16)
            pan = run_lesson("panorama", "panorama", "majority_vote", spec_pan,
                             Xp, yp, segp, cols, cfg, embargo_p,
                             stable_seed(cfg.SEED, "panorama"), {})
            library.add(pan)
            spec_lookup[pan.key] = spec_pan
            journal_rows.append({"explorer": "panorama", "lesson_idx": 0, "stage": "panorama",
                                 "key": pan.key, "family": "top", "transform": "sign_only",
                                 "k": k_pan, "oof_corr": pan.oof_corr, "width": pan.width,
                                 "wf_corr": pan.wf_corr, "wf_width": pan.wf_width,
                                 "stability": pan.stability, "seed_var": pan.seed_var,
                                 "overfit_ratio": pan.overfit_ratio, "decision": pan.decision,
                                 "reason": pan.reason, "budget_left": 0})
            log("panorama", k=k_pan, oof_corr=round(pan.oof_corr, 4),
                width=round(pan.width, 4), decision=pan.decision,
                note="the floor and the horizon, established before anyone walks")

        # ---- AUDITION PARADE (v12): every skill gets one measured lesson ------
        # v9 measured: relay_caravan and swell_rider were NEVER picked by any
        # bandit; in v11 swell_rider finally ran (via evolution) and instantly
        # WON the run. Unmeasured is not failed -- a registry entry that never
        # runs is a silent prior. Every skill now auditions once through the
        # SAME doors (draft gate + car rung included) before UCB free play.
        auditioned = 0
        if cfg.AUDITION_ALL_SKILLS:
            k_aud = min(cfg.AUDITION_K, len(cols))
            for skill in list(SKILL_REGISTRY):
                if any(l.skill == skill for l in library.lessons):
                    continue                      # already measured (e.g. panorama)
                spec_a = ViewportSpec(name=f"top{k_aud}_identity", family="top",
                                      k=k_aud, transform="identity", proj_dim=16)
                status, _, r_spec, payload = attempt_lesson("audition", "audition",
                                                            skill, spec_a, library.oofs())
                key_a = f"{skill}|{spec_a.name}"
                if status == "culled":
                    library.note_draft_cull(key_a)
                    journal_rows.append({"explorer": "audition", "lesson_idx": auditioned,
                                         "stage": "audition", "key": key_a, "family": "top",
                                         "transform": "identity", "k": k_aud,
                                         "oof_corr": payload["draft_corr"],
                                         "width": payload["draft_width"],
                                         "wf_corr": np.nan, "wf_width": np.nan,
                                         "stability": np.nan, "seed_var": np.nan,
                                         "overfit_ratio": np.nan, "decision": "draft_culled",
                                         "reason": "audition_below_draft_bar", "budget_left": 0})
                    log("audition_culled", skill=skill,
                        draft_width=round(payload["draft_width"], 4))
                    continue
                library.add(payload)
                spec_lookup[payload.key] = r_spec
                auditioned += 1
                journal_rows.append({"explorer": "audition", "lesson_idx": auditioned,
                                     "stage": "audition", "key": payload.key, "family": "top",
                                     "transform": "identity", "k": k_aud,
                                     "oof_corr": payload.oof_corr, "width": payload.width,
                                     "wf_corr": payload.wf_corr, "wf_width": payload.wf_width,
                                     "stability": payload.stability, "seed_var": payload.seed_var,
                                     "overfit_ratio": payload.overfit_ratio,
                                     "decision": payload.decision, "reason": payload.reason,
                                     "budget_left": 0})
                log("audition", skill=skill, key=payload.key,
                    oof_corr=round(payload.oof_corr, 4), width=round(payload.width, 4),
                    decision=payload.decision)
            # v13: transforms audition too -- a way of seeing that never runs
            # is just as silent a prior as a skill that never runs. One
            # linear_assoc lesson per unseen transform, at its bit-frontier k.
            for tf in ALL_TRANSFORMS:
                if any(l.transform == tf for l in library.lessons):
                    continue
                k_t = min(cfg.AUDITION_K, len(cols),
                          max(CFG.K_MIN, cfg.BIT_BUDGET // TRANSFORM_BITS.get(tf, 32)))
                spec_t = ViewportSpec(name=f"top{k_t}_{tf}", family="top", k=k_t,
                                      transform=tf, proj_dim=16)
                if not library.can_run("linear_assoc", spec_t):
                    continue
                status, _, r_spec, payload = attempt_lesson("audition", "audition",
                                                            "linear_assoc", spec_t, library.oofs())
                key_t = f"linear_assoc|{spec_t.name}"
                if status == "culled":
                    library.note_draft_cull(key_t)
                    log("audition_culled", transform=tf,
                        draft_width=round(payload["draft_width"], 4))
                    continue
                library.add(payload)
                spec_lookup[payload.key] = r_spec
                auditioned += 1
                journal_rows.append({"explorer": "audition", "lesson_idx": auditioned,
                                     "stage": "audition", "key": payload.key, "family": "top",
                                     "transform": tf, "k": k_t,
                                     "oof_corr": payload.oof_corr, "width": payload.width,
                                     "wf_corr": payload.wf_corr, "wf_width": payload.wf_width,
                                     "stability": payload.stability, "seed_var": payload.seed_var,
                                     "overfit_ratio": payload.overfit_ratio,
                                     "decision": payload.decision, "reason": payload.reason,
                                     "budget_left": 0})
                log("audition", transform=tf, key=payload.key,
                    oof_corr=round(payload.oof_corr, 4), width=round(payload.width, 4),
                    decision=payload.decision)
            # v16: FAMILIES audition too -- a place to look that never runs is
            # as silent a prior as a skill or transform that never runs. One
            # linear_assoc identity lesson per unseen family (its own pool).
            for fam in FAMILIES:
                if any(l.family == fam for l in library.lessons):
                    continue
                k_f = min(cfg.AUDITION_K, len(cols))
                spec_f = ViewportSpec(name=f"{fam}{k_f}_identity", family=fam, k=k_f,
                                      transform="identity", proj_dim=16)
                if not library.can_run("linear_assoc", spec_f):
                    continue
                status, _, r_spec, payload = attempt_lesson("audition", "audition",
                                                            "linear_assoc", spec_f, library.oofs())
                key_f = f"linear_assoc|{spec_f.name}"
                if status == "culled":
                    library.note_draft_cull(key_f)
                    log("audition_culled", family=fam,
                        draft_width=round(payload["draft_width"], 4))
                    continue
                library.add(payload)
                spec_lookup[payload.key] = r_spec
                auditioned += 1
                journal_rows.append({"explorer": "audition", "lesson_idx": auditioned,
                                     "stage": "audition", "key": payload.key, "family": fam,
                                     "transform": "identity", "k": k_f,
                                     "oof_corr": payload.oof_corr, "width": payload.width,
                                     "wf_corr": payload.wf_corr, "wf_width": payload.wf_width,
                                     "stability": payload.stability, "seed_var": payload.seed_var,
                                     "overfit_ratio": payload.overfit_ratio,
                                     "decision": payload.decision, "reason": payload.reason,
                                     "budget_left": 0})
                log("audition", family=fam, key=payload.key,
                    oof_corr=round(payload.oof_corr, 4), width=round(payload.width, 4),
                    decision=payload.decision)
            log("audition_parade_done", auditioned=auditioned,
                note="every registered skill, transform AND family measured once (v9/v11/v13/v16 fix)")

        def run_explorer(ex: Explorer) -> None:
            stage_gains: list[float] = []
            lesson_idx = 0
            while ex.budget > 0 and META.allow("explore"):
                pick = ex.ucb_pick(library, len(cols))
                if pick is None:
                    if ex.stage_idx < ex.max_stage:
                        ex.stage_idx += 1
                        log("explorer_graduates", explorer=ex.name, to=STAGES[ex.stage_idx][0],
                            via="menu_exhausted", best=round(max(stage_gains or [0.0]), 4))
                        stage_gains = []
                        continue
                    break
                skill, spec = pick
                stage_name = STAGES[ex.stage_idx][0]

                # heterogeneous pairing: grab a second pick from the OPPOSITE
                # lane and run both lessons simultaneously
                picks = [(skill, spec)]
                if hetero:
                    other = "cpu" if lesson_lane(skill) == "gpu" else "gpu"
                    pick2 = ex.ucb_pick(library, len(cols), lane=other,
                                        exclude_key=f"{skill}|{spec.name}")
                    if pick2 is not None:
                        picks.append(pick2)

                oofs_snap = library.oofs()
                if len(picks) == 2:
                    log("hetero_pair", explorer=ex.name,
                        gpu_lane=next(f"{s}|{sp.name}" for s, sp in picks if lesson_lane(s) == "gpu"),
                        cpu_lane=next(f"{s}|{sp.name}" for s, sp in picks if lesson_lane(s) == "cpu"))
                    with ThreadPoolExecutor(max_workers=2) as exe:
                        futs = [exe.submit(attempt_lesson, ex.name, stage_name, s, sp, oofs_snap)
                                for s, sp in picks]
                        results = [f.result() for f in futs]
                else:
                    results = [attempt_lesson(ex.name, stage_name, skill, spec, oofs_snap)]

                for status, r_skill, r_spec, payload in results:
                    key = f"{r_skill}|{r_spec.name}"
                    if status == "culled":
                        stalled = bool(payload.get("car_stalled"))
                        library.note_draft_cull(key)
                        ex.budget -= 2 if stalled else 1
                        journal_rows.append({"explorer": ex.name, "lesson_idx": lesson_idx,
                                             "stage": stage_name, "key": key, "family": r_spec.family,
                                             "transform": r_spec.transform, "k": r_spec.k,
                                             "oof_corr": payload["draft_corr"], "width": payload["draft_width"],
                                             "wf_corr": np.nan, "wf_width": np.nan,
                                             "stability": np.nan, "seed_var": np.nan,
                                             "overfit_ratio": np.nan, "decision": "draft_culled",
                                             "reason": "car_stalled" if stalled else "below_draft_bar",
                                             "budget_left": ex.budget})
                        log("car_stalled" if stalled else "draft_culled", explorer=ex.name, key=key,
                            draft_width=round(payload["draft_width"], 4), budget=ex.budget)
                        continue
                    lesson = payload
                    library.add(lesson)
                    spec_lookup[lesson.key] = r_spec
                    ex.budget -= lesson.cost
                    stage_gains.append(lesson.oof_corr)
                    lesson_idx += 1
                    log("lesson", explorer=ex.name, stage=stage_name, key=key,
                        lane=lesson_lane(r_skill),
                        oof_corr=round(lesson.oof_corr, 4), wf=round(lesson.wf_corr, 4),
                        width=round(lesson.width, 4), wf_width=round(lesson.wf_width, 4),
                        overfit=round(lesson.overfit_ratio, 2),
                        decision=lesson.decision, budget=ex.budget)
                    journal_rows.append({"explorer": ex.name, "lesson_idx": lesson_idx, "stage": stage_name,
                                         "key": key, "family": r_spec.family, "transform": r_spec.transform,
                                         "k": r_spec.k, "oof_corr": lesson.oof_corr, "width": lesson.width,
                                         "wf_corr": lesson.wf_corr, "wf_width": lesson.wf_width,
                                         "stability": lesson.stability, "seed_var": lesson.seed_var,
                                         "overfit_ratio": lesson.overfit_ratio, "decision": lesson.decision,
                                         "reason": lesson.reason, "budget_left": ex.budget})
                    growth_rows.append({"explorer": ex.name, "lesson_idx": lesson_idx, "stage": stage_name,
                                        "best_corr_so_far": max(l.oof_corr for l in library.lessons
                                                                if l.explorer == ex.name)})
                if ex.maybe_graduate(stage_gains):
                    stage_gains = []

        # ---- SEASONS (v12): the roster is reborn while the larder has food ----
        # Newborns of season N+1 are recruited by season N's waggle dances,
        # walk on its mycelium, inherit its quorum switches -- and the dedup
        # caps force each new season into UNWALKED cells of the menu. v11
        # measured that mycelium only wins AFTER deposits thicken the network;
        # seasons are how deposits compound inside one run.
        META.begin("explore", cfg.MET_EXPLORE_SHARE)
        explorers: list[Explorer] = []
        season = 0
        while True:
            season += 1
            n_before = len(library.lessons)
            for ei in range(cfg.N_EXPLORERS):
                if META.enabled and not META.allow("explore"):
                    break
                ex = Explorer(EXPLORER_TRAITS[ei % len(EXPLORER_TRAITS)], cfg)
                ex.season = season
                explorers.append(ex)
                log("explorer_born", season=season, explorer=ex.name, species=ex.species,
                    metaheuristic=ex.traits["metaheuristic"], behavior=ex.behavior,
                    stage=STAGES[ex.stage_idx][0])
                run_explorer(ex)
            META.seasons = season
            if not META.wants_more("explore"):
                break
            if season >= cfg.MAX_SEASONS:
                log("seasons_capped", at=season)
                break
            if len(library.lessons) == n_before:
                log("seasons_exhausted", season=season, note="the menu is fully walked")
                break
            write_csv(pd.DataFrame(journal_rows), "explorer_journal.csv")  # crash-safe partial
            META.heartbeat(f"season_{season}_done")
            log("season_turns", next_season=season + 1, lessons_so_far=len(library.lessons),
                promoted_so_far=len(library.promoted()), elapsed_min=round(META.now(), 1))

        # ---- ATTENTION MARKET (v10): Charnov's marginal value theorem --------
        # The held-back pool flows to the explorer with the highest recent
        # marginal yield -- attention follows whoever is still learning.
        attention_grantee = None
        if cfg.ATTENTION_POOL > 0 and library.lessons and META.allow("explore"):
            def marginal_yield(name: str) -> float:
                last = [l for l in library.lessons if l.explorer == name][-3:]
                return (float(np.mean([lesson_fitness(l) / max(l.cost, 1) for l in last]))
                        if last else -1e9)
            ranked_ex = sorted(explorers, key=lambda e: marginal_yield(e.name), reverse=True)
            winner = ranked_ex[0]
            winner.budget = cfg.ATTENTION_POOL
            attention_grantee = winner.name
            log("attention_grant", explorer=winner.name, pool=cfg.ATTENTION_POOL,
                marginal_yield=round(marginal_yield(winner.name), 5),
                note="MVT: the pool goes to whoever is still learning fastest")
            run_explorer(winner)

        # ---- LEARNING-RATE CURVE (v10): information intake, measured ---------
        lr_rows, cum_cost, best_w = [], 0, 0.0
        for i, l in enumerate(library.lessons):
            cum_cost += max(l.cost, 1)
            best_w = max(best_w, l.width if np.isfinite(l.width) else 0.0)
            lr_rows.append({"lesson_n": i + 1, "explorer": l.explorer, "cum_cost": cum_cost,
                            "width": l.width, "best_width_so_far": best_w,
                            "best_width_per_cost": best_w / max(cum_cost, 1)})
        write_csv(pd.DataFrame(lr_rows), "learning_rate_curve.csv")

        # ---- v11 menagerie reports ------------------------------------------
        quorum_rows = [{"family": fam, "distinct_voters": len(voters),
                        "reached_quorum": len(voters) >= cfg.QUORUM_MIN,
                        "voters": "|".join(sorted(voters))}
                       for fam, voters in sorted(QUORUM.items(), key=lambda kv: -len(kv[1]))]
        if quorum_rows:
            write_csv(pd.DataFrame(quorum_rows), "quorum_report.csv")
        n_quorum = sum(1 for r in quorum_rows if r["reached_quorum"])
        log("quorum_sensing", families_voted=len(quorum_rows), reached_quorum=n_quorum,
            dances=len(DANCES), gene_pool=len(GENE_POOL))
        sp_rows = []
        seen_names: set[str] = set()
        for ex in reversed(explorers):           # last instance per name = final state
            if ex.name in seen_names:
                continue
            seen_names.add(ex.name)
            mine = [l for l in library.lessons if l.explorer == ex.name]
            prom = [l for l in mine if l.decision == "promote"]
            sp_rows.append({"explorer": ex.name, "species": ex.species, "behavior": ex.behavior,
                            "seasons_lived": sum(1 for e in explorers if e.name == ex.name),
                            "birth_stage": STAGES[ex.birth_stage][0], "max_stage": STAGES[ex.max_stage][0],
                            "final_stage": STAGES[ex.stage_idx][0], "lessons": len(mine),
                            "promoted": len(prom),
                            "best_oof": round(max((l.oof_corr for l in mine), default=0.0), 4)})
        sp_rows.reverse()
        write_csv(pd.DataFrame(sp_rows), "species_report.csv")

        circadian(phase_frac("explore", 0.45), "post_phase1")
        gc.collect()
        mem_status("post_phase1")

        # ---- RAID 1 (v10): the predator strikes early so venom shapes search --
        raid1 = PredatorEngine(cfg, library, spec_lookup, budget=cfg.PREDATOR_BUDGET // 3)
        pred_df1 = raid1.run(Xp, yp, segp, cols, embargo_p)
        attacked = set(pred_df1["key"]) if not pred_df1.empty else set()
        log("raid1_done", attacked=len(attacked), taboo_motifs=len(TABOO))

        # ---- PHASE 2: metaheuristic evolution (v12: EPOCHS while fed) ---------
        # v4, v8, v9 AND v11 all ended evolution still climbing at budget
        # exhaustion (v11: g_best in the final generation, budget -1). The
        # metabolism re-fuels evolution in epochs: each epoch re-seeds its
        # population from the full library (seasons' champions included),
        # resets the annealing temperature, and spends a fresh budget.
        META.begin("evolve", cfg.MET_EVOLVE_SHARE)
        evo = EvolutionEngine(cfg, library, spec_lookup, gate)
        while True:
            n_hist = len(evo.history)
            evo.run(Xp, yp, segp, cols, embargo_p, journal_rows)
            META.epochs = evo.epoch + 1
            if not META.wants_more("evolve"):
                break
            if evo.epoch + 1 >= cfg.MAX_EPOCHS:
                log("epochs_capped", at=evo.epoch + 1)
                break
            if len(evo.history) == n_hist:
                log("epochs_exhausted", epoch=evo.epoch + 1,
                    note="an epoch produced nothing new; the lineage has converged")
                break
            evo.epoch += 1
            evo.budget = cfg.EVOLUTION_BUDGET
            write_csv(pd.DataFrame(journal_rows), "explorer_journal.csv")  # crash-safe partial
            META.heartbeat(f"epoch_{evo.epoch}_refuel")
            log("evolution_epoch_refuel", epoch=evo.epoch, budget=evo.budget,
                elapsed_min=round(META.now(), 1))
        if evo.history:
            write_csv(pd.DataFrame(evo.history), "evolution_history.csv")
            write_csv(evo.operator_report(), "evolution_operator_report.csv")
        write_json(gate.report(), "draft_gate_report.json")
        log("draft_gate", **gate.report())
        circadian(phase_frac("evolve", 0.70), "post_evolution")
        gc.collect()
        mem_status("post_evolution")

        # ---- RAID 2: the predator attacks evolution's promotions --------------
        predator = PredatorEngine(cfg, library, spec_lookup,
                                  budget=cfg.PREDATOR_BUDGET - cfg.PREDATOR_BUDGET // 3,
                                  exclude=attacked)
        pred_df2 = predator.run(Xp, yp, segp, cols, embargo_p)
        pred_df = pd.concat([pred_df1, pred_df2], ignore_index=True) \
            if not pred_df1.empty or not pred_df2.empty else pd.DataFrame()
        if not pred_df.empty:
            write_csv(pred_df, "predator_report.csv")

        # ---- CHAMPION ABLATION (v10): vary one thing at a time ----------------
        ablation_rows = []
        champs = sorted(library.promoted(), key=lesson_fitness, reverse=True)
        if champs:
            champ = champs[0]
            c_spec = spec_lookup[champ.key]
            variants = []
            if c_spec.transform != "identity" and not SKILL_REGISTRY[champ.skill]["needs_identity"]:
                variants.append(("transform->identity", Genome(champ.skill, c_spec.family, "identity", c_spec.k)))
            if c_spec.family != "top":
                variants.append(("family->top", Genome(champ.skill, "top", c_spec.transform, c_spec.k)))
            if c_spec.k > cfg.K_MIN * 2:
                variants.append(("k->half", Genome(champ.skill, c_spec.family, c_spec.transform, c_spec.k // 2)))
            spent = 0
            for label, g in variants:
                g = g.repaired()
                cost_g = SKILL_REGISTRY[g.skill]["cost"]
                if spent + cost_g > cfg.ABLATION_BUDGET:
                    break
                if library.can_run(g.skill, g.spec()):
                    les = run_lesson("ablation", "ablation", g.skill, g.spec(), Xp, yp, segp,
                                     cols, cfg, embargo_p,
                                     stable_seed(cfg.SEED, "ablation", g.key), library.oofs())
                    library.add(les)
                    spec_lookup[les.key] = g.spec()
                    spent += cost_g
                    a_oof, a_w = les.oof_corr, les.width
                else:
                    a_oof, a_w = library.mean_gain(g.key), float("nan")
                ablation_rows.append({"variant": label, "key": g.key,
                                      "oof_corr": a_oof, "width": a_w,
                                      "champion_key": champ.key,
                                      "delta_oof_vs_champion": a_oof - champ.oof_corr})
            if ablation_rows:
                write_csv(pd.DataFrame(ablation_rows), "champion_ablation.csv")
                log("champion_ablation", champion=champ.key, variants=len(ablation_rows),
                    note="the edge, attributed to its parts")

        # ---- DIVE PHASE (v10): submarines under the visible surface -----------
        # The provisional champion's fold-honest OOF defines the surface; dive
        # lessons hunt the residual y - slope*z(champion_oof). Same doors,
        # scored in their own residual world; the stacker judges the blend.
        dive_added = 0
        if champs and cfg.DIVE_BUDGET > 0:
            champ = champs[0]
            oz = (champ.oof - champ.oof.mean()) / (champ.oof.std() + 1e-9)
            slope = float(np.mean(oz * (yp - yp.mean())))
            y_dive = (yp - slope * oz).astype(np.float32)
            DIVE_GRID = [("linear_assoc", "decor", "identity"),
                         ("codebook", "terrain", "quantize4"),
                         ("bagged_linear", "shadow", "quantize8"),
                         ("scout_lattice", "anon", "fold_abs"),
                         ("relay_caravan", "weather", "identity"),
                         ("majority_vote", "mycelium", "sign_only"),
                         ("steepness_gate", "top", "doppler"),
                         ("swell_rider", "both_clocks", "rank")]
            META.begin("dive", cfg.MET_DIVE_SHARE)
            dive_spent = 0
            # v12: a fed metabolism sends the submarines on a SECOND descent
            # with wider periscopes (k 32 -> 64) -- new keys, same honest doors
            dive_rounds = 2 if META.wants_more("dive") else 1
            for rnd in range(dive_rounds):
                k_d = min(32 * (rnd + 1), len(cols))
                allowance = cfg.DIVE_BUDGET * (rnd + 1)
                for sk, fam, tf in DIVE_GRID:
                    if rnd > 0 and not META.allow("dive"):
                        break
                    cost_d = SKILL_REGISTRY[sk]["cost"]
                    if dive_spent + cost_d > allowance:
                        continue
                    spec_d = ViewportSpec(name=f"{fam}{k_d}_{tf}", family=fam, k=k_d,
                                          transform=tf, proj_dim=16)
                    les = run_lesson("submarine", "dive", sk, spec_d, Xp, y_dive, segp,
                                     cols, cfg, embargo_p,
                                     stable_seed(cfg.SEED, "dive", sk, fam, tf, rnd), {})
                    library.add(les)
                    spec_lookup[les.key] = spec_d
                    dive_spent += cost_d
                    dive_added += 1
                    log("dive_lesson", descent=rnd + 1, skill=sk, family=fam, transform=tf,
                        resid_oof=round(les.oof_corr, 4), width=round(les.width, 4),
                        decision=les.decision, spent=dive_spent)
            log("dive_phase_done", dives=dive_added, surface=champ.key,
                surface_slope=round(slope, 5))
        circadian(phase_frac("dive", 0.80), "post_dive")

        # ---- TOPOGRAPHY: texture every trail, cluster into families -----------
        tex_df, tex_fam = texture_layer(library.lessons, yp, segp, terr_p, wth_p, cfg)
        if not tex_df.empty:
            write_csv(tex_df, "path_texture_report.csv")
            log("path_textures", trails=len(tex_df),
                families=int(tex_df["trail_family"].nunique()),
                features="|".join(TEXTURE_FEATURES))
        tt_df = terrain_trail_report(library.lessons, yp, terr_p, cfg.TERRAIN_MIN_ROWS)
        if not tt_df.empty:
            write_csv(tt_df, "terrain_trail_report.csv")
        tg_df = texture_generalization_report(tex_df)
        if not tg_df.empty:
            write_csv(tg_df, "texture_generalization.csv")

        # ---- RED PHEROMONE (v14): the repellent channel, reported ------------
        if RED_MYCELIUM:
            top_red = sorted(RED_MYCELIUM.items(), key=lambda kv: -kv[1])[:40]
            write_csv(pd.DataFrame([{"col_idx": int(ci),
                                     "feature": cols[int(ci)] if int(ci) < len(cols) else str(ci),
                                     "repellent": round(float(v), 3),
                                     "is_trap": int(ci) in TRAPS}
                                    for ci, v in top_red]), "red_pheromone_report.csv")
            log("red_pheromone", poisoned_columns=len(RED_MYCELIUM),
                from_traps=len(TRAPS), note="repellent subtracted from corr-driven rankings")

        # ---- DREAM REPLAY (v9): bootstrap every promoted trail, free ----------
        rng_dream = np.random.default_rng(stable_seed(cfg.SEED, "dreams"))
        dream_omens = 0
        for l in library.promoted():
            l.dream_p05, l.dream_p50 = dream_replay(l.oof, yp, segp,
                                                    cfg.DREAM_REPLAYS, rng_dream)
            if l.dream_p05 < 0:
                dream_omens += 1
        log("dream_replay", trails=len(library.promoted()), replays=cfg.DREAM_REPLAYS,
            omens=dream_omens, note="dream_p05<0 = the trail fails in some dreamed world")

        # Xp's last consumer was the predator; the probe matrix goes back to
        # the OS before the report/ensemble half of the run (memory doctrine).
        # attempt_lesson's closure also pins Xp -- both must go.
        del Xp, attempt_lesson
        gc.collect()
        mem_status("post_predator")

        # ---- study reports -----------------------------------------------------
        ledger = []
        for l in library.lessons:
            row = asdict(l)
            row.pop("oof")
            row.pop("used_cols", None)   # mycelium substrate, too wide for the ledger
            row["fold_corrs"] = "|".join(f"{c:.5f}" for c in l.fold_corrs)
            row["wf_fold_corrs"] = "|".join(f"{c:.5f}" for c in l.wf_fold_corrs)
            row["fitness"] = lesson_fitness(l)
            row["bits_per_feature"] = TRANSFORM_BITS.get(l.transform, 32)
            row["total_bits"] = TRANSFORM_BITS.get(l.transform, 32) * max(l.k, 1)
            ledger.append(row)
        ledger_df = pd.DataFrame(ledger)
        write_csv(ledger_df, "explorer_lessons.csv")
        write_csv(pd.DataFrame(journal_rows), "explorer_journal.csv")
        write_csv(pd.DataFrame(growth_rows), "explorer_growth_curve.csv")
        write_csv(pd.DataFrame([{"key": k, "tries": library.runs[k], "mean_corr": library.mean_gain(k)}
                                for k in library.runs]).sort_values("mean_corr", ascending=False),
                  "shared_library.csv")
        if not ledger_df.empty:
            for dim in ("skill", "family", "transform", "stage", "explorer"):
                rep = (ledger_df.groupby(dim, as_index=False)
                       .agg(lessons=("key", "count"), mean_corr=("oof_corr", "mean"),
                            best_corr=("oof_corr", "max"), mean_width=("width", "mean"),
                            mean_wf_corr=("wf_corr", "mean"), mean_era_corr=("era_corr", "mean"),
                            promote_rate=("decision", lambda s: float((s == "promote").mean())))
                       .sort_values("best_corr", ascending=False))
                write_csv(rep, f"study_{dim}_report.csv")

        # ---- members: regime + uniqueness filters, z-scored --------------------
        promoted = sorted(library.promoted(), key=lambda l: -l.oof_corr)
        if not promoted:
            promoted = sorted(library.lessons, key=lambda l: -l.oof_corr)[:1]
            log("WARNING_no_promotions", note="falling back to best lesson regardless of gates")
        segs_u = np.unique(segp)
        members: dict[str, np.ndarray] = {}
        member_lessons: dict[str, Lesson] = {}
        name_counts: dict[str, int] = {}
        fam_counts: dict[int, int] = {}        # trail-texture family counts (v8)
        vfam_counts: dict[str, int] = {}       # v23: viewport-FAMILY counts (the monoculture cap)
        dive_members = 0

        def _try_admit(l: "Lesson", enforce_vfam: bool) -> bool:
            """Admit a promoted trail to the blend if it clears every diversity
            door. v23 adds a hard viewport-family cap; the two-pass caller below
            relaxes ONLY that cap to reach the member floor."""
            nonlocal dive_members
            if len(members) >= cfg.MAX_MEMBERS:
                return False
            sd = float(np.std(l.oof))
            if sd <= 1e-9:
                return False
            is_dive = l.stage == "dive"
            if is_dive and dive_members >= cfg.DIVE_MEMBER_CAP:
                return False
            if not is_dive:
                # dive lessons predict the RESIDUAL: near-zero corr vs y is
                # their job description, so only surface trails face this gate
                per_seg = [pearson(yp[segp == s], l.oof[segp == s]) for s in segs_u]
                if float(np.mean([c <= 0 for c in per_seg])) > cfg.MAX_SEG_NEG_FRAC:
                    return False
            if members and max(abs(pearson(l.oof, o)) for o in members.values()) > cfg.MEMBER_CORR_CAP:
                return False
            # v14 jamming avoidance (CONSERVATIVE): fires only for a confirmed
            # near-duplicate -- high input overlap AND high output agreement.
            lcols = set(l.used_cols)
            if lcols and members and not is_dive:
                for nm in members:
                    oc = abs(pearson(l.oof, member_lessons[nm].oof))
                    if oc < 0.80:
                        continue
                    mcols = set(member_lessons[nm].used_cols)
                    jac = len(lcols & mcols) / max(1, len(lcols | mcols))
                    if jac >= cfg.JAMMING_JACCARD:
                        log("member_skipped_jamming", key=l.key,
                            twin=nm, input_jaccard=round(jac, 3), output_corr=round(oc, 3))
                        return False
            # v8 texture diversity: cap members sharing one trail-texture family
            fam_t = tex_fam.get(l.key, -1)
            if fam_t >= 0 and fam_counts.get(fam_t, 0) >= cfg.TEXTURE_FAMILY_CAP:
                log("member_skipped_texture_family", key=l.key, trail_family=fam_t,
                    cap=cfg.TEXTURE_FAMILY_CAP)
                return False
            # v23 VIEWPORT-FAMILY CAP -- the biting input-space diversity gate
            # the v12/v19 monocultures needed. The texture cap above is fooled
            # (12 mycelium viewports register as 10 "trail families"); this
            # counts the actual viewport family (mycelium/tail/weather/...). A
            # blend may carry at most MAX_FAMILY_MEMBERS of any one, so it can
            # NEVER be the single-family bet that gamed sealed and collapsed
            # out-of-period in v12 (8/8 mycelium) and v19 (100% mycelium, 0.0749).
            if enforce_vfam and vfam_counts.get(l.family, 0) >= cfg.MAX_FAMILY_MEMBERS:
                log("member_skipped_viewport_family", key=l.key, family=l.family,
                    cap=cfg.MAX_FAMILY_MEMBERS)
                return False
            base = l.key
            name_counts[base] = name_counts.get(base, 0) + 1
            name = base if name_counts[base] == 1 else f"{base}#r{name_counts[base]}"
            mu0 = float(np.mean(l.oof))
            members[name] = ((l.oof - mu0) / sd).astype(np.float32)
            member_lessons[name] = l
            if is_dive:
                dive_members += 1
            if fam_t >= 0:
                fam_counts[fam_t] = fam_counts.get(fam_t, 0) + 1
            vfam_counts[l.family] = vfam_counts.get(l.family, 0) + 1
            return True

        # pass 1: diversity-first, viewport-family cap enforced
        for l in promoted:
            if len(members) >= cfg.MAX_MEMBERS:
                break
            _try_admit(l, enforce_vfam=True)
        # pass 2: backfill ONLY if the cap left the blend under the floor (one
        # family genuinely dominates the promoted pool). Relaxes the viewport-
        # family cap alone; every other door still holds.
        if len(members) < cfg.MIN_BLEND_MEMBERS:
            admitted = {id(member_lessons[nm]) for nm in members}
            for l in promoted:
                if len(members) >= cfg.MIN_BLEND_MEMBERS:
                    break
                if id(l) in admitted:
                    continue
                if _try_admit(l, enforce_vfam=False):
                    admitted.add(id(l))
            log("blend_backfilled", to=len(members), floor=cfg.MIN_BLEND_MEMBERS,
                note="viewport-family cap relaxed to reach the member floor")
        # v27 COMPLEXITY-ANCHOR ADMISSION: reserve a few LOW-complexity, HIGH-
        # walk-forward members into the pool so the runtime governor has a
        # genuinely simple alternative to ship IF measurement says capacity decays
        # here. The 11th-place simple-linear recipes (linear_ols/greedy_ols on the
        # tail block, tiny-k quantize4) promote every run but never made the oof-
        # ranked top-MAX_MEMBERS pool -- this makes them REACHABLE without forcing
        # them; lambda (the measured decay~complexity penalty) decides if they win.
        if cfg.COMPLEXITY_GOVERNOR:
            present = {id(member_lessons[nm]) for nm in members}
            anchor_pool = sorted(
                (l for l in promoted
                 if id(l) not in present and l.oof_corr > 0 and float(np.std(l.oof)) > 1e-9
                 and member_complexity(l, cfg) <= cfg.GOV_SIMPLE_C),
                key=lambda l: -(l.wf_corr if np.isfinite(l.wf_corr) else l.oof_corr))
            added_anchor = 0
            for l in anchor_pool:
                if added_anchor >= cfg.GOV_ANCHOR_MEMBERS:
                    break
                if members and max(abs(pearson(l.oof, o)) for o in members.values()) > cfg.MEMBER_CORR_CAP:
                    continue
                base = l.key
                name_counts[base] = name_counts.get(base, 0) + 1
                name = base if name_counts[base] == 1 else f"{base}#r{name_counts[base]}"
                mu0, sd0 = float(np.mean(l.oof)), float(np.std(l.oof))
                members[name] = ((l.oof - mu0) / sd0).astype(np.float32)
                member_lessons[name] = l
                vfam_counts[l.family] = vfam_counts.get(l.family, 0) + 1
                added_anchor += 1
            if added_anchor:
                log("complexity_anchor_admitted", n=added_anchor,
                    note="low-complexity high-wf members reserved so the governor can ship simple if it generalizes")
        n_trail_families = len(set(tex_fam.get(member_lessons[nm].key, -1) for nm in members))
        n_view_families = len(set(member_lessons[nm].family for nm in members))
        log("ensemble_members", n=len(members), names=list(members),
            trail_families=n_trail_families, viewport_families=n_view_families,
            dive_members=dive_members)

        result = nested_ensemble(members, yp, segp, cfg, embargo_p, wth=wth_p)
        write_json(result["honest"] | {"winner": result["winner"]}, "ensemble_nested_assessment.json")
        log("ensemble_selected", winner=result["winner"],
            honest=round(result["honest"][result["winner"]], 5),
            best_single=round(result["honest"]["best_single"], 5),
            weather_conditional=result["weather_states"] is not None)

        w = result["weights"]
        blend_oof = apply_weights_rows(members, w, result["is_median"],
                                       result["weather_states"], wth_p).astype(np.float64)

        # weather report: how the blend (and best single) fare in each sky (v9)
        best_full = max(members, key=lambda nm: pearson(yp, members[nm]))
        write_csv(pd.DataFrame(
            [{"state": int(s), "rows": int((wth_p == s).sum()),
              "frac": float((wth_p == s).mean()),
              "blend_corr": pearson(yp[wth_p == s], blend_oof[wth_p == s]),
              "best_single_corr": pearson(yp[wth_p == s], members[best_full][wth_p == s])}
             for s in np.unique(wth_p)]), "weather_report.csv")

        wins = winsorize_audit(blend_oof, yp, cfg.WINSOR_QS)
        write_json(wins, "winsorize_audit.json")
        log("winsorize", apply=wins["apply"], raw=round(wins["raw_corr"], 5), best=round(wins["best_corr"], 5))

        write_csv(dominance_report(w, members, yp), "dominance_report.csv")
        write_csv(regime_stress(members, yp, segp, volp, cfg.VOLUME_SLICE_Q), "member_regime_stress.csv")
        # v19 truth-serum reports: many-worlds survival + MAP-Elites biodiversity
        mw_df = many_worlds_report(members, yp, segp, terr_p, wth_p, cfg)
        write_csv(mw_df, "many_worlds_cv.csv")
        if not mw_df.empty:
            log("many_worlds_cv", members=len(mw_df),
                best_survival=round(float(mw_df["world_survival_min"].max()), 4),
                worst_survival=round(float(mw_df["world_survival_min"].min()), 4),
                note="per-member corr floor across time/terrain/weather worlds")
        me_df = map_elites_archive(library.lessons)
        write_csv(me_df, "map_elites_archive.csv")
        log("map_elites", niches=len(me_df),
            note="best lesson per (family x transform x k x overfit) cell -- biodiversity, not monoculture")
        write_csv(pd.DataFrame([{"member": nm, "fit_corr": member_lessons[nm].fit_corr,
                                 "oof_corr": member_lessons[nm].oof_corr,
                                 "wf_corr": member_lessons[nm].wf_corr,
                                 "deflated_corr": member_lessons[nm].deflated_corr,
                                 "overfit_ratio": member_lessons[nm].overfit_ratio,
                                 "flag": "ALARM" if member_lessons[nm].overfit_ratio > cfg.MAX_OVERFIT_RATIO else "ok"}
                                for nm in members]), "train_cv_gap.csv")

        # ---- forward-drift check + forward gate (within WORKING region) --------
        best_cv_name = max(members, key=lambda nm: pearson(yp, members[nm]))
        names_for_fwd = list(dict.fromkeys(list(w) + [best_cv_name]))
        cut = int((1 - cfg.FORWARD_FRACTION) * n_work)
        past = np.arange(0, max(2, cut - cfg.EMBARGO_ROWS))
        future = np.arange(cut, n_work)
        fwd_rows, fwd_parts = [], {}
        for nm in names_for_fwd:
            l = member_lessons[nm]
            st = fit_skill(l.skill, spec_lookup[l.key], X_full[past], y_full[past],
                           seg_full[past], cols, np.random.default_rng(l.seed), cfg, l.seed)
            p = predict_skill(st, X_full[future])
            sd = float(np.std(p)) + 1e-9
            fwd_parts[nm] = ((p - float(np.mean(p))) / sd).astype(np.float64)
            fwd_rows.append({"member": nm, "cv_oof_corr": l.oof_corr,
                             "forward_corr": pearson(y_full[future], p),
                             "drift_gap": l.oof_corr - pearson(y_full[future], p)})
        wth_future = GAUGE.assign(X_full[future]) if GAUGE is not None else None
        fwd_blend = apply_weights_rows(fwd_parts, w, result["is_median"],
                                       result["weather_states"], wth_future)
        forward_blend_corr = score_metric(y_full[future], fwd_blend)
        single_fwd_corr = score_metric(y_full[future], fwd_parts[best_cv_name])
        fwd_rows.append({"member": "__BLEND__", "cv_oof_corr": result["honest"][result["winner"]],
                         "forward_corr": forward_blend_corr,
                         "drift_gap": result["honest"][result["winner"]] - forward_blend_corr})
        write_csv(pd.DataFrame(fwd_rows), "forward_holdout_report.csv")
        log("forward_holdout", blend=round(forward_blend_corr, 5),
            best_cv_single=best_cv_name, single=round(single_fwd_corr, 5))

        # ---- v27 RUNTIME COMPLEXITY-GENERALIZATION GOVERNOR -------------------
        # Measure whether THIS dataset rewards or punishes capacity -- from every
        # lesson's OWN out-of-period decay (oof_corr - wf_corr) regressed on its
        # complexity -- then set the shipping penalty lambda accordingly. The cure
        # for the measured complexity ratchet (v19/v24/v25 all converged on the
        # highest-capacity, highest-sealed, worst-private models) WITHOUT hard-
        # coding simplicity: a capacity-friendly dataset yields beta<=0, lambda 0.
        GOVERNOR.clear()
        if cfg.COMPLEXITY_GOVERNOR:
            gov_pts = [(member_complexity(l, cfg), float(l.oof_corr - l.wf_corr))
                       for l in library.lessons
                       if l.oof_corr > 0.02 and np.isfinite(l.wf_corr)]
            gov_cmap = {nm: member_complexity(member_lessons[nm], cfg) for nm in members}
            gov_beta = gov_lambda = 0.0
            if len(gov_pts) >= cfg.GOV_MIN_LESSONS:
                Cv = np.array([p[0] for p in gov_pts], np.float64)
                Dv = np.array([p[1] for p in gov_pts], np.float64)
                if float(Cv.std()) > 1e-6:
                    gov_beta = float(np.cov(Cv, Dv, bias=True)[0, 1] / (Cv.var() + 1e-12))
                gov_lambda = float(np.clip(gov_beta * cfg.GOV_LAMBDA_SCALE, 0.0, cfg.GOV_LAMBDA_MAX))
                edges = np.quantile(Cv, [1.0 / 3, 2.0 / 3])
                bucket = np.searchsorted(edges, Cv)
                curve_rows = []
                for b, lab in enumerate(("low", "mid", "high")):
                    m = bucket == b
                    if m.any():
                        curve_rows.append({"stratum": lab, "n": int(m.sum()),
                                           "mean_complexity": round(float(Cv[m].mean()), 4),
                                           "mean_decay_oof_minus_wf": round(float(Dv[m].mean()), 5)})
                write_csv(pd.DataFrame(curve_rows), "complexity_generalization_curve.csv")
            # v27 self-improvement: blend the measured beta with the cross-run
            # ledger's accumulated beta (shrunk by evidence count) so lambda is
            # stable across runs and doesn't whipsaw on one noisy measurement.
            gov_beta_meas = gov_beta
            if cfg.SELF_IMPROVE and (LEDGER_PRIOR.get("governor") or {}).get("count", 0):
                pb = float(LEDGER_PRIOR["governor"].get("beta", 0.0))
                pc = float(LEDGER_PRIOR["governor"].get("count", 0)) * cfg.LEDGER_SHRINK * cfg.GOV_MIN_LESSONS
                nm = float(len(gov_pts))
                if nm + pc > 0:
                    gov_beta = (nm * gov_beta + pc * pb) / (nm + pc)
                gov_lambda = float(np.clip(gov_beta * cfg.GOV_LAMBDA_SCALE, 0.0, cfg.GOV_LAMBDA_MAX))
                log("governor_beta_blend", measured=round(gov_beta_meas, 4), prior=round(pb, 4),
                    blended=round(gov_beta, 4), prior_runs=int(LEDGER_PRIOR["governor"].get("count", 0)))
            GOVERNOR.update({"lambda": gov_lambda, "beta": gov_beta, "complexity": gov_cmap})
            write_json({"beta_decay_vs_complexity": round(gov_beta, 5),
                        "lambda_penalty": round(gov_lambda, 5),
                        "lessons_measured": len(gov_pts),
                        "member_complexity": {nm: round(c, 4) for nm, c in gov_cmap.items()},
                        "note": "lambda * config-complexity is subtracted from each candidate's robust score; "
                                "beta>0 => this dataset punishes capacity (ship simpler); beta<=0 => capacity free"},
                       "complexity_governor.json")
            log("complexity_governor", beta=round(gov_beta, 4), lam=round(gov_lambda, 4),
                lessons=len(gov_pts),
                note="shipping-complexity penalty set by the measured decay~complexity slope (runtime-adaptive)")

        # ---- v21 FORENSIC REGIME-SCIENCE: self-tuning, forward-validated -------
        # Writes the full forensic suite and MEASURES regime-aware shipping
        # configs on the forward slice. Overrides the incumbent weights ONLY if
        # a worst-world + input-diversity reselection strictly beats it out-of-
        # sample (the measured cure for the v12 monoculture); strict no-op else.
        forensic_dec = forensic_regime_science(
            members, member_lessons, spec_lookup, w, result["is_median"],
            result["weather_states"], forward_blend_corr, fwd_parts,
            yp, segp, terr_p, wth_p, volp, X_full, y_full, seg_full, n_work,
            cols, past, future, cfg)
        fwd_parts = forensic_dec["fwd_parts"]
        if forensic_dec["override"]:
            w = forensic_dec["weights"]
            result["is_median"] = forensic_dec["is_median"]
            result["weather_states"] = forensic_dec["weather_states"]
            forward_blend_corr = forensic_dec["forward_blend_corr"]
            fwd_blend = sum(w[nm] * fwd_parts[nm] for nm in w if nm in fwd_parts)
            single_fwd_corr = score_metric(y_full[future], fwd_parts[best_cv_name]) \
                if best_cv_name in fwd_parts else single_fwd_corr
            names_for_fwd = list(dict.fromkeys(list(w) + [best_cv_name]))

        # v18 FORWARD-GATE ERROR BARS: a point margin means a coin-flip can
        # pick the captain. Block-bootstrap the forward slice and require the
        # single to beat the blend SIGNIFICANTLY (in >= GATE_BOOT_CONF of
        # resamples), not just on a noisy point estimate -- a noisy gate can
        # undo the whole search. Default-safe: if bootstrap is degenerate the
        # old point rule stands.
        gate_point = bool(cfg.USE_FORWARD_GATE
                          and single_fwd_corr > forward_blend_corr + cfg.FORWARD_GATE_MARGIN)
        gate_fired = gate_point
        if gate_point and best_cv_name in fwd_parts:
            yf = y_full[future]
            seg_f = seg_full[future]
            usegs = np.unique(seg_f)
            if len(usegs) >= 4:
                rng_g = np.random.default_rng(stable_seed(cfg.SEED, "gate_boot"))
                seg_idx_f = [np.where(seg_f == s)[0] for s in usegs]
                single_p = fwd_parts[best_cv_name]
                wins_single = 0
                B = 200
                for _ in range(B):
                    pick = rng_g.integers(0, len(usegs), len(usegs))
                    bidx = np.concatenate([seg_idx_f[int(p)] for p in pick])
                    sc_s = pearson(yf[bidx], single_p[bidx])
                    sc_b = pearson(yf[bidx], fwd_blend[bidx])
                    if sc_s > sc_b + cfg.FORWARD_GATE_MARGIN:
                        wins_single += 1
                conf = wins_single / B
                gate_fired = conf >= cfg.GATE_BOOT_CONF
                log("forward_gate_error_bars", point_fire=gate_point, boot_conf=round(conf, 3),
                    required=cfg.GATE_BOOT_CONF, fired=gate_fired,
                    note="gate requires bootstrap significance, not a point margin")
        if gate_fired:
            final_weights = {best_cv_name: 1.0}
            final_is_median = False
            final_weather = None
            log("FORWARD_GATE_OVERRIDE", shipped=best_cv_name,
                single_fwd=round(single_fwd_corr, 5), blend_fwd=round(forward_blend_corr, 5))
        else:
            final_weights = w
            final_is_median = result["is_median"]
            final_weather = result["weather_states"]

        # ---- v27 ANTI-OVERFIT SHIPPING COURT ---------------------------------
        # Channels the best of the regime-criticality / overfit-gravity-well /
        # CV-reality-distortion / prediction-crowding ideas into ONE cheap, out-of-
        # sample-grounded SELECTION HARDENER (adds NO capacity -- it only raises the
        # bar where overfit risk is measured): members that fail their local "escape
        # velocity" (width vs decay + reality-distortion + complexity + crowding, with
        # a non-positive worst-world floor) are down-weighted, and high regime
        # CRITICALITY (residual autocorrelation) shrinks the blend toward equal-weight.
        # Conservative => near-no-op on a healthy blend; fully reported.
        if not gate_fired and not final_is_median and final_weather is None:
            final_weights = shipping_court(final_weights, members, member_lessons,
                                           yp, segp, terr_p, wth_p, cfg)

        # ---- v16 SHRUNK BLEND (no-op-safe anti-decay) ------------------------
        # The whole story of this dataset is the CV->forward gap (regime decay):
        # honest CV ~0.144 but forward/sealed ~0.108. CV-optimal weights are
        # brittle; EQUAL weighting is more robust (v9's equal_top won outright).
        # So shrink the shipped weights toward equal by a factor chosen on the
        # FORWARD slice -- this attacks the gap directly and DEFAULTS TO lambda=0
        # (no change) unless forward STRICTLY improves. It can only help or
        # no-op; it never trades away a CV-justified edge the forward slice
        # does not confirm. Only fires for a real multi-member global blend.
        shrunk_lambda = 0.0
        if (not gate_fired and not final_is_median and final_weather is None
                and len(final_weights) >= 2
                and all(nm in fwd_parts for nm in final_weights)):
            eq = 1.0 / len(final_weights)
            best_fc = forward_blend_corr
            for lam in (0.25, 0.5, 0.75, 1.0):
                shr = {nm: (1.0 - lam) * v + lam * eq for nm, v in final_weights.items()}
                fb = sum(shr[nm] * fwd_parts[nm] for nm in shr)
                fc = pearson(y_full[future], fb)
                if fc > best_fc + 1e-6:
                    best_fc, shrunk_lambda = fc, lam
            if shrunk_lambda > 0:
                final_weights = {nm: (1.0 - shrunk_lambda) * v + shrunk_lambda * eq
                                 for nm, v in final_weights.items()}
                log("shrunk_blend", lam=shrunk_lambda,
                    forward_before=round(forward_blend_corr, 5), forward_after=round(best_fc, 5),
                    note="weights shrunk toward equal on the forward slice (no-op unless it helps)")

        # ---- v18 CHORUS SHRINKAGE (no-op-safe) -------------------------------
        # Pearson is won on the rows where you make BIG calls -- so only make
        # big calls where independent members CONCUR. Scale each row of the
        # blend by committee agreement; the strength beta is chosen on the
        # FORWARD slice and DEFAULTS TO 0 (no shrink) unless forward improves.
        chorus_beta = 0.0
        if (not gate_fired and not final_is_median and final_weather is None
                and len(final_weights) >= 2
                and all(nm in fwd_parts for nm in final_weights)):
            fwd_base = sum(final_weights[nm] * fwd_parts[nm] for nm in final_weights)
            best_fc = pearson(y_full[future], fwd_base)
            for beta in (0.5, 1.0, 2.0, 4.0):
                fac = chorus_factor(fwd_parts, final_weights, beta)
                fc = pearson(y_full[future], fwd_base * fac)
                if fc > best_fc + 1e-6:
                    best_fc, chorus_beta = fc, beta
            if chorus_beta > 0:
                log("chorus_shrinkage", beta=chorus_beta, forward_after=round(best_fc, 5),
                    note="blend scaled by member agreement on the forward slice (no-op unless it helps)")

        # ---- v19 PREDICTION SHAPE ALCHEMY (no-op-safe) -----------------------
        # Audition output shapes (rank / power / tanh) on the FORWARD blend;
        # ship the best, default 'raw' (no-op). Financial labels often reward
        # ORDER over amplitude, so a rank/tanh remap can be more robust.
        ship_shape = "raw"
        if not final_is_median and final_weather is None and len(future) > 100 \
                and all(nm in fwd_parts for nm in final_weights):
            fwd_final = sum(final_weights[nm] * fwd_parts[nm] for nm in final_weights)
            best_sc = pearson(y_full[future], fwd_final)
            for sh in SHIP_SHAPES[1:]:
                c = pearson(y_full[future], _shape_pred(fwd_final, sh))
                if c > best_sc + cfg.SHAPE_MARGIN:   # v23: real forward gain, not a noisy overfit surface
                    best_sc, ship_shape = c, sh
            if ship_shape != "raw":
                log("prediction_shape_alchemy", shape=ship_shape, forward_after=round(best_sc, 5),
                    note="output remap chosen on the forward slice (no-op unless it helps)")

        monitor = HealthMonitor(cfg)
        if len(members) >= 2:
            # v8: a blend whose every member shares one trail family is one
            # texture-correlated bet, however decorrelated the predictions look
            monitor.check("trail_family_diversity", float(n_trail_families), 1.5, "below",
                          "all blend members share a single path-texture family")
        ship_dreams = [member_lessons[nm].dream_p05 for nm in final_weights
                       if nm in member_lessons and np.isfinite(member_lessons[nm].dream_p05)]
        if ship_dreams:
            # v9: a shipped trail that fails in its 5th-percentile dreamed
            # world is a fragility warning the pooled numbers hide
            monitor.check("shipped_dream_p05", float(min(ship_dreams)), 0.0, "below",
                          "a shipped trail goes negative in bootstrapped replays of the world")
        ship_gaps = [member_lessons[nm].sense_gap / max(abs(member_lessons[nm].oof_corr), 1e-6)
                     for nm in final_weights
                     if nm in member_lessons and np.isfinite(member_lessons[nm].sense_gap)]
        if ship_gaps:
            # v10: the two senses disagree -> tail-driven alpha in the blend
            monitor.check("shipped_sense_gap_ratio", float(max(ship_gaps)), 0.75, "above",
                          "a shipped trail looks strong to one sense only (tail-driven)")
        alarms = monitor.run_checks(result, members, yp, library.lessons, forward_blend_corr)
        write_csv(alarms, "ensemble_health_alarms.csv")

        # ---- SEALED HOLDOUT: evaluated ONCE, after the blend is frozen ----------
        # Refit shipped members on the working region only; score the sealed tail.
        # This number gates NOTHING in this run -- it is the once-per-version
        # generalization audit for the harness itself.
        sealed_corr = None
        if len(sealed_idx) >= 500:
            past_seal = np.arange(0, max(2, seal_cut - cfg.EMBARGO_ROWS))
            seal_parts = {}
            for nm in final_weights:
                l = member_lessons[nm]
                st = fit_skill(l.skill, spec_lookup[l.key], X_full[past_seal], y_full[past_seal],
                               seg_full[past_seal], cols, np.random.default_rng(l.seed), cfg, l.seed)
                p = predict_skill(st, X_full[sealed_idx])
                sd = float(np.std(p)) + 1e-9
                seal_parts[nm] = ((p - float(np.mean(p))) / sd).astype(np.float64)
            wth_seal = GAUGE.assign(X_full[sealed_idx]) if GAUGE is not None else None
            seal_blend = apply_weights_rows(seal_parts, final_weights, final_is_median,
                                            final_weather, wth_seal)
            sealed_corr = score_metric(y_full[sealed_idx], seal_blend)
            write_json({"sealed_rows": int(len(sealed_idx)), "sealed_corr": sealed_corr,
                        "shipped_weights": final_weights,
                        "note": "audited once per kernel version; never used for decisions"},
                       "sealed_holdout_report.json")
            log("SEALED_HOLDOUT_AUDIT", sealed_corr=round(sealed_corr, 5),
                rows=len(sealed_idx), note="not_a_gate")

        # ---- final refit on FULL train (incl. sealed) + test predictions --------
        gc.collect()
        mem_status("pre_final_refit")
        test_parts = {}
        for nm in final_weights:
            l = member_lessons[nm]
            st = fit_skill(l.skill, spec_lookup[l.key], X_full, y_full, seg_full, cols,
                           np.random.default_rng(l.seed), cfg, l.seed)
            p = predict_skill(st, X_test)
            sd = float(np.std(p)) + 1e-9
            test_parts[nm] = ((p - float(np.mean(p))) / sd).astype(np.float64)
            log("final_member_refit", member=nm, weight=round(final_weights[nm], 3))
        wth_test = GAUGE.assign(X_test) if GAUGE is not None else None
        test_pred = apply_weights_rows(test_parts, final_weights, final_is_median,
                                       final_weather, wth_test)
        if chorus_beta > 0 and all(nm in test_parts for nm in final_weights):
            test_pred = test_pred * chorus_factor(test_parts, final_weights, chorus_beta)
            log("chorus_applied", beta=chorus_beta, note="test blend shrunk by member agreement")
        if ship_shape != "raw":
            test_pred = _shape_pred(test_pred, ship_shape)
            log("shape_applied", shape=ship_shape, note="test prediction remapped (forward-chosen)")
        if wins["apply"]:
            lo, hi = np.quantile(test_pred, wins["best_q"]), np.quantile(test_pred, 1 - wins["best_q"])
            test_pred = np.clip(test_pred, lo, hi)
        write_submission(np.asarray(test_pred, np.float32), root)
        n_test = len(X_test)
        del X_full, X_test                 # nothing below needs the matrices
        gc.collect()
        free_gpu_mem()
        mem_status("post_submission")

        # ---- CAIRN (v10): fingerprint this world for the next visitor ---------
        champ_key = max(library.promoted(), key=lesson_fitness).key if library.promoted() else None
        # v14 seed bank: the run's best measured LOSERS -- positive-corr trails
        # that neither became the champion nor shipped. Written for the NEXT
        # run to germinate (temporal biodiversity against regime decay).
        shipped_keys = {member_lessons[nm].key for nm in final_weights if nm in member_lessons}
        loser_pool = sorted((l for l in library.promoted()
                             if l.key != champ_key and l.key not in shipped_keys
                             and l.oof_corr > 0),
                            key=lambda l: -l.oof_corr)
        seed_bank, seen_sb = [], set()
        for l in loser_pool:
            if l.key in seen_sb:
                continue
            seen_sb.add(l.key)
            seed_bank.append(l.key)
            if len(seed_bank) >= cfg.SEEDBANK_SIZE:
                break
        cairn = {"version": "v25", "data_source": data_source,
                 "gauge_edges": [float(e) for e in (GAUGE.edges if GAUGE is not None else [])],
                 "terrain_populations": t_pop, "weather_populations": w_pop,
                 "even_dominant": n_even, "trap_count": len(TRAPS),
                 "jnd": jnd["jnd"], "champion": champ_key,
                 "seed_bank": seed_bank,        # v14: measured losers for the next run
                 "honest": result["honest"][result["winner"]]}
        # v27 self-improvement: distil this run's OUT-OF-SAMPLE-grounded learnings
        # into the cross-run ledger (merged with the prior by evidence count) so the
        # NEXT run starts smarter. survivors = low-decay shipped genomes (warm starts);
        # decayers = high-decay motifs (anti-priors); + governor beta + the per-family/
        # skill generalization track record + the data profile.
        if cfg.SELF_IMPROVE:
            prev_led = LEDGER_PRIOR or {}
            surv = sorted((member_lessons[nm] for nm in final_weights if nm in member_lessons),
                          key=lambda l: (l.oof_corr - l.wf_corr) if np.isfinite(l.wf_corr) else 0.0)
            survivors = list(dict.fromkeys(l.key for l in surv))[: cfg.LEDGER_MAX_SURVIVORS]
            decj = sorted((l for l in library.lessons
                           if l.oof_corr > 0.03 and np.isfinite(l.wf_corr) and (l.oof_corr - l.wf_corr) > 0.03),
                          key=lambda l: -(l.oof_corr - l.wf_corr))
            decayers = list(dict.fromkeys(f"{l.skill}|{l.family}" for l in decj))[: cfg.LEDGER_MAX_DECAYERS]
            gcount = int((prev_led.get("governor") or {}).get("count", 0)) + 1
            ledger = {"version": "v27", "data_source": data_source,
                      "governor": {"beta": round(float(GOVERNOR.get("beta", 0.0)), 5),
                                   "lambda": round(float(GOVERNOR.get("lambda", 0.0)), 5),
                                   "count": gcount},
                      "family_decay": _ledger_merge(prev_led.get("family_decay", {}),
                                                    _ledger_decay_stats(library.lessons, "family")),
                      "skill_decay": _ledger_merge(prev_led.get("skill_decay", {}),
                                                   _ledger_decay_stats(library.lessons, "skill")),
                      "survivors": survivors, "decayers": decayers,
                      "profile": {"metric": PROFILE.get("metric"), "temporal": PROFILE.get("temporal"),
                                  "target_kind": PROFILE.get("target_kind")}}
            cairn["ledger"] = ledger
            write_json(ledger, "learning_ledger.json")
            log("learning_ledger_written", governor_runs=gcount, survivors=len(survivors),
                decayers=len(decayers), tracked_families=len(ledger["family_decay"]),
                tracked_skills=len(ledger["skill_decay"]))
        write_json(cairn, "world_cairn.json")
        cairn_drift = None
        prev_paths = [Path(p) for p in cfg.CAIRN_PATHS]
        try:
            prev_paths += list(Path("/kaggle/input").glob("*/world_cairn.json"))
        except Exception:
            pass
        for pth in prev_paths:
            try:
                if pth.exists():
                    prev = json.loads(pth.read_text())
                    pe, ce = prev.get("gauge_edges") or [], cairn["gauge_edges"]
                    edge_drift = (float(np.mean([abs(a - b) / (abs(a) + 1e-9)
                                                 for a, b in zip(pe, ce)]))
                                  if pe and len(pe) == len(ce) else None)
                    cairn_drift = {"prev_cairn": str(pth),
                                   "gauge_edge_drift": edge_drift,
                                   "prev_champion": prev.get("champion"),
                                   "champion_changed": prev.get("champion") != champ_key,
                                   "trap_count_delta": len(TRAPS) - int(prev.get("trap_count", 0))}
                    log("cairn_comparison", **{k: v for k, v in cairn_drift.items()})
                    break
            except Exception:
                continue

        evo_summary = {}
        full_evo = [h for h in evo.history if np.isfinite(h.get("fitness", float("nan")))]
        if full_evo:
            best_evo = max(full_evo, key=lambda h: h["fitness"])
            evo_summary = {"lessons": len(full_evo),
                           "draft_culled": sum(1 for h in evo.history if h["decision"] == "draft_culled"),
                           "g_best_key": best_evo["key"],
                           "g_best_fitness": best_evo["fitness"],
                           "g_best_operator": best_evo["operator"]}

        # ---- WORLD CHRONICLE (v9): the run as a written story -----------------
        explorer_lines = []
        for t in EXPLORER_TRAITS[: cfg.N_EXPLORERS]:
            mine = [l for l in library.lessons if l.explorer == t["name"]]
            if mine:
                best_l = max(mine, key=lambda l: l.oof_corr)
                explorer_lines.append(
                    f"{t['name']} [{t.get('species', t['name'])}/{t['metaheuristic']}]: "
                    f"{len(mine)} trails, best {best_l.key} ({best_l.oof_corr:+.4f})")
        champion_lines = []
        key2lesson = {l.key: l for l in library.lessons}
        if not tex_df.empty:
            for _, r in tex_df[tex_df["decision"] == "promote"].head(3).iterrows():
                lsn = key2lesson.get(r["key"])
                verb = trail_verb(lsn.skill, lsn.family, lsn.transform) if lsn else "walks"
                champion_lines.append(
                    f"{r['key']} ({r['oof_corr']:+.4f}, family T{int(r['trail_family'])}) "
                    f"{verb} -- {texture_words(r, tex_df)}")
        kill_lines = [f"{l.key} -- {l.predator_verdict}"
                      for l in library.lessons if l.decision == "predator_killed"]
        dream_lines = []
        for l in sorted(library.promoted(), key=lambda l: l.dream_p05 if np.isfinite(l.dream_p05) else 1e9)[:3]:
            if np.isfinite(l.dream_p05):
                omen = "an OMEN" if l.dream_p05 < 0 else "steady"
                dream_lines.append(f"{l.key}: dream_p05={l.dream_p05:+.4f}, "
                                   f"dream_p50={l.dream_p50:+.4f} -- {omen}")
        ship_desc = (f"The party shipped '{result['winner']}'"
                     + (" (weather-conditional)" if final_weather is not None else "")
                     + (f" after the forward gate overrode to {best_cv_name}" if gate_fired else "")
                     + ": " + ", ".join(f"{nm} ({final_weights[nm]:.2f})" for nm in final_weights)
                     + f". Forward corr {forward_blend_corr:+.5f}.")
        species_run = ", ".join(dict.fromkeys(ex.species for ex in explorers))
        embodiment_lines = [
            (f"The expedition carried {int(cfg.TIME_BUDGET_MIN)} minutes of provisions: "
             f"{META.seasons} season(s) of foraging, {META.epochs} evolutionary epoch(s), "
             f"{auditioned} skill audition(s); camp was made with "
             f"{max(0.0, round(cfg.TIME_BUDGET_MIN - META.now(), 1))} minutes to spare."
             if META.enabled else
             "No time budget was set: fixed rations -- one season, one epoch."),
            f"The expedition was a menagerie: {species_run}.",
            (f"{n_quorum} feature-families reached colony quorum (>= {cfg.QUORUM_MIN} species agreed); "
             f"{len(DANCES)} waggle dances were posted; the gene pool held {len(GENE_POOL)} plasmids."
             if QUORUM else ""),
            f"The satellites surveyed {len(SURVEY)} families from orbit; "
            f"'{max(SURVEY, key=SURVEY.get)}' showed the densest signal." if SURVEY else "",
            f"The trap map marked {len(TRAPS)} mirages before anyone walked.",
            (f"Hearing test: the expedition can detect planted alpha down to "
             f"s={jnd['jnd']}." if jnd["jnd"] is not None else
             "Hearing test: no planted strength on the grid was detectable -- "
             "the world is loud and the budget is small."),
            (f"The attention pool went to {attention_grantee} "
             f"(highest marginal yield)." if attention_grantee else ""),
            (f"{dive_added} submarines dove beneath the champion's surface; "
             f"{dive_members} made the shipped party." if dive_added else ""),
            (f"Venom memory holds {len(TABOO)} motifs." if TABOO else ""),
            (f"The circadian governor cut: {', '.join(self._circadian_cuts)}."
             if self._circadian_cuts else ""),
            (f"A cairn from a previous visit was found; gauge drift "
             f"{cairn_drift['gauge_edge_drift']}." if cairn_drift else
             "No previous cairn was found; ours now stands."),
        ]
        write_chronicle({
            "title": f"DRW world-explorer v25 ({data_source})",
            "features": len(cols), "train_rows": n, "sealed_rows": int(len(sealed_idx)),
            "data_source": data_source, "terrain_pop": t_pop, "weather_pop": w_pop,
            "even_dominant": n_even, "explorer_lines": explorer_lines,
            "embodiment_lines": [s for s in embodiment_lines if s],
            "n_lessons": len(library.lessons), "n_promoted": len(library.promoted()),
            "n_families": int(tex_df["trail_family"].nunique()) if not tex_df.empty else 0,
            "champion_lines": champion_lines, "kill_lines": kill_lines,
            "dream_lines": dream_lines, "shipping_line": ship_desc,
            "sealed_line": (f"Behind the glass, the sealed {len(sealed_idx)} minutes answered: "
                            f"{sealed_corr:+.5f}." if sealed_corr is not None else ""),
        })

        summary = {
            "data_source": data_source, "train_rows": n, "working_rows": n_work,
            "sealed_rows": int(len(sealed_idx)), "test_rows": n_test,
            "features": len(cols), "segments": cfg.N_SEGMENTS,
            "splits": cfg.N_SPLITS, "wf_folds": cfg.WF_FOLDS,
            "gbdt_backend": GBDT_BACKEND, "nnls_available": HAVE_NNLS,
            "hardware": {"torch": HAVE_TORCH, "gpus": N_GPUS,
                         "gpu_names": _gpu_names(),
                         "hetero_pairing": hetero,
                         "schedule": "hetero" if hetero else ("gpu" if N_GPUS > 0 else "cpu")},
            "lessons": len(library.lessons), "promoted": len(library.promoted()),
            "predator": {"targets": int(len(pred_df)) if not pred_df.empty else 0,
                         "killed": int((ledger_df["decision"] == "predator_killed").sum()) if not ledger_df.empty else 0,
                         "budget_left": predator.budget},
            "draft_gate": gate.report(),
            "evolution": evo_summary,
            "topography": {"terrain_clusters": len(t_pop), "terrain_populations": t_pop,
                           "even_dominant_features": n_even,
                           "textured_trails": int(len(tex_df)),
                           "trail_families_total": int(tex_df["trail_family"].nunique()) if not tex_df.empty else 0,
                           "trail_families_in_blend": n_trail_families,
                           "predator_terrain_kills": int(sum("dead_terrain" in (l.reason or "")
                                                             for l in library.lessons))},
            "ecology": {"weather_states": len(w_pop), "weather_populations": w_pop,
                        "weather_conditional_blend": final_weather is not None,
                        "mycelium_columns": len(MYCELIUM),
                        "dream_omens": dream_omens,
                        "predator_weather_kills": int(sum("dead_weather" in (l.reason or "")
                                                          for l in library.lessons))},
            "embodiment": {"trap_mirages": len(TRAPS),
                           "survey_best_family": max(SURVEY, key=SURVEY.get) if SURVEY else None,
                           "jnd_threshold": jnd["jnd"],
                           "attention_grantee": attention_grantee,
                           "car_stalls": int(sum(1 for r in journal_rows
                                                 if r.get("reason") == "car_stalled")
                                             + sum(1 for h in evo.history
                                                   if h.get("decision") == "car_stalled")),
                           "taboo_motifs": len(TABOO),
                           "dive_lessons": dive_added,
                           "dive_members_shipped": dive_members,
                           "circadian_cuts": self._circadian_cuts,
                           "cairn_drift": cairn_drift},
            "menagerie": {"species_run": [ex.species for ex in explorers],
                          "families_reached_quorum": n_quorum,
                          "waggle_dances": len(DANCES),
                          "gene_pool_size": len(GENE_POOL),
                          "evolution_op_counts": {op: len(g) for op, g in evo.op_gains.items()
                                                  if g and op in ("chemotaxis", "flock", "plasmid",
                                                                  "antiflock")}},
            "bio_ecology": {"red_pheromone_columns": len(RED_MYCELIUM),
                            "red_from_traps": len(TRAPS),
                            "seedbank_germinated": len(SEEDBANK),
                            "seedbank_written": len(seed_bank),
                            "islands": bool(cfg.ISLANDS),
                            "antiflock_runs": len(evo.op_gains.get("antiflock", [])),
                            "member_input_jaccard_cap": cfg.JAMMING_JACCARD},
            "anti_decay": {"shrunk_blend_lambda": shrunk_lambda,
                           "cv_forward_gap": round(result["honest"][result["winner"]]
                                                   - forward_blend_corr, 5),
                           "note": "shrunk_blend defaults to 0; >0 only if forward strictly improves"},
            "forensics": {"enabled": bool(cfg.FORENSIC_ENABLED),
                          "override": bool(forensic_dec["override"]),
                          "shipped_selector": forensic_dec["winner"],
                          "forward_blend_corr": round(forensic_dec["forward_blend_corr"], 5),
                          "feature_clusters": int(len(np.unique(FCLUST))) if FCLUST is not None else 0,
                          "note": "full forensic suite written; override is forward-validated, no-op unless it strictly improves the CV->forward gap"},
            "beacons": {"dropped": len(BEACONS.kinds) if BEACONS is not None else 0,
                        "rare": sum(k == "rare" for k in BEACONS.kinds) if BEACONS is not None else 0,
                        "novelty": sum(k == "novelty" for k in BEACONS.kinds) if BEACONS is not None else 0,
                        "field_channels_added": len(BEACONS.kinds) if BEACONS is not None else 0,
                        "predator_beacon_kills": int(sum("dead_beacon" in (l.reason or "")
                                                         for l in library.lessons))},
            "metabolism": {"time_budget_min": cfg.TIME_BUDGET_MIN,
                           "enabled": META.enabled,
                           "shipping_reserve_min": round(META.reserve, 1),
                           "seasons": META.seasons, "epochs": META.epochs,
                           "auditioned_skills": auditioned,
                           "elapsed_min": round(META.now(), 1),
                           "deadlines_min": {k: round(v, 1) for k, v in META.deadline.items()}},
            "members": list(members), "ensemble_winner": result["winner"],
            "honest_scores": result["honest"], "weights": w,
            "forward_blend_corr": forward_blend_corr,
            "forward_best_single": {best_cv_name: single_fwd_corr},
            "forward_gate_fired": gate_fired,
            "shipped_weights": final_weights,
            "sealed_holdout_corr": sealed_corr,
            "winsorize_q": wins["best_q"] if wins["apply"] else None,
            "alarms": int((alarms["status"] == "ALARM").sum()) if not alarms.empty else 0,
        }
        write_json(summary, "explorer_run_summary.json")
        META.heartbeat("run_end")
        mem_status("run_end")
        log("run_end", winner=result["winner"],
            honest=round(result["honest"][result["winner"]], 5),
            forward=round(forward_blend_corr, 5),
            sealed=round(sealed_corr, 5) if sealed_corr is not None else None,
            gate=gate_fired, alarms=summary["alarms"])
        print(json.dumps({k: v for k, v in summary.items() if k != "honest_scores"}, indent=2, default=str))
        return summary


