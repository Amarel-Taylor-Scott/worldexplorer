class _RunState:
    """Method-object state for ExplorerHarness.run: every cross-phase
    local of the old monolithic run() lives here as an attribute
    (rs.<name>), so the phase methods share dataflow without a
    1500-line scope."""


class ExplorerHarness:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        np.random.seed(cfg.SEED)

    def run(self) -> dict[str, Any]:
        rs = _RunState()
        self._setup(rs)
        self._load_data(rs)
        self._quarantine_probe(rs)
        self._build_atlas(rs)
        self._beacons(rs)
        self._pre_scans(rs)
        self._phase1_explore(rs)
        self._raid1(rs)
        self._phase2_evolve(rs)
        self._raid2(rs)
        self._ablation_dive(rs)
        self._trail_reports(rs)
        self._select_members(rs)
        self._ensemble(rs)
        self._forward_holdout(rs)
        self._governor(rs)
        self._forensics(rs)
        self._forward_gate(rs)
        self._shipping_court(rs)
        self._shrink_chorus_shape(rs)
        self._health_alarms(rs)
        self._sealed_audit(rs)
        self._final_refit_submit(rs)
        self._cairn_ledger(rs)
        self._chronicle(rs)
        return self._summarize(rs)

    def _setup(self, rs: "_RunState") -> None:
        global ATLAS, GAUGE
        rs.cfg = self.cfg
        MYCELIUM.clear()                 # fresh pheromone network every run
        TRAPS.clear()                    # fresh threat map (v10)
        TABOO.clear()                    # fresh venom memory (v10)
        SURVEY.clear()                   # fresh satellite map (v10)
        QUORUM.clear()                   # fresh colony consensus (v11)
        DANCES.clear()                   # fresh waggle floor (v11)
        GENE_POOL.clear()                # fresh plasmid pool (v11)
        RED_MYCELIUM.clear()             # fresh repellent channel (v14)
        GOVERNOR.clear()                 # v27: fresh runtime complexity-generalization governor
        WIDTH_BIAS["n"] = 0              # v30: fresh wide-path annealing clock
        if rs.cfg.WIDE_PERSONA:          # v33: the albatross occupies roster slot 7
            rs.cfg.N_EXPLORERS = max(rs.cfg.N_EXPLORERS, 8)
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
            rs.seed_paths = [Path(p) for p in rs.cfg.CAIRN_PATHS]
            try:
                rs.seed_paths += list(Path("/kaggle/input").glob("*/world_cairn.json"))
            except Exception:
                pass
            for rs.pth in rs.seed_paths:
                if rs.pth.exists():
                    rs.prev = json.loads(rs.pth.read_text())
                    for rs.wk in (rs.prev.get("seed_bank") or [])[: rs.cfg.SEED_GERMINATE]:
                        rs.g = parse_genome_key(rs.wk)
                        if rs.g is not None:
                            SEEDBANK.append(rs.g)
                    # v27 self-improvement: load the cross-run learning ledger as priors
                    if rs.cfg.SELF_IMPROVE and rs.prev.get("ledger"):
                        LEDGER_PRIOR.clear(); LEDGER_PRIOR.update(rs.prev["ledger"])
                        for rs.wk in (LEDGER_PRIOR.get("survivors") or [])[: rs.cfg.SEED_GERMINATE]:
                            rs.gs = parse_genome_key(rs.wk)
                            if rs.gs is not None and all(s.key != rs.gs.key for s in SEEDBANK):
                                SEEDBANK.append(rs.gs)
                        for rs.mk in (LEDGER_PRIOR.get("decayers") or [])[: rs.cfg.LEDGER_MAX_DECAYERS]:
                            TABOO[rs.mk] = TABOO.get(rs.mk, 0.0) + 1.0     # cross-run anti-prior: avoid decayed motifs
                        rs.gp = LEDGER_PRIOR.get("governor") or {}
                        log("learning_ledger_loaded", governor_runs=int(rs.gp.get("count", 0)),
                            prior_beta=round(float(rs.gp.get("beta", 0.0)), 4),
                            families=len(LEDGER_PRIOR.get("family_decay") or {}),
                            survivors=len(LEDGER_PRIOR.get("survivors") or []),
                            decayers=len(LEDGER_PRIOR.get("decayers") or []))
                    if SEEDBANK:
                        log("seedbank_loaded", count=len(SEEDBANK), from_cairn=str(rs.pth),
                            keys="|".join(g.key for g in SEEDBANK))
                    break
        except Exception:
            pass

        def circadian(frac_target: float, tag: str) -> None:
            """v10 governor: the body has a clock. If the elapsed fraction of
            RUN_DEADLINE_MIN exceeds this phase's scheduled fraction, shed
            cost gracefully -- the organism always makes camp before dark."""
            if not rs.cfg.RUN_DEADLINE_MIN or rs.cfg.RUN_DEADLINE_MIN <= 0:
                return
            frac = (time.monotonic() - RUN_START) / 60.0 / rs.cfg.RUN_DEADLINE_MIN
            if frac <= frac_target:
                return
            over = frac / max(frac_target, 1e-9)
            cuts = []
            if rs.cfg.SEED_REPS_STOCHASTIC > 0:
                rs.cfg.SEED_REPS_STOCHASTIC = 0
                cuts.append("seed_reps->0")
            new_gen = max(1, int(rs.cfg.EVOLUTION_MAX_GENERATIONS / over))
            if new_gen < rs.cfg.EVOLUTION_MAX_GENERATIONS:
                rs.cfg.EVOLUTION_MAX_GENERATIONS = new_gen
                cuts.append(f"generations->{new_gen}")
            if rs.cfg.DREAM_REPLAYS > 30:
                rs.cfg.DREAM_REPLAYS = max(30, rs.cfg.DREAM_REPLAYS // 2)
                cuts.append(f"dream_replays->{rs.cfg.DREAM_REPLAYS}")
            if rs.cfg.DIVE_BUDGET > 2:
                rs.cfg.DIVE_BUDGET = max(2, rs.cfg.DIVE_BUDGET // 2)
                cuts.append(f"dive_budget->{rs.cfg.DIVE_BUDGET}")
            if rs.cfg.PREDATOR_MAX_TARGETS > 4:
                rs.cfg.PREDATOR_MAX_TARGETS = max(4, rs.cfg.PREDATOR_MAX_TARGETS // 2)
                cuts.append(f"predator_targets->{rs.cfg.PREDATOR_MAX_TARGETS}")
            self._circadian_cuts += cuts
            log("CIRCADIAN_CUT", tag=tag, elapsed_frac=round(frac, 3),
                target_frac=frac_target, cuts="|".join(cuts) if cuts else "none_left")
        rs.circadian = circadian

        self._circadian_cuts: list[str] = []

        # ---- v12 METABOLISM: arm the energy ledger ---------------------------
        global META
        META = Metabolism(rs.cfg)
        if META.enabled and not rs.cfg.RUN_DEADLINE_MIN:
            # the circadian governor becomes the metabolism's late-running
            # backstop: shed-cost cuts engage just past the metabolic budget
            rs.cfg.RUN_DEADLINE_MIN = round(rs.cfg.TIME_BUDGET_MIN * 1.04, 1)
            log("circadian_backstop_armed", run_deadline_min=rs.cfg.RUN_DEADLINE_MIN,
                note="sheds cost only if shipping runs past the metabolic plan")

        def phase_frac(phase: str, fallback: float) -> float:
            """When the metabolism is armed, the circadian backstop's phase
            targets follow the metabolic plan instead of the fixed v10 map."""
            if META.enabled and rs.cfg.RUN_DEADLINE_MIN and phase in META.deadline:
                return min(1.0, META.deadline[phase] / rs.cfg.RUN_DEADLINE_MIN + 0.05)
            return fallback
        rs.phase_frac = phase_frac

        log("run_start", out=str(OUT), seed=rs.cfg.SEED,
            time_budget_min=rs.cfg.TIME_BUDGET_MIN, explorers=rs.cfg.N_EXPLORERS,
            lesson_budget=rs.cfg.LESSON_BUDGET, evolution_budget=rs.cfg.EVOLUTION_BUDGET,
            gbdt_backend=str(GBDT_BACKEND), nnls=HAVE_NNLS,
            torch=HAVE_TORCH, gpus=N_GPUS)

        # hardware profile: same objective, the expensive skills reshape per device
        if N_GPUS > 0:
            rs.cfg.MLP_HIDDEN = rs.cfg.GPU_MLP_HIDDEN
            rs.cfg.MLP_MAX_ITER = rs.cfg.GPU_MLP_MAX_ITER
            rs.cfg.MLP_MAX_ROWS = rs.cfg.GPU_MLP_MAX_ROWS
            rs.cfg.MLP_BATCH = rs.cfg.GPU_MLP_BATCH
            rs.cfg.GBDT_ESTIMATORS = rs.cfg.GPU_GBDT_ESTIMATORS
            log("hardware_profile", schedule="gpu", gpus=N_GPUS,
                names="|".join(_gpu_names()),
                mlp_hidden=str(rs.cfg.MLP_HIDDEN), mlp_rows=rs.cfg.MLP_MAX_ROWS,
                mlp_iters=rs.cfg.MLP_MAX_ITER, gbdt_estimators=rs.cfg.GBDT_ESTIMATORS,
                two_gpu_concurrency=N_GPUS >= 2)
        else:
            log("hardware_profile", schedule="cpu",
                note="identical_v5_behavior" if not HAVE_TORCH else "torch_cpu_mlp_with_pearson_loss")
        rs.hetero = bool(rs.cfg.HETERO_PAIRING) and N_GPUS > 0
        if rs.hetero:
            log("hetero_lanes_enabled",
                note="gpu-lane and cpu-lane lessons run simultaneously (phase 1 + evolution)")
        rs.personas = [{k: v for k, v in t.items() if not isinstance(v, dict)} for t in EXPLORER_TRAITS]
        rs.personas.append({"name": "predator_skeptic", "metaheuristic": "immune_system_falsifier",
                         "curiosity": 0.0, "caution": 1.0, "sociality": 1.0})
        write_csv(pd.DataFrame(rs.personas), "explorer_personas.csv")


    def _load_data(self, rs: "_RunState") -> None:
        # ---- data ----------------------------------------------------------
        rs.root = find_data_root()
        if rs.root is not None:
            rs.train, rs.test = load_competition(rs.root)
            rs.data_source = "competition"
        else:
            if not rs.cfg.ALLOW_SYNTHETIC_FALLBACK:
                raise SystemExit("DRW data not found and synthetic fallback disabled.")
            print("=" * 78)
            print("!! NO COMPETITION DATA FOUND -> RUNNING ON A LABELED SYNTHETIC SMOKE FIXTURE")
            print("!! Attach the DRW dataset for real results. All artifacts are tagged.")
            print("=" * 78)
            rs.train, rs.test = make_synthetic(rs.cfg.SYN_ROWS, rs.cfg.SYN_ANON, rs.cfg.SEED)
            rs.data_source = "SYNTHETIC_SMOKE"
        log("data_loaded", source=rs.data_source, train_rows=len(rs.train), test_rows=len(rs.test))

        rs.eng_tr = add_market_features(rs.train)
        rs.eng_te = add_market_features(rs.test)
        if not rs.eng_tr.empty:
            rs.train = pd.concat([rs.eng_tr, rs.train], axis=1)
            rs.test = pd.concat([rs.eng_te, rs.test], axis=1)
        del rs.eng_tr, rs.eng_te
        gc.collect()

        rs.cols = [c for c in rs.train.columns
                if c not in ("label", "timestamp") and pd.api.types.is_numeric_dtype(rs.train[c])]
        rs.cols = [c for c in rs.cols if c in rs.test.columns]
        rs.y_full = rs.train["label"].to_numpy(np.float32)
        rs.n = len(rs.train)
        rs.seg_full = (np.arange(rs.n) * rs.cfg.N_SEGMENTS // rs.n).astype(np.int32)
        rs.medians = rs.train[rs.cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        rs.vol_full = rs.train["volume"].to_numpy(np.float32) if "volume" in rs.train.columns else None

        log("building_matrices", features=len(rs.cols))
        rs.X_full = build_matrix(rs.train, rs.cols, rs.medians)
        rs.X_test = build_matrix(rs.test, rs.cols, rs.medians)
        del rs.train, rs.test
        gc.collect()
        mem_status("post_data_matrices")

        # ---- v26 DATA PROFILE + GEOMETRY SEAM (generalization) ---------------
        # Detect target type + time-order; under cfg.METRIC/GEOMETRY="auto" this
        # picks the active metric and CV geometry. DEFAULTS (pearson/temporal)
        # reproduce DRW exactly. For NON-temporal data, ONE row-permutation turns
        # all positional CV (segments, walk-forward, sealed/forward tails) into
        # honest RANDOM CV -- so the same harness runs on any tabular dataset.
        global PROFILE
        rs._prof = profile_data(rs.X_full, rs.y_full, rs.cols, rs.cfg)
        PROFILE = resolve_profile(rs._prof, rs.cfg)
        write_json({**rs._prof, "active_metric": PROFILE["metric"],
                    "active_temporal": PROFILE["temporal"],
                    "metric_cfg": rs.cfg.METRIC, "geometry_cfg": rs.cfg.GEOMETRY}, "data_profile.json")
        log("data_profile", target_kind=PROFILE["target_kind"], metric=PROFILE["metric"],
            temporal=PROFILE["temporal"], feature_autocorr=rs._prof["feature_autocorr"],
            note="defaults (pearson/temporal) reproduce v25; 'auto' adapts to any dataset")
        if not PROFILE["temporal"]:
            rs._perm = np.random.default_rng(rs.cfg.SEED + 777).permutation(rs.n)
            rs.X_full = rs.X_full[rs._perm]; rs.y_full = rs.y_full[rs._perm]
            if rs.vol_full is not None:
                rs.vol_full = rs.vol_full[rs._perm]
            log("geometry_randomized",
                note="non-temporal data: rows permuted so positional CV (segments/WF/sealed) becomes random CV")


    def _quarantine_probe(self, rs: "_RunState") -> None:
        # ---- SEALED HOLDOUT quarantine ---------------------------------------
        # The final SEALED_FRACTION of rows is invisible to every decision below.
        rs.seal_cut = int((1 - rs.cfg.SEALED_FRACTION) * rs.n)
        rs.n_work = rs.seal_cut
        rs.sealed_idx = np.arange(rs.seal_cut, rs.n)
        log("sealed_holdout_quarantined", sealed_rows=len(rs.sealed_idx), working_rows=rs.n_work,
            note="evaluated exactly once after the blend is frozen; never gated on")

        # ---- probe subsample from the WORKING region only --------------------
        if rs.n_work > rs.cfg.PROBE_MAX_ROWS:
            rs.keep = np.unique(np.linspace(0, rs.n_work - 1, rs.cfg.PROBE_MAX_ROWS).round().astype(int))
        else:
            rs.keep = np.arange(rs.n_work)
        rs.Xp, rs.yp, rs.segp = rs.X_full[rs.keep], rs.y_full[rs.keep], rs.seg_full[rs.keep]
        rs.volp = rs.vol_full[rs.keep] if rs.vol_full is not None else None
        rs.embargo_p = max(1, int(rs.cfg.EMBARGO_ROWS * len(rs.keep) / max(1, rs.n_work)))
        log("probe_ready", probe_rows=len(rs.keep), segments=int(len(np.unique(rs.segp))),
            embargo_rows=rs.embargo_p, splits=rs.cfg.N_SPLITS, wf_folds=rs.cfg.WF_FOLDS)

        # ---- v31 TEST-LIKENESS sensor (IDEAS_ZOO B1): X-only, target-free ------
        # Working-vs-test classifier on features alone; per-row 'looks like
        # test' scores feed the robust selector's testlike partitions (flag)
        # and a drift report. Labels and the leaderboard never touch this.
        global TESTLIKE
        TESTLIKE = None
        if rs.cfg.TESTLIKE_REPORT or rs.cfg.TESTLIKE_PARTITIONS:
            try:
                TESTLIKE, rs.testlike_info = fit_testlike(rs.X_full, rs.X_test, rs.n_work, rs.cfg)
                write_json({**rs.testlike_info,
                            "note": "target-free working-vs-test classifier; AUC~0.5 = no shift; "
                                    "feeds robust-selector partitions when TESTLIKE_PARTITIONS"},
                           "testlike_report.json")
                log("testlike_sensor", **rs.testlike_info)
            except Exception as e:
                TESTLIKE = None
                log("testlike_skipped", err=str(e)[:80])


    def _build_atlas(self, rs: "_RunState") -> None:
        global ATLAS, GAUGE, PRESSURE
        # ---- TERRAIN ATLAS: target-free map of the space (working X only) -----
        # y never touches the atlas, so its ids are leak-free everywhere; the
        # 'terrain' family, terrain_router skill, predator terrain attack and
        # all per-terrain reports read from this one map.
        ATLAS = TerrainAtlas(rs.cfg.TERRAIN_CLUSTERS, rs.cfg.SEED).fit(rs.Xp, rs.cols, rs.cfg.TERRAIN_FIT_ROWS)
        rs.terr_p = ATLAS.assign(rs.Xp)
        rs.t_pop = {int(t): int((rs.terr_p == t).sum()) for t in np.unique(rs.terr_p)}
        rs.alt_p = ATLAS.altitude(rs.Xp)
        write_csv(pd.DataFrame([{"terrain": t, "rows": npop,
                                 "frac": npop / len(rs.terr_p),
                                 "mean_altitude": float(np.mean(rs.alt_p[rs.terr_p == t])),
                                 "max_altitude": float(np.max(rs.alt_p[rs.terr_p == t]))}
                                for t, npop in sorted(rs.t_pop.items())]),
                  "terrain_atlas_report.csv")
        log("terrain_atlas_built", clusters=len(rs.t_pop), populations=str(rs.t_pop),
            note="target_free__leakfree_everywhere")

        # ---- WEATHER GAUGE (v9): row-local volatility states, target-free ----
        GAUGE = WeatherGauge(rs.cfg.WEATHER_STATES).fit(rs.Xp, rs.cols)
        rs.wth_p = GAUGE.assign(rs.Xp)
        rs.w_pop = {int(s): int((rs.wth_p == s).sum()) for s in np.unique(rs.wth_p)}
        log("weather_gauge_built", states=len(rs.w_pop), populations=str(rs.w_pop),
            note="row_local__order_free__target_free")

        # ---- PRESSURE GAUGE (v20, FIT FIXED in v32): the order-book twin of the
        # weather gauge. Built in v20 but never fit by the harness -- the
        # pressure family silently corr-fell-back in EVERY run since. Fitting it
        # activates the family for real and enables pressure_moe blending.
        PRESSURE = PressureGauge(rs.cfg.WEATHER_STATES).fit(rs.Xp, rs.cols)
        rs.prs_p = PRESSURE.assign(rs.Xp)
        rs.p_pop = {int(s): int((rs.prs_p == s).sum()) for s in np.unique(rs.prs_p)}
        log("pressure_gauge_built", states=len(rs.p_pop), populations=str(rs.p_pop),
            note="v20 gauge, fit for the first time (latent bug fixed in v32)")

        # ---- v31 latent FACTORS (target-free, IDEAS_ZOO B2): top PCA factor
        # scores of the probe matrix -- the exposure substrate for the member
        # redundancy/crowding report ("different names, same latent bet").
        rs.factor_scores = None
        if rs.cfg.REDUNDANCY_REPORT:
            try:
                sub_f = rs.Xp[:: max(1, len(rs.Xp) // 20_000)]
                mu_f = sub_f.mean(axis=0, keepdims=True)
                _, _, vt_f = np.linalg.svd(sub_f - mu_f, full_matrices=False)
                comps_f = vt_f[: int(rs.cfg.FACTOR_COUNT)].astype(np.float32)
                rs.factor_scores = ((rs.Xp - mu_f) @ comps_f.T).astype(np.float32)
                rs.factor_mu, rs.factor_comps = mu_f.astype(np.float32), comps_f
                rs.factor_dpre = int(rs.Xp.shape[1])   # pre-beacon column count
                log("latent_factors_built", n_factors=int(comps_f.shape[0]),
                    note="target-free PCA factors; exposure substrate for crowding report")
            except Exception as e:
                rs.factor_scores = None
                log("latent_factors_skipped", err=str(e)[:80])

        # ---- ADVERSARIAL VALIDATION (v24): early-vs-late covariate-shift map --
        try:
            rs.adv_df, rs.adv_auc = adversarial_validation_report(rs.Xp, rs.cols, rs.cfg)
            if not rs.adv_df.empty:
                write_csv(rs.adv_df, "adversarial_validation.csv")
            log("adversarial_validation", auc=round(rs.adv_auc, 4),
                top_shift=str(rs.adv_df["feature"].iloc[0]) if not rs.adv_df.empty else None,
                note="early-vs-late drift; AUC~0.5 stable, >>0.5 strong shift; target-free, gates nothing")
        except Exception as e:
            log("adversarial_validation_skipped", err=str(e)[:80])


    def _beacons(self, rs: "_RunState") -> None:
        # ---- BEACON FIELD (v15): drop items at unique typologies -------------
        # Items placed at rare-terrain + novelty-altitude coordinates (all
        # target-free, from the atlas) emit a radial RBF field; the field
        # becomes NEW FEATURE CHANNELS appended to the matrix, so every
        # explorer can see and bend around the landmarks. Leak-free everywhere.
        global BEACONS
        if rs.cfg.BEACON_DROP and ATLAS is not None:
            BEACONS = BeaconField(ATLAS, rs.cfg).fit(rs.Xp)
            rs.fld_full = BEACONS.field(rs.X_full)
            rs.fld_test = BEACONS.field(rs.X_test)
            rs.beacon_names = [f"beacon_{i}_{BEACONS.kinds[i]}" for i in range(rs.fld_full.shape[1])]
            rs.X_full = np.ascontiguousarray(np.hstack([rs.X_full, rs.fld_full]).astype(np.float32))
            rs.X_test = np.ascontiguousarray(np.hstack([rs.X_test, rs.fld_test]).astype(np.float32))
            rs.cols = rs.cols + rs.beacon_names
            rs.Xp = rs.X_full[rs.keep]            # the probe now carries the field channels too
            rs.bf_p = BEACONS.field(rs.Xp)
            rs.b_assign = BEACONS.assign(rs.Xp)
            rs.b_pop = {int(b): int((rs.b_assign == b).sum()) for b in np.unique(rs.b_assign)}
            write_csv(pd.DataFrame([{"beacon": i, "kind": BEACONS.kinds[i],
                                     "sigma": round(float(BEACONS.sigmas[i]), 4),
                                     "basin_rows": rs.b_pop.get(i, 0),
                                     "mean_activation": round(float(rs.bf_p[:, i].mean()), 4),
                                     "corr_to_y": round(pearson(rs.yp, rs.bf_p[:, i]), 4)}
                                    for i in range(len(BEACONS.kinds))]),
                      "beacon_atlas_report.csv")
            log("beacons_dropped", n=len(BEACONS.kinds),
                rare=sum(k == "rare" for k in BEACONS.kinds),
                novelty=sum(k == "novelty" for k in BEACONS.kinds),
                features=len(rs.cols), note="field channels appended; target_free__leakfree")
        else:
            BEACONS = None


    def _pre_scans(self, rs: "_RunState") -> None:
        # ---- SYMMETRY FIELD: even-vs-odd response of the strongest features ---
        rs.sym_df = symmetry_field_report(rs.Xp, rs.yp, rs.cols)
        write_csv(rs.sym_df, "symmetry_field_report.csv")
        # v18 label archaeology (train-side forensics on the anonymized target)
        try:
            rs.arch_df = label_archaeology(rs.Xp, rs.yp, rs.segp, rs.cols)
            write_csv(rs.arch_df, "label_archaeology.csv")
            if not rs.arch_df.empty:
                rs.bestrow = rs.arch_df.iloc[rs.arch_df["best_lagged_corr"].abs().argmax()]
                log("label_archaeology", best_lag=int(rs.bestrow["lag"]),
                    best_feature=str(rs.bestrow["best_feature"]),
                    best_lagged_corr=round(float(rs.bestrow["best_lagged_corr"]), 4),
                    note="is y a feature's future value? train-side map, gates nothing")
        except Exception as e:
            log("label_archaeology_skipped", err=str(e)[:80])
        rs.n_even = int((rs.sym_df["even_excess"] > 0).sum()) if not rs.sym_df.empty else 0
        log("symmetry_field", features=len(rs.sym_df), even_dominant=rs.n_even,
            note="motivates fold_abs/fold_pairs viewports (report, not a gate)")

        # ---- TRAP MAP (v10): fear scans before hunger ------------------------
        rs.traps, rs.trap_df = build_trap_map(rs.Xp, rs.yp, rs.segp, rs.cfg)
        TRAPS.update(rs.traps)
        for rs.ci in rs.traps:                 # v14: mirages seed the repellent channel
            RED_MYCELIUM[rs.ci] = RED_MYCELIUM.get(rs.ci, 0.0) + 1.0
        write_csv(rs.trap_df, "trap_map.csv")
        log("trap_map", scanned=len(rs.trap_df), mirages=len(TRAPS),
            note="mirage features demoted in corr-driven rankings")

        # ---- SATELLITE SURVEY (v10): orbit every family before walking -------
        rs.sat_cfg = replace(rs.cfg, DRAFT_ROWS=max(2_000, len(rs.keep) // rs.cfg.SAT_STRIDE),
                          DRAFT_FOLDS=rs.cfg.SAT_FOLDS)
        rs.sat_rows = []
        rs.k_sat = min(len(rs.cols), max(8, rs.cfg.BIT_BUDGET // TRANSFORM_BITS["quantize2"]))
        for rs.fam in FAMILIES:
            rs.spec_s = ViewportSpec(name=f"{rs.fam}{rs.k_sat}_quantize2", family=rs.fam, k=rs.k_sat,
                                  transform="quantize2", proj_dim=16)
            rs.d = run_draft("majority_vote", rs.spec_s, rs.Xp, rs.yp, rs.segp, rs.cols, rs.sat_cfg,
                          rs.embargo_p, stable_seed(rs.cfg.SEED, "satellite", rs.fam))
            SURVEY[rs.fam] = max(0.0, float(rs.d["draft_corr"]))
            rs.sat_rows.append({"family": rs.fam, "survey_corr": rs.d["draft_corr"],
                             "survey_width": rs.d["draft_width"], "k": rs.k_sat})
        write_csv(pd.DataFrame(rs.sat_rows).sort_values("survey_corr", ascending=False),
                  "survey_map.csv")
        log("satellite_survey", families=len(SURVEY),
            best=max(SURVEY, key=SURVEY.get), note="feeds bandit family priors")

        # ---- SENSORY THRESHOLD (v10): how quiet a signal can we hear? --------
        rs.jnd = jnd_probe(rs.Xp, rs.yp, rs.segp, rs.cols, rs.cfg, rs.embargo_p)
        write_json(rs.jnd, "sensory_threshold.json")
        log("sensory_threshold", jnd=rs.jnd["jnd"],
            grid="|".join(str(s) for s in rs.cfg.JND_STRENGTHS), note="calibration only")
        gc.collect()
        mem_status("post_atlas")


    def _phase1_explore(self, rs: "_RunState") -> None:
        # ---- PHASE 1: developmental explorers (with draft culling) -----------
        rs.library = SharedLibrary()
        rs.gate = DraftGate(rs.cfg)
        rs.spec_lookup: dict[str, ViewportSpec] = {}
        rs.journal_rows, rs.growth_rows = [], []

        def attempt_lesson(ex_name: str, stage_name: str, skill: str, spec: ViewportSpec,
                           oofs_snap: dict[str, np.ndarray]) -> tuple:
            """Draft-gate + full lesson, with NO shared-state mutation beyond the
            draft gate's append (GIL-atomic) -- safe to run in a lane thread.
            Library/journal/budget updates happen in the main thread after join."""
            key = f"{skill}|{spec.name}"
            d_width = float("nan")
            if SKILL_REGISTRY[skill]["cost"] >= rs.cfg.DRAFT_MIN_COST:
                d = run_draft(skill, spec, rs.Xp, rs.yp, rs.segp, rs.cols, rs.cfg, rs.embargo_p,
                              stable_seed(rs.cfg.SEED, "draft", key))
                d_width = d["draft_width"]
                if not rs.gate.passes(d_width):
                    return ("culled", skill, spec, d)
                car = run_car(skill, spec, rs.Xp, rs.yp, rs.segp, rs.cols, rs.cfg, rs.embargo_p,
                              stable_seed(rs.cfg.SEED, "car", key))
                if car["draft_width"] <= 0:           # v10: stalled on the road
                    car["car_stalled"] = 1.0
                    return ("culled", skill, spec, car)
            seed = stable_seed(rs.cfg.SEED, key, rs.library.runs.get(key, 0))
            lesson = run_lesson(ex_name, stage_name, skill, spec, rs.Xp, rs.yp, rs.segp, rs.cols,
                                rs.cfg, rs.embargo_p, seed, oofs_snap, draft_width=d_width)
            return ("done", skill, spec, lesson)
        rs.attempt_lesson = attempt_lesson

        # ---- PANORAMA (v10): the 360-degree freeze-and-orient scan -----------
        if rs.cfg.PANORAMA_FIRST:
            rs.k_pan = min(len(rs.cols), max(8, rs.cfg.BIT_BUDGET // TRANSFORM_BITS["sign_only"]))
            rs.spec_pan = ViewportSpec(name=f"top{rs.k_pan}_sign_only", family="top", k=rs.k_pan,
                                    transform="sign_only", proj_dim=16)
            rs.pan = run_lesson("panorama", "panorama", "majority_vote", rs.spec_pan,
                             rs.Xp, rs.yp, rs.segp, rs.cols, rs.cfg, rs.embargo_p,
                             stable_seed(rs.cfg.SEED, "panorama"), {})
            rs.library.add(rs.pan)
            rs.spec_lookup[rs.pan.key] = rs.spec_pan
            rs.journal_rows.append({"explorer": "panorama", "lesson_idx": 0, "stage": "panorama",
                                 "key": rs.pan.key, "family": "top", "transform": "sign_only",
                                 "k": rs.k_pan, "oof_corr": rs.pan.oof_corr, "width": rs.pan.width,
                                 "wf_corr": rs.pan.wf_corr, "wf_width": rs.pan.wf_width,
                                 "stability": rs.pan.stability, "seed_var": rs.pan.seed_var,
                                 "overfit_ratio": rs.pan.overfit_ratio, "decision": rs.pan.decision,
                                 "reason": rs.pan.reason, "budget_left": 0})
            log("panorama", k=rs.k_pan, oof_corr=round(rs.pan.oof_corr, 4),
                width=round(rs.pan.width, 4), decision=rs.pan.decision,
                note="the floor and the horizon, established before anyone walks")

        # ---- AUDITION PARADE (v12): every skill gets one measured lesson ------
        # v9 measured: relay_caravan and swell_rider were NEVER picked by any
        # bandit; in v11 swell_rider finally ran (via evolution) and instantly
        # WON the run. Unmeasured is not failed -- a registry entry that never
        # runs is a silent prior. Every skill now auditions once through the
        # SAME doors (draft gate + car rung included) before UCB free play.
        rs.auditioned = 0
        if rs.cfg.AUDITION_ALL_SKILLS:
            rs.k_aud = min(rs.cfg.AUDITION_K, len(rs.cols))
            for rs.skill in list(SKILL_REGISTRY):
                if any(l.skill == rs.skill for l in rs.library.lessons):
                    continue                      # already measured (e.g. panorama)
                rs.spec_a = ViewportSpec(name=f"top{rs.k_aud}_identity", family="top",
                                      k=rs.k_aud, transform="identity", proj_dim=16)
                rs.status, rs._, rs.r_spec, rs.payload = rs.attempt_lesson("audition", "audition",
                                                            rs.skill, rs.spec_a, rs.library.oofs())
                rs.key_a = f"{rs.skill}|{rs.spec_a.name}"
                if rs.status == "culled":
                    rs.library.note_draft_cull(rs.key_a)
                    rs.journal_rows.append({"explorer": "audition", "lesson_idx": rs.auditioned,
                                         "stage": "audition", "key": rs.key_a, "family": "top",
                                         "transform": "identity", "k": rs.k_aud,
                                         "oof_corr": rs.payload["draft_corr"],
                                         "width": rs.payload["draft_width"],
                                         "wf_corr": np.nan, "wf_width": np.nan,
                                         "stability": np.nan, "seed_var": np.nan,
                                         "overfit_ratio": np.nan, "decision": "draft_culled",
                                         "reason": "audition_below_draft_bar", "budget_left": 0})
                    log("audition_culled", skill=rs.skill,
                        draft_width=round(rs.payload["draft_width"], 4))
                    continue
                rs.library.add(rs.payload)
                rs.spec_lookup[rs.payload.key] = rs.r_spec
                rs.auditioned += 1
                rs.journal_rows.append({"explorer": "audition", "lesson_idx": rs.auditioned,
                                     "stage": "audition", "key": rs.payload.key, "family": "top",
                                     "transform": "identity", "k": rs.k_aud,
                                     "oof_corr": rs.payload.oof_corr, "width": rs.payload.width,
                                     "wf_corr": rs.payload.wf_corr, "wf_width": rs.payload.wf_width,
                                     "stability": rs.payload.stability, "seed_var": rs.payload.seed_var,
                                     "overfit_ratio": rs.payload.overfit_ratio,
                                     "decision": rs.payload.decision, "reason": rs.payload.reason,
                                     "budget_left": 0})
                log("audition", skill=rs.skill, key=rs.payload.key,
                    oof_corr=round(rs.payload.oof_corr, 4), width=round(rs.payload.width, 4),
                    decision=rs.payload.decision)
            # v13: transforms audition too -- a way of seeing that never runs
            # is just as silent a prior as a skill that never runs. One
            # linear_assoc lesson per unseen transform, at its bit-frontier k.
            for rs.tf in ALL_TRANSFORMS:
                if any(l.transform == rs.tf for l in rs.library.lessons):
                    continue
                rs.k_t = min(rs.cfg.AUDITION_K, len(rs.cols),
                          max(CFG.K_MIN, rs.cfg.BIT_BUDGET // TRANSFORM_BITS.get(rs.tf, 32)))
                rs.spec_t = ViewportSpec(name=f"top{rs.k_t}_{rs.tf}", family="top", k=rs.k_t,
                                      transform=rs.tf, proj_dim=16)
                if not rs.library.can_run("linear_assoc", rs.spec_t):
                    continue
                rs.status, rs._, rs.r_spec, rs.payload = rs.attempt_lesson("audition", "audition",
                                                            "linear_assoc", rs.spec_t, rs.library.oofs())
                rs.key_t = f"linear_assoc|{rs.spec_t.name}"
                if rs.status == "culled":
                    rs.library.note_draft_cull(rs.key_t)
                    log("audition_culled", transform=rs.tf,
                        draft_width=round(rs.payload["draft_width"], 4))
                    continue
                rs.library.add(rs.payload)
                rs.spec_lookup[rs.payload.key] = rs.r_spec
                rs.auditioned += 1
                rs.journal_rows.append({"explorer": "audition", "lesson_idx": rs.auditioned,
                                     "stage": "audition", "key": rs.payload.key, "family": "top",
                                     "transform": rs.tf, "k": rs.k_t,
                                     "oof_corr": rs.payload.oof_corr, "width": rs.payload.width,
                                     "wf_corr": rs.payload.wf_corr, "wf_width": rs.payload.wf_width,
                                     "stability": rs.payload.stability, "seed_var": rs.payload.seed_var,
                                     "overfit_ratio": rs.payload.overfit_ratio,
                                     "decision": rs.payload.decision, "reason": rs.payload.reason,
                                     "budget_left": 0})
                log("audition", transform=rs.tf, key=rs.payload.key,
                    oof_corr=round(rs.payload.oof_corr, 4), width=round(rs.payload.width, 4),
                    decision=rs.payload.decision)
            # v16: FAMILIES audition too -- a place to look that never runs is
            # as silent a prior as a skill or transform that never runs. One
            # linear_assoc identity lesson per unseen family (its own pool).
            for rs.fam in FAMILIES:
                if any(l.family == rs.fam for l in rs.library.lessons):
                    continue
                rs.k_f = min(rs.cfg.AUDITION_K, len(rs.cols))
                rs.spec_f = ViewportSpec(name=f"{rs.fam}{rs.k_f}_identity", family=rs.fam, k=rs.k_f,
                                      transform="identity", proj_dim=16)
                if not rs.library.can_run("linear_assoc", rs.spec_f):
                    continue
                rs.status, rs._, rs.r_spec, rs.payload = rs.attempt_lesson("audition", "audition",
                                                            "linear_assoc", rs.spec_f, rs.library.oofs())
                rs.key_f = f"linear_assoc|{rs.spec_f.name}"
                if rs.status == "culled":
                    rs.library.note_draft_cull(rs.key_f)
                    log("audition_culled", family=rs.fam,
                        draft_width=round(rs.payload["draft_width"], 4))
                    continue
                rs.library.add(rs.payload)
                rs.spec_lookup[rs.payload.key] = rs.r_spec
                rs.auditioned += 1
                rs.journal_rows.append({"explorer": "audition", "lesson_idx": rs.auditioned,
                                     "stage": "audition", "key": rs.payload.key, "family": rs.fam,
                                     "transform": "identity", "k": rs.k_f,
                                     "oof_corr": rs.payload.oof_corr, "width": rs.payload.width,
                                     "wf_corr": rs.payload.wf_corr, "wf_width": rs.payload.wf_width,
                                     "stability": rs.payload.stability, "seed_var": rs.payload.seed_var,
                                     "overfit_ratio": rs.payload.overfit_ratio,
                                     "decision": rs.payload.decision, "reason": rs.payload.reason,
                                     "budget_left": 0})
                log("audition", family=rs.fam, key=rs.payload.key,
                    oof_corr=round(rs.payload.oof_corr, 4), width=round(rs.payload.width, 4),
                    decision=rs.payload.decision)
            log("audition_parade_done", auditioned=rs.auditioned,
                note="every registered skill, transform AND family measured once (v9/v11/v13/v16 fix)")

        def run_explorer(ex: Explorer) -> None:
            stage_gains: list[float] = []
            lesson_idx = 0
            while ex.budget > 0 and META.allow("explore"):
                pick = ex.ucb_pick(rs.library, len(rs.cols))
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
                if rs.hetero:
                    other = "cpu" if lesson_lane(skill) == "gpu" else "gpu"
                    pick2 = ex.ucb_pick(rs.library, len(rs.cols), lane=other,
                                        exclude_key=f"{skill}|{spec.name}")
                    if pick2 is not None:
                        picks.append(pick2)

                oofs_snap = rs.library.oofs()
                if len(picks) == 2:
                    log("hetero_pair", explorer=ex.name,
                        gpu_lane=next(f"{s}|{sp.name}" for s, sp in picks if lesson_lane(s) == "gpu"),
                        cpu_lane=next(f"{s}|{sp.name}" for s, sp in picks if lesson_lane(s) == "cpu"))
                    with ThreadPoolExecutor(max_workers=2) as exe:
                        futs = [exe.submit(rs.attempt_lesson, ex.name, stage_name, s, sp, oofs_snap)
                                for s, sp in picks]
                        results = [f.result() for f in futs]
                else:
                    results = [rs.attempt_lesson(ex.name, stage_name, skill, spec, oofs_snap)]

                for status, r_skill, r_spec, payload in results:
                    key = f"{r_skill}|{r_spec.name}"
                    if status == "culled":
                        stalled = bool(payload.get("car_stalled"))
                        rs.library.note_draft_cull(key)
                        ex.budget -= 2 if stalled else 1
                        rs.journal_rows.append({"explorer": ex.name, "lesson_idx": lesson_idx,
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
                    rs.library.add(lesson)
                    rs.spec_lookup[lesson.key] = r_spec
                    ex.budget -= lesson.cost
                    stage_gains.append(lesson.oof_corr)
                    lesson_idx += 1
                    log("lesson", explorer=ex.name, stage=stage_name, key=key,
                        lane=lesson_lane(r_skill),
                        oof_corr=round(lesson.oof_corr, 4), wf=round(lesson.wf_corr, 4),
                        width=round(lesson.width, 4), wf_width=round(lesson.wf_width, 4),
                        overfit=round(lesson.overfit_ratio, 2),
                        decision=lesson.decision, budget=ex.budget)
                    rs.journal_rows.append({"explorer": ex.name, "lesson_idx": lesson_idx, "stage": stage_name,
                                         "key": key, "family": r_spec.family, "transform": r_spec.transform,
                                         "k": r_spec.k, "oof_corr": lesson.oof_corr, "width": lesson.width,
                                         "wf_corr": lesson.wf_corr, "wf_width": lesson.wf_width,
                                         "stability": lesson.stability, "seed_var": lesson.seed_var,
                                         "overfit_ratio": lesson.overfit_ratio, "decision": lesson.decision,
                                         "reason": lesson.reason, "budget_left": ex.budget})
                    rs.growth_rows.append({"explorer": ex.name, "lesson_idx": lesson_idx, "stage": stage_name,
                                        "best_corr_so_far": max(l.oof_corr for l in rs.library.lessons
                                                                if l.explorer == ex.name)})
                if ex.maybe_graduate(stage_gains):
                    stage_gains = []
        rs.run_explorer = run_explorer

        # ---- SEASONS (v12): the roster is reborn while the larder has food ----
        # Newborns of season N+1 are recruited by season N's waggle dances,
        # walk on its mycelium, inherit its quorum switches -- and the dedup
        # caps force each new season into UNWALKED cells of the menu. v11
        # measured that mycelium only wins AFTER deposits thicken the network;
        # seasons are how deposits compound inside one run.
        META.begin("explore", rs.cfg.MET_EXPLORE_SHARE)
        rs.explorers: list[Explorer] = []
        rs.season = 0
        while True:
            rs.season += 1
            rs.n_before = len(rs.library.lessons)
            for rs.ei in range(rs.cfg.N_EXPLORERS):
                if META.enabled and not META.allow("explore"):
                    break
                rs.ex = Explorer(EXPLORER_TRAITS[rs.ei % len(EXPLORER_TRAITS)], rs.cfg)
                rs.ex.season = rs.season
                rs.explorers.append(rs.ex)
                log("explorer_born", season=rs.season, explorer=rs.ex.name, species=rs.ex.species,
                    metaheuristic=rs.ex.traits["metaheuristic"], behavior=rs.ex.behavior,
                    stage=STAGES[rs.ex.stage_idx][0])
                rs.run_explorer(rs.ex)
            META.seasons = rs.season
            if not META.wants_more("explore"):
                break
            if rs.season >= rs.cfg.MAX_SEASONS:
                log("seasons_capped", at=rs.season)
                break
            if len(rs.library.lessons) == rs.n_before:
                log("seasons_exhausted", season=rs.season, note="the menu is fully walked")
                break
            write_csv(pd.DataFrame(rs.journal_rows), "explorer_journal.csv")  # crash-safe partial
            META.heartbeat(f"season_{rs.season}_done")
            log("season_turns", next_season=rs.season + 1, lessons_so_far=len(rs.library.lessons),
                promoted_so_far=len(rs.library.promoted()), elapsed_min=round(META.now(), 1))

        # ---- ATTENTION MARKET (v10): Charnov's marginal value theorem --------
        # The held-back pool flows to the explorer with the highest recent
        # marginal yield -- attention follows whoever is still learning.
        rs.attention_grantee = None
        if rs.cfg.ATTENTION_POOL > 0 and rs.library.lessons and META.allow("explore"):
            def marginal_yield(name: str) -> float:
                last = [l for l in rs.library.lessons if l.explorer == name][-3:]
                return (float(np.mean([lesson_fitness(l) / max(l.cost, 1) for l in last]))
                        if last else -1e9)
            rs.marginal_yield = marginal_yield
            rs.ranked_ex = sorted(rs.explorers, key=lambda e: rs.marginal_yield(e.name), reverse=True)
            rs.winner = rs.ranked_ex[0]
            rs.winner.budget = rs.cfg.ATTENTION_POOL
            rs.attention_grantee = rs.winner.name
            log("attention_grant", explorer=rs.winner.name, pool=rs.cfg.ATTENTION_POOL,
                marginal_yield=round(rs.marginal_yield(rs.winner.name), 5),
                note="MVT: the pool goes to whoever is still learning fastest")
            rs.run_explorer(rs.winner)

        # ---- LEARNING-RATE CURVE (v10): information intake, measured ---------
        rs.lr_rows, rs.cum_cost, rs.best_w = [], 0, 0.0
        for rs.i, rs.l in enumerate(rs.library.lessons):
            rs.cum_cost += max(rs.l.cost, 1)
            rs.best_w = max(rs.best_w, rs.l.width if np.isfinite(rs.l.width) else 0.0)
            rs.lr_rows.append({"lesson_n": rs.i + 1, "explorer": rs.l.explorer, "cum_cost": rs.cum_cost,
                            "width": rs.l.width, "best_width_so_far": rs.best_w,
                            "best_width_per_cost": rs.best_w / max(rs.cum_cost, 1)})
        write_csv(pd.DataFrame(rs.lr_rows), "learning_rate_curve.csv")

        # ---- v11 menagerie reports ------------------------------------------
        rs.quorum_rows = [{"family": fam, "distinct_voters": len(voters),
                        "reached_quorum": len(voters) >= rs.cfg.QUORUM_MIN,
                        "voters": "|".join(sorted(voters))}
                       for fam, voters in sorted(QUORUM.items(), key=lambda kv: -len(kv[1]))]
        if rs.quorum_rows:
            write_csv(pd.DataFrame(rs.quorum_rows), "quorum_report.csv")
        rs.n_quorum = sum(1 for r in rs.quorum_rows if r["reached_quorum"])
        log("quorum_sensing", families_voted=len(rs.quorum_rows), reached_quorum=rs.n_quorum,
            dances=len(DANCES), gene_pool=len(GENE_POOL))
        rs.sp_rows = []
        rs.seen_names: set[str] = set()
        for rs.ex in reversed(rs.explorers):           # last instance per name = final state
            if rs.ex.name in rs.seen_names:
                continue
            rs.seen_names.add(rs.ex.name)
            rs.mine = [l for l in rs.library.lessons if l.explorer == rs.ex.name]
            rs.prom = [l for l in rs.mine if l.decision == "promote"]
            rs.sp_rows.append({"explorer": rs.ex.name, "species": rs.ex.species, "behavior": rs.ex.behavior,
                            "seasons_lived": sum(1 for e in rs.explorers if e.name == rs.ex.name),
                            "birth_stage": STAGES[rs.ex.birth_stage][0], "max_stage": STAGES[rs.ex.max_stage][0],
                            "final_stage": STAGES[rs.ex.stage_idx][0], "lessons": len(rs.mine),
                            "promoted": len(rs.prom),
                            "best_oof": round(max((l.oof_corr for l in rs.mine), default=0.0), 4)})
        rs.sp_rows.reverse()
        write_csv(pd.DataFrame(rs.sp_rows), "species_report.csv")

        rs.circadian(rs.phase_frac("explore", 0.45), "post_phase1")
        gc.collect()
        mem_status("post_phase1")


    def _raid1(self, rs: "_RunState") -> None:
        # ---- RAID 1 (v10): the predator strikes early so venom shapes search --
        rs.raid1 = PredatorEngine(rs.cfg, rs.library, rs.spec_lookup, budget=rs.cfg.PREDATOR_BUDGET // 3)
        rs.pred_df1 = rs.raid1.run(rs.Xp, rs.yp, rs.segp, rs.cols, rs.embargo_p)
        rs.attacked = set(rs.pred_df1["key"]) if not rs.pred_df1.empty else set()
        log("raid1_done", attacked=len(rs.attacked), taboo_motifs=len(TABOO))


    def _phase2_evolve(self, rs: "_RunState") -> None:
        # ---- PHASE 2: metaheuristic evolution (v12: EPOCHS while fed) ---------
        # v4, v8, v9 AND v11 all ended evolution still climbing at budget
        # exhaustion (v11: g_best in the final generation, budget -1). The
        # metabolism re-fuels evolution in epochs: each epoch re-seeds its
        # population from the full library (seasons' champions included),
        # resets the annealing temperature, and spends a fresh budget.
        META.begin("evolve", rs.cfg.MET_EVOLVE_SHARE)
        rs.evo = EvolutionEngine(rs.cfg, rs.library, rs.spec_lookup, rs.gate)
        while True:
            rs.n_hist = len(rs.evo.history)
            rs.evo.run(rs.Xp, rs.yp, rs.segp, rs.cols, rs.embargo_p, rs.journal_rows)
            META.epochs = rs.evo.epoch + 1
            if not META.wants_more("evolve"):
                break
            if rs.evo.epoch + 1 >= rs.cfg.MAX_EPOCHS:
                log("epochs_capped", at=rs.evo.epoch + 1)
                break
            if len(rs.evo.history) == rs.n_hist:
                log("epochs_exhausted", epoch=rs.evo.epoch + 1,
                    note="an epoch produced nothing new; the lineage has converged")
                break
            rs.evo.epoch += 1
            rs.evo.budget = rs.cfg.EVOLUTION_BUDGET
            write_csv(pd.DataFrame(rs.journal_rows), "explorer_journal.csv")  # crash-safe partial
            META.heartbeat(f"epoch_{rs.evo.epoch}_refuel")
            log("evolution_epoch_refuel", epoch=rs.evo.epoch, budget=rs.evo.budget,
                elapsed_min=round(META.now(), 1))
        if rs.evo.history:
            write_csv(pd.DataFrame(rs.evo.history), "evolution_history.csv")
            write_csv(rs.evo.operator_report(), "evolution_operator_report.csv")
        write_json(rs.gate.report(), "draft_gate_report.json")
        log("draft_gate", **rs.gate.report())
        rs.circadian(rs.phase_frac("evolve", 0.70), "post_evolution")
        gc.collect()
        mem_status("post_evolution")


    def _raid2(self, rs: "_RunState") -> None:
        # ---- RAID 2: the predator attacks evolution's promotions --------------
        rs.predator = PredatorEngine(rs.cfg, rs.library, rs.spec_lookup,
                                  budget=rs.cfg.PREDATOR_BUDGET - rs.cfg.PREDATOR_BUDGET // 3,
                                  exclude=rs.attacked)
        rs.pred_df2 = rs.predator.run(rs.Xp, rs.yp, rs.segp, rs.cols, rs.embargo_p)
        rs.pred_df = pd.concat([rs.pred_df1, rs.pred_df2], ignore_index=True) \
            if not rs.pred_df1.empty or not rs.pred_df2.empty else pd.DataFrame()
        if not rs.pred_df.empty:
            write_csv(rs.pred_df, "predator_report.csv")


    def _ablation_dive(self, rs: "_RunState") -> None:
        # ---- CHAMPION ABLATION (v10): vary one thing at a time ----------------
        rs.ablation_rows = []
        rs.champs = sorted(rs.library.promoted(), key=lesson_fitness, reverse=True)
        if rs.champs:
            rs.champ = rs.champs[0]
            rs.c_spec = rs.spec_lookup[rs.champ.key]
            rs.variants = []
            if rs.c_spec.transform != "identity" and not SKILL_REGISTRY[rs.champ.skill]["needs_identity"]:
                rs.variants.append(("transform->identity", Genome(rs.champ.skill, rs.c_spec.family, "identity", rs.c_spec.k)))
            if rs.c_spec.family != "top":
                rs.variants.append(("family->top", Genome(rs.champ.skill, "top", rs.c_spec.transform, rs.c_spec.k)))
            if rs.c_spec.k > rs.cfg.K_MIN * 2:
                rs.variants.append(("k->half", Genome(rs.champ.skill, rs.c_spec.family, rs.c_spec.transform, rs.c_spec.k // 2)))
            rs.spent = 0
            for rs.label, rs.g in rs.variants:
                rs.g = rs.g.repaired()
                rs.cost_g = SKILL_REGISTRY[rs.g.skill]["cost"]
                if rs.spent + rs.cost_g > rs.cfg.ABLATION_BUDGET:
                    break
                if rs.library.can_run(rs.g.skill, rs.g.spec()):
                    rs.les = run_lesson("ablation", "ablation", rs.g.skill, rs.g.spec(), rs.Xp, rs.yp, rs.segp,
                                     rs.cols, rs.cfg, rs.embargo_p,
                                     stable_seed(rs.cfg.SEED, "ablation", rs.g.key), rs.library.oofs())
                    rs.library.add(rs.les)
                    rs.spec_lookup[rs.les.key] = rs.g.spec()
                    rs.spent += rs.cost_g
                    rs.a_oof, rs.a_w = rs.les.oof_corr, rs.les.width
                else:
                    rs.a_oof, rs.a_w = rs.library.mean_gain(rs.g.key), float("nan")
                rs.ablation_rows.append({"variant": rs.label, "key": rs.g.key,
                                      "oof_corr": rs.a_oof, "width": rs.a_w,
                                      "champion_key": rs.champ.key,
                                      "delta_oof_vs_champion": rs.a_oof - rs.champ.oof_corr})
            if rs.ablation_rows:
                write_csv(pd.DataFrame(rs.ablation_rows), "champion_ablation.csv")
                log("champion_ablation", champion=rs.champ.key, variants=len(rs.ablation_rows),
                    note="the edge, attributed to its parts")

        # ---- DIVE PHASE (v10): submarines under the visible surface -----------
        # The provisional champion's fold-honest OOF defines the surface; dive
        # lessons hunt the residual y - slope*z(champion_oof). Same doors,
        # scored in their own residual world; the stacker judges the blend.
        rs.dive_added = 0
        if rs.champs and rs.cfg.DIVE_BUDGET > 0:
            rs.champ = rs.champs[0]
            rs.oz = (rs.champ.oof - rs.champ.oof.mean()) / (rs.champ.oof.std() + 1e-9)
            rs.slope = float(np.mean(rs.oz * (rs.yp - rs.yp.mean())))
            rs.y_dive = (rs.yp - rs.slope * rs.oz).astype(np.float32)
            rs.DIVE_GRID = [("linear_assoc", "decor", "identity"),
                         ("codebook", "terrain", "quantize4"),
                         ("bagged_linear", "shadow", "quantize8"),
                         ("scout_lattice", "anon", "fold_abs"),
                         ("relay_caravan", "weather", "identity"),
                         ("majority_vote", "mycelium", "sign_only"),
                         ("steepness_gate", "top", "doppler"),
                         ("swell_rider", "both_clocks", "rank")]
            META.begin("dive", rs.cfg.MET_DIVE_SHARE)
            rs.dive_spent = 0
            # v12: a fed metabolism sends the submarines on a SECOND descent
            # with wider periscopes (k 32 -> 64) -- new keys, same honest doors
            rs.dive_rounds = 2 if META.wants_more("dive") else 1
            for rs.rnd in range(rs.dive_rounds):
                rs.k_d = min(32 * (rs.rnd + 1), len(rs.cols))
                rs.allowance = rs.cfg.DIVE_BUDGET * (rs.rnd + 1)
                for rs.sk, rs.fam, rs.tf in rs.DIVE_GRID:
                    if rs.rnd > 0 and not META.allow("dive"):
                        break
                    rs.cost_d = SKILL_REGISTRY[rs.sk]["cost"]
                    if rs.dive_spent + rs.cost_d > rs.allowance:
                        continue
                    rs.spec_d = ViewportSpec(name=f"{rs.fam}{rs.k_d}_{rs.tf}", family=rs.fam, k=rs.k_d,
                                          transform=rs.tf, proj_dim=16)
                    rs.les = run_lesson("submarine", "dive", rs.sk, rs.spec_d, rs.Xp, rs.y_dive, rs.segp,
                                     rs.cols, rs.cfg, rs.embargo_p,
                                     stable_seed(rs.cfg.SEED, "dive", rs.sk, rs.fam, rs.tf, rs.rnd), {})
                    rs.library.add(rs.les)
                    rs.spec_lookup[rs.les.key] = rs.spec_d
                    rs.dive_spent += rs.cost_d
                    rs.dive_added += 1
                    log("dive_lesson", descent=rs.rnd + 1, skill=rs.sk, family=rs.fam, transform=rs.tf,
                        resid_oof=round(rs.les.oof_corr, 4), width=round(rs.les.width, 4),
                        decision=rs.les.decision, spent=rs.dive_spent)
            log("dive_phase_done", dives=rs.dive_added, surface=rs.champ.key,
                surface_slope=round(rs.slope, 5))
        rs.circadian(rs.phase_frac("dive", 0.80), "post_dive")


    def _trail_reports(self, rs: "_RunState") -> None:
        # ---- TOPOGRAPHY: texture every trail, cluster into families -----------
        rs.tex_df, rs.tex_fam = texture_layer(rs.library.lessons, rs.yp, rs.segp, rs.terr_p, rs.wth_p, rs.cfg)
        if not rs.tex_df.empty:
            write_csv(rs.tex_df, "path_texture_report.csv")
            log("path_textures", trails=len(rs.tex_df),
                families=int(rs.tex_df["trail_family"].nunique()),
                features="|".join(TEXTURE_FEATURES))
        rs.tt_df = terrain_trail_report(rs.library.lessons, rs.yp, rs.terr_p, rs.cfg.TERRAIN_MIN_ROWS)
        if not rs.tt_df.empty:
            write_csv(rs.tt_df, "terrain_trail_report.csv")
        rs.tg_df = texture_generalization_report(rs.tex_df)
        if not rs.tg_df.empty:
            write_csv(rs.tg_df, "texture_generalization.csv")

        # ---- v30.1 WINNER NETWORK (IDEAS.md 1a): the promoted trails as a graph
        # -- observation only; communities feed the queued 1b selection cap.
        try:
            if rs.cfg.NETWORK_REPORT:
                rs.net_df, rs.net_summary = winner_network_report(rs.library.promoted(), rs.cfg)
                if not rs.net_df.empty:
                    write_csv(rs.net_df, "winner_network.csv")
                log("winner_network", **rs.net_summary,
                    note="prediction-space graph of promoted trails; one community = one bet")
        except Exception as e:
            log("winner_network_skipped", err=str(e)[:80])

        # ---- RED PHEROMONE (v14): the repellent channel, reported ------------
        if RED_MYCELIUM:
            rs.top_red = sorted(RED_MYCELIUM.items(), key=lambda kv: -kv[1])[:40]
            write_csv(pd.DataFrame([{"col_idx": int(ci),
                                     "feature": rs.cols[int(ci)] if int(ci) < len(rs.cols) else str(ci),
                                     "repellent": round(float(v), 3),
                                     "is_trap": int(ci) in TRAPS}
                                    for ci, v in rs.top_red]), "red_pheromone_report.csv")
            log("red_pheromone", poisoned_columns=len(RED_MYCELIUM),
                from_traps=len(TRAPS), note="repellent subtracted from corr-driven rankings")

        # ---- DREAM REPLAY (v9): bootstrap every promoted trail, free ----------
        rs.rng_dream = np.random.default_rng(stable_seed(rs.cfg.SEED, "dreams"))
        rs.dream_omens = 0
        for rs.l in rs.library.promoted():
            rs.l.dream_p05, rs.l.dream_p50 = dream_replay(rs.l.oof, rs.yp, rs.segp,
                                                    rs.cfg.DREAM_REPLAYS, rs.rng_dream)
            if rs.l.dream_p05 < 0:
                rs.dream_omens += 1
        log("dream_replay", trails=len(rs.library.promoted()), replays=rs.cfg.DREAM_REPLAYS,
            omens=rs.dream_omens, note="dream_p05<0 = the trail fails in some dreamed world")

        # Xp's last consumer was the predator; the probe matrix goes back to
        # the OS before the report/ensemble half of the run (memory doctrine).
        # attempt_lesson's closure also pins Xp -- both must go.
        del rs.Xp, rs.attempt_lesson
        gc.collect()
        mem_status("post_predator")

        # ---- study reports -----------------------------------------------------
        rs.ledger = []
        for rs.l in rs.library.lessons:
            rs.row = asdict(rs.l)
            rs.row.pop("oof")
            rs.row.pop("used_cols", None)   # mycelium substrate, too wide for the ledger
            rs.row["fold_corrs"] = "|".join(f"{c:.5f}" for c in rs.l.fold_corrs)
            rs.row["wf_fold_corrs"] = "|".join(f"{c:.5f}" for c in rs.l.wf_fold_corrs)
            rs.row["fitness"] = lesson_fitness(rs.l)
            rs.row["bits_per_feature"] = TRANSFORM_BITS.get(rs.l.transform, 32)
            rs.row["total_bits"] = TRANSFORM_BITS.get(rs.l.transform, 32) * max(rs.l.k, 1)
            rs.ledger.append(rs.row)
        rs.ledger_df = pd.DataFrame(rs.ledger)
        write_csv(rs.ledger_df, "explorer_lessons.csv")
        write_csv(pd.DataFrame(rs.journal_rows), "explorer_journal.csv")
        write_csv(pd.DataFrame(rs.growth_rows), "explorer_growth_curve.csv")
        write_csv(pd.DataFrame([{"key": k, "tries": rs.library.runs[k], "mean_corr": rs.library.mean_gain(k)}
                                for k in rs.library.runs]).sort_values("mean_corr", ascending=False),
                  "shared_library.csv")
        if not rs.ledger_df.empty:
            for rs.dim in ("skill", "family", "transform", "stage", "explorer"):
                rs.rep = (rs.ledger_df.groupby(rs.dim, as_index=False)
                       .agg(lessons=("key", "count"), mean_corr=("oof_corr", "mean"),
                            best_corr=("oof_corr", "max"), mean_width=("width", "mean"),
                            mean_wf_corr=("wf_corr", "mean"), mean_era_corr=("era_corr", "mean"),
                            promote_rate=("decision", lambda s: float((s == "promote").mean())))
                       .sort_values("best_corr", ascending=False))
                write_csv(rs.rep, f"study_{rs.dim}_report.csv")


    def _select_members(self, rs: "_RunState") -> None:
        # ---- members: regime + uniqueness filters, z-scored --------------------
        rs.promoted = sorted(rs.library.promoted(), key=lambda l: -l.oof_corr)
        if not rs.promoted:
            rs.promoted = sorted(rs.library.lessons, key=lambda l: -l.oof_corr)[:1]
            log("WARNING_no_promotions", note="falling back to best lesson regardless of gates")
        rs.segs_u = np.unique(rs.segp)
        rs.members: dict[str, np.ndarray] = {}
        rs.member_lessons: dict[str, Lesson] = {}
        rs.name_counts: dict[str, int] = {}
        rs.fam_counts: dict[int, int] = {}        # trail-texture family counts (v8)
        rs.vfam_counts: dict[str, int] = {}       # v23: viewport-FAMILY counts (the monoculture cap)
        rs.dive_members = 0

        def _try_admit(l: "Lesson", enforce_vfam: bool) -> bool:
            """Admit a promoted trail to the blend if it clears every diversity
            door. v23 adds a hard viewport-family cap; the two-pass caller below
            relaxes ONLY that cap to reach the member floor."""
            if len(rs.members) >= rs.cfg.MAX_MEMBERS:
                return False
            sd = float(np.std(l.oof))
            if sd <= 1e-9:
                return False
            is_dive = l.stage == "dive"
            if is_dive and rs.dive_members >= rs.cfg.DIVE_MEMBER_CAP:
                return False
            if not is_dive:
                # dive lessons predict the RESIDUAL: near-zero corr vs y is
                # their job description, so only surface trails face this gate
                per_seg = [pearson(rs.yp[rs.segp == s], l.oof[rs.segp == s]) for s in rs.segs_u]
                if float(np.mean([c <= 0 for c in per_seg])) > rs.cfg.MAX_SEG_NEG_FRAC:
                    return False
            if rs.members and max(abs(pearson(l.oof, o)) for o in rs.members.values()) > rs.cfg.MEMBER_CORR_CAP:
                return False
            # v32 REDUNDANCY FLOOR (IDEAS_ZOO v53 sec8): the candidate must carry
            # information the admitted members do not already SPAN -- stricter
            # than pairwise corr caps (a blend of two members can replicate a
            # third that correlates with neither). 0.0 = exact no-op.
            if rs.members and rs.cfg.REDUNDANCY_MIN_NEW_INFO > 0.0:
                Mz = np.vstack([rs.members[nm0] for nm0 in rs.members]).astype(np.float64)
                cz = (l.oof - float(np.mean(l.oof))) / (float(np.std(l.oof)) + 1e-12)
                beta_r, *_ = np.linalg.lstsq(Mz.T, cz, rcond=None)
                new_info = max(0.0, float(np.var(cz - Mz.T @ beta_r)))
                if new_info < rs.cfg.REDUNDANCY_MIN_NEW_INFO:
                    log("member_skipped_redundancy", key=l.key, new_info=round(new_info, 4),
                        floor=rs.cfg.REDUNDANCY_MIN_NEW_INFO)
                    return False
            # v14 jamming avoidance (CONSERVATIVE): fires only for a confirmed
            # near-duplicate -- high input overlap AND high output agreement.
            lcols = set(l.used_cols)
            if lcols and rs.members and not is_dive:
                for nm in rs.members:
                    oc = abs(pearson(l.oof, rs.member_lessons[nm].oof))
                    if oc < 0.80:
                        continue
                    mcols = set(rs.member_lessons[nm].used_cols)
                    jac = len(lcols & mcols) / max(1, len(lcols | mcols))
                    if jac >= rs.cfg.JAMMING_JACCARD:
                        log("member_skipped_jamming", key=l.key,
                            twin=nm, input_jaccard=round(jac, 3), output_corr=round(oc, 3))
                        return False
            # v8 texture diversity: cap members sharing one trail-texture family
            fam_t = rs.tex_fam.get(l.key, -1)
            if fam_t >= 0 and rs.fam_counts.get(fam_t, 0) >= rs.cfg.TEXTURE_FAMILY_CAP:
                log("member_skipped_texture_family", key=l.key, trail_family=fam_t,
                    cap=rs.cfg.TEXTURE_FAMILY_CAP)
                return False
            # v23 VIEWPORT-FAMILY CAP -- the biting input-space diversity gate
            # the v12/v19 monocultures needed. The texture cap above is fooled
            # (12 mycelium viewports register as 10 "trail families"); this
            # counts the actual viewport family (mycelium/tail/weather/...). A
            # blend may carry at most MAX_FAMILY_MEMBERS of any one, so it can
            # NEVER be the single-family bet that gamed sealed and collapsed
            # out-of-period in v12 (8/8 mycelium) and v19 (100% mycelium, 0.0749).
            if enforce_vfam and rs.vfam_counts.get(l.family, 0) >= rs.cfg.MAX_FAMILY_MEMBERS:
                log("member_skipped_viewport_family", key=l.key, family=l.family,
                    cap=rs.cfg.MAX_FAMILY_MEMBERS)
                return False
            base = l.key
            rs.name_counts[base] = rs.name_counts.get(base, 0) + 1
            name = base if rs.name_counts[base] == 1 else f"{base}#r{rs.name_counts[base]}"
            mu0 = float(np.mean(l.oof))
            rs.members[name] = ((l.oof - mu0) / sd).astype(np.float32)
            rs.member_lessons[name] = l
            if is_dive:
                rs.dive_members += 1
            if fam_t >= 0:
                rs.fam_counts[fam_t] = rs.fam_counts.get(fam_t, 0) + 1
            rs.vfam_counts[l.family] = rs.vfam_counts.get(l.family, 0) + 1
            return True
        rs._try_admit = _try_admit

        # pass 1: diversity-first, viewport-family cap enforced
        for rs.l in rs.promoted:
            if len(rs.members) >= rs.cfg.MAX_MEMBERS:
                break
            rs._try_admit(rs.l, enforce_vfam=True)
        # pass 2: backfill ONLY if the cap left the blend under the floor (one
        # family genuinely dominates the promoted pool). Relaxes the viewport-
        # family cap alone; every other door still holds.
        if len(rs.members) < rs.cfg.MIN_BLEND_MEMBERS:
            rs.admitted = {id(rs.member_lessons[nm]) for nm in rs.members}
            for rs.l in rs.promoted:
                if len(rs.members) >= rs.cfg.MIN_BLEND_MEMBERS:
                    break
                if id(rs.l) in rs.admitted:
                    continue
                if rs._try_admit(rs.l, enforce_vfam=False):
                    rs.admitted.add(id(rs.l))
            log("blend_backfilled", to=len(rs.members), floor=rs.cfg.MIN_BLEND_MEMBERS,
                note="viewport-family cap relaxed to reach the member floor")
        # v27 COMPLEXITY-ANCHOR ADMISSION: reserve a few LOW-complexity, HIGH-
        # walk-forward members into the pool so the runtime governor has a
        # genuinely simple alternative to ship IF measurement says capacity decays
        # here. The 11th-place simple-linear recipes (linear_ols/greedy_ols on the
        # tail block, tiny-k quantize4) promote every run but never made the oof-
        # ranked top-MAX_MEMBERS pool -- this makes them REACHABLE without forcing
        # them; lambda (the measured decay~complexity penalty) decides if they win.
        if rs.cfg.COMPLEXITY_GOVERNOR:
            rs.present = {id(rs.member_lessons[nm]) for nm in rs.members}
            rs.anchor_pool = sorted(
                (l for l in rs.promoted
                 if id(l) not in rs.present and l.oof_corr > 0 and float(np.std(l.oof)) > 1e-9
                 and member_complexity(l, rs.cfg) <= rs.cfg.GOV_SIMPLE_C),
                key=lambda l: -(l.wf_corr if np.isfinite(l.wf_corr) else l.oof_corr))
            rs.added_anchor = 0
            for rs.l in rs.anchor_pool:
                if rs.added_anchor >= rs.cfg.GOV_ANCHOR_MEMBERS:
                    break
                if rs.members and max(abs(pearson(rs.l.oof, o)) for o in rs.members.values()) > rs.cfg.MEMBER_CORR_CAP:
                    continue
                rs.base = rs.l.key
                rs.name_counts[rs.base] = rs.name_counts.get(rs.base, 0) + 1
                rs.name = rs.base if rs.name_counts[rs.base] == 1 else f"{rs.base}#r{rs.name_counts[rs.base]}"
                rs.mu0, rs.sd0 = float(np.mean(rs.l.oof)), float(np.std(rs.l.oof))
                rs.members[rs.name] = ((rs.l.oof - rs.mu0) / rs.sd0).astype(np.float32)
                rs.member_lessons[rs.name] = rs.l
                rs.vfam_counts[rs.l.family] = rs.vfam_counts.get(rs.l.family, 0) + 1
                rs.added_anchor += 1
            if rs.added_anchor:
                log("complexity_anchor_admitted", n=rs.added_anchor,
                    note="low-complexity high-wf members reserved so the governor can ship simple if it generalizes")
        rs.n_trail_families = len(set(rs.tex_fam.get(rs.member_lessons[nm].key, -1) for nm in rs.members))
        rs.n_view_families = len(set(rs.member_lessons[nm].family for nm in rs.members))
        log("ensemble_members", n=len(rs.members), names=list(rs.members),
            trail_families=rs.n_trail_families, viewport_families=rs.n_view_families,
            dive_members=rs.dive_members)


    def _ensemble(self, rs: "_RunState") -> None:
        rs.result = nested_ensemble(rs.members, rs.yp, rs.segp, rs.cfg, rs.embargo_p,
                                    wth=rs.wth_p, prs=getattr(rs, "prs_p", None))
        rs.moe_gauge = rs.result.get("moe_gauge")
        write_json(rs.result["honest"] | {"winner": rs.result["winner"]}, "ensemble_nested_assessment.json")
        log("ensemble_selected", winner=rs.result["winner"],
            honest=round(rs.result["honest"][rs.result["winner"]], 5),
            best_single=round(rs.result["honest"]["best_single"], 5),
            weather_conditional=rs.result["weather_states"] is not None)

        rs.w = rs.result["weights"]
        rs.blend_oof = apply_weights_rows(rs.members, rs.w, rs.result["is_median"],
                                       rs.result["weather_states"], rs.wth_p).astype(np.float64)

        # weather report: how the blend (and best single) fare in each sky (v9)
        rs.best_full = max(rs.members, key=lambda nm: pearson(rs.yp, rs.members[nm]))
        write_csv(pd.DataFrame(
            [{"state": int(s), "rows": int((rs.wth_p == s).sum()),
              "frac": float((rs.wth_p == s).mean()),
              "blend_corr": pearson(rs.yp[rs.wth_p == s], rs.blend_oof[rs.wth_p == s]),
              "best_single_corr": pearson(rs.yp[rs.wth_p == s], rs.members[rs.best_full][rs.wth_p == s])}
             for s in np.unique(rs.wth_p)]), "weather_report.csv")

        rs.wins = winsorize_audit(rs.blend_oof, rs.yp, rs.cfg.WINSOR_QS)
        write_json(rs.wins, "winsorize_audit.json")
        log("winsorize", apply=rs.wins["apply"], raw=round(rs.wins["raw_corr"], 5), best=round(rs.wins["best_corr"], 5))

        write_csv(dominance_report(rs.w, rs.members, rs.yp), "dominance_report.csv")
        write_csv(regime_stress(rs.members, rs.yp, rs.segp, rs.volp, rs.cfg.VOLUME_SLICE_Q), "member_regime_stress.csv")
        # v19 truth-serum reports: many-worlds survival + MAP-Elites biodiversity
        rs.mw_df = many_worlds_report(rs.members, rs.yp, rs.segp, rs.terr_p, rs.wth_p, rs.cfg)
        write_csv(rs.mw_df, "many_worlds_cv.csv")
        if not rs.mw_df.empty:
            log("many_worlds_cv", members=len(rs.mw_df),
                best_survival=round(float(rs.mw_df["world_survival_min"].max()), 4),
                worst_survival=round(float(rs.mw_df["world_survival_min"].min()), 4),
                note="per-member corr floor across time/terrain/weather worlds")
        rs.me_df = map_elites_archive(rs.library.lessons)
        write_csv(rs.me_df, "map_elites_archive.csv")
        log("map_elites", niches=len(rs.me_df),
            note="best lesson per (family x transform x k x overfit) cell -- biodiversity, not monoculture")
        write_csv(pd.DataFrame([{"member": nm, "fit_corr": rs.member_lessons[nm].fit_corr,
                                 "oof_corr": rs.member_lessons[nm].oof_corr,
                                 "wf_corr": rs.member_lessons[nm].wf_corr,
                                 "deflated_corr": rs.member_lessons[nm].deflated_corr,
                                 "overfit_ratio": rs.member_lessons[nm].overfit_ratio,
                                 "flag": "ALARM" if rs.member_lessons[nm].overfit_ratio > rs.cfg.MAX_OVERFIT_RATIO else "ok"}
                                for nm in rs.members]), "train_cv_gap.csv")

        # ---- v31 redundancy + factor crowding (IDEAS_ZOO B2, observation) -----
        try:
            if rs.cfg.REDUNDANCY_REPORT:
                rf_df = redundancy_factor_report(rs.members, getattr(rs, "factor_scores", None))
                if not rf_df.empty:
                    write_csv(rf_df, "redundancy_factor_report.csv")
                    log("redundancy_factor", members=len(rf_df),
                        min_new_info=round(float(rf_df["new_info"].min()), 4),
                        max_crowding=(round(float(rf_df["crowding_cos"].max()), 4)
                                      if "crowding_cos" in rf_df.columns else None))
        except Exception as e:
            log("redundancy_factor_skipped", err=str(e)[:80])

        # ---- v32 SEGMENT SENATE (IDEAS_ZOO C2, observation): per-segment votes
        try:
            if rs.cfg.SENATE_REPORT:
                sen_df = segment_senate_report(rs.members, rs.yp, rs.segp, rs.cfg)
                if not sen_df.empty:
                    write_csv(sen_df, "segment_senate.csv")
                    log("segment_senate", members=len(sen_df),
                        max_veto=int(sen_df["veto"].max()),
                        note="a great mean hiding many vetoed segments is the era-mean blind spot")
        except Exception as e:
            log("segment_senate_skipped", err=str(e)[:80])


    def _forward_holdout(self, rs: "_RunState") -> None:
        # ---- forward-drift check + forward gate (within WORKING region) --------
        rs.best_cv_name = max(rs.members, key=lambda nm: pearson(rs.yp, rs.members[nm]))
        rs.names_for_fwd = list(dict.fromkeys(list(rs.w) + [rs.best_cv_name]))
        rs.cut = int((1 - rs.cfg.FORWARD_FRACTION) * rs.n_work)
        rs.past = np.arange(0, max(2, rs.cut - rs.cfg.EMBARGO_ROWS))
        rs.future = np.arange(rs.cut, rs.n_work)
        rs.fwd_rows, rs.fwd_parts = [], {}
        for rs.nm in rs.names_for_fwd:
            rs.l = rs.member_lessons[rs.nm]
            rs.st = fit_skill(rs.l.skill, rs.spec_lookup[rs.l.key], rs.X_full[rs.past], rs.y_full[rs.past],
                           rs.seg_full[rs.past], rs.cols, np.random.default_rng(rs.l.seed), rs.cfg, rs.l.seed)
            rs.p = predict_skill(rs.st, rs.X_full[rs.future])
            rs.sd = float(np.std(rs.p)) + 1e-9
            rs.fwd_parts[rs.nm] = ((rs.p - float(np.mean(rs.p))) / rs.sd).astype(np.float64)
            rs.fwd_rows.append({"member": rs.nm, "cv_oof_corr": rs.l.oof_corr,
                             "forward_corr": pearson(rs.y_full[rs.future], rs.p),
                             "drift_gap": rs.l.oof_corr - pearson(rs.y_full[rs.future], rs.p)})
        rs.wth_future = moe_states(getattr(rs, "moe_gauge", None), rs.X_full[rs.future])
        rs.fwd_blend = apply_weights_rows(rs.fwd_parts, rs.w, rs.result["is_median"],
                                       rs.result["weather_states"], rs.wth_future)
        rs.forward_blend_corr = score_metric(rs.y_full[rs.future], rs.fwd_blend)
        rs.single_fwd_corr = score_metric(rs.y_full[rs.future], rs.fwd_parts[rs.best_cv_name])
        rs.fwd_rows.append({"member": "__BLEND__", "cv_oof_corr": rs.result["honest"][rs.result["winner"]],
                         "forward_corr": rs.forward_blend_corr,
                         "drift_gap": rs.result["honest"][rs.result["winner"]] - rs.forward_blend_corr})
        write_csv(pd.DataFrame(rs.fwd_rows), "forward_holdout_report.csv")
        log("forward_holdout", blend=round(rs.forward_blend_corr, 5),
            best_cv_single=rs.best_cv_name, single=round(rs.single_fwd_corr, 5))


    def _governor(self, rs: "_RunState") -> None:
        # ---- v27 RUNTIME COMPLEXITY-GENERALIZATION GOVERNOR -------------------
        # Measure whether THIS dataset rewards or punishes capacity -- from every
        # lesson's OWN out-of-period decay (oof_corr - wf_corr) regressed on its
        # complexity -- then set the shipping penalty lambda accordingly. The cure
        # for the measured complexity ratchet (v19/v24/v25 all converged on the
        # highest-capacity, highest-sealed, worst-private models) WITHOUT hard-
        # coding simplicity: a capacity-friendly dataset yields beta<=0, lambda 0.
        GOVERNOR.clear()
        if rs.cfg.COMPLEXITY_GOVERNOR:
            rs.gov_pts = [(member_complexity(l, rs.cfg), float(l.oof_corr - l.wf_corr))
                       for l in rs.library.lessons
                       if l.oof_corr > 0.02 and np.isfinite(l.wf_corr)]
            rs.gov_cmap = {nm: member_complexity(rs.member_lessons[nm], rs.cfg) for nm in rs.members}
            rs.gov_beta = rs.gov_lambda = 0.0
            if len(rs.gov_pts) >= rs.cfg.GOV_MIN_LESSONS:
                rs.Cv = np.array([p[0] for p in rs.gov_pts], np.float64)
                rs.Dv = np.array([p[1] for p in rs.gov_pts], np.float64)
                if float(rs.Cv.std()) > 1e-6:
                    rs.gov_beta = float(np.cov(rs.Cv, rs.Dv, bias=True)[0, 1] / (rs.Cv.var() + 1e-12))
                rs.gov_lambda = float(np.clip(rs.gov_beta * rs.cfg.GOV_LAMBDA_SCALE, 0.0, rs.cfg.GOV_LAMBDA_MAX))
                rs.edges = np.quantile(rs.Cv, [1.0 / 3, 2.0 / 3])
                rs.bucket = np.searchsorted(rs.edges, rs.Cv)
                rs.curve_rows = []
                for rs.b, rs.lab in enumerate(("low", "mid", "high")):
                    rs.m = rs.bucket == rs.b
                    if rs.m.any():
                        rs.curve_rows.append({"stratum": rs.lab, "n": int(rs.m.sum()),
                                           "mean_complexity": round(float(rs.Cv[rs.m].mean()), 4),
                                           "mean_decay_oof_minus_wf": round(float(rs.Dv[rs.m].mean()), 5)})
                write_csv(pd.DataFrame(rs.curve_rows), "complexity_generalization_curve.csv")
            # v27 self-improvement: blend the measured beta with the cross-run
            # ledger's accumulated beta (shrunk by evidence count) so lambda is
            # stable across runs and doesn't whipsaw on one noisy measurement.
            rs.gov_beta_meas = rs.gov_beta
            if rs.cfg.SELF_IMPROVE and (LEDGER_PRIOR.get("governor") or {}).get("count", 0):
                rs.pb = float(LEDGER_PRIOR["governor"].get("beta", 0.0))
                rs.pc = float(LEDGER_PRIOR["governor"].get("count", 0)) * rs.cfg.LEDGER_SHRINK * rs.cfg.GOV_MIN_LESSONS
                rs.nm = float(len(rs.gov_pts))
                if rs.nm + rs.pc > 0:
                    rs.gov_beta = (rs.nm * rs.gov_beta + rs.pc * rs.pb) / (rs.nm + rs.pc)
                rs.gov_lambda = float(np.clip(rs.gov_beta * rs.cfg.GOV_LAMBDA_SCALE, 0.0, rs.cfg.GOV_LAMBDA_MAX))
                log("governor_beta_blend", measured=round(rs.gov_beta_meas, 4), prior=round(rs.pb, 4),
                    blended=round(rs.gov_beta, 4), prior_runs=int(LEDGER_PRIOR["governor"].get("count", 0)))
            GOVERNOR.update({"lambda": rs.gov_lambda, "beta": rs.gov_beta, "complexity": rs.gov_cmap})
            # v30: MEASURE the wide-path hypothesis -- does a trail's WIDTH (robust
            # lower-bound strength) predict LOWER out-of-period decay on THIS
            # dataset? Negative corr = wide paths decay less = the initial bias is
            # justified; the ledger carries it so the next run can recalibrate
            # WIDTH_BIAS_START from accumulated evidence instead of a prior.
            rs.width_decay_corr = None
            wd = [(float(min(l.width, l.wf_width if np.isfinite(l.wf_width) else l.width)),
                   float(l.oof_corr - l.wf_corr))
                  for l in rs.library.lessons
                  if l.oof_corr > 0.02 and np.isfinite(l.wf_corr) and np.isfinite(l.width)]
            if len(wd) >= rs.cfg.GOV_MIN_LESSONS:
                rs.width_decay_corr = float(pearson(
                    np.array([a for a, _ in wd], np.float64),
                    np.array([b for _, b in wd], np.float64)))
            write_json({"beta_decay_vs_complexity": round(rs.gov_beta, 5),
                        "lambda_penalty": round(rs.gov_lambda, 5),
                        "lessons_measured": len(rs.gov_pts),
                        "width_decay_corr": (round(rs.width_decay_corr, 5)
                                             if rs.width_decay_corr is not None else None),
                        "width_share_now": round(width_share(), 4),
                        "member_complexity": {nm: round(c, 4) for nm, c in rs.gov_cmap.items()},
                        "note": "lambda * config-complexity is subtracted from each candidate's robust score; "
                                "beta>0 => this dataset punishes capacity (ship simpler); beta<=0 => capacity free; "
                                "width_decay_corr<0 => wide paths decay less (the v30 initial bias is justified)"},
                       "complexity_governor.json")
            log("complexity_governor", beta=round(rs.gov_beta, 4), lam=round(rs.gov_lambda, 4),
                lessons=len(rs.gov_pts),
                width_decay_corr=(round(rs.width_decay_corr, 4)
                                  if rs.width_decay_corr is not None else None),
                note="shipping-complexity penalty set by the measured decay~complexity slope (runtime-adaptive)")


    def _forensics(self, rs: "_RunState") -> None:
        # ---- v21 FORENSIC REGIME-SCIENCE: self-tuning, forward-validated -------
        # Writes the full forensic suite and MEASURES regime-aware shipping
        # configs on the forward slice. Overrides the incumbent weights ONLY if
        # a worst-world + input-diversity reselection strictly beats it out-of-
        # sample (the measured cure for the v12 monoculture); strict no-op else.
        rs.forensic_dec = forensic_regime_science(
            rs.members, rs.member_lessons, rs.spec_lookup, rs.w, rs.result["is_median"],
            rs.result["weather_states"], rs.forward_blend_corr, rs.fwd_parts,
            rs.yp, rs.segp, rs.terr_p, rs.wth_p, rs.volp, rs.X_full, rs.y_full, rs.seg_full, rs.n_work,
            rs.cols, rs.past, rs.future, rs.cfg)
        rs.fwd_parts = rs.forensic_dec["fwd_parts"]
        if rs.forensic_dec["override"]:
            rs.w = rs.forensic_dec["weights"]
            rs.result["is_median"] = rs.forensic_dec["is_median"]
            rs.result["weather_states"] = rs.forensic_dec["weather_states"]
            rs.forward_blend_corr = rs.forensic_dec["forward_blend_corr"]
            rs.fwd_blend = sum(rs.w[nm] * rs.fwd_parts[nm] for nm in rs.w if nm in rs.fwd_parts)
            rs.single_fwd_corr = score_metric(rs.y_full[rs.future], rs.fwd_parts[rs.best_cv_name]) \
                if rs.best_cv_name in rs.fwd_parts else rs.single_fwd_corr
            rs.names_for_fwd = list(dict.fromkeys(list(rs.w) + [rs.best_cv_name]))


    def _forward_gate(self, rs: "_RunState") -> None:
        # v18 FORWARD-GATE ERROR BARS: a point margin means a coin-flip can
        # pick the captain. Block-bootstrap the forward slice and require the
        # single to beat the blend SIGNIFICANTLY (in >= GATE_BOOT_CONF of
        # resamples), not just on a noisy point estimate -- a noisy gate can
        # undo the whole search. Default-safe: if bootstrap is degenerate the
        # old point rule stands.
        rs.gate_point = bool(rs.cfg.USE_FORWARD_GATE
                          and rs.single_fwd_corr > rs.forward_blend_corr + rs.cfg.FORWARD_GATE_MARGIN)
        rs.gate_fired = rs.gate_point
        if rs.gate_point and rs.best_cv_name in rs.fwd_parts:
            rs.yf = rs.y_full[rs.future]
            rs.seg_f = rs.seg_full[rs.future]
            rs.usegs = np.unique(rs.seg_f)
            if len(rs.usegs) >= 4:
                rs.rng_g = np.random.default_rng(stable_seed(rs.cfg.SEED, "gate_boot"))
                rs.seg_idx_f = [np.where(rs.seg_f == s)[0] for s in rs.usegs]
                rs.single_p = rs.fwd_parts[rs.best_cv_name]
                rs.wins_single = 0
                rs.B = 200
                for rs._ in range(rs.B):
                    rs.pick = rs.rng_g.integers(0, len(rs.usegs), len(rs.usegs))
                    rs.bidx = np.concatenate([rs.seg_idx_f[int(p)] for p in rs.pick])
                    rs.sc_s = pearson(rs.yf[rs.bidx], rs.single_p[rs.bidx])
                    rs.sc_b = pearson(rs.yf[rs.bidx], rs.fwd_blend[rs.bidx])
                    if rs.sc_s > rs.sc_b + rs.cfg.FORWARD_GATE_MARGIN:
                        rs.wins_single += 1
                rs.conf = rs.wins_single / rs.B
                rs.gate_fired = rs.conf >= rs.cfg.GATE_BOOT_CONF
                log("forward_gate_error_bars", point_fire=rs.gate_point, boot_conf=round(rs.conf, 3),
                    required=rs.cfg.GATE_BOOT_CONF, fired=rs.gate_fired,
                    note="gate requires bootstrap significance, not a point margin")
        if rs.gate_fired:
            rs.final_weights = {rs.best_cv_name: 1.0}
            rs.final_is_median = False
            rs.final_weather = None
            log("FORWARD_GATE_OVERRIDE", shipped=rs.best_cv_name,
                single_fwd=round(rs.single_fwd_corr, 5), blend_fwd=round(rs.forward_blend_corr, 5))
        else:
            rs.final_weights = rs.w
            rs.final_is_median = rs.result["is_median"]
            rs.final_weather = rs.result["weather_states"]


    def _shipping_court(self, rs: "_RunState") -> None:
        # ---- v27 ANTI-OVERFIT SHIPPING COURT ---------------------------------
        # Channels the best of the regime-criticality / overfit-gravity-well /
        # CV-reality-distortion / prediction-crowding ideas into ONE cheap, out-of-
        # sample-grounded SELECTION HARDENER (adds NO capacity -- it only raises the
        # bar where overfit risk is measured): members that fail their local "escape
        # velocity" (width vs decay + reality-distortion + complexity + crowding, with
        # a non-positive worst-world floor) are down-weighted, and high regime
        # CRITICALITY (residual autocorrelation) shrinks the blend toward equal-weight.
        # Conservative => near-no-op on a healthy blend; fully reported.
        if not rs.gate_fired and not rs.final_is_median and rs.final_weather is None:
            rs.final_weights = shipping_court(rs.final_weights, rs.members, rs.member_lessons,
                                           rs.yp, rs.segp, rs.terr_p, rs.wth_p, rs.cfg)


    def _shrink_chorus_shape(self, rs: "_RunState") -> None:
        # ---- v16 SHRUNK BLEND (no-op-safe anti-decay) ------------------------
        # The whole story of this dataset is the CV->forward gap (regime decay):
        # honest CV ~0.144 but forward/sealed ~0.108. CV-optimal weights are
        # brittle; EQUAL weighting is more robust (v9's equal_top won outright).
        # So shrink the shipped weights toward equal by a factor chosen on the
        # FORWARD slice -- this attacks the gap directly and DEFAULTS TO lambda=0
        # (no change) unless forward STRICTLY improves. It can only help or
        # no-op; it never trades away a CV-justified edge the forward slice
        # does not confirm. Only fires for a real multi-member global blend.
        rs.shrunk_lambda = 0.0
        if (not rs.gate_fired and not rs.final_is_median and rs.final_weather is None
                and len(rs.final_weights) >= 2
                and all(nm in rs.fwd_parts for nm in rs.final_weights)):
            rs.eq = 1.0 / len(rs.final_weights)
            rs.best_fc = rs.forward_blend_corr
            for rs.lam in (0.25, 0.5, 0.75, 1.0):
                rs.shr = {nm: (1.0 - rs.lam) * v + rs.lam * rs.eq for nm, v in rs.final_weights.items()}
                rs.fb = sum(rs.shr[nm] * rs.fwd_parts[nm] for nm in rs.shr)
                rs.fc = pearson(rs.y_full[rs.future], rs.fb)
                if rs.fc > rs.best_fc + 1e-6:
                    rs.best_fc, rs.shrunk_lambda = rs.fc, rs.lam
            if rs.shrunk_lambda > 0:
                rs.final_weights = {nm: (1.0 - rs.shrunk_lambda) * v + rs.shrunk_lambda * rs.eq
                                 for nm, v in rs.final_weights.items()}
                log("shrunk_blend", lam=rs.shrunk_lambda,
                    forward_before=round(rs.forward_blend_corr, 5), forward_after=round(rs.best_fc, 5),
                    note="weights shrunk toward equal on the forward slice (no-op unless it helps)")

        # ---- v18 CHORUS SHRINKAGE (no-op-safe) -------------------------------
        # Pearson is won on the rows where you make BIG calls -- so only make
        # big calls where independent members CONCUR. Scale each row of the
        # blend by committee agreement; the strength beta is chosen on the
        # FORWARD slice and DEFAULTS TO 0 (no shrink) unless forward improves.
        rs.chorus_beta = 0.0
        if (not rs.gate_fired and not rs.final_is_median and rs.final_weather is None
                and len(rs.final_weights) >= 2
                and all(nm in rs.fwd_parts for nm in rs.final_weights)):
            rs.fwd_base = sum(rs.final_weights[nm] * rs.fwd_parts[nm] for nm in rs.final_weights)
            rs.best_fc = pearson(rs.y_full[rs.future], rs.fwd_base)
            for rs.beta in (0.5, 1.0, 2.0, 4.0):
                rs.fac = chorus_factor(rs.fwd_parts, rs.final_weights, rs.beta)
                rs.fc = pearson(rs.y_full[rs.future], rs.fwd_base * rs.fac)
                if rs.fc > rs.best_fc + 1e-6:
                    rs.best_fc, rs.chorus_beta = rs.fc, rs.beta
            if rs.chorus_beta > 0:
                log("chorus_shrinkage", beta=rs.chorus_beta, forward_after=round(rs.best_fc, 5),
                    note="blend scaled by member agreement on the forward slice (no-op unless it helps)")

        # ---- v32 FACTOR-NEUTRAL blend (IDEAS_ZOO v55 sec14, no-op-safe) --------
        # Residualize the blend against the dominant TARGET-FREE factors by a
        # fraction chosen on the FORWARD slice (default 0 = raw). The blend's
        # loading on a crowded latent factor is exactly what decays when the
        # factor rotates; neutralization must clear SHAPE_MARGIN to ship.
        rs.fn_frac, rs.fn_beta = 0.0, None
        if (rs.cfg.FACTOR_NEUTRAL and not rs.gate_fired and not rs.final_is_median
                and rs.final_weather is None and getattr(rs, "factor_comps", None) is not None
                and len(rs.future) > 100
                and all(nm in rs.fwd_parts for nm in rs.final_weights)):
            try:
                F_fut = ((rs.X_full[rs.future][:, : rs.factor_dpre] - rs.factor_mu)
                         @ rs.factor_comps.T).astype(np.float64)
                Fz_fut = (F_fut - F_fut.mean(0)) / (F_fut.std(0) + 1e-12)
                fwd_b = sum(rs.final_weights[nm] * rs.fwd_parts[nm] for nm in rs.final_weights)
                base_fc = pearson(rs.y_full[rs.future], fwd_b)
                beta_f, *_ = np.linalg.lstsq(Fz_fut, fwd_b, rcond=None)
                best_fc = base_fc
                for frac in rs.cfg.FACTOR_NEUTRAL_FRACS:
                    fc = pearson(rs.y_full[rs.future], fwd_b - float(frac) * (Fz_fut @ beta_f))
                    if fc > best_fc + rs.cfg.SHAPE_MARGIN:
                        best_fc, rs.fn_frac, rs.fn_beta = fc, float(frac), beta_f
                if rs.fn_frac > 0:
                    log("factor_neutral_blend", frac=rs.fn_frac,
                        forward_before=round(base_fc, 5), forward_after=round(best_fc, 5),
                        note="blend residualized against target-free factors (forward-chosen, margin-gated)")
            except Exception as e:
                rs.fn_frac, rs.fn_beta = 0.0, None
                log("factor_neutral_skipped", err=str(e)[:80])

        # ---- v19 PREDICTION SHAPE ALCHEMY (no-op-safe) -----------------------
        # Audition output shapes (rank / power / tanh) on the FORWARD blend;
        # ship the best, default 'raw' (no-op). Financial labels often reward
        # ORDER over amplitude, so a rank/tanh remap can be more robust.
        rs.ship_shape = "raw"
        if not rs.final_is_median and rs.final_weather is None and len(rs.future) > 100 \
                and all(nm in rs.fwd_parts for nm in rs.final_weights):
            rs.fwd_final = sum(rs.final_weights[nm] * rs.fwd_parts[nm] for nm in rs.final_weights)
            rs.best_sc = pearson(rs.y_full[rs.future], rs.fwd_final)
            for rs.sh in SHIP_SHAPES[1:]:
                rs.c = pearson(rs.y_full[rs.future], _shape_pred(rs.fwd_final, rs.sh))
                if rs.c > rs.best_sc + rs.cfg.SHAPE_MARGIN:   # v23: real forward gain, not a noisy overfit surface
                    rs.best_sc, rs.ship_shape = rs.c, rs.sh
            if rs.ship_shape != "raw":
                log("prediction_shape_alchemy", shape=rs.ship_shape, forward_after=round(rs.best_sc, 5),
                    note="output remap chosen on the forward slice (no-op unless it helps)")


    def _health_alarms(self, rs: "_RunState") -> None:
        rs.monitor = HealthMonitor(rs.cfg)
        if len(rs.members) >= 2:
            # v8: a blend whose every member shares one trail family is one
            # texture-correlated bet, however decorrelated the predictions look
            rs.monitor.check("trail_family_diversity", float(rs.n_trail_families), 1.5, "below",
                          "all blend members share a single path-texture family")
        rs.ship_dreams = [rs.member_lessons[nm].dream_p05 for nm in rs.final_weights
                       if nm in rs.member_lessons and np.isfinite(rs.member_lessons[nm].dream_p05)]
        if rs.ship_dreams:
            # v9: a shipped trail that fails in its 5th-percentile dreamed
            # world is a fragility warning the pooled numbers hide
            rs.monitor.check("shipped_dream_p05", float(min(rs.ship_dreams)), 0.0, "below",
                          "a shipped trail goes negative in bootstrapped replays of the world")
        rs.ship_gaps = [rs.member_lessons[nm].sense_gap / max(abs(rs.member_lessons[nm].oof_corr), 1e-6)
                     for nm in rs.final_weights
                     if nm in rs.member_lessons and np.isfinite(rs.member_lessons[nm].sense_gap)]
        if rs.ship_gaps:
            # v10: the two senses disagree -> tail-driven alpha in the blend
            rs.monitor.check("shipped_sense_gap_ratio", float(max(rs.ship_gaps)), 0.75, "above",
                          "a shipped trail looks strong to one sense only (tail-driven)")
        rs.alarms = rs.monitor.run_checks(rs.result, rs.members, rs.yp, rs.library.lessons, rs.forward_blend_corr)
        write_csv(rs.alarms, "ensemble_health_alarms.csv")


    def _sealed_audit(self, rs: "_RunState") -> None:
        # ---- SEALED HOLDOUT: evaluated ONCE, after the blend is frozen ----------
        # Refit shipped members on the working region only; score the sealed tail.
        # This number gates NOTHING in this run -- it is the once-per-version
        # generalization audit for the harness itself.
        rs.sealed_corr = None
        if len(rs.sealed_idx) >= 500:
            rs.past_seal = np.arange(0, max(2, rs.seal_cut - rs.cfg.EMBARGO_ROWS))
            rs.seal_parts = {}
            for rs.nm in rs.final_weights:
                rs.l = rs.member_lessons[rs.nm]
                rs.st = fit_skill(rs.l.skill, rs.spec_lookup[rs.l.key], rs.X_full[rs.past_seal], rs.y_full[rs.past_seal],
                               rs.seg_full[rs.past_seal], rs.cols, np.random.default_rng(rs.l.seed), rs.cfg, rs.l.seed)
                rs.p = predict_skill(rs.st, rs.X_full[rs.sealed_idx])
                rs.sd = float(np.std(rs.p)) + 1e-9
                rs.seal_parts[rs.nm] = ((rs.p - float(np.mean(rs.p))) / rs.sd).astype(np.float64)
            rs.wth_seal = moe_states(getattr(rs, "moe_gauge", None), rs.X_full[rs.sealed_idx])
            rs.seal_blend = apply_weights_rows(rs.seal_parts, rs.final_weights, rs.final_is_median,
                                            rs.final_weather, rs.wth_seal)
            rs.sealed_corr = score_metric(rs.y_full[rs.sealed_idx], rs.seal_blend)
            write_json({"sealed_rows": int(len(rs.sealed_idx)), "sealed_corr": rs.sealed_corr,
                        "shipped_weights": rs.final_weights,
                        "note": "audited once per kernel version; never used for decisions"},
                       "sealed_holdout_report.json")
            log("SEALED_HOLDOUT_AUDIT", sealed_corr=round(rs.sealed_corr, 5),
                rows=len(rs.sealed_idx), note="not_a_gate")


    def _final_refit_submit(self, rs: "_RunState") -> None:
        # ---- final refit on FULL train (incl. sealed) + test predictions --------
        gc.collect()
        mem_status("pre_final_refit")
        rs.test_parts = {}
        for rs.nm in rs.final_weights:
            rs.l = rs.member_lessons[rs.nm]
            rs.st = fit_skill(rs.l.skill, rs.spec_lookup[rs.l.key], rs.X_full, rs.y_full, rs.seg_full, rs.cols,
                           np.random.default_rng(rs.l.seed), rs.cfg, rs.l.seed)
            rs.p = predict_skill(rs.st, rs.X_test)
            rs.sd = float(np.std(rs.p)) + 1e-9
            rs.test_parts[rs.nm] = ((rs.p - float(np.mean(rs.p))) / rs.sd).astype(np.float64)
            log("final_member_refit", member=rs.nm, weight=round(rs.final_weights[rs.nm], 3))
        rs.wth_test = moe_states(getattr(rs, "moe_gauge", None), rs.X_test)
        rs.test_pred = apply_weights_rows(rs.test_parts, rs.final_weights, rs.final_is_median,
                                       rs.final_weather, rs.wth_test)
        if rs.chorus_beta > 0 and all(nm in rs.test_parts for nm in rs.final_weights):
            rs.test_pred = rs.test_pred * chorus_factor(rs.test_parts, rs.final_weights, rs.chorus_beta)
            log("chorus_applied", beta=rs.chorus_beta, note="test blend shrunk by member agreement")
        if getattr(rs, "fn_frac", 0.0) > 0 and rs.fn_beta is not None:
            F_te = ((rs.X_test[:, : rs.factor_dpre] - rs.factor_mu)
                    @ rs.factor_comps.T).astype(np.float64)
            Fz_te = (F_te - F_te.mean(0)) / (F_te.std(0) + 1e-12)
            rs.test_pred = rs.test_pred - rs.fn_frac * (Fz_te @ rs.fn_beta)
            log("factor_neutral_applied", frac=rs.fn_frac,
                note="test blend residualized against target-free factors (forward-chosen)")
        if rs.ship_shape != "raw":
            rs.test_pred = _shape_pred(rs.test_pred, rs.ship_shape)
            log("shape_applied", shape=rs.ship_shape, note="test prediction remapped (forward-chosen)")
        if rs.wins["apply"]:
            rs.lo, rs.hi = np.quantile(rs.test_pred, rs.wins["best_q"]), np.quantile(rs.test_pred, 1 - rs.wins["best_q"])
            rs.test_pred = np.clip(rs.test_pred, rs.lo, rs.hi)
        # ---- v32 PREDICTION-DISTRIBUTION SHIFT (IDEAS_ZOO v65 sec43) ----------
        try:
            if rs.cfg.PRED_DIST_REPORT:
                pd_df = prediction_distribution_report(rs.blend_oof, rs.test_pred)
                write_csv(pd_df, "prediction_distribution_shift.csv")
                log("prediction_distribution",
                    test_tail3sd=float(pd_df.iloc[1]["tail_mass_3sd"]),
                    work_tail3sd=float(pd_df.iloc[0]["tail_mass_3sd"]),
                    note="test predictions more extreme than anything validation scored = amplitude risk")
        except Exception as e:
            log("prediction_distribution_skipped", err=str(e)[:80])
        write_submission(np.asarray(rs.test_pred, np.float32), rs.root)
        rs.n_test = len(rs.X_test)
        del rs.X_full, rs.X_test                 # nothing below needs the matrices
        gc.collect()
        free_gpu_mem()
        mem_status("post_submission")


    def _cairn_ledger(self, rs: "_RunState") -> None:
        # ---- CAIRN (v10): fingerprint this world for the next visitor ---------
        rs.champ_key = max(rs.library.promoted(), key=lesson_fitness).key if rs.library.promoted() else None
        # v14 seed bank: the run's best measured LOSERS -- positive-corr trails
        # that neither became the champion nor shipped. Written for the NEXT
        # run to germinate (temporal biodiversity against regime decay).
        rs.shipped_keys = {rs.member_lessons[nm].key for nm in rs.final_weights if nm in rs.member_lessons}
        rs.loser_pool = sorted((l for l in rs.library.promoted()
                             if l.key != rs.champ_key and l.key not in rs.shipped_keys
                             and l.oof_corr > 0),
                            key=lambda l: -l.oof_corr)
        rs.seed_bank, rs.seen_sb = [], set()
        for rs.l in rs.loser_pool:
            if rs.l.key in rs.seen_sb:
                continue
            rs.seen_sb.add(rs.l.key)
            rs.seed_bank.append(rs.l.key)
            if len(rs.seed_bank) >= rs.cfg.SEEDBANK_SIZE:
                break
        rs.cairn = {"version": "v33", "data_source": rs.data_source,
                 "gauge_edges": [float(e) for e in (GAUGE.edges if GAUGE is not None else [])],
                 "terrain_populations": rs.t_pop, "weather_populations": rs.w_pop,
                 "even_dominant": rs.n_even, "trap_count": len(TRAPS),
                 "jnd": rs.jnd["jnd"], "champion": rs.champ_key,
                 "seed_bank": rs.seed_bank,        # v14: measured losers for the next run
                 "honest": rs.result["honest"][rs.result["winner"]]}
        # v27 self-improvement: distil this run's OUT-OF-SAMPLE-grounded learnings
        # into the cross-run ledger (merged with the prior by evidence count) so the
        # NEXT run starts smarter. survivors = low-decay shipped genomes (warm starts);
        # decayers = high-decay motifs (anti-priors); + governor beta + the per-family/
        # skill generalization track record + the data profile.
        if rs.cfg.SELF_IMPROVE:
            rs.prev_led = LEDGER_PRIOR or {}
            rs.surv = sorted((rs.member_lessons[nm] for nm in rs.final_weights if nm in rs.member_lessons),
                          key=lambda l: (l.oof_corr - l.wf_corr) if np.isfinite(l.wf_corr) else 0.0)
            rs.survivors = list(dict.fromkeys(l.key for l in rs.surv))[: rs.cfg.LEDGER_MAX_SURVIVORS]
            rs.decj = sorted((l for l in rs.library.lessons
                           if l.oof_corr > 0.03 and np.isfinite(l.wf_corr) and (l.oof_corr - l.wf_corr) > 0.03),
                          key=lambda l: -(l.oof_corr - l.wf_corr))
            rs.decayers = list(dict.fromkeys(f"{l.skill}|{l.family}" for l in rs.decj))[: rs.cfg.LEDGER_MAX_DECAYERS]
            rs.gcount = int((rs.prev_led.get("governor") or {}).get("count", 0)) + 1
            rs.ledger = {"version": "v33", "data_source": rs.data_source,
                      "governor": {"beta": round(float(GOVERNOR.get("beta", 0.0)), 5),
                                   "lambda": round(float(GOVERNOR.get("lambda", 0.0)), 5),
                                   "width_decay_corr": (round(rs.width_decay_corr, 5)
                                                        if getattr(rs, "width_decay_corr", None) is not None
                                                        else None),   # v30: wide-path evidence for the next run
                                   "count": rs.gcount},
                      "family_decay": _ledger_merge(rs.prev_led.get("family_decay", {}),
                                                    _ledger_decay_stats(rs.library.lessons, "family")),
                      "skill_decay": _ledger_merge(rs.prev_led.get("skill_decay", {}),
                                                   _ledger_decay_stats(rs.library.lessons, "skill")),
                      "survivors": rs.survivors, "decayers": rs.decayers,
                      "profile": {"metric": PROFILE.get("metric"), "temporal": PROFILE.get("temporal"),
                                  "target_kind": PROFILE.get("target_kind")}}
            rs.cairn["ledger"] = rs.ledger
            write_json(rs.ledger, "learning_ledger.json")
            log("learning_ledger_written", governor_runs=rs.gcount, survivors=len(rs.survivors),
                decayers=len(rs.decayers), tracked_families=len(rs.ledger["family_decay"]),
                tracked_skills=len(rs.ledger["skill_decay"]))
        write_json(rs.cairn, "world_cairn.json")
        rs.cairn_drift = None
        rs.prev_paths = [Path(p) for p in rs.cfg.CAIRN_PATHS]
        try:
            rs.prev_paths += list(Path("/kaggle/input").glob("*/world_cairn.json"))
        except Exception:
            pass
        for rs.pth in rs.prev_paths:
            try:
                if rs.pth.exists():
                    rs.prev = json.loads(rs.pth.read_text())
                    rs.pe, rs.ce = rs.prev.get("gauge_edges") or [], rs.cairn["gauge_edges"]
                    rs.edge_drift = (float(np.mean([abs(a - b) / (abs(a) + 1e-9)
                                                 for a, b in zip(rs.pe, rs.ce)]))
                                  if rs.pe and len(rs.pe) == len(rs.ce) else None)
                    rs.cairn_drift = {"prev_cairn": str(rs.pth),
                                   "gauge_edge_drift": rs.edge_drift,
                                   "prev_champion": rs.prev.get("champion"),
                                   "champion_changed": rs.prev.get("champion") != rs.champ_key,
                                   "trap_count_delta": len(TRAPS) - int(rs.prev.get("trap_count", 0))}
                    log("cairn_comparison", **{k: v for k, v in rs.cairn_drift.items()})
                    break
            except Exception:
                continue


    def _chronicle(self, rs: "_RunState") -> None:
        rs.evo_summary = {}
        rs.full_evo = [h for h in rs.evo.history if np.isfinite(h.get("fitness", float("nan")))]
        if rs.full_evo:
            rs.best_evo = max(rs.full_evo, key=lambda h: h["fitness"])
            rs.evo_summary = {"lessons": len(rs.full_evo),
                           "draft_culled": sum(1 for h in rs.evo.history if h["decision"] == "draft_culled"),
                           "g_best_key": rs.best_evo["key"],
                           "g_best_fitness": rs.best_evo["fitness"],
                           "g_best_operator": rs.best_evo["operator"]}

        # ---- WORLD CHRONICLE (v9): the run as a written story -----------------
        rs.explorer_lines = []
        for rs.t in EXPLORER_TRAITS[: rs.cfg.N_EXPLORERS]:
            rs.mine = [l for l in rs.library.lessons if l.explorer == rs.t["name"]]
            if rs.mine:
                rs.best_l = max(rs.mine, key=lambda l: l.oof_corr)
                rs.explorer_lines.append(
                    f"{rs.t['name']} [{rs.t.get('species', rs.t['name'])}/{rs.t['metaheuristic']}]: "
                    f"{len(rs.mine)} trails, best {rs.best_l.key} ({rs.best_l.oof_corr:+.4f})")
        rs.champion_lines = []
        rs.key2lesson = {l.key: l for l in rs.library.lessons}
        if not rs.tex_df.empty:
            for rs._, rs.r in rs.tex_df[rs.tex_df["decision"] == "promote"].head(3).iterrows():
                rs.lsn = rs.key2lesson.get(rs.r["key"])
                rs.verb = trail_verb(rs.lsn.skill, rs.lsn.family, rs.lsn.transform) if rs.lsn else "walks"
                rs.champion_lines.append(
                    f"{rs.r['key']} ({rs.r['oof_corr']:+.4f}, family T{int(rs.r['trail_family'])}) "
                    f"{rs.verb} -- {texture_words(rs.r, rs.tex_df)}")
        rs.kill_lines = [f"{l.key} -- {l.predator_verdict}"
                      for l in rs.library.lessons if l.decision == "predator_killed"]
        rs.dream_lines = []
        for rs.l in sorted(rs.library.promoted(), key=lambda l: l.dream_p05 if np.isfinite(l.dream_p05) else 1e9)[:3]:
            if np.isfinite(rs.l.dream_p05):
                rs.omen = "an OMEN" if rs.l.dream_p05 < 0 else "steady"
                rs.dream_lines.append(f"{rs.l.key}: dream_p05={rs.l.dream_p05:+.4f}, "
                                   f"dream_p50={rs.l.dream_p50:+.4f} -- {rs.omen}")
        rs.ship_desc = (f"The party shipped '{rs.result['winner']}'"
                     + (" (weather-conditional)" if rs.final_weather is not None else "")
                     + (f" after the forward gate overrode to {rs.best_cv_name}" if rs.gate_fired else "")
                     + ": " + ", ".join(f"{nm} ({rs.final_weights[nm]:.2f})" for nm in rs.final_weights)
                     + f". Forward corr {rs.forward_blend_corr:+.5f}.")
        rs.species_run = ", ".join(dict.fromkeys(ex.species for ex in rs.explorers))
        rs.embodiment_lines = [
            (f"The expedition carried {int(rs.cfg.TIME_BUDGET_MIN)} minutes of provisions: "
             f"{META.seasons} season(s) of foraging, {META.epochs} evolutionary epoch(s), "
             f"{rs.auditioned} skill audition(s); camp was made with "
             f"{max(0.0, round(rs.cfg.TIME_BUDGET_MIN - META.now(), 1))} minutes to spare."
             if META.enabled else
             "No time budget was set: fixed rations -- one season, one epoch."),
            f"The expedition was a menagerie: {rs.species_run}.",
            (f"{rs.n_quorum} feature-families reached colony quorum (>= {rs.cfg.QUORUM_MIN} species agreed); "
             f"{len(DANCES)} waggle dances were posted; the gene pool held {len(GENE_POOL)} plasmids."
             if QUORUM else ""),
            f"The satellites surveyed {len(SURVEY)} families from orbit; "
            f"'{max(SURVEY, key=SURVEY.get)}' showed the densest signal." if SURVEY else "",
            f"The trap map marked {len(TRAPS)} mirages before anyone walked.",
            (f"Hearing test: the expedition can detect planted alpha down to "
             f"s={rs.jnd['jnd']}." if rs.jnd["jnd"] is not None else
             "Hearing test: no planted strength on the grid was detectable -- "
             "the world is loud and the budget is small."),
            (f"The attention pool went to {rs.attention_grantee} "
             f"(highest marginal yield)." if rs.attention_grantee else ""),
            (f"{rs.dive_added} submarines dove beneath the champion's surface; "
             f"{rs.dive_members} made the shipped party." if rs.dive_added else ""),
            (f"Venom memory holds {len(TABOO)} motifs." if TABOO else ""),
            (f"The circadian governor cut: {', '.join(self._circadian_cuts)}."
             if self._circadian_cuts else ""),
            (f"A cairn from a previous visit was found; gauge drift "
             f"{rs.cairn_drift['gauge_edge_drift']}." if rs.cairn_drift else
             "No previous cairn was found; ours now stands."),
        ]
        write_chronicle({
            "title": f"DRW world-explorer v33 ({rs.data_source})",
            "features": len(rs.cols), "train_rows": rs.n, "sealed_rows": int(len(rs.sealed_idx)),
            "data_source": rs.data_source, "terrain_pop": rs.t_pop, "weather_pop": rs.w_pop,
            "even_dominant": rs.n_even, "explorer_lines": rs.explorer_lines,
            "embodiment_lines": [s for s in rs.embodiment_lines if s],
            "n_lessons": len(rs.library.lessons), "n_promoted": len(rs.library.promoted()),
            "n_families": int(rs.tex_df["trail_family"].nunique()) if not rs.tex_df.empty else 0,
            "champion_lines": rs.champion_lines, "kill_lines": rs.kill_lines,
            "dream_lines": rs.dream_lines, "shipping_line": rs.ship_desc,
            "sealed_line": (f"Behind the glass, the sealed {len(rs.sealed_idx)} minutes answered: "
                            f"{rs.sealed_corr:+.5f}." if rs.sealed_corr is not None else ""),
        })


    def _summarize(self, rs: "_RunState") -> dict[str, Any]:
        rs.summary = {
            "data_source": rs.data_source, "train_rows": rs.n, "working_rows": rs.n_work,
            "sealed_rows": int(len(rs.sealed_idx)), "test_rows": rs.n_test,
            "features": len(rs.cols), "segments": rs.cfg.N_SEGMENTS,
            "splits": rs.cfg.N_SPLITS, "wf_folds": rs.cfg.WF_FOLDS,
            "gbdt_backend": GBDT_BACKEND, "nnls_available": HAVE_NNLS,
            "hardware": {"torch": HAVE_TORCH, "gpus": N_GPUS,
                         "gpu_names": _gpu_names(),
                         "hetero_pairing": rs.hetero,
                         "schedule": "hetero" if rs.hetero else ("gpu" if N_GPUS > 0 else "cpu")},
            "lessons": len(rs.library.lessons), "promoted": len(rs.library.promoted()),
            "predator": {"targets": int(len(rs.pred_df)) if not rs.pred_df.empty else 0,
                         "killed": int((rs.ledger_df["decision"] == "predator_killed").sum()) if not rs.ledger_df.empty else 0,
                         "budget_left": rs.predator.budget},
            "draft_gate": rs.gate.report(),
            "evolution": rs.evo_summary,
            "topography": {"terrain_clusters": len(rs.t_pop), "terrain_populations": rs.t_pop,
                           "even_dominant_features": rs.n_even,
                           "textured_trails": int(len(rs.tex_df)),
                           "trail_families_total": int(rs.tex_df["trail_family"].nunique()) if not rs.tex_df.empty else 0,
                           "trail_families_in_blend": rs.n_trail_families,
                           "predator_terrain_kills": int(sum("dead_terrain" in (l.reason or "")
                                                             for l in rs.library.lessons))},
            "ecology": {"weather_states": len(rs.w_pop), "weather_populations": rs.w_pop,
                        "weather_conditional_blend": rs.final_weather is not None,
                        "mycelium_columns": len(MYCELIUM),
                        "dream_omens": rs.dream_omens,
                        "predator_weather_kills": int(sum("dead_weather" in (l.reason or "")
                                                          for l in rs.library.lessons))},
            "embodiment": {"trap_mirages": len(TRAPS),
                           "survey_best_family": max(SURVEY, key=SURVEY.get) if SURVEY else None,
                           "jnd_threshold": rs.jnd["jnd"],
                           "attention_grantee": rs.attention_grantee,
                           "car_stalls": int(sum(1 for r in rs.journal_rows
                                                 if r.get("reason") == "car_stalled")
                                             + sum(1 for h in rs.evo.history
                                                   if h.get("decision") == "car_stalled")),
                           "taboo_motifs": len(TABOO),
                           "dive_lessons": rs.dive_added,
                           "dive_members_shipped": rs.dive_members,
                           "circadian_cuts": self._circadian_cuts,
                           "cairn_drift": rs.cairn_drift},
            "menagerie": {"species_run": [ex.species for ex in rs.explorers],
                          "families_reached_quorum": rs.n_quorum,
                          "waggle_dances": len(DANCES),
                          "gene_pool_size": len(GENE_POOL),
                          "evolution_op_counts": {op: len(g) for op, g in rs.evo.op_gains.items()
                                                  if g and op in ("chemotaxis", "flock", "plasmid",
                                                                  "antiflock")}},
            "bio_ecology": {"red_pheromone_columns": len(RED_MYCELIUM),
                            "red_from_traps": len(TRAPS),
                            "seedbank_germinated": len(SEEDBANK),
                            "seedbank_written": len(rs.seed_bank),
                            "islands": bool(rs.cfg.ISLANDS),
                            "antiflock_runs": len(rs.evo.op_gains.get("antiflock", [])),
                            "member_input_jaccard_cap": rs.cfg.JAMMING_JACCARD},
            "anti_decay": {"shrunk_blend_lambda": rs.shrunk_lambda,
                           "cv_forward_gap": round(rs.result["honest"][rs.result["winner"]]
                                                   - rs.forward_blend_corr, 5),
                           "note": "shrunk_blend defaults to 0; >0 only if forward strictly improves"},
            "forensics": {"enabled": bool(rs.cfg.FORENSIC_ENABLED),
                          "override": bool(rs.forensic_dec["override"]),
                          "shipped_selector": rs.forensic_dec["winner"],
                          "forward_blend_corr": round(rs.forensic_dec["forward_blend_corr"], 5),
                          "feature_clusters": int(len(np.unique(FCLUST))) if FCLUST is not None else 0,
                          "note": "full forensic suite written; override is forward-validated, no-op unless it strictly improves the CV->forward gap"},
            "beacons": {"dropped": len(BEACONS.kinds) if BEACONS is not None else 0,
                        "rare": sum(k == "rare" for k in BEACONS.kinds) if BEACONS is not None else 0,
                        "novelty": sum(k == "novelty" for k in BEACONS.kinds) if BEACONS is not None else 0,
                        "field_channels_added": len(BEACONS.kinds) if BEACONS is not None else 0,
                        "predator_beacon_kills": int(sum("dead_beacon" in (l.reason or "")
                                                         for l in rs.library.lessons))},
            "metabolism": {"time_budget_min": rs.cfg.TIME_BUDGET_MIN,
                           "enabled": META.enabled,
                           "shipping_reserve_min": round(META.reserve, 1),
                           "seasons": META.seasons, "epochs": META.epochs,
                           "auditioned_skills": rs.auditioned,
                           "elapsed_min": round(META.now(), 1),
                           "deadlines_min": {k: round(v, 1) for k, v in META.deadline.items()}},
            "members": list(rs.members), "ensemble_winner": rs.result["winner"],
            "honest_scores": rs.result["honest"], "weights": rs.w,
            "forward_blend_corr": rs.forward_blend_corr,
            "forward_best_single": {rs.best_cv_name: rs.single_fwd_corr},
            "forward_gate_fired": rs.gate_fired,
            "shipped_weights": rs.final_weights,
            "sealed_holdout_corr": rs.sealed_corr,
            "winsorize_q": rs.wins["best_q"] if rs.wins["apply"] else None,
            "alarms": int((rs.alarms["status"] == "ALARM").sum()) if not rs.alarms.empty else 0,
        }
        write_json(rs.summary, "explorer_run_summary.json")
        META.heartbeat("run_end")
        mem_status("run_end")
        log("run_end", winner=rs.result["winner"],
            honest=round(rs.result["honest"][rs.result["winner"]], 5),
            forward=round(rs.forward_blend_corr, 5),
            sealed=round(rs.sealed_corr, 5) if rs.sealed_corr is not None else None,
            gate=rs.gate_fired, alarms=rs.summary["alarms"])
        print(json.dumps({k: v for k, v in rs.summary.items() if k != "honest_scores"}, indent=2, default=str))
        return rs.summary


