# ----------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    DATA_ROOTS: tuple[str, ...] = (
        "/kaggle/input/drw-crypto-market-prediction",
        "/kaggle/input/competitions/drw-crypto-market-prediction",
    )
    OUT_DIR: str = "/kaggle/working" if Path("/kaggle/working").exists() else "explorer_out"
    ALLOW_SYNTHETIC_FALLBACK: bool = True
    SYN_ROWS: int = 30_000
    SYN_ANON: int = 48

    SEED: int = 42
    N_SEGMENTS: int = 12
    N_SPLITS: int = 5                   # leave-segments-out folds
    WF_FOLDS: int = 4                   # walk-forward validation segments
    EMBARGO_ROWS: int = 720
    PROBE_MAX_ROWS: int = 120_000
    FORWARD_FRACTION: float = 0.25      # of the WORKING region (gate slice)
    SEALED_FRACTION: float = 0.10       # final rows: evaluated once, never gated

    # phase 1: developmental curriculum (v11: 7 of the species roster run by
    # default -- proven 5 + bacterium + starling; raise to run more menagerie)
    N_EXPLORERS: int = 7
    LESSON_BUDGET: int = 24             # per explorer (cost units)
    UCB_C: float = 0.02

    # phase 2: metaheuristic evolution (budget raised: the v4 real run was
    # still climbing -- gen2 produced 0.126-0.130 accepts -- when it hit 0)
    EVOLUTION_BUDGET: int = 56
    EVOLUTION_POP: int = 10
    EVOLUTION_OFFSPRING: int = 6
    EVOLUTION_MAX_GENERATIONS: int = 6
    # measured champions, refreshed from the v11 REAL run (2026-06, T4x2,
    # private LB 0.08969 -- the project best): the mycelium x swell_rider x
    # quantize4 motif won outright (g_best + 0.498 of the shipped blend via
    # the ablation k->half variant), with dual_exposure and rank carrying the
    # rest of the weights. v9's frontier champions stay behind them; every
    # warm genome is re-measured through the same doors at generation 0.
    WARM_GENOMES: tuple[str, ...] = (
        "swell_rider|mycelium30_quantize4",         # v11 g_best (oof 0.1374, wf 0.1438)
        "swell_rider|mycelium15_quantize4",         # v11 shipped 0.498 (ablation k->half)
        "bagged_linear|mycelium195_dual_exposure",  # v11 shipped 0.320
        "bagged_linear|top160_rank",                # v11 shipped 0.182
        "bagged_linear|mycelium131_quantize4",      # v11 evolution accept (oof 0.1350)
        "scout_lattice|top160_quantize8",           # v9 best single lesson (frontier_surf)
        "bagged_linear|top135_quantize4",           # v9 g_best
        "scout_lattice|top128_quantize8",           # 0.1327 in v8 AND v9 (determinism witness)
        "linear_ols|tail120_identity",              # v22: plain OLS on the trailing feature block (11th-place class)
        "linear_ols|tail40_identity",               # v22: smaller trailing block
        "linear_ols|top50_identity",                # v22: plain OLS on corr-top-50 (discoverable everywhere)
        "huber_linear|tail120_identity",            # v23: fat-tail-robust linear on the trailing block
        "elastic_net|top64_identity",               # v23: sparse in-fit feature selection (which of 800 are real)
        "elastic_net|tail120_identity",             # v23: sparse on the trailing block
        "elastic_net|stabsel64_identity",           # v24: sparse fit on the stability-selected feature set
        "linear_ols|stabsel50_identity",            # v24: plain OLS on the stably-selected features
        "recency_weighted|tail120_identity",        # v24: exp-recency-weighted ridge on the trailing block
        "shift_linear|mycelium160_identity",        # v24: covariate-shift-reweighted linear
        "bagged_linear|irm128_quantize4",           # v24: bagged on invariant-risk-selected features
        "terrace|mycelium7_moire",                  # v21 real run: tiny-k terrace+moire champion region
        "gpu_ridge_swarm|mycelium7_moire",          # v21 real run g_best (fitness 0.1449, highest measured)
        "terrace|mycelium8_moire",                  # v21 real run: tiny-k terrace+moire champion
        "greedy_ols|tail50_identity",               # v25: the 11th-place 0.111 recipe (greedy suffix OLS on the tail)
        "greedy_ols|tail120_identity",              # v25: greedy suffix OLS on a larger trailing block
        "greedy_ols|top64_identity",                # v25: greedy suffix OLS on corr-top features (general)
        "pls|decor64_identity",                     # v27: PLS supervised SVD-regression (top-solution learning)
        "pls|tail120_identity",                     # v27: PLS on the trailing block
        "bayes_ridge|tail120_identity",             # v27: evidence-tuned Bayesian ridge on the tail
        "ard_linear|medoid64_identity",             # v27: ARD sparse Bayesian linear on the cluster-medoid set
    )
    EVOLUTION_PATIENCE: int = 3         # v8 real run: patience (not budget) ended evolution, 7 units left
    SA_T0: float = 0.02
    SA_DECAY: float = 0.6
    DE_F: float = 0.8
    OPERATOR_SOFTMAX_TAU: float = 0.01
    OPERATOR_FLOOR: float = 0.10
    K_MIN: int = 4
    K_MAX: int = 224                    # v8 real run: top160_quantize8 was accepted AT the old cap and
                                        # took 0.295 blend weight -- the boundary was binding; widen it

    # successive-halving drafts
    DRAFT_ROWS: int = 20_000
    DRAFT_FOLDS: int = 2
    DRAFT_WARMUP: int = 5               # first drafts always pass (calibration)
    DRAFT_PASS_QUANTILE: float = 0.5    # must beat this quantile of prior drafts
    DRAFT_ABS_PASS: float = 0.06        # v4 real run: median draft width 0.072; 0.02 passed everything
    DRAFT_MIN_COST: int = 4             # v4 real run: drafting cheap skills cost more than it saved

    # predator persona (executes the null tax + perturbation + sub-period attacks)
    PREDATOR_BUDGET: int = 28           # v9+v11 measured: nulls ate every wallet, perturbs never ran,
                                        # units stranded -- PERTURB_ALL doubles per-target spend
    PREDATOR_MAX_TARGETS: int = 14      # top promoted lessons per raid (seasons grow the pool)
    PREDATOR_WORST3_FLOOR: float = -0.01  # kill if worst 3-consecutive-seg mean corr below
    NULL_Q: float = 0.95

    # bit-budget frontier (quantization-aware viewports)
    BIT_BUDGET: int = 4096              # genomes must satisfy k * bits_per_feature <= this

    # v8 topography layer
    TERRAIN_CLUSTERS: int = 6           # target-free regimes ("valleys") in the atlas
    TERRAIN_FIT_ROWS: int = 60_000
    CODEBOOK_SIZE: int = 64             # vector-quantization centroids (space LUT)
    CODEBOOK_SHRINK: float = 25.0       # per-centroid target-mean shrinkage mass
    SCOUT_COUNT: int = 6                # speculative lattice scouts per lesson
    SCOUT_ACCEPT: float = 0.01          # verifier acceptance bar (inner-split corr)
    TEXTURE_FAMILY_RADIUS: float = 1.0  # leader-cluster radius in z-scored texture space
    TEXTURE_FAMILY_CAP: int = 4         # max blend members sharing one trail family
    PREDATOR_TERRAIN_FLOOR: float = -0.02  # kill if worst populated-terrain corr below
    FOLD_PAIRS: int = 8                 # anti-correlated pairs folded by fold_pairs
    TERRAIN_EXPERT_MIN: int = 500       # min rows in a terrain before it earns a router expert
    TERRAIN_EXPERT_SHRINK: float = 2000.0  # rows-scale shrinkage of experts toward the global model
    TERRAIN_MIN_ROWS: int = 200         # a terrain counts as populated above this (reports + predator)

    # v9 ecology layer
    WEATHER_STATES: int = 3             # calm / mid / storm (quantile bands of row dispersion)
    WEATHER_MIN_ROWS: int = 200         # a weather band counts as populated above this
    WEATHER_MOE_SHRINK: float = 500.0   # rows-scale shrinkage of per-state weights toward global
    PREDATOR_WEATHER_FLOOR: float = -0.02  # kill if worst populated-weather corr below
    RELAY_TAUS: tuple[float, ...] = (0.25, 1.0, 4.0)  # caravan shrinkage grid (x lam)
    SWELL_SPANS: tuple[int, ...] = (1, 8, 32)  # EMA spans; 1 = raw label (plain ridge)
    DREAM_REPLAYS: int = 150            # block-bootstrap replays per promoted trail
    SHADOW_VAR_Q: float = 0.5           # shadow family: variance must exceed this quantile

    # v10 embodiment layer
    PANORAMA_FIRST: bool = True         # 1-bit all-features orient scan before phase 1
    SAT_STRIDE: int = 6                 # satellite survey: probe-row stride
    SAT_FOLDS: int = 2                  # satellite survey: purged folds
    CAR_ROWS: int = 60_000              # car rung row tile (between airplane 20k and full hike)
    CAR_FOLDS: int = 3                  # car rung purged folds
    TRAP_SCAN_TOP: int = 128            # mirage scan: strongest-|corr| features examined
    TRAP_FLIP_RATE: float = 0.4         # trap if >= this fraction of folds flip corr sign
    SURPRISE_W: float = 0.01            # bandit bonus per unit of coordinate surprise EMA
    ATTENTION_POOL: int = 12            # held-back budget granted to the highest-yield explorer
    JND_STRENGTHS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08)  # planted-alpha grid
    TABOO_W: float = 0.02               # bandit penalty per unit of venom on a motif
    TABOO_SKIP: float = 3.0             # evolution skips candidates with taboo load >= this
    RUN_DEADLINE_MIN: float = 0.0       # wall-clock budget in minutes; 0 disables the governor
    DIVE_BUDGET: int = 8                # submarine phase cost units (residual prospecting)
    DIVE_MEMBER_CAP: int = 2            # max dive lessons admitted to the blend
    ABLATION_BUDGET: int = 6            # cost units for the champion ablation panel
    CAIRN_PATHS: tuple[str, ...] = ("world_cairn_prev.json",)  # prior-world fingerprints

    # v11 menagerie layer
    QUORUM_MIN: int = 3                 # distinct species promoting a family before the colony switches it on
    QUORUM_BOOST: float = 0.02         # ucb prior lift for a family that reached quorum
    WAGGLE_MIN: float = 0.05           # promoted width that earns a waggle dance
    WAGGLE_W: float = 0.03             # newborn recruitment strength toward the best dance
    HGT_RATE: float = 0.25             # fraction of evolution candidates formed by plasmid transfer
    LATERAL_NEIGHBORS: int = 6         # correlated-neighbor count for the lateral_line transform
    COMPASS_POOL: int = 256            # candidate features the compass cross-references

    # v12 metabolism layer -- ONE knob sets the run length: a wall-clock
    # budget in minutes. 0 = disabled (fixed v11 pacing, ~70 min on T4x2).
    # 330 (default) ~ 5.5 h, comfortably inside Kaggle's 12 h GPU session.
    # v11 measured units (T4x2): setup ~1.6 min, one phase-1 roster ~35 min,
    # one evolution epoch ~20 min, shipping ~11 min -- the governor plans in
    # those units and ALWAYS reserves the shipping time first.
    TIME_BUDGET_MIN: float = 690.0      # v22: an 11.5 h Kaggle session (690 min)
    RESERVE_MIN: float = 25.0           # shipping reserve floor (forward+sealed+refits+robust-select+reports)
    MET_EXPLORE_SHARE: float = 0.52     # share of remaining usable time for phase-1 seasons
    MET_EVOLVE_SHARE: float = 0.60      # share of what remains after explore for evolution epochs
    MET_DIVE_SHARE: float = 0.30        # share of what remains after the predator for dives
    MAX_SEASONS: int = 12               # roster rebirth cap (a backstop, not a target)
    MAX_EPOCHS: int = 10                # evolution refuel cap (a backstop, not a target)
    AUDITION_ALL_SKILLS: bool = True    # v9 measured: relay_caravan/swell_rider NEVER picked; in
    AUDITION_K: int = 64                # v11 swell_rider then WON -- every skill auditions once
    PERTURB_ALL: bool = True            # v9+v11 measured: the perturbation attack never fired

    # v13 sensorium layer
    TIDE_SPAN: int = 64                 # causal EMA span the tide transform subtracts
    TERRACE_BINS: int = 16              # quantile steps cut into the ridge score
    TERRACE_SHRINK: float = 25.0        # per-step target-mean shrinkage mass
    RAPIDS_TOP_FRAC: float = 0.30       # roughest-|residual| rows the 2nd stage refits
    RAPIDS_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 0.5)  # re-entry weights tried on the inner split
    PREDATOR_RECENT_FLOOR: float = -0.02  # fading_trail kill: mean corr of final 2 segments below this
    MLP_DROPOUT: float = 0.10           # v13: 0 promotes / overfit 11-56x in EVERY real run
    MLP_PATIENCE: int = 8               # torch early-stop patience on the causal validation tail

    # v14 ecology kernel
    RED_PHEROMONE_W: float = 0.5        # weight of the repellent channel in corr-driven rankings
    JAMMING_JACCARD: float = 0.85       # skip a member sharing >= this input-column Jaccard with one already shipped
    SEEDBANK_SIZE: int = 8              # measured losers written to the cairn for the next run
    SEED_GERMINATE: int = 3             # cairn losers re-tried as warm genomes next run
    ISLANDS: bool = True                # each evolution epoch is an island with its own operator bias

    # v15 beacon layer -- items DROPPED at unique typologies that emit a radial
    # field warping the feature space around them (target-free, leak-free)
    BEACON_DROP: bool = True
    BEACON_MAX: int = 12                # cap on dropped items (rare-terrain + novelty-peak)
    BEACON_RARE_FRAC: float = 0.06      # a terrain rarer than this earns a beacon at its centroid
    BEACON_ALTITUDE_PEAKS: int = 6      # novelty beacons at the highest-altitude clusters
    BEACON_BANDWIDTH: float = 1.0       # x median-distance RBF bandwidth (the field's reach)
    PREDATOR_BEACON_FLOOR: float = -0.02  # dead_beacon kill: worst populated-basin corr below this

    # v17 silicon layer -- the idle T4s do ridge-family work at scale
    SWARM_SCOUTS: int = 64              # GPU ridge models fit per gpu_ridge_swarm lesson
    SWARM_ALPHA: float = 10.0           # ridge regularization for the swarm solves

    # v19 civilization layer
    RESERVOIR_SIZE: int = 96            # echo-state reservoir hidden units
    RESERVOIR_SPECTRAL: float = 0.9     # spectral radius (echo-state property < 1)
    RESERVOIR_LEAK: float = 0.3         # leak rate of the reservoir state
    RASHOMON_EPS: float = 0.002         # strategies within this of the winner join the Rashomon centroid
    MANY_WORLDS_MIN_ROWS: int = 300     # min rows for a leave-state-out world to count

    # skill knobs
    HGB_ITERS: int = 120
    HGB_LR: float = 0.05
    HGB_LEAVES: int = 15
    HGB_MIN_LEAF: int = 200
    HGB_L2: float = 2.0
    KNN_BANK: int = 25_000
    LADDER_MAX_ROUNDS: int = 80
    LADDER_SHRINK: float = 0.2
    BAG_SUBSETS: int = 8
    MLP_MAX_ROWS: int = 60_000
    MLP_MAX_ITER: int = 25
    MLP_HIDDEN: tuple[int, ...] = (64, 32)
    MLP_BATCH: int = 1024
    MLP_PEARSON_LOSS_W: float = 0.4     # torch path: loss = (1-w)*MSE + w*(1-corr), DRW 1st place
    GBDT_ESTIMATORS: int = 300

    # GPU schedule (applied automatically when CUDA is detected; CPU otherwise)
    GPU_MLP_HIDDEN: tuple[int, ...] = (128, 64, 32)   # 3-layer, 1st-place shaped
    GPU_MLP_MAX_ITER: int = 40
    GPU_MLP_MAX_ROWS: int = 120_000
    GPU_MLP_BATCH: int = 4096
    GPU_GBDT_ESTIMATORS: int = 500
    HETERO_PAIRING: bool = True         # run gpu-lane + cpu-lane lessons simultaneously
    SEED_REPS_STOCHASTIC: int = 1
    STABILITY_NOISE: float = 0.05
    STABILITY_TOL: float = 0.35
    STABILITY_PENALTY: float = 0.05

    # viewport knobs
    PAIR_BASE: int = 8
    PAIR_KEEP: int = 16
    MEDOID_THRESHOLD: float = 0.6
    MEDOID_POOL: int = 300
    LASTN_BLOCK: int = 50
    POSITIONAL_BLOCK: int = 160         # v22: size of head/mid/tail feature-order blocks

    # v22 ROBUST MULTI-PARTITION BOOTSTRAP OUT-OF-SAMPLE SELECTION (the spine).
    # A shipping config is chosen only if it is robustly best across MANY
    # train/test geometries of different makeup -- time windows (expanding /
    # sliding-recent / reversed) AND structure-aware splits drawn from the
    # explorer's own map (leave-one-terrain-out, leave-one-weather-out) -- each
    # with a block-bootstrap CI. Score = mean - std ACROSS partitions (stable
    # everywhere, not lucky once). Ties within ROBUST_HEDGE_BAND are HEDGED
    # (blend the tied configs) rather than bet on one. No-op-safe: the incumbent
    # is one of the candidates, so it ships unless a config robustly beats it.
    ROBUST_OOS_SELECT: bool = True
    ROBUST_SAMPLE_ROWS: int = 90_000    # strided working sample used for partition refits
    ROBUST_MIN_TRAIN: int = 3000        # min rows in a train partition
    ROBUST_MIN_TEST: int = 500          # min rows in a test partition
    ROBUST_BOOT: int = 200              # block-bootstrap resamples per partition (segment blocks)
    ROBUST_BOOT_Q: float = 0.25         # lower-tail bootstrap quantile per partition
    ROBUST_HEDGE_BAND: float = 0.004    # robust-score tie band -> hedge (blend) the tied configs
    ROBUST_MAX_MEMBERS: int = 14        # cap distinct members refit across partitions (cost guard)

    # ensemble / shipping
    MAX_MEMBERS: int = 12               # v12: seasons grow the promoted pool; strategies still pick top-8
    MEMBER_CORR_CAP: float = 0.98
    MAX_SEG_NEG_FRAC: float = 0.5

    # v23 ANTI-MONOCULTURE -- the measured cure for the v12 AND v19 regressions.
    # BOTH shipped a single-viewport-family blend that gamed the in-regime
    # sealed holdout and collapsed out-of-period: v19 = 100% `mycelium` members,
    # private 0.0749 (WORST real run) WITH the HIGHEST sealed (0.129) and the
    # biggest CV->forward gap ever (0.046). Diversity must be enforced in
    # viewport-FAMILY space -- output-corr (0.98), texture-family (the v19 blend
    # looked like "10 trail families") and input-Jaccard caps ALL missed it
    # because 12 mycelium viewports differ in output yet share the family that
    # decays together. A blend may carry at most MAX_FAMILY_MEMBERS of any one
    # viewport family; the cap relaxes only to reach MIN_BLEND_MEMBERS.
    MAX_FAMILY_MEMBERS: int = 3
    MIN_BLEND_MEMBERS: int = 6
    MYCELIUM_SATURATE: bool = True      # sqrt-saturate pheromone + let |corr| co-rank (breaks the runaway feedback)
    ROBUST_DEFLATE: float = 0.0010      # selection deflation: override bar rises with log(#configs compared)
    SHAPE_MARGIN: float = 0.003         # forward gain a shape remap must clear before shipping (was a free overfit surface)

    # v27 RUNTIME COMPLEXITY-GENERALIZATION GOVERNOR -- the cure for the MEASURED
    # "complexity ratchet": across the v19/v24/v25 real runs the search ALWAYS
    # converged on the HIGHEST-capacity models (gpu_ridge_swarm/linear_pearson/
    # steepness_gate at k160+ on engineered decor/mycelium viewports) because
    # EVERY selection signal is computed in the working region and rewards in-
    # distribution fit -- nothing measured out-of-period DECAY, so more search ->
    # higher sealed -> WORSE private (v11 sealed 0.111 -> 0.0897 BEST; v25 sealed
    # 0.133 -> 0.0783). The governor does NOT hard-code simplicity (that would
    # lose on a capacity-friendly dataset); it MEASURES this dataset's decay~
    # complexity slope (beta) from every lesson's own oof_corr-wf_corr, and
    # penalizes SHIPPED complexity by lambda=f(beta). beta>0 (DRW) pulls the ship
    # toward the strata that generalize; beta<=0 leaves capacity free. The machine
    # keeps ALL capability and ADJUSTS at runtime. No-op-safe (lambda 0 = v25).
    COMPLEXITY_GOVERNOR: bool = True
    GOV_LAMBDA_SCALE: float = 0.5       # lambda = clip(beta * this, 0, GOV_LAMBDA_MAX)
    GOV_LAMBDA_MAX: float = 0.04        # cap on the complexity penalty (robust-score units)
    GOV_SIMPLE_C: float = 0.40          # a lesson with complexity index <= this is a low-complexity "anchor"
    GOV_ANCHOR_MEMBERS: int = 3         # low-complexity high-walk-forward members reserved into the blend pool
    GOV_MIN_LESSONS: int = 30           # min measured lessons before beta is trusted

    # v27 CROSS-RUN SELF-IMPROVEMENT LEDGER -- learnings distilled each run feed the
    # next as EVIDENCE-SHRUNK priors (governor beta warm-start, per-family/skill
    # generalization lift, survivor warm genomes, decayer anti-priors). No-op with no
    # prior ledger present. Persisted in world_cairn.json["ledger"] + learning_ledger.json;
    # on Kaggle attach the prior run's output so the next run reads it.
    SELF_IMPROVE: bool = True
    LEDGER_SHRINK: float = 1.0          # prior pseudo-count multiplier (higher = trust history more)
    LEDGER_PRIOR_W: float = 0.015       # bandit lift per unit of cross-run generalization track record
    LEDGER_MAX_SURVIVORS: int = 12      # low-decay shipped genomes carried to the next run as warm starts
    LEDGER_MAX_DECAYERS: int = 12       # high-decay motifs carried as cross-run anti-priors (taboo)

    # v27 ANTI-OVERFIT SHIPPING COURT -- a cheap, OUT-OF-SAMPLE-grounded selection
    # hardener distilled from the regime-criticality / overfit-gravity-well /
    # CV-reality-distortion / prediction-crowding ideas (the 100-idea brainstorm,
    # triaged against the measured law: only ADD machinery that REMOVES overfit).
    # Raises the promotion bar where overfit risk is MEASURED; adds NO capacity.
    # Conservative -> near-no-op on a healthy blend. SHIPPING_COURT=False disables it.
    SHIPPING_COURT: bool = True
    COURT_BASE: float = 0.0
    COURT_W_DECAY: float = 0.5         # escape-velocity weight on out-of-period decay (oof-wf)
    COURT_W_RDI: float = 0.3           # ... on CV reality-distortion (oof - worst-world floor)
    COURT_W_CPLX: float = 0.04        # ... on governor complexity
    COURT_W_CROWD: float = 0.2         # ... on excess loading of the condensation axis
    COURT_PENALTY: float = 0.5         # weight multiplier for a member that fails escape velocity
    COURT_CRIT_HI: float = 0.15        # residual-autocorr (criticality) threshold for "regime critical"
    COURT_CRIT_SHRINK: float = 0.30    # shrink the blend toward equal-weight when critical

    # v24 GENERALIZATION LEVERS -- all no-op-safe, judged by the robust selector,
    # cannot form a monoculture (v23 family cap) or ship unless robustly better.
    STABSEL_BOOT: int = 12              # stability selection: L1 bootstrap subsamples (Meinshausen-Buhlmann)
    STABSEL_POOL: int = 256            # pre-screen to top-N by |corr| before stabilizing (cost guard)
    STABSEL_ALPHA: float = 0.001       # Lasso penalty in the bootstrap fits
    IRM_SIGNFLIP_PENALTY: float = 1.0  # invariant-risk: penalty on per-environment slope sign flips
    SHIFT_CLIP: float = 5.0            # adversarial covariate-shift importance-weight clip
    CPCV_GROUPS: int = 6               # combinatorial purged CV: segment groups
    CPCV_TEST_GROUPS: int = 2          # test groups per CPCV path
    CPCV_MAX_PATHS: int = 12           # cap on CPCV partitions added to the robust selector

    # v28 PRIVATE-LB-GOLD selection hardeners (from the published 4th-place
    # recipe; both add ZERO capacity -- one re-RANKS inputs, one adds robust-
    # selector partitions -- so they cannot feed the measured complexity ratchet).
    SIGNSTAB_FAMILY: bool = True       # 'sign_stability' ranker family: demote features whose corr-sign flips across segments
    SIGNSTAB_MAX_FLIP: float = 0.25    # max fraction of segments whose corr sign may disagree with the pooled sign
    ROBUST_INTERIOR: bool = True       # interior-block partitions (train oldest+newest, validate bracketed middle)

    # v29 PLS-AS-SELECTOR (kuzn137, private 0.099): rank features by MULTIVARIATE
    # PLS |coefficient| -- usefulness net of collinear copies -- where corr-ranking
    # is univariate and double-counts duplicates. Ranking only; zero capacity.
    PLSRANK_FAMILY: bool = True        # 'pls_weight' ranker family
    PLSRANK_POOL: int = 192            # pre-screen to top-N by |corr| before the PLS fit (cost guard)
    PLSRANK_COMPONENTS: int = 8        # PLS components for the selector fit

    # v30 INITIAL WIDE-PATH BIAS (user-directed): the search currency's WIDTH
    # share starts high and ANNEALS (half-life in lessons) back to the v4 0.5
    # balance -- early exploration prefers WIDE robust trails (high lower-bound
    # strength) over narrow lucky ridgelines, and the run's own measured
    # evidence takes over as lessons accumulate. 0.5 = exact no-op (the
    # historical fitness); the run also MEASURES corr(width, decay) so the
    # ledger can recalibrate this prior from real evidence.
    WIDTH_BIAS_START: float = 0.8      # initial width share of the search currency (0.5 = off)
    WIDTH_BIAS_HALFLIFE: int = 60      # lessons until the extra width bias halves

    # v30.1 WINNER NETWORK (observation only, IDEAS.md 1a): the promoted trails
    # as a graph -- output-corr edges, leader-cluster communities. Feeds the
    # queued network-aware member-selection cap (1b) with measurements first.
    NETWORK_REPORT: bool = True        # write winner_network.csv each run
    NETWORK_MAX_NODES: int = 120       # top promoted lessons (by fitness) in the graph
    NETWORK_EDGE_CORR: float = 0.5     # |output corr| >= this = an edge
    NETWORK_COMMUNITY_CORR: float = 0.7  # leader-cluster radius (one prediction community)

    # v21 FORENSIC REGIME-SCIENCE layer (self-tuning, forward-validated, no-op-safe).
    # Motivated by the v12 monoculture regression: an 8/8 single-family blend
    # looked great in-regime and decayed out-of-regime, invisible to every
    # in-working-region door. The forensic layer MEASURES input-space
    # concentration + worst-world behaviour and lets the FORWARD slice (the
    # CV->forward gap, this dataset's real lever) pick the blend. It OVERRIDES
    # v20's shipping choice only when a regime-aware config strictly beats the
    # incumbent on the forward slice; otherwise it is a strict no-op. Every
    # rejected candidate/action is kept and reported with its measured delta.
    FORENSIC_ENABLED: bool = True
    FORENSIC_DIVERSITY_JACCARD: float = 0.60  # worst-world blend: reject member sharing > this of its cols
    FORENSIC_WORLD_Q: float = 0.10            # distributional floor quantile across worlds
    FORENSIC_MARGIN: float = 0.0015           # forward improvement required to override the incumbent
    FORENSIC_FEATURE_CLUSTERS: int = 8        # feature-cluster sensor (partial dynamics + diversity + why)
    FORENSIC_MIN_WORLD_ROWS: int = 200        # a world (time/terrain/weather/habitat cell) must exceed this
    FORENSIC_INFLUENCE_Q: float = 0.99        # top (1-this) |residual| rows are bad-row suspects
    FORENSIC_ACTIONS: bool = True             # measure repair actions (row-quarantine, regime-split) on forward
    MAX_SEG_NEG_FRAC_KEEP: float = 0.5        # (reserved) kept for forensic row-court alignment
    MAX_OVERFIT_RATIO: float = 6.0
    MAX_TOP_WEIGHT: float = 0.70
    MIN_EFFECTIVE_MEMBERS: float = 1.5
    VOLUME_SLICE_Q: float = 0.70
    WINSOR_QS: tuple[float, ...] = (0.0, 0.001, 0.005)
    USE_FORWARD_GATE: bool = True
    FORWARD_GATE_MARGIN: float = 0.005
    GATE_BOOT_CONF: float = 0.80        # v18: forward gate fires only if single beats blend in >= this frac of bootstraps
    PREDATOR_PALINDROME_FLOOR: float = -0.04  # v18: kill only on a CLEAR time-reversal inversion (conservative)
    PARLIAMENT_HHI: float = 0.03        # v18 parliament: penalty on weight concentration (HHI)
    PARLIAMENT_MAXW: float = 0.03       # v18 parliament: penalty on the single largest weight

    # v26 GENERALIZATION SEAMS -- defaults reproduce DRW EXACTLY (Pearson metric
    # + time-ordered geometry). Set "auto" (or a specific value) to adapt to any
    # dataset: a data-profile sensor writes data_profile.json and, under "auto",
    # picks the metric from the target type and the CV geometry from time-order.
    METRIC: str = "pearson"             # "auto" | "pearson" | "gini" | "spearman" | "rmse"
    GEOMETRY: str = "temporal"          # "auto" | "temporal" | "random"
    PROFILE_MIN_AC1: float = 0.15       # mean feature lag-1 autocorr >= this => temporal (under "auto")


CFG = HarnessConfig()
OUT = Path(CFG.OUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)


