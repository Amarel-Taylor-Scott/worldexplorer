#!/usr/bin/env python3
# ============================================================================
# WORLD-EXPLORER LEARNING HARNESS v25 -- DRW CRYPTO (single Kaggle cell)
# ----------------------------------------------------------------------------
# v25 -- THE DRW-PERFORMANCE BUILD: the full v1-v24 stack (NOTHING removed --
# the v23 anti-monoculture cap + v24 generalization levers stay load-bearing),
# tuned to BREAK THE ~0.09 PRIVATE PLATEAU on DRW before any cross-dataset
# generalization. Two targeted moves, both through the same doors, no-op-safe:
#   * greedy_ols skill -- the published 11th-place 0.111 recipe AS A SKILL:
#     greedy SUFFIX feature selection (add features in viewport order, keep
#     each only if out-of-sample corr improves) + plain OLS. Strongest on the
#     trailing `tail` block; the robust selector ships it only if it wins.
#   * warm genomes refreshed with the v21 REAL-RUN champions -- the tiny-k
#     (7-8) terrace/gpu_ridge_swarm + moire mycelium motif (g_best 0.1449, the
#     highest evolution fitness measured) -- so evolution starts at the peak.
# Data-agnostic generalization (metric/geometry auto-detection) is DEFERRED to
# a later version per the "make it work for DRW, then generalize" order.
# ----------------------------------------------------------------------------
# v24 -- THE GENERALIZATION-LEVERS BUILD: the full v1-v23 stack (NOTHING
# removed -- the v23 anti-monoculture spine stays load-bearing) + five new
# levers that attack regime decay and feature noise directly. Each is a
# competing HYPOTHESIS judged by the v22 robust multi-partition selector,
# gated by the overfit door, target-free where possible, and -- under the v23
# viewport-family cap -- unable to form the monoculture that sank v12/v19:
#   * stabsel family   -- STABILITY SELECTION (Meinshausen-Buhlmann): rank
#     features by how often they survive an L1/Lasso fit across bootstrap
#     subsamples (finite-sample false-discovery control -- which of 800 are real).
#   * irm family       -- INVARIANT-RISK selection: keep features whose per-
#     environment SLOPE is invariant across time/terrain/weather, punishing
#     sign flips -- the causally stable signal (the slope twin of 'invariant').
#   * recency_weighted -- exp(-lam*age) RECENCY-weighted ridge, lam on the inner
#     split (the continuous form of the 11th-place monthly models).
#   * shift_linear     -- ADVERSARIAL covariate-shift reweighting: weight train
#     rows by their feature-resemblance to the LATE/future regime (target-free).
#   * CPCV partitions  -- COMBINATORIAL PURGED CV paths added to the robust
#     selector (many backtest paths, not one), + adversarial_validation.csv
#     (early-vs-late AUC drift map). Permanent guardrail kept: NO leaderboard-oracle.
# ----------------------------------------------------------------------------
# v23 -- THE ANTI-MONOCULTURE BUILD. Driven by a MEASURED disaster: the v19
# civilization run scored private 0.0749 (the WORST real run) while posting the
# HIGHEST honest CV (0.175) and HIGHEST sealed (0.129) ever -- the sealed->
# private ladder INVERTED. Cause: the shipped blend was 100% one viewport
# family (mycelium); the pheromone feedback (scent->reuse->more scent) +
# high-capacity gpu_ridge_swarm/foveated formed a monoculture that GAMED the
# in-regime sealed tail and COLLAPSED on the out-of-period private set
# (CV->forward gap 0.046, the biggest ever). This is the v12 failure amplified.
# v23 makes a single-family blend STRUCTURALLY IMPOSSIBLE and biases toward
# robust simplicity -- all through the same doors, no-op-safe:
#   * HARD viewport-FAMILY cap at member selection (MAX_FAMILY_MEMBERS=3) with a
#     floor-backfill -- the biting input-space diversity gate v12/v19 needed
#     (output-corr / texture-family / input-Jaccard caps ALL missed it: 12
#     mycelium viewports looked like "10 trail families").
#   * PHEROMONE SATURATION -- sqrt-saturate scent + let |corr| co-rank, breaking
#     the runaway monoculture feedback at the source.
#   * `diverse_families` candidate (one best member per family) added to the
#     robust selector, which is now the shipping spine; SELECTION DEFLATION
#     raises the override bar by the number of configs compared.
#   * SIMPLICITY LEVERS toward the 11th-place 0.111 recipe: `huber_linear`
#     (fat-tail-robust) + `elastic_net` (L1 in-fit feature selection) skills,
#     gated by the overfit door, judged by the robust selector.
#   * shape-alchemy now must clear a real forward margin (was a free overfit
#     surface). Permanent guardrail kept: NO leaderboard-oracle anywhere.
# Everything from v1-v22 is intact; nothing pruned. Default TIME_BUDGET_MIN=690.
# ----------------------------------------------------------------------------
# THE ROBUST-SELECTION BUILD: the FULL v1-v21 stack (nothing pruned, nothing
# deferred) + a STRUCTURE-AWARE, MULTI-PARTITION, BOOTSTRAPPED out-of-sample
# selection spine. Runs as-is on Kaggle T4 x2; default budget 11.5 h (690 min).
#
#   v22 ROBUST OOS SELECTION (new shipping spine; resolves "the 0.09 run might
#     have been LUCKY / DRW-specific"). Noisy signals make any SINGLE
#     out-of-sample slice a coin-flip, so v22 does not bet on one. It scores
#     every candidate shipping config across MANY train/test geometries of
#     different MAKEUP -- time windows (expanding / sliding-recent / REVERSED:
#     train late, test early) AND structure-aware splits drawn from the
#     explorer's own discovered map (leave-one-TERRAIN-out, leave-one-WEATHER-
#     out -- "does the signal survive a regime it was never trained on?") --
#     each with a block-BOOTSTRAP confidence interval over segment blocks. The
#     robust score is mean - std ACROSS partitions (stable everywhere, not
#     lucky once); ties within a band are HEDGED (blend the tied configs)
#     rather than bet on one. No-op-safe: the incumbent is one candidate, so it
#     ships unless a config ROBUSTLY beats it. Self-tuning that cannot overfit,
#     because nothing is selected on a slice it was tuned against, and nothing
#     dataset-specific is assumed -- model class (plain OLS vs self-tuned
#     Ridge), feature ORDER (head/mid/tail blocks), and selection are all
#     hypotheses that compete by robust measurement. (Writes robust_oos_*.csv.)
#
#   v22 GENERAL CAPABILITY (so the right answer is REACHABLE on any dataset):
#     + linear_ols skill -- plain LinearRegression, the model class the 11th-
#       place 0.111 used; gated by the same overfit-ratio door so it only
#       ships where an unregularized fit truly generalizes.
#     + head/mid/tail families -- contiguous feature-ORDER blocks (feature
#       order is signal); general, all tested, measurement picks.
#     + every Ridge self-tunes its shrinkage down to near-OLS (alpha grid now
#       1e-2..1e5) when the data supports it -- regularization is measured,
#       not assumed.
#
#   THE FORENSIC BUILD (v21): a self-tuning FORENSIC REGIME-SCIENCE layer
#     (regime passports, gap-as-symptom decomposition, partial-dynamics tensor,
#     support-lattice + sheaf, row-influence court, distributional VOI, expert-
#     dispatch actions, mistake memory). v22 feeds those diagnostics INTO the
#     robust selector (the discovered structure defines the partitions).
#
#   v21 FORENSIC REGIME-SCIENCE LAYER (new; runs in the shipping phase) --
#     a CV/train(forward) gap is a SYMPTOM, not a number. The layer diagnoses
#     it and SELF-TUNES the shipped blend, all validated on the forward slice
#     (the CV->forward gap is this dataset's real lever). It writes the full
#     forensic suite -- regime_change_passports (where/when/why a block boundary
#     shifted; complete vs partial), cv_train_gap_passports (per-member gap
#     DECOMPOSED across time/terrain/weather/habitat worlds + a measured cause:
#     overfit / regime_fragile / forward_decay / no_signal / honest),
#     partial_dynamics_tensor (block x feature-cluster -- which subspace decayed
#     and when), support_lattice_cells + support_sheaf_consistency (super-lattice
#     intersections + incompatible local truths), row_influence_court (are a few
#     bad rows lying?), distributional_voi (worst-world LCB, not a scalar -- the
#     cure for single-metric overfitting), expert_dispatch_log +
#     backward_feedback_actions (repair actions -- row-quarantine, regime-split
#     -- MEASURED on the forward slice), mistake_memory_bank (members that looked
#     strong in-regime and decayed). Then a forward-validated CANDIDATE BAKE-OFF
#     reselects the blend by WORST-WORLD q10 corr under an INPUT-feature-overlap
#     DIVERSITY cap, and OVERRIDES v20's shipping choice ONLY when it strictly
#     beats the incumbent out-of-sample -- a strict no-op otherwise. This is the
#     measured cure for the v12 monoculture (an 8/8 single-family blend that
#     decayed out-of-regime, invisible to every in-working-region door): every
#     mechanism is MEASURED, even the rejected ones are kept and reported with
#     their deltas, and measurement -- not intuition -- decides what ships.
#
#   THE CIVILIZATION BUILD (v19): the full v1-v18 stack (nothing pruned) + a
#   layer that treats the harness as a scientific CIVILIZATION -- representation
#   discovery, governance, falsification, and biodiversity, not just more
#   transforms. The guiding insight (from the design doc): the highest-value
#   additions are the ones that make the crazier ones SAFE. No-op-safe.
#   invariant family          -- the CAUSAL COURT: rank features by signal that
#     survives MANY environments (segments + terrain + weather), mean|corr|
#     minus cross-world instability. Distinct from 'stable' (temporal only):
#     these are the features least likely to break under regime shift.
#   lorentz_boost transform   -- RELATIVITY: a moving-observer mix of level and
#     velocity, boost beta = clipped row volatility (target-free). Calm rows
#     see level (degrades to doppler); storm rows see velocity-boosted level.
#   echo_state_ridge skill    -- RESERVOIR COMPUTING: a random, frozen recurrent
#     reservoir (echo-state) with a LINEAR ridge readout -- cheap temporal
#     memory that cannot overfit like an RNN (only the readout is fit).
#   prediction shape alchemy  -- audition output remaps (rank/power/tanh) on the
#     forward blend; financial labels often reward ORDER over amplitude.
#     Forward-chosen, DEFAULT raw (no-op).
#   many_worlds_cv report     -- per shipped member, the corr FLOOR across every
#     time/terrain/weather world + frac of worlds it helps in. A member strong
#     on average but deeply negative in one world is a private-LB landmine.
#   map_elites_archive report -- best lesson per behavioral niche (family x
#     transform x k x overfit), so we can SEE biodiversity vs monoculture.
#   (Queued for v20, door-safe, from the same doc: wormhole/graph-analog memory,
#   rashomon centroid, bayesian-bootstrap blend, elder council / candidate
#   judge, symbolic alpha grammar, adversarial-splitter + state-permutation +
#   gauge + twin-earth predators, constitutional MoE, risk-parity blend.)
#   DO-NOT-ADD (honored): leaderboard-oracle / public-LB label inference.
# ----------------------------------------------------------------------------
# THE FABRIC BUILD: the full v1-v17 stack (nothing pruned) + a SPACETIME-FABRIC,
# MODEL-DEMOCRACY, and TRUTH-SERUM batch. Everything here still enters through
# the same doors (draft -> dual-geometry -> forward gate -> predator -> sealed)
# and defaults to a no-op when it does not help. The additions:
#   random_fourier transform  -- a Monte-Carlo RBF feature map: ridge on cos(Wz+b)
#     approximates KERNEL ridge (curved feature-space capacity, winning family).
#   curvature transform       -- the SECOND causal difference (acceleration, the
#     dual of doppler's velocity): where the world's motion is itself changing.
#   minimax_era strategy      -- a nested-ensemble competitor that maximizes the
#     WORST per-segment corr (antifragile: breed for the decay regimes).
#   parliament strategy       -- ANTITRUST for the blend: corr minus penalties on
#     weight concentration (HHI) + the single largest weight; one model ships
#     dominant only if its edge outweighs the democracy tax. Competes honestly.
#   chorus_shrinkage          -- scale the shipped blend per-row by member
#     AGREEMENT (big calls only where the committee concurs); beta chosen on the
#     forward slice, DEFAULT 0 (no-op unless it helps).
#   forward-gate error bars   -- the captain-picking gate now requires BOOTSTRAP
#     significance (single beats blend in >= GATE_BOOT_CONF of segment resamples),
#     not a coin-flip point margin -- a noisy gate can undo the whole search.
#   palindrome predator       -- refit on TIME-REVERSED training; momentum
#     mirages invert, structure survives. Kills only on a clear inversion.
#   label_archaeology report  -- train-side forensics: is the anonymized y a
#     feature's FUTURE value? Sweeps lagged corr + y autocorrelation. Gates
#     nothing; it is a map.
#   (Deferred to a verified v19, from the same design doc: metric-tensor
#   viewports, lorentz_boost, wormhole memory, rashomon centroid, bayesian-
#   bootstrap blend, CV tensor ledger, elder council, colony distill, the
#   gauge/state-permutation/twin-earth predators -- all door-safe, queued.)
# ----------------------------------------------------------------------------
# THE SILICON BUILD: the full v1-v16 stack (nothing pruned) + a GPU-UTILISATION
# layer. MEASURED FACT behind it: in EVERY real run the ONLY thing touching the
# two T4 GPUs is the torch MLP, and the MLP has NEVER promoted (overfit 11-56x);
# GBDT is pinned to CPU by the crash fix. So the entire shipped blend, every
# time, is RIDGE-FAMILY linear models computed in microseconds on CPU -- two
# T4s and 11 hours sitting ~95% idle. v17 puts that silicon to work IN THE
# WINNING FAMILY. Both new skills enter through the same honest doors and have
# a hard CPU fallback (a GPU OOM degrades, never crashes -- a crash = lost
# submission, the catastrophe the shipping reserve exists to prevent):
#   LINEAR_PEARSON skill   a single Linear layer trained with the DRW 1st-place
#     loss (1-w)*MSE + w*(1-corr) on GPU, with the v13 causal early stop. The
#     MLP fails because it is NONLINEAR, not because the loss is wrong; this
#     pairs the winning loss with the winning (linear) model class. RidgeCV
#     fallback when torch is absent.
#   GPU_RIDGE_SWARM skill  scout_lattice (the v8 champion family) at swarm
#     scale: SWARM_SCOUTS ridge models on random column-subsets solved with
#     on-device linear algebra, verified on the inner time split, survivors
#     merged corr-weighted. numpy full-ridge fallback on any device error.
#   Both route to the gpu lane (hetero-paired with CPU ridge lessons) and are
#   guaranteed a measured audition by the ungated v16 parade.
# ----------------------------------------------------------------------------
# THE MORPHOGENESIS BUILD: the full v1-v15 stack (nothing pruned) + the
# PATTERNS-IN-NATURE catalog (Wikipedia's taxonomy: symmetry, trees/fractals,
# spirals, meanders, waves, foam, tessellations, CRACKS, spots/stripes). Most
# of that taxonomy is ALREADY load-bearing here; v16 adds the four that were
# missing, each a new way to SEE. Same honest doors throughout.
#   NATURE-PATTERN CENSUS -- already present: symmetry=fold_abs/fold_pairs;
#     trees=GBDT; meanders=relay_caravan+doppler; waves/dunes=swell_rider+tide;
#     foam/Voronoi/tessellation=codebook+terrain; sensitivity=stability_probe.
#   MORPHOGENESIS -- the missing four (v16):
#     FRACTAL transform (trees)      self-similar 3-scale pyramid: each feature
#       at level + EMA-8 + EMA-32 concatenated (coarse & fine at once, 96 bits).
#     PHYLLOTAXIS family (spirals)   golden-ratio seed packing: order by |corr|
#       then select by a low-discrepancy golden stride -- coverage over the
#       whole spectrum, the spiral dual of greedy 'decor'.
#     FAULT family (cracks)          rank by the largest discontinuity in per-
#       segment corr between ADJACENT segments -- the regime BREAKS themselves,
#       surfaced for a regime-aware skill. The complement of stable/springs.
#     REACTION_DIFFUSION transform (spots/stripes, Turing)  activator (EMA-4)
#       minus inhibitor (EMA-32): the band-pass morphogen where standing-wave
#       patterns live (distinct from tide's high-pass).
#   v15 beacon layer (items dropped at unique typologies emit an RBF field that
#       becomes leak-free feature channels); v14 bio-ecology kernel below:
# ----------------------------------------------------------------------------
# THE BIO-ECOLOGY BUILD: the full v1-v13 stack (nothing pruned -- the novel,
# creative, and strange components are load-bearing) + an ECOLOGY KERNEL.
# The thesis (the user's, 2026-06-10): stop treating explorers as reasoning
# PERSONAS and treat them as living SENSING + LEARNING ORGANISMS -- they
# should sense like bats, move like bacteria, share like microbes, prune like
# slime mold, defend like immune systems, and remember like a seed bank.
# Much of that biology is ALREADY load-bearing here (see the LIVING CENSUS
# below); v14 adds the missing primitives that fit the honest doors and
# attack the two MEASURED enemies -- regime decay and correlated error.
# Every mechanism still enters through the same honest doors: draft gate ->
# dual-geometry width -> predator -> sealed silence.
#
#   LIVING CENSUS -- biology ALREADY in the harness (do not rebuild):
#     bacterial chemotaxis      = op_chemotaxis (run-and-tumble evo operator)
#     horizontal gene transfer  = op_plasmid + GENE_POOL (lateral motif jump)
#     quorum sensing            = QUORUM (>= K species switch a family on)
#     bee waggle dance          = DANCES + newborn recruitment
#     ant pheromone / stigmergy = MYCELIUM (promoted trails deposit on columns)
#     starling flocking         = op_flock (topological-neighbor alignment)
#     immune clonal+negative    = PredatorEngine + TABOO venom memory
#     bat/electric active sense = stability_probe (perturb, read the echo)
#     dreaming / replay         = dream_replay (block-bootstrap promoted trails)
#     circadian clock           = the v10 governor + v12 metabolic windows
#     metabolic ATP / MVT       = lesson costs + attention market (yield/cost)
#     threat learning           = TRAPS (mirage features marked before walking)
#
#   ECOLOGY KERNEL -- the new primitives (v14):
#     TYPED PHEROMONE (ant)      mycelium was one scalar (attraction). v14
#       makes the trail chemistry TWO-CHANNEL: a GREEN attractant (promoted
#       trails, as before) and a RED REPELLENT deposited by predator KILLS and
#       trap MIRAGES on the exact columns they used. Corr-driven family
#       rankings now subtract the repellent -- explorers are warned away from
#       poisoned ground, not just drawn to rich ground. Generalizes traps +
#       venom into the stigmergic substrate. red_pheromone_report.csv.
#     JAMMING AVOIDANCE (bat)    bats shift frequency when rivals crowd their
#       band; a blend of members that all read the SAME input columns
#       hallucinates consensus and fails together. v14 caps the INPUT-space
#       overlap of shipped members (Jaccard of used columns >= JAMMING_JACCARD
#       skips), a sharper filter than output-corr and texture-family alone --
#       it fights correlated error at the SOURCE, not just the prediction.
#     SEED BANK (temporal biodiversity)  warm starts carried only WINNERS, so
#       a strategy that lost to this run's champion was forgotten -- exactly
#       the diversity a regime shift later rewards. v14 writes the run's best
#       measured LOSERS (positive-corr non-champions) into the cairn's seed
#       bank; the next run GERMINATES a few as extra warm genomes (re-measured
#       through the doors). Anti-catastrophic-forgetting across runs.
#     ISLAND EVOLUTION (allopatry) one panmictic population converges early.
#       v14 makes each evolutionary EPOCH an ISLAND with its own operator
#       bias -- pioneer (levy/woa explorers), exploiter (frontier/gwo),
#       recombinator (plasmid/de/flock) -- and MIGRATION between epochs is the
#       existing re-seed from the full library (every island's champions flow
#       in). Allopatric diversity with measured gene flow.
#     ANTI-FLOCK SCOUT (contrarian)  most explorers chase the leaders; a few
#       must walk AGAINST the crowd to catch a regime turn before consensus
#       does. op_antiflock builds a child by pushing AWAY from the population
#       centroid in genome space (rare skill/family/transform, k reflected
#       across the mean). A new evolution operator, the dual of op_flock.
#     CHEMOTAXIS BASELINE (E. coli adaptation)  bacteria chase concentration
#       above the LOCAL background, not absolute. op_chemotaxis now compares a
#       parent pair against the population's MEDIAN fitness (the local
#       baseline) -- it runs while beating the room, tumbles when merely
#       average, so it cannot get trapped in a globally-mediocre basin.
#
#   --- v13 SENSORIUM (perception layer, all still active) ---
#   v13 grew new SENSE ORGANS and a richer LANGUAGE for what they perceive:
#   new places to look (springs, watershed, echo), new eyes (prism, moire,
#   tide), new gaits to walk what is seen (terrace, rapids), a measured GAIT
#   for every trail, a predator that kills fading trails, an ensemble
#   strategy that stands on the most recent ground, an MLP that finally knows
#   when to stop, and a chronicle that speaks in verbs.
#
#   NEW PLACES TO LOOK (v13 viewport families)
#     springs    : persistence wells -- features ranked by lag-1 self-
#                  autocorrelation x |corr|. Slow geology, not fast weather:
#                  a spring that flows today flowed yesterday too.
#     watershed  : valley specialists -- features ranked by how much their
#                  BEST single-terrain |corr| exceeds their pooled |corr|.
#                  The exact complement of 'weather' (which wants all-band
#                  robustness): watershed wants the expert of ONE valley --
#                  terrain_router's natural diet.
#     echo       : features ranked by |corr(x_t, y_{t-1})| on the training
#                  fold -- columns still ringing with YESTERDAY'S outcome.
#                  The model maps x->y as always; only the RANKING listens
#                  backward (fold-local; no test-time y is ever needed).
#   NEW EYES (v13 transforms)
#     prism      : refraction -- each feature split into three spectral
#                  bands (x; x where x<=q33; x where x>=q66; train-fold
#                  quantiles): piecewise-linear light for linear skills.
#                  96 bits/feature, so the bit-budget frontier pushes prism
#                  into the tiny-k regime v11 just discovered (k* ~ 42).
#     moire      : interference -- each column paired with itself times the
#                  viewport's OWN row-dispersion (local agitation): regime-
#                  conditional slopes with no gate model. 64 bits/feature.
#     tide       : the slow swell subtracted -- x minus its causal EMA
#                  (TIDE_SPAN): what the water does AGAINST the tide.
#                  doppler hears 1-step velocity; tide sees the long set
#                  (same documented stride caveat as doppler).
#   NEW GAITS (v13 skills)
#     terrace    : terraced fields -- the ridge score is quantile-cut into
#                  TERRACE_BINS steps, each step's shrunken target mean
#                  forms a LUT; prediction = half slope, half steps. The
#                  codebook idea applied to the model's OWN 1-D score.
#     rapids     : two-stage water -- ridge the river, find where it runs
#                  roughest (top-|residual| rows), fit a second ridge THERE
#                  on the residual, re-enter at a weight chosen on the
#                  inner split (0 is allowed: calm rivers keep one stage).
#   GAIT OF A TRAIL (v13)      every trail's per-segment corr profile gets a
#     measured SLOPE ('gait'): ascending trails strengthen toward the
#     present, fading trails weaken. Joins the texture vector, the trail-
#     family clustering, and the chronicle adjectives -- and the doors:
#     + predator attack 6 'fading_trail' (free): a trail whose mean corr
#       over the FINAL 2 segments is clearly negative dies. Regime decay is
#       this dataset's measured enemy; the predator now hunts it directly.
#     + ensemble strategy 'late_era_hill': hill-climbed weights maximizing
#       mean per-segment corr over the MOST RECENT third of segments,
#       competing in the SAME nested honest assessment as every strategy.
#   MLP LEARNS TO STOP (v13)   measured 0 promotions in EVERY real run, with
#     overfit ratios 11-56x: the net trained to the bitter end. The torch
#     MLP gains dropout (MLP_DROPOUT) and CAUSAL early stopping (the last
#     20% of its training rows is a time-ordered validation tail; patience
#     MLP_PATIENCE; best-epoch weights restored). The sklearn fallback
#     enables its own early stopping. Same lane, same doors, same budget.
#   TRANSFORM AUDITIONS (v13)  v12 auditioned every SKILL after swell_rider
#     sat unmeasured for two versions and then won a run; v13 extends the
#     parade to every TRANSFORM (one linear_assoc lesson each at its bit-
#     frontier k) -- a way of seeing that never runs is just as silent a
#     prior as a skill that never runs.
#   TRAIL VERBS (v13)          the chronicle learns verbs: each champion is
#     described by HOW it moves -- rides the swell, follows pheromone,
#     scouts ahead, runs the rapids, terraces the slope, listens for
#     motion, squints through quantization -- plus its gait adjective
#     (ascending / fading). The map of the world deserves a living language.
#
#   METABOLISM (new in v12)    the energy ledger. TIME_BUDGET_MIN minutes are
#     planned as: shipping reserve FIRST (forward gate, sealed audit, final
#     refits, reports -- exploration can never eat it), then each spending
#     phase opens a window sized as a SHARE OF WHAT REMAINS when it starts
#     (marginal-value pacing: a phase that finishes early donates its slack
#     to every phase after it). Phase loops ask allow(phase) between units
#     and stop on the clock. TIME_BUDGET_MIN=0 disables the ledger entirely:
#     fixed v11 pacing, one season, one epoch. metabolism_heartbeat.json is
#     the live ledger. The v10 circadian governor stays armed a few percent
#     PAST the metabolic budget as the shed-cost backstop -- clock to cut,
#     metabolism to feast; two halves of one body.
#   SEASONS (v12)              while the explore window has time, the WHOLE
#     species roster is reborn for another season. Newborns are recruited by
#     the waggle dances of the previous season, walk on its mycelium, vote
#     quorum into its families -- and the dedup caps force every new season
#     into UNWALKED cells of the menu. The colony's social learning finally
#     compounds within a single run.
#   EPOCHS (v12)               the v4, v8 AND v9 real runs ALL ended evolution
#     still climbing (v9: g_best set in the FINAL generation, budget at -8).
#     The metabolism re-fuels evolution in epochs: each re-seeds its
#     population from the full library (seasons included), resets the
#     annealing temperature, and spends a fresh budget -- until the window
#     closes, the cap is hit, or an epoch produces nothing new.
#   AUDITION PARADE (v12)      v9 measured: relay_caravan and swell_rider got
#     ZERO lessons -- never picked by any bandit. Unmeasured is not failed,
#     but a registry entry that never runs is a silent prior. Every skill is
#     now guaranteed one audition lesson (same doors, draft-gated) before
#     UCB free play begins.
#   PERTURB FOR ALL (v12)      v9 measured: the perturbation attack NEVER
#     fired -- it was gated on cost >= DRAFT_MIN_COST while every champion
#     cost 2-3; 10 nulls ran, 4 budget units died stranded. PERTURB_ALL
#     frees the attack for every cost class, the predator wallet grows to
#     afford null + perturb on every target, and raids stop on the clock.
#
#   TROPHIC ONTOGENY (replaces human stages)  the developmental ladder is no
#     longer infant/child/adolescent/adult but a SENSORY-RANGE ascent that is
#     what the ladder always mechanically WAS (k = perceptual reach grows):
#       microbe (k4) -> forager (k16) -> navigator (k64) -> apex (k160).
#     Each SPECIES has its own ontogeny BOUNDS: a bacterium is born a microbe
#     and never grows past forager (it adapts and divides, it does not grow a
#     bigger eye); an apex navigator starts already ranging. birth_stage and
#     max_stage per species; the k-growth machinery is unchanged underneath.
#   THE MENAGERIE (new species personas)  alongside the proven v1-v10
#     personas (kept intact -- their priors are measured), v11 adds organisms
#     whose NAMESAKE LEARNING STRATEGY becomes a real mechanism:
#       bacterium  : chemotaxis + horizontal gene transfer + quorum
#       starling   : topological flocking (murmuration)
#       electric_fish / shark : active electrolocation + lateral-line flow
#       honeybee   : waggle-dance recruitment + quorum decision
#       nutcracker : scatter-hoard caching memory (what-where-when)
#     The mechanisms below are UNIVERSAL (every explorer benefits); the
#     species personas add tuned priors and activate as N_EXPLORERS rises.
#   CHEMOTAXIS (bacterial run-and-tumble)  a new evolution operator: bacteria
#     cannot sense a spatial gradient (too small) so they sense TEMPORALLY --
#     run straight while things improve, tumble (randomize) when they stall.
#     op_chemotaxis steps k along the population's fitness gradient and EXTENDS
#     the run while improving, tumbling transform/family when it is not. A
#     temporal-difference local search no other operator expresses.
#   HORIZONTAL GENE TRANSFER (plasmid)  bacteria do not only inherit by
#     descent -- they swap plasmids (working genes) laterally. op_plasmid
#     builds a child by grafting a high-fitness MOTIF FRAGMENT (skill, or
#     transform, or k) from ANY library lesson (the GENE_POOL) onto a current
#     parent. A proven gene jumps straight across the population.
#   QUORUM SENSING (bacterial / bee democracy)  pheromone (mycelium) is a
#     gradient; quorum is a SWITCH. When >= QUORUM_MIN DISTINCT species
#     promote lessons in the same family, the colony "switches on" that
#     family -- a collective prior boost that no single explorer could vote.
#     quorum_report.csv records which regions reached consensus.
#   WAGGLE DANCE (honeybee recruitment)  a bee that finds a rich patch returns
#     and DANCES its direction + quality; recruits pile on proportionally. A
#     promoted lesson whose width clears WAGGLE_MIN posts a dance (its genome
#     region + quality); each NEWBORN explorer is recruited toward the best
#     current dance, biasing its first picks. Active, directional, quality-
#     weighted recruitment -- distinct from passive pheromone.
#   FLOCKING (starling murmuration)  a new evolution operator: starlings track
#     ~7 nearest neighbors TOPOLOGICALLY (not metrically) and align -- emergent
#     order, no leader. op_flock aligns a child toward the centroid of the
#     recipient's k nearest neighbors in GENOME space. Distinct from PSO/GWO
#     (which chase the global best); flocking is purely local and leaderless.
#   COMPASS (bird multi-sensor navigation)  migratory birds cross-reference a
#     magnetic compass, a sun compass, and a star compass and trust the
#     CONSENSUS. The 'compass' viewport family ranks features by agreement
#     across the three target-free frames already in the harness: terrain
#     separation (atlas), weather robustness (gauge), and time stability
#     (per-segment). Features all three compasses point at are the true north.
#   LATERAL LINE (fish flow sensing)  a fish feels the water move along its
#     body -- near-field FLOW, not far-field sight. The 'lateral_line'
#     transform replaces each feature with its deviation from the local
#     consensus of its most-correlated neighbor features: you sense the EDDY,
#     the divergence from the school, rather than the absolute position. A new
#     information class (neighbor-relative), the spatial-in-feature-space twin
#     of v10's doppler (neighbor-relative-in-time).
#   ELECTROLOCATION (shark / mormyrid active sensing)  sharks sense prey by
#     bioelectric field; electric fish emit their OWN field and read the
#     distortion. Folded into the bacterium/electric_fish species' use of the
#     existing JND active-probe + a sensitivity prior: rank stability under
#     self-generated perturbation (already measured by the stability probe).
#   IMMUNE / ANT (already native)  the predator IS clonal selection + negative
#     selection (kill the self-binding false positives); mycelium IS ant
#     stigmergy. v11 names them in the phylogeny and leaves them intact.
#
#   THE VEHICLE FLEET (new in v10)  modes of transport = resolution profiles
#     (time-resolution x bit-resolution x terrain access). The fleet:
#       satellite  : pre-phase orbital survey -- every family scanned at
#                    quantize2 / frontier-k / strided rows / 2 folds. Output
#                    is survey_map.csv (signal density per family), which
#                    feeds the bandit's family priors. Sees geography, not
#                    detail.
#       airplane   : the existing successive-halving draft (20k-row tile,
#                    2 folds) -- now named for what it always was.
#       car        : NEW middle rung for expensive skills (cost >= 4): after
#                    the airplane pass clears the gate, a coarse-time pass
#                    (CAR_ROWS rows, CAR_FOLDS purged folds) must keep width
#                    positive ('car_stalled' culls) before paying full hike
#                    price. Cars are fast on mapped terrain, useless off-road.
#       hike       : the full dual-geometry lesson. Unchanged. Slow, any
#                    terrain, every sense available.
#       submarine  : DIVE PHASE -- after evolution, the provisional champion
#                    defines the visible surface; submarines run lessons on
#                    the fold-honest RESIDUAL (y - slope*champion_oof),
#                    prospecting what the surface fleet CANNOT see. Dive
#                    promotions pass the same doors (scored in their own
#                    residual world), enter the member pool under a capped
#                    quota (DIVE_MEMBER_CAP) and bypass only the seg-negative
#                    filter (their value is incremental by construction --
#                    the nested stacker and forward gate judge the blend).
#   PANORAMA RITUAL            before any explorer is born: one 1-bit,
#     all-features majority_vote lesson -- the 360-degree freeze-and-orient
#     scan that establishes the floor and the horizon.
#   TRAP MAP (threat-first)    fear scans before hunger: a pre-phase pass
#     marks MIRAGE features (high full-sample |corr|, high fold-to-fold sign
#     flip rate). Corr-driven families demote traps to the back of their
#     rankings; trap_map.csv reports them. One-trial threat learning before
#     any budget is spent walking toward water that is not there.
#   DOPPLER SENSE (transform)  the harness gains HEARING/motion perception:
#     'doppler' appends causal row deltas (Z_t - Z_{t-1}, first row zero) --
#     the VELOCITY of the world, a new information CLASS, not a new view of
#     the same class. CAVEAT (documented, measured): on the strided probe the
#     deltas are coarser than the minute deltas the shipped refit sees; the
#     forward holdout + gate catch any measured-vs-shipped drift.
#   CROSS-SENSE GAP (free)     every lesson now perceives with TWO senses:
#     Pearson (sight) and Spearman (rank-sense). The |gap| is stored per
#     lesson, joins the texture vector, and a shipped member whose gap ratio
#     is extreme raises a health alarm -- tail-driven alpha LOOKS strong to
#     one sense and dies on regime change.
#   ATTENTION MARKET (MVT)     Charnov's marginal value theorem: explorers
#     each spend LESSON_BUDGET, then a held-back ATTENTION_POOL is granted to
#     the explorer with the highest recent marginal yield (fitness per cost
#     of its last lessons) -- attention flows to whoever is still learning.
#     learning_rate_curve.csv reports cumulative width-per-cost over time:
#     the run's information learning rate, measured.
#   SURPRISE CURIOSITY         predictive coding: the library PREDICTS each
#     candidate's width from genome-space neighbors before it runs; the
#     bandit pays an exploration bonus toward coordinates whose recent
#     |predicted - measured| (surprise EMA) is high -- attention goes where
#     the map is wrong. Lesson.surprise lands in the ledger.
#   SENSORY THRESHOLD (JND)    active experimentation: plant synthetic alpha
#     of known strength into a COPY of the labels (shuffled-y + s*z(feature))
#     and measure whether the standard draft pipeline DETECTS it, for a grid
#     of s. sensory_threshold.json reports the psychometric curve and the
#     just-noticeable difference: the harness's measured information
#     processing floor at current budget. Pure calibration; gates nothing.
#   VENOM MEMORY (taboo)       one-trial fear learning, asymmetric by design:
#     predator kills deposit taboo penalties on the killed motif's
#     (skill|transform) and (skill|family) coordinates; the bandit subtracts
#     TABOO_W * penalty and evolution skips candidates whose taboo load is
#     extreme. Reward still needs repetition; fear learns ONCE. The predator
#     now raids TWICE (after phase 1 and after evolution) so venom shapes the
#     later search, like an animal stung early in the day. Killed lessons no
#     longer seed the evolution population (a measured kill, not intuition).
#   CIRCADIAN GOVERNOR         the body has a clock and a deadline. When
#     RUN_DEADLINE_MIN is set, phase boundaries compare elapsed fraction to
#     schedule; behind-schedule runs gracefully shed cost (seed reps, then
#     generations, then dream replays, then dive budget), every cut logged.
#     A Kaggle timeout is death with no submission; the governor exists so
#     the organism always makes camp before dark.
#   CHAMPION ABLATION          vary-one-thing-at-a-time on the final
#     champion (transform->identity, family->top, k->k/2): the controlled
#     experiment that attributes the champion's edge to its parts.
#     champion_ablation.csv.
#   CAIRNS (cross-run memory)  world_cairn.json fingerprints this world
#     (gauge edges, terrain populations, symmetry counts, trap count,
#     champion); if a previous cairn is found (CAIRN_PATHS), the run reports
#     how much the world CHANGED since the last visit -- drift detection
#     across runs, culture beyond warm genomes.
#   PERIPHERY FAMILY           the eye gets a periphery: features ranked by
#     |corr| SHIFT between the early 75% and late 25% of the training fold,
#     discounted by mycelium pheromone -- movement in regions nobody is
#     fixating. Motion in the corner of the eye earns a saccade.
#
#   WEATHER SYSTEM (new in v9) target-free, ROW-LOCAL volatility states --
#     the temporal twin of v8's terrain. WeatherGauge measures each row's
#     instantaneous dispersion (mean |z| over market/gauge columns; row-local
#     so it is order-free and leak-free on any slice) and quantile-bins it
#     into calm / mid / storm states. Provides:
#       + 'weather' viewport family : features ranked by their WEAKEST
#         per-state |corr| -- alpha that survives storms (the two-clock
#         robust-intersection idea applied to weather instead of time).
#       + predator 'dead_weather'   : free fifth attack -- a lesson clearly
#         harmful inside one whole weather band is killed.
#       + weather_moe ensemble      : per-state blend weights shrunk toward
#         the global weights by state population, competing in the SAME
#         nested honest assessment as every other strategy; if it wins, the
#         shipped blend is weather-conditional at test time.
#       + per-state texture channel (weather_spread) + chronicle entries.
#   MYCELIUM (stigmergy made real)  promoted lessons DEPOSIT PHEROMONE on the
#     exact columns their full fit used; the 'mycelium' viewport family ranks
#     features by accumulated pheromone (falling back to |corr| while the
#     network is young). Explorers literally follow each other's trails now
#     -- the terrain_surveyor persona's stigmergy finally has a substrate.
#     Self-reinforcement is bounded by the usual doors: dedup caps, draft
#     gate, predator, and the uniqueness penalty.
#   SHADOW FAMILY (negative space) features with HIGH variance but LOW |corr|
#     -- the big quiet regions of the world. Almost always nothing... but the
#     cost of looking is one lesson, and nobody else is looking there.
#   RELAY CARAVAN (skill)          the measured enemy of this dataset is
#     regime decay (honest CV overstates sealed by ~50%). relay_caravan walks
#     the training fold segment by segment, each block's ridge coefficients
#     shrunk toward the previous block's (a random-walk-over-coefficients
#     model); the prediction is made from the caravan's FINAL position. The
#     shrinkage strength tau is chosen on the inner time split.
#   SWELL RIDER (skill)            fits the EMA-smoothed label (spans from
#     SWELL_SPANS, chosen on the inner split scored against the RAW label)
#     -- ride the swell underneath the chop. span=1 degrades to plain ridge.
#   DUAL EXPOSURE (transform)      rank(5 bits) + quantize4(4 bits) of the
#     SAME features concatenated -- one eye for order, one eye for magnitude
#     (9 bits/feature on the bit-budget frontier).
#   DREAM REPLAY (free)            every promoted trail is block-bootstrapped
#     (segments resampled with replacement, DREAM_REPLAYS times) into the
#     runs the world COULD have shown us; dream_p05/dream_p50 land in the
#     ledger and a shipped member whose 5th-percentile dream is negative
#     raises a health alarm. Reports describe; doors stay fixed.
#   FRONTIER SURF (evolution op)   fifth operator: copy the best parent and
#     push k to the bit-budget boundary for its transform (k* = BUDGET/bits,
#     jittered) -- the v4 champion lived near the frontier edge; this operator
#     exploits that measured prior.
#   TEXTURE -> GENERALIZATION      meta-report: across all dual-geometry
#     lessons, how each texture dimension correlates with walk-forward drift
#     (oof_corr - wf_corr) and width -- the harness studying which kinds of
#     trails keep their promises. Purely descriptive.
#   WORLD CHRONICLE                world_chronicle.md -- a written, human-
#     readable story of the run: births, graduations, champions and their
#     texture words, terrain and weather maps, predator kills, dream omens,
#     the sealed verdict. The map becomes a narrative artifact.
#
#   TERRAIN ATLAS (new in v8)  target-free MiniBatchKMeans regimes over
#     market + high-variance columns, fit on WORKING-region X only (y never
#     touches it -> terrain ids are leak-free everywhere, including sealed
#     rows and the test set). Provides: terrain ids ("which valley is this
#     minute in"), altitude (distance to own terrain centroid -- "how far up
#     the mountain"), and f_rank (features ranked by how strongly they
#     SEPARATE the terrains -- the mountains' defining minerals -> the new
#     fully-leak-free 'terrain' viewport family).
#   PATH TEXTURE SUB-MODELS    every positive lesson's OOF trace is described
#     post-hoc, the way a hiker reads a trail: roughness (dispersion of the
#     per-segment corr profile), SIDE TEXTURES (residual std above vs below
#     the path + their log-ratio tilt -- what the terrain looks like on each
#     side of the trail), residual lag-1 WAKE (structure the path failed to
#     model trailing behind it), steepness affinity (does it bet bigger where
#     |y| is large), per-terrain altitude profile, and a per-terrain trail
#     report. Trails are then leader-clustered into TRAIL FAMILIES in
#     z-scored texture space; member selection caps how many blend members
#     may share one family (two decorrelated paths can still FAIL the same
#     way), and a HealthMonitor alarm fires if the shipped blend is
#     single-family.
#   FOLDING & SYMMETRY         + fold_abs  : global space folding -- z-score
#       then |.|, identifying x with its mirror image. The exact dual of
#       sign_only: sign_only keeps direction and discards magnitude, fold_abs
#       keeps magnitude and discards direction (the even-response detector).
#     + fold_pairs: regional folding -- discover the most ANTI-correlated
#       feature pairs on the training fold (mirror planes, e.g. bid vs ask
#       pressure) and append (a-b)/2 (the coherent axis where mirror pairs
#       reinforce) and |a+b|/2 (the folded even channel) to the z-scored base.
#     + symmetry_field_report.csv: per-feature even-vs-odd response field --
#       corr(y, z) vs corr(y, |z|); even-dominant features are the measured
#       motivation for fold viewports. DIAGNOSTIC ONLY -- gates nothing.
#   DEEPER QUANTIZATION        + quantize2 (4 levels, 2 bits) completes the
#     precision sweep 32/8/6/4/2/1 under the BIT_BUDGET frontier; + codebook
#     skill: vector quantization of SPACE itself (k-means codebook on the
#     viewport, shrunken per-centroid target means -- a learned LUT).
#   TERRAIN-AWARE SUB-MODELS   + terrain_router skill: per-terrain ridge
#     experts shrunk toward a global ridge by terrain population (regime
#     mixture-of-experts, atlas ids are target-free so fully fold-honest);
#     + steepness_gate skill: a second sub-model predicts |y| ("where are
#     the mountains") and the direction prediction is scaled by the z-scored
#     predicted steepness (clipped) -- bet bigger where the ground is steep.
#   SPECULATIVE PATH LATTICE   + scout_lattice skill: M cheap scouts, each
#     drafting a path through a DIFFERENT random sub-viewport (a third of the
#     columns on a third of the rows), then VERIFIED on the inner time split;
#     scouts clearing SCOUT_ACCEPT merge corr-weighted; if no scout survives
#     the skill falls back to the full-viewport ridge. "Small view when
#     stable, larger view when fragile" -- coverage 1 - prod(1 - r_m) from
#     scout diversity.
#     Together with drafts (speculate-then-verify), WARM_GENOMES (the
#     precomputed path library), and path width (the stability signal that
#     decides when to widen), the lattice idea is now executable end to end.
#   PREDATOR TERRAIN ATTACK    free fourth attack: worst per-terrain corr
#     from the stored OOF; a lesson strongly negative in a populated terrain
#     is killed ('dead_terrain').
#
#   MEMORY DOCTRINE (new in v8)  memory is managed by measurement, like
#     everything else: mem_status() logs host RSS + per-GPU reserved bytes at
#     every phase boundary; each lesson releases its device tensors and
#     empties the CUDA cache (free_gpu_mem) so long runs cannot ratchet; the
#     probe matrix is freed after its last consumer (the predator) and the
#     full matrices after the submission is written.
#
#   HETEROGENEOUS LANES (v7, lane contents corrected in v8)
#     Every skill is classified into a lane:
#       gpu lane : torch MLP ONLY (v8 -- see the measured-crash note above
#                  the GBDT_BACKEND selection: xgboost-cuda in a lane thread
#                  aborted the whole process on the 2026-06 T4 x2 run)
#       cpu lane : ridge, voting, ladders, kNN, HGB, ALL GBDT, primitives,
#                  codebook/router/steepness/scout
#     PHASE 1   : at each explorer turn the bandit makes its normal UCB pick,
#                 then a SECOND pick constrained to the opposite lane; both
#                 lessons run concurrently in threads (torch/xgboost release
#                 the GIL on GPU work; sklearn/numpy release it inside BLAS/
#                 OpenMP). The bandit sees results one step late -- standard
#                 delayed-feedback bandit, accepted for ~2x throughput.
#     EVOLUTION : each generation's offspring are generated first, then
#                 evaluated in lane-paired concurrent batches; SA acceptance
#                 and operator credit stay sequential and deterministic in
#                 batch order.
#     Stacked with v6's within-lesson concurrency, the maximal state on a
#     T4 x2 box is: CPU cores fitting a ridge/vote lesson while cuda:0 runs
#     an MLP's main CV and cuda:1 runs its seed-repetition CV.
#     HONEST NOTES: (1) hetero mode trades strict run-to-run determinism for
#     throughput (draft-gate bar and bandit state depend on completion
#     interleaving; per-lesson seeds remain deterministic, and seed_var still
#     measures stochastic skills). (2) With 0 GPUs there is no pairing --
#     lanes collapse to cpu and behavior is sequential v5/v6. Set
#     HETERO_PAIRING=False to force sequential on GPU boxes for A/B runs.
# ============================================================================
#
# LINEAGE (measurements, not vibes):
#   v1  baseline harness REAL-DATA RUN: honest CV 0.08944 (corr_weighted),
#       forward 0.12643 (full tail, pre-sealing), 52 lessons, ~19 min.
#   v4  REAL-DATA RUN (2026-06): honest CV 0.14114 (nnls_stack), forward
#       0.10581, SEALED HOLDOUT 0.09450, 67 lessons / 51 promoted / 0 taxed /
#       0 alarms, ~43 min. +58% honest CV over v1 on identical data; sealed
#       lands in contention range of the known ~0.10-0.11 private ceiling.
#   v8  REAL-DATA RUN (2026-06, T4x2, 41 min): honest CV 0.14411 (nnls_stack),
#       forward 0.11582, sealed 0.10058, private LB 0.08537. scout_lattice =
#       best phase-1 family ever measured (oof 0.1327), 5 of 10 blend slots.
#   v9  REAL-DATA RUN (2026-06, T4x2, 55 min): honest CV 0.14409 (equal_top),
#       forward 0.10797, sealed 0.10899, private LB 0.08653 (best at the time).
#       Honest CV identical to v8; the private gain came from WHAT shipped:
#       quantize4 took 5/8 slots, the weather family 4/8, steepness_gate 3/8,
#       and equal_top dethroned nnls_stack (1/N at the ensemble door).
#   v11 REAL-DATA RUN (2026-06, T4x2, 70 min): honest CV 0.16287 (nnls_stack),
#       forward 0.12400, sealed 0.11088, PRIVATE LB 0.08969 -- PROJECT BEST.
#       The menagerie's first outing: the bacterium farmed the mycelium
#       family, the starling was waggle-recruited INTO it (first recruitment
#       ever fired), evolution swarmed it, and swell_rider -- NEVER PICKED in
#       any previous run -- became g_best (swell_rider|mycelium30_quantize4,
#       fitness 0.1209, wf 0.1438) AND took 0.498 of the shipped nnls blend
#       via its champion-ablation k->half variant (mycelium15_quantize4).
#       7 of 10 members were mycelium-family; tiny-k (15-33) quantize4
#       viewports opened a SECOND champion regime far below the k~100-160
#       frontier; doppler promoted (~0.114); dual_exposure carried 0.32.
#       THE SEALED LADDER IS 3-FOR-3 ON ORDERING: sealed 0.10058 < 0.10899 <
#       0.11088 <-> private 0.08537 < 0.08653 < 0.08969;
#       private ~ 0.79-0.85 x sealed ~ 0.55-0.60 x honest.
#
# MEASURED LEARNINGS FROM THE v9 + v11 REAL RUNS, APPLIED TO v12:
#   * Evolution AGAIN still climbing at exhaustion (third consecutive run:
#     g_best bagged_linear|top135_quantize4 set in the final generation,
#     budget overshot to -8, patience never fired) -> metabolism EPOCHS.
#   * Predator perturbation structurally dead (cost gate excluded every
#     cost-2/3 champion; 10 nulls, 0 perturbs, 4 units stranded) ->
#     PERTURB_ALL + wallet 20 -> 28; raids stop on the clock, not greed.
#   * relay_caravan + swell_rider NEVER PICKED (zero lessons; unmeasured,
#     not failed) -> AUDITION PARADE: every skill gets one measured lesson.
#   * Warm genomes refreshed with v9 champions: scout_lattice|top160_quantize8
#     (best single lesson, oof 0.1362, found by frontier_surf),
#     bagged_linear|top135_quantize4 (g_best), steepness_gate|
#     weather106_quantize4 (the shipped weather x steepness motif).
#     quantize4 is a first-class champion transform now.
#   * scout_lattice|top128_quantize8 reproduced oof 0.1327 EXACTLY across
#     v8/v9: per-lesson determinism holds under hetero lanes.
#   * v11: evolution ended at budget -1 with g_best set in the FINAL
#     generation -- the FOURTH consecutive run still climbing -> EPOCHS.
#   * v11: perturbation never fired again (perturb_w=None on all 20 targets)
#     and raid1's wallet (6) covered only 6 of 10 nulls -> PERTURB_ALL plus
#     raid wallets sized to 2x targets whenever the metabolism is armed.
#   * v11: the champion-ablation k->half variant SHIPPED with the TOP weight
#     (0.498) -- the ablation panel is a champion factory, not just a report.
#   * v11: mycelium only wins AFTER phase-1 deposits thicken the network ->
#     seasons multiply deposits; WARM_GENOMES refreshed with the mycelium /
#     swell champions (re-measured through the same doors at gen 0).
#
# MEASURED LEARNINGS FROM THE v4 REAL RUN, APPLIED TO v7 DEFAULTS:
#   * Evolution's champion: bagged_linear + quantize8 + k=123..138 (oof up to
#     0.1301, gwo_guided, still climbing at budget exhaustion) -> adult menu
#     max_k 96 -> 128, EVOLUTION_BUDGET 40 -> 56, generations 4 -> 6, and the
#     discovered genomes are WARM-STARTED into generation 0 (CFG.WARM_GENOMES)
#     -- the population now learns across runs.
#   * rand_proj redeemed (recency|top96_rand_proj took 0.343 of the shipped
#     blend); quantize8/pca priors raised. v1's "exotica loses" verdict was
#     conditional on the old skill set.
#   * local_interp/kNN never promoted in ANY run (overfit 15-27x, ~80s each):
#     cost raised to 4, priors cut. bin_association early-stage failures:
#     priors cut. market family rejected again: priors stay floored.
#   * Draft economics inverted on real data (57 drafts -> 2 culls, 3.5%):
#     drafts now only tax skills costing >= 4, and DRAFT_ABS_PASS 0.02 -> 0.06
#     (median real draft width was 0.072).
#   * nnls_stack won the ensemble (0.1411 vs best_single 0.1371); the
#     quantized-bagged family has a 2x null floor (0.018 vs 0.009) -- watch
#     the predator's deflated margins on that family.
#   v2  writeup-driven upgrades (medoid/lastN viewports, pair_aug/pca_aug,
#       MLP/bagged-linear/GBDT skills, nnls stacking, regime gate, forward
#       gate, global lesson dedup, adult k capped at 96 after k=64 beat k=160).
#   v3  metaheuristic evolution phase over lesson genomes (DE / GWO / WOA /
#       Levy operators + simulated-annealing acceptance + adaptive operator
#       selection), mapped from the mealpy taxonomy, dependency-free.
#   v4  validation-geometry build (all mechanisms kept intact below):
#   v5  primitive/quantized/foveated/two-clock/predator batch (kept intact):
#   v6  hardware-adaptive execution ("one mathematical objective, many
#       hardware schedules") -- GPU schedule + torch Pearson-loss MLP:
#   v7  heterogeneous lanes -- CPU and GPU work simultaneously:
#   v8  topography layer (terrain atlas, path texture, folding/symmetry,
#       codebook/router/steepness/scout skills) + memory doctrine + the
#       measured v7-crash fix:
#   v9  ecology layer (weather, mycelium, shadow, caravan, swell, dual
#       exposure, dreams, frontier surf, chronicle):
#   v10 embodiment layer (vehicle fleet, panorama, trap map, doppler, cross-
#       sense gap, attention market, surprise curiosity, JND, venom, circadian
#       governor, champion ablation, cairns, periphery):
#   v11 menagerie layer (trophic ontogeny + species; chemotaxis / HGT /
#       quorum from bacteria; flocking + compass from birds; lateral line
#       from fish; waggle dance + quorum from bees):
#   v12 metabolism layer (time-budget energy ledger, seasons, evolutionary
#       epochs, audition parade, perturb-for-all):
#   v13 sensorium layer (springs/watershed/echo families; prism/moire/tide
#       transforms; terrace/rapids skills; gait texture + fading_trail attack
#       + late_era_hill strategy; MLP dropout + causal early stop; transform
#       auditions; trail verbs):
#   v14 THIS FILE -- the ecology kernel above (typed/repellent pheromone,
#       jamming avoidance, seed bank, island evolution, anti-flock scout,
#       chemotaxis baseline). The methodology only ever GROWS; mechanisms are
#       demoted by measurements, never pruned on intuition.
#
#   GPU DETECTION             torch probes CUDA at import. 0 GPUs -> the CPU
#                             schedule, byte-identical behavior to v5.
#                             1-2 GPUs (Kaggle T4 x2) -> the GPU schedule
#                             below. All GPU paths are guarded with CPU
#                             fallbacks, so the same cell runs on any
#                             accelerator setting.
#   TORCH MLP                 mlp_assoc switches from sklearn MLPRegressor to
#                             a torch MLP placed round-robin across cuda:0 /
#                             cuda:1. This is not just speed: the torch path
#                             trains with the DRW 1st-place loss
#                             (1-w)*MSE + w*(1 - Pearson), which sklearn
#                             cannot express. GPU schedule also upgrades the
#                             net to 3 layers (128,64,32), 120k rows, 40
#                             epochs, batch 4096 -- the 1st-place protagonist
#                             at full size instead of the CPU-starved one.
#   GBDT ON CPU (v8)          v6/v7 ran XGBoost with device=cuda; the 2026-06
#                             T4 x2 real run proved that fatal inside a lane
#                             thread (uncatchable C++ thrust abort, kernel
#                             dead at 127s). GBDT is now pinned to CPU
#                             permanently -- it never promoted in any real
#                             run, so the GPU loses nothing that measured.
#   TWO-GPU CONCURRENCY       the one place a single-process kernel gets TRUE
#                             2x: for the torch MLP, the main CV and the
#                             seed-repetition CV run concurrently in two
#                             threads, one per T4 (torch releases the GIL
#                             during GPU compute). HONEST NOTE: aggregate
#                             utilization is NOT 2x -- rank caches, ridge,
#                             voting, ladders are numpy/CPU by design (they
#                             are microseconds, not the bill). The T4s
#                             accelerate exactly the lesson family that
#                             dominates wall-clock: the MLP, ~5-10x.
#   HARDWARE PROFILE LOG      the run logs which schedule it chose; the
#                             study reports stay comparable across substrates
#                             because the validation geometry is unchanged.
#
#   v5 batch below, all still active -- every entrant measured through the
#   same doors (draft gate -> dual-geometry width -> predator/null tax ->
#   sealed silence):
#
#   PRIMITIVE CLUSTER         the data already voted for simple (k=64 ridge
#                             beat k=160; exotic projections culled), so v5
#                             races radical simplicity:
#                             + rank transform      : per-fold ECDF rank, ~6
#                               bits, monotone-invariant -> immune to regime
#                               rescaling and fat tails.
#                             + sign_only transform : sign(x - fold_median),
#                               1-bit features ("is the alpha just direction
#                               agreement?").
#                             + majority_vote skill : k features each vote
#                               sign(fold_corr)*sign(x - median); NO fitted
#                               weights at all (1/N-portfolio logic).
#                             + theil_sen skill     : median-of-pairwise-
#                               slopes single factor; outlier-proof primitive.
#   BIT-BUDGET FRONTIER       quantize4 joins quantize8/rank/sign_only; each
#                             transform has a bits-per-feature cost and every
#                             genome must satisfy k * bits <= BIT_BUDGET, so
#                             evolution searches the wide-coarse vs
#                             narrow-fine frontier with one constraint. The
#                             ledger records bits_per_feature / total_bits so
#                             the frontier is measured, not assumed.
#   FOVEATED VIEWPORT         3-level-of-detail transform: fovea = top-8
#                             features at full precision; periphery = next 64
#                             compressed to 4 PCA components quantized to 8
#                             bits; background = ALL remaining viewport
#                             features collapsed into one corr-weighted
#                             summary column (the mipmap). The graphics
#                             analogy made executable and fold-honest.
#   TWO-CLOCK FAMILIES        + dawn        : features whose recent-clock
#                               (last 25% of the training fold) corr EXCEEDS
#                               their full-clock corr -- a measured bet on
#                               the new regime, judged by walk-forward.
#                             + both_clocks : rank by the weaker of the two
#                               clocks -- the robust intersection.
#   PREDATOR PERSONA          a falsification engine that spends its budget
#                             KILLING promoted lessons instead of finding new
#                             ones (immune-system analog). Attacks per target
#                             (four since v8): sub-period (worst 3-consecutive
#                             -segment corr from stored OOF, free), terrain
#                             (worst populated-valley corr vs the atlas, free,
#                             v8), null probe (full CV on within-segment-
#                             shuffled labels -- this EXECUTES the v4 null
#                             tax), and perturbation (shrunken-viewport draft;
#                             collapse = fragile).
#                             Kills become decision='predator_killed'.
#
#   v4 mechanisms below, all still active:
#
#   DUAL-GEOMETRY PROMOTION   every lesson is scored under BOTH
#                             leave-segments-out CV (stationarity estimate)
#                             and purged expanding walk-forward CV
#                             (deployability estimate). Promotion now requires
#                             positive width under BOTH geometries. Evolution
#                             fitness = 0.5*oof_corr + 0.5*min(width, wf_width)
#                             -- optimize the weaker geometry.
#   ERA-MEAN OBJECTIVE        path width promoted from referee to coach: new
#                             era_mean_hill ensemble strategy chooses weights
#                             by maximizing the MEAN of per-segment corrs
#                             (Numerai-style era objective) on inner folds;
#                             all strategies still scored on outer folds by
#                             pooled Pearson (the competition metric). Ledger
#                             gains era_mean_corr per lesson.
#   NULL TAX                  top promoted lessons re-run once on
#                             within-segment-shuffled labels (same pipeline,
#                             same viewport rebuild on the shuffled fold) ->
#                             pooled |null| distribution; deflated_corr =
#                             oof_corr - q95(null). deflated <= 0 demotes the
#                             lesson ("null_taxed"). This is the
#                             multiplicity-honest number for a run that
#                             examines ~50 candidate lessons.
#   SEALED HOLDOUT            final 10% of rows is quarantined from EVERY
#                             decision: probe, CV, evolution, ensemble
#                             selection, forward gate. It is evaluated exactly
#                             once, after the shipped blend is frozen, and
#                             reported -- never gated on. The forward-drift
#                             slice (last 25% of the WORKING region) remains
#                             the actionable gate; the sealed slice is the
#                             once-per-version truth serum.
#   SUCCESSIVE-HALVING DRAFTS lessons costing >=2 units first run as a draft:
#                             ~20k-row tile, 2 purged folds, reduced iters.
#                             Drafts below the running median draft-width are
#                             culled for 1 budget unit instead of full price
#                             (frustum culling + speculate-then-verify) ->
#                             roughly 2x candidates examined per budget.
#                             Cull rate and draft/full agreement are reported.
#   VIEWPORT RANK CACHE       per-(fold-signature, family) memoization of
#                             feature rankings (the actual repeated cost:
#                             re-ranking ~800 columns per lesson per fold).
#                             The fold signature hashes strided y CONTENT, so
#                             null-tax runs with shuffled labels correctly
#                             rebuild their own rankings. NOTE: the roadmap's
#                             Gram-fusion for ridge is deliberately skipped --
#                             sklearn RidgeCV already SVD-shares across alphas
#                             internally; ranking was the uncached hot path.
#
# Everything remains fold-honest. Dependencies: numpy, pandas, scikit-learn
# (lightgbm/xgboost/scipy optional, guarded). Offline. Loud synthetic
# fallback. Runtime = TIME_BUDGET_MIN (default 330 min ~ 5.5 h, safe inside
# Kaggle's 12 h session; set 60 for v9-parity, 0 for fixed v11 pacing).
# HONEST NOTE on determinism: per-lesson seeds remain deterministic, but
# under a time budget WHICH lessons run depends on the host's speed -- the
# same documented trade hetero lanes already made for throughput.
#
# Outputs in /kaggle/working:
#   submission.csv                    explorer_run_summary.json
#   explorer_lessons.csv              explorer_journal.csv
#   evolution_history.csv             evolution_operator_report.csv
#   predator_report.csv               sealed_holdout_report.json
#   draft_gate_report.json            study_{skill,family,transform,...}.csv
#   ensemble_nested_assessment.json   forward_holdout_report.csv
#   dominance_report.csv              member_regime_stress.csv
#   ensemble_health_alarms.csv        winsorize_audit.json
#   terrain_atlas_report.csv          symmetry_field_report.csv      (v8)
#   path_texture_report.csv           terrain_trail_report.csv       (v8)
#   weather_report.csv                texture_generalization.csv     (v9)
#   world_chronicle.md                                               (v9)
#   survey_map.csv                    trap_map.csv                   (v10)
#   sensory_threshold.json            learning_rate_curve.csv        (v10)
#   champion_ablation.csv             world_cairn.json               (v10)
#   quorum_report.csv                 species_report.csv             (v11)
#   metabolism_heartbeat.json         (v12: the live energy ledger)
# ============================================================================

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, replace, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNetCV, HuberRegressor,
                                  Lasso, LinearRegression, RidgeCV)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---- hardware detection: same objective, schedule reshapes per device ------
try:
    import torch
    HAVE_TORCH = True
    N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    torch = None
    HAVE_TORCH = False
    N_GPUS = 0

try:
    from lightgbm import LGBMRegressor
    HAVE_LGB = True
except Exception:
    LGBMRegressor = None
    HAVE_LGB = False
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    XGBRegressor = None
    HAVE_XGB = False

# MEASURED FATAL CRASH (2026-06, Kaggle T4 x2, v7 real run): XGBoost with
# device=cuda inside a lane thread aborted the WHOLE process with a C++
# thrust::system_error -- 'terminate called', uncatchable from Python, kernel
# dead at 127s. GBDT therefore stays on CPU permanently (it never promoted in
# any real run anyway); the GPU lane is the torch MLP only, whose threaded
# CUDA use is well-behaved.
if HAVE_LGB:
    GBDT_BACKEND = "lightgbm"
elif HAVE_XGB:
    GBDT_BACKEND = "xgboost"
else:
    GBDT_BACKEND = None

_GPU_RR = {"i": 0}


def next_device() -> str:
    """Round-robin across visible GPUs (cuda:0 / cuda:1 on a Kaggle T4 x2 box).
    Spreads memory and keeps both devices warm; 'cpu' when no CUDA."""
    if N_GPUS == 0:
        return "cpu"
    d = f"cuda:{_GPU_RR['i'] % N_GPUS}"
    _GPU_RR["i"] += 1
    return d


def _gpu_names() -> list[str]:
    names = []
    for i in range(N_GPUS):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append(f"cuda:{i}")
    return names


try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    HAVE_PSUTIL = False


def free_gpu_mem() -> None:
    """Release torch's CUDA caching-allocator reservations between lessons so
    long runs cannot ratchet GPU memory upward (v8 memory-management layer)."""
    if HAVE_TORCH and N_GPUS > 0:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def mem_status(tag: str) -> None:
    """Measured memory accounting at phase boundaries: host RSS + per-GPU
    reserved bytes. Memory is managed by measurement, like everything else."""
    host_gb = round(psutil.Process().memory_info().rss / 1e9, 2) if HAVE_PSUTIL else -1.0
    gpu = []
    if HAVE_TORCH and N_GPUS > 0:
        for i in range(N_GPUS):
            try:
                gpu.append(round(torch.cuda.memory_reserved(i) / 1e9, 2))
            except Exception:
                gpu.append(-1.0)
    log("memory_status", tag=tag, host_rss_gb=host_gb, gpu_reserved_gb=str(gpu))

try:
    from scipy.optimize import nnls
    HAVE_NNLS = True
except Exception:
    HAVE_NNLS = False

RUN_START = time.monotonic()


