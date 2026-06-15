# worldexplorer

**Zero‑config, self‑improving, overfit‑resistant tabular ML.** You supply a dataframe (or a path) and the name of the target column. It figures out *everything else* — the metric, the validation geometry, the id columns, the categorical encodings, the compute budget — and runs an entire civilization of bounded explorers that promote only signal that survives out of sample.

```python
import worldexplorer as wx

result = wx.explore(df, target="label")     # the whole API
result.predictions      # id + prediction
result.score            # honest holdout score (if you didn't pass a test set)
result.profile          # everything it auto-detected
result.artifacts_dir    # every CSV/JSON report it wrote
```

or from the shell:

```bash
worldexplorer train.parquet --target label --test test.parquet --out ./out
worldexplorer train.parquet --profile-only      # just show what it detected
```

---

## What "figures out everything" means

`wx.profile(df, target)` (run automatically inside `explore`) detects:

| Decision | How |
|---|---|
| **Target column** | your arg, or an obvious name (`label/target/y/...`), or the column in train but not test |
| **Task + metric** | 2 classes → `gini`; continuous/high‑cardinality → `pearson`; small‑int → rank/`spearman` |
| **Validation geometry** | a datetime column, or mean feature lag‑1 autocorrelation → **temporal** (purged walk‑forward / sealed tail) vs **tabular** (random CV) |
| **Id columns** | unique / monotonic integer / `id`‑named → dropped from features, kept for output |
| **Categoricals** | object / low‑cardinality → label‑encoded (train+test jointly) |
| **Group/era column** | `era/group/symbol/date/...` with repeats → flagged (panel CV on the roadmap) |
| **Compute budget** | scaled from `rows × features`, 3–45 min default, or pass `time_budget=<minutes>` |

Nothing about the *modeling* needs configuring. The engine self‑tunes which models, transforms, feature combinations, and ensemble it ships.

---

## The framework: a civilization that won't fool itself

The engine is **not** "more models." It's a society of bounded explorers moving through compressed feature / latent worlds, governed by one law learned the hard way on real data:

> At a low signal ceiling, a real pattern and a spurious one *fit equally well in‑sample*. The only thing that separates them is **invariance** — does the relationship to the target survive when you change the world (time block, regime, feature clustering)? **Judge by stability across environments, never by in‑sample magnitude.**

So every promoted path must be **wide** (stable across folds), **stable** (sign‑ and strength‑invariant across regimes), **unique** (not a duplicate direction), **cheap** (worth its hardware), and **hard to kill** (survives adversarial validators).

### The members of the civilization
- **Explorers** form candidate paths through *bounded* feature viewports with strange instruments (rank, quantize, doppler, prism, supervised projections…). They combine features in **gauge‑invariant** ways — sign‑aligned sums, ratios, PLS/ARD projections — and *across* decorrelated clusters, not within. Free‑form interactions are allowed but **penalized in proportion to their measured decay**, never banned.
- **Artifacts** measure each path's **stability spectrum** — per‑environment correlation (time × terrain × weather × pressure), sign‑flip rate, worst‑world floor, null‑shuffle margin, and out‑of‑period decay (`oof − walk‑forward`). Promotion requires *sign‑stable AND positive‑worst‑world AND clears‑null*, not "high score."
- **Scientists** (forensic court) try to *kill* every path — interior‑block CV, leave‑one‑cluster‑out, drop‑its‑dominant‑feature, refit under a different clustering. Only survivors ship.
- **Predators** falsify promotions with null/permutation/time‑reversal/regime attacks.
- **The governor** measures this dataset's *complexity → decay* slope at runtime and ships at the complexity the data rewards (simpler where capacity overfits, free where it doesn't).
- **The shipping court** down‑weights members that fail their local *escape velocity* (width vs decay + CV‑distortion + crowding) and shrinks toward equal‑weight when regime‑criticality is high.
- **The ledger** distils each run's *out‑of‑sample‑grounded* learnings (survivors → warm starts, decayers → anti‑priors, governor β) and feeds them, evidence‑shrunk, into the next run. **It self‑improves and can't overfit to one run.**

### Anti‑overfit is structural
Selection is never done on a slice the path was tuned against; complexity is penalized by *measured* decay; diversity is enforced in feature‑family space; everything rejected is kept and reported with its deltas. Reports include `complexity_governor.json`, `shipping_court_report.csv`, `regime_criticality.json`, `learning_ledger.json`, `many_worlds_cv.csv`, and dozens more.

---

## How the code works

WorldExplorer has three layers:

```text
1. Front ends
   Python API, CLI, and Kaggle bootstrap.

2. Engine
   The feature-space explorer, generated from engine_src/ into
   worldexplorer/_engine.py.

3. Tooling
   Fleet launchers, memory-matrix builders, telemetry review, route-carve
   experiments, publishing, and source audits.
```

The normal call path is:

```text
wx.explore(...) or wx.kaggle.run(CONFIG)
  -> autoconfig/profile data
  -> adapter writes engine-shaped train/test/sample_submission files
  -> ExplorerHarness(HarnessConfig).run()
  -> submission.csv + reports + durable run memory
```

On Kaggle, the preferred notebook is only a launcher:

```text
kaggle/bootstrap_kernel.py
  -> install package from GitHub
  -> import worldexplorer as wx
  -> wx.kaggle.run(CONFIG)
```

That keeps all logic in GitHub. The Kaggle Dataset wheel/source mirror exists
only for internet-disabled notebooks.

### Main entrypoints

| Entrypoint | File | Purpose |
|---|---|---|
| `wx.profile` | `worldexplorer/autoconfig.py` | Detect target, metric, validation geometry, ids, categorical columns, and default budget. |
| `wx.explore` | `worldexplorer/adapter.py` | Generic dataframe/path interface. Converts arbitrary tabular data into the engine's expected layout. |
| `wx.kaggle.run` | `worldexplorer/kaggle.py` | Kaggle-specific path resolver. Finds train/test/sample submission, runs the engine, writes `submission.csv`. |
| `worldexplorer` CLI | `worldexplorer/cli.py` | Shell wrapper around `wx.profile` and `wx.explore`. |
| `ExplorerHarness.run` | `engine_src/16_harness.py` | Main engine runtime. Builds topology, searches, attacks, selects, ships, and writes memory. |
| Slim bootstrap | `kaggle/bootstrap_kernel.py` | Fetches the package and calls `wx.kaggle.run(CONFIG)`. |

### Engine source model

The editable engine source is in `engine_src/`. The installable package imports
the generated file:

```text
engine_src/*.py -> python sync_engine.py -> worldexplorer/_engine.py
```

`worldexplorer/_engine.py` is committed so `pip install` from GitHub is
self-contained. When changing engine logic, edit `engine_src/` and then run:

```bash
python sync_engine.py
```

Important modules:

| Area | Files |
|---|---|
| Config and runtime budget | `engine_src/01_configuration.py`, `engine_src/03_metabolism.py` |
| Data discovery and profiling | `engine_src/05_data_discovery_synthetic_f.py`, `worldexplorer/autoconfig.py` |
| Row and feature topology | `engine_src/06_viewports__*.py` |
| Model and transform skills | `engine_src/07_skills__*.py` |
| Candidate path bookkeeping | `engine_src/08_lessons.py` |
| Search phases | `engine_src/09_phase_1.py`, `engine_src/10_phase_2.py` |
| Adversarial validators | `engine_src/11_predator_persona.py` |
| Reports and terrain summaries | `engine_src/12_topography_reports__*.py` |
| Ensembles and many-world scoring | `engine_src/13_ensemble__*.py` |
| Robust OOS and shipping court | `engine_src/15_v21_forensic_regime_scienc__*.py` |
| Main runtime | `engine_src/16_harness.py` |

### Runtime sequence

An engine run does this:

```text
1. Read config, data, prior memory, and time budget.
2. Build validation geometry: temporal, random, sealed, or auto-detected.
3. Build row-space maps: terrain, weather, pressure, beacons, test-likeness.
4. Build feature-space maps: feature graph, communities, shift, agreement, stability.
5. Generate bounded viewports: selected columns, ranks, quantized views, PLS views,
   topology residuals, tail/order blocks, beacon fields, and routed regions.
6. Train candidate lessons with model/skill families.
7. Promote only candidates that pass width, stability, uniqueness, and cost gates.
8. Attack promoted candidates with predator validators.
9. Re-score survivors across many worlds and robust OOS courts.
10. Run the shipping court to down-weight fragile or crowded members.
11. Write `submission.csv`.
12. Write reports, ledgers, atlas artifacts, and next-run memory.
```

The search is intentionally branchy. Failed paths are not thrown away
silently; they become anti-priors or evidence for future runs.

### The world model

WorldExplorer treats a dataset as multiple overlapping spaces:

```text
X = raw feature matrix
Y = target or label
E = environment/regime assignments
Z = transformed coordinates
P = row-to-region membership
G_row = row topology
G_feat = feature topology
R = residual/uncertainty surfaces
Theta = model parameters and ensemble weights
```

The code does not store all of these as full durable tensors yet. Some are
runtime objects and some are reports. The `tools/memory_matrices.py` direction is
to turn more of them into durable matrices/operators that later runs can reuse.

### Creative worldview: material, terrain, and reversible movement

WorldExplorer's research metaphor is deliberate: a tabular dataset is not a
clean spreadsheet of independent columns. It is a loose heterogeneous material
field. Signal, proxy signal, confounding, missingness artifacts, local pockets,
regime effects, delayed effects, duplicated measurements, and pure noise are
mixed together like different grains in the same pile.

The engine therefore studies material before deleting it. A suspicious feature
or residual pocket can be:

```text
active
suspect
discounted
region-limited
route-limited
quarantined
revived
discarded for one scope only
```

The raw evidence is never rewritten. "Moving material" means creating a derived
coordinate system, local route, mask, blend, projection, or residual correction
that treats one region differently while preserving provenance and reversal
paths.

Important material units:

| Material unit | Meaning |
|---|---|
| Feature grain | One raw column or transformed column. |
| Feature community | A group of columns that move together in target-free topology. |
| Row region | A terrain/weather/pressure/testlike pocket of examples. |
| Residual pocket | A localized region where predictions fail in a structured way. |
| Route material | A model or submission's unique prediction direction. |
| Projection material | A PCA/PLS/rank/quantized/latent coordinate system. |
| Quarantined material | Signal not trusted globally, but preserved for scoped retest. |
| Grokking material | Hard structure that may need longer training or a different representation. |

Creative operations are treated as replayable operators:

| Operation | ML equivalent |
|---|---|
| Sift | Select, rank, or filter features/material. |
| Wash | Denoise, impute, shrink, or remove obvious artifacts. |
| Grind | Rank, bin, quantize, or reduce resolution. |
| Fold | Symmetry transforms, absolute folds, pair folds, latent encoders. |
| Carve | Create a local mask, region split, or route-specific correction. |
| Move | Reassign a region in a derived representation or route it to a specialist. |
| Alloy | Combine features, routes, residuals, projections, or model families. |
| Crystallize | Find stable feature communities or repeated route motifs. |
| Quarantine | Discount without deleting; preserve scope, reason, and revival conditions. |
| Revive | Retry old material through new validation worlds or new operators. |
| Anneal | Train longer with regularization, replay, noise, dropout, or lower LR. |
| Quench | Freeze and attack with robust validation before promotion. |

The most important rule is that material movement is not a license to chase
noise. A move must carry evidence:

```text
local gain
global gain or bounded global damage
neighbor/region side effects
public/private or CV/forward gap
false-agreement risk
false-disagreement opportunity
overfit risk
foundation stress
reversibility
```

Unsupported moves stay branch-only.

### Carving patterns: pockets, grids, cubes, and lattices

WorldExplorer can carve freeform pockets, but the safer long-term abstraction is
a lattice atlas: project the data into a small set of meaningful coordinates and
operate on cells instead of arbitrary blobs.

Useful atlas axes include:

```text
stable target-alignment direction
test-likeness / train-test shift
residual pressure
model disagreement
uncertainty
local density
time / regime / terrain
feature-community membership
```

The grid/cube is built over these atlas coordinates, not over raw 800D feature
space. Each cell stores a material inventory:

```text
density
mean prediction
mean residual
uncertainty
model disagreement
calibration error
feature agreement/disagreement
label-noise estimate
testlike score
route success history
grokking potential
```

Then a candidate move can be cell-local:

```text
split a mixed cell
merge neighboring stable cells
smooth a jagged boundary
route one cell to a specialist
discount a suspicious material only inside a cell
diffuse a residual correction across adjacent cells
train longer on high-grokking cells
restore quarantined material inside a newly stable cell
```

Cell moves are judged with neighbor regularization. A local improvement that
breaks adjacent cells, increases surface roughness too much, or invalidates many
old claims becomes a branch or a foundation challenge, not a default path.

### False agreement, false disagreement, and foundation stress

WorldExplorer assumes agreement can be fake. Two features or routes can agree
because they share real signal, but also because they share leakage, missingness,
collinearity, train-test shift, or the same overrepresented region.

False agreement checks ask:

```text
Does agreement collapse after conditioning?
Does it vanish in a time split?
Is it driven by testlike shift?
Is it just duplicate material from one feature community?
Does dropping the dominant feature destroy the route?
```

False disagreement is also useful. Two signals can appear to conflict because a
hidden region mixes two stable local laws. Those conflicts can become carve,
split, specialist, or grokking candidates if the disagreement becomes coherent
after conditioning on region or environment.

Foundation stress tracks when local repairs imply the current coordinate system
may be wrong:

```text
topology shock
feature-graph shock
partition shock
new contradictions
old claims invalidated
boundary complexity growth
bootstrap instability
model-family rank flips
```

High local gain plus high foundation stress is not a normal promotion. It means:

```text
branch from an earlier world
try a different coordinate system
try a different partition
try a different validation world
or keep the move route-limited
```

### Grokking and long-horizon incubation

Some hard material is not noise. It may be structure that needs a longer
training horizon, a different representation, or a neural model with dropout,
noise, and regularization. WorldExplorer treats this as a quarantined research
lane, not as normal early-stopping patience.

A grokking candidate earns runtime when it has evidence like:

```text
structured residuals
low estimated label noise
consistent gradient direction
improving representation separation
systematic model disagreement
slow validation improvement under longer training
similar regions solved by previous transformations
```

The `revive` fleet mode seeds MLP/dropout warm genomes and older successful
motifs into generation zero. Neural members are explicitly marked
`GROK_INCUBATION` and `GROK_SHIP_ELIGIBLE=False`: they may continue exploring,
but they cannot become the shipped path unless a later independent robust court
promotes them.

### Revival, immune memory, and proof-carrying paths

Failed or old paths are not erased. They become memory:

```text
survivors       -> warm starts and route priors
decayers        -> anti-priors
predator kills  -> hazard signatures
weak witnesses  -> possible material for scoped revival
old champions   -> retest candidates under new worlds
```

This is why `tools/fleet.py revive` exists. It asks whether older material that
was weak globally, public/private unstable, or merely outdated can become useful
when viewed through new topology, testlike partitions, residual lenses, or
grokking schedules. The result must still pass current evidence gates.

Every strong path should eventually carry a proof object:

```text
what it used
where it worked
where it failed
what validation worlds judged it
what risks remain
what scope it is valid under
what would invalidate it
```

This keeps the system from blindly trusting either the latest score or an old
memory. The goal is a versioned computational atlas: states, operators, tensors,
models, residual fields, route strengths, contradictions, and validation budget
all stored as reusable scientific evidence.

### Row-space topology

Rows are mapped into regions so the engine can distinguish global signal from
local pockets.

Examples:

```text
TerrainAtlas   -> target-free row clusters and altitude/novelty
WeatherGauge   -> volatility-like row regimes
PressureGauge  -> microstructure or local-pressure regimes
BeaconField    -> radial basis landmarks around useful terrain locations
TESTLIKE       -> target-free probability that a row resembles the test set
```

These maps let the engine ask:

```text
Does this path work everywhere, or only in one terrain?
Does a local fix help a weird region but damage nearby plateaus?
Does a model fail exactly where rows look test-like?
```

### Feature-space topology

Features are also mapped as a graph.

Examples:

```text
FeatureGraph                 -> feature-feature communities
sign_stability               -> whether feature-target direction flips
PLS / target-alignment ranks -> supervised low-dimensional directions
testlike stability           -> whether a signal survives domain shift
redundancy/crowding          -> whether two members are the same signal twice
```

This is how the engine tries to avoid false agreement from collinear features or
duplicate routes. A feature is rarely just "good" or "bad"; it can be globally
weak, locally useful, shifted, redundant, route-limited, or suspicious.

### Candidate lessons

A lesson is one measured candidate path:

```text
explorer
stage
skill/model family
viewport
transform
feature subset
OOF score
walk-forward or sealed score
stability diagnostics
predator verdict
promotion decision
```

Candidate paths are created by explorers, then measured and filtered. The
engine favors paths that are:

```text
wide across folds
stable across regimes
positive in worst worlds
not redundant with existing members
not too complex for the measured decay slope
hard to kill by null/perturbation/regime attacks
```

### Adversarial validation

The validator stack is deliberately hostile:

```text
draft gates
dual-geometry checks
forward/sealed checks
terrain and weather checks
null and permutation attacks
perturbation attacks
time-reversal style checks
many-world reports
robust OOS selection
shipping court
```

Key artifacts:

```text
predator_report.csv
many_worlds_cv.csv
robust_oos_selection.csv
shipping_court_report.csv
sealed_holdout_report.json
regime_criticality.json
```

The main adversarial question is always:

```text
Did this path learn a stable relationship, or did it learn one validation shape?
```

### Cross-run memory

Each run writes memory for the next run:

```text
learning_ledger.json
world_cairn.json
explorer_findings_graph.json
complexity_governor.json
feature_topology_report.csv
path_texture_report.csv
shipping_court_report.csv
many_worlds_cv.csv
```

The current memory is mostly report-backed. The intended direction is a
computational atlas:

```text
actual tensors and matrices
feature-space operators
projection bases
route-strength estimates
residual fields
validation-budget ledgers
evidence gates
proof-carrying paths
grokking candidates
contradiction graphs
checkpoint lineage
```

The main tool for this direction is:

```bash
python tools/memory_matrices.py
```

The material-field framing is deliberately general:

```text
features, regions, projections, residual pockets, and route effects are
derived material particles with support, composition, evidence, risk, and
provenance.
```

This does not mean every strange signal becomes a new foundation. The atlas
uses an evidence ladder:

```text
A supported candidate
B branch with independent retest
C route-limited or quarantined
D rejected or hazard memory
```

Unsupported material operations stay branch-only until they survive independent
validation, parent/sibling ablation, false-agreement checks, and foundation
stress checks.

### Fleet and Kaggle workflow

`tools/fleet.py` generates slim Kaggle kernels. The generated kernels are not
source code; they are launch artifacts.

GitHub-first breaker fleet:

```bash
python tools/fleet.py breaker \
  --count 5 \
  --prefix breaker-github-master \
  --time-budget 180
```

Old-material + MLP grokking revival fleet:

```bash
python tools/fleet.py revive \
  --count 6 \
  --prefix revival-old-mlp \
  --gpu-frac 0.67 \
  --time-budget 240
```

`revive` is intentionally conservative. It seeds older useful motifs and MLP
dropout/noise warm genomes into generation zero, but every path is remeasured
through the current validation worlds. Neural revival members are marked as
grokking incubators: they may train longer and explore delayed structure, but
they are branch-only until an independent current-world retest passes the robust
OOS and shipping courts.

Offline wheel-backed bootstrap:

```bash
python tools/fleet.py bootstrap \
  --name wx-offline-wheel \
  --offline \
  --source-policy wheel_first \
  --engine-dataset /kaggle/input/worldexplorer-engine \
  --dataset taylorsamarel/worldexplorer-engine \
  --time-budget 120
```

## Claude Code / Research-Agent Handoff

This section is intentionally pasteable into another coding agent. It gives the
current research state, validated lessons, safe extension points, and guardrails.

### Current research snapshot

Date of this snapshot: **2026-06-15**.

WorldExplorer is being developed on the DRW Crypto Market Prediction setting as
a stress test for low-signal, shifting tabular data. Scores below are Kaggle
public/private after-deadline probes. They are useful for research memory, but
must not be used as training labels, target leakage, or direct optimization
targets.

| Submission / route | Private | Public | Lesson |
|---|---:|---:|---|
| External reference: `Fork of DRW Crypto Market Prediction Arunemble 2.0` | `0.11164` | `0.10872` | Reference only. Do not weight as WorldExplorer evidence unless source/artifacts are available. |
| Historical `Rejoin - Version 10` | `0.08979` | `0.07615` | Historical teacher/reference. Useful as context, not direct proof of current engine behavior. |
| `DRW WX gpu-wide - Version 2` / raw gpu-wide | `0.08548` | `0.06689` | Early strong WorldExplorer basin. |
| `grok-atlas2-03-weird` route carves | about `0.08619` | about `0.06682` | Weird/topology branch carried reusable material. |
| `breaker_long_master_01_tail_sharp` | `0.08661` | `0.07318` | Framework-native tail/sharp run improved public stability. |
| `gpu-wide_rank_resid_add_a0.05` | `0.08740` | `0.07293` | First strong residual-add route from breaker/gpu-wide material. |
| `gpu-wide_rank_resid_add_a0_05_pow15_resid_add_a0.05` | `0.08881` | `0.07469` | Best public-stable WorldExplorer candidate so far. |
| `revive old breaker rank-reduce rank residual add 0.05` | `0.08919` | `0.07395` | Old material revival works. |
| `revive old governor rank-reduce raw residual add 0.05` | `0.08906` | `0.07396` | Independent support for old-material revival. |
| `revive old gpu-wide conflict rank residual add 0.05` | `0.08934` | `0.07399` | Current best WorldExplorer private score. |
| `revive stage2 prior public stabilizer clip residual add 0.04` | `0.08839` | `0.07436` | Second-stage stacking preserved public somewhat but diluted private. |
| `revive stage2 breaker/governor rank residual add 0.04` | `0.08814` / `0.08806` | `0.07241` / `0.07246` | Second-stage revival mostly hurt. Treat as hazard/branch evidence. |

Current interpretation:

```text
Old material is not dead.
First-layer revival is productive.
Second-layer residual stacking is risky.
Public-stable and private-sharp paths are related but not identical.
MLP/grokking paths are worth incubating, but cannot ship without robust retest.
```

If choosing two submissions from this snapshot, the preferred pair is:

```text
private-best:
  submission_submission_gpu-wide_raw_conflict_sub_a0_06_rank_resid_add_a0.05.csv
  private=0.08934 public=0.07399

public-stable hedge:
  submission_submission_gpu-wide_rank_resid_add_a0_05_pow15_resid_add_a0.05.csv
  private=0.08881 public=0.07469
```

### Active hypotheses

1. **Old route material can revive under new operators.**
   The revival probes show old gpu-wide/governor/breaker routes still contain
   usable signal when viewed through current residual/rank lenses.

2. **One layer of surgery helps; two layers can over-smooth or overfit.**
   The stage-2 carves reduced private score. Future work should test first-layer
   moves, smaller alphas, or different axes before stacking more residual edits.

3. **Public/private divergence is a terrain signal.**
   Public-stable candidates are not necessarily private-best. Treat the gap as
   a validation-world disagreement, not as random noise.

4. **MLP/dropout grokking may still matter.**
   Earlier competition versions suggested MLPs with noise/dropout could do well.
   Current MLP branches must be incubated behind `GROK_INCUBATION` and promoted
   only after robust retest.

5. **Lattice/cell carving is the next safer form of material movement.**
   Instead of arbitrary pockets, project into atlas coordinates and operate on
   cells with adjacency, support, roughness, and neighbor damage checks.

### Current creative strategy stack

WorldExplorer's strategy stack is:

```text
row topology
feature topology
material inventory
bounded viewports
model/skill diversity
route-carve output-space probes
old-material revival
grokking incubation
false-agreement checks
foundation-stress checks
validation-budget accounting
proof-carrying paths
computational atlas memory
```

Specific methods already in the code or tooling:

```text
TerrainAtlas
WeatherGauge
PressureGauge
BeaconField
FeatureGraph
testlike partitions
sign-stability rankers
PLS-weight rankers
consensus feature-family rankers
greedy OLS tail/order motifs
Bayesian ridge / ARD / PLS paths
GPU ridge swarm
small MLP skill with dropout and Pearson+MSE loss
predator null/regime attacks
forensic robust OOS selection
shipping court
route_carve residual/rank/clip/tanh/pow transforms
revive fleet mode
grokking incubation reports
memory matrices / computational atlas artifacts
```

### Safe improvement modules to add

External agents should prefer additive, flag-gated modules. Good targets:

| Module | Purpose | Likely files |
|---|---|---|
| `LatticeAtlas` | Build grid/cube/cell atlas over stable coordinates; store cell stats and adjacency. | `engine_src/06_viewports__*.py`, `tools/memory_matrices.py` |
| `CellSurgeryCourt` | Score cell-local moves by local gain, neighbor damage, roughness, support, and foundation stress. | `engine_src/15_*`, `tools/memory_matrices.py` |
| `FalseAgreementCourt` | Detect agreement caused by collinearity, shift, missingness, or duplicate communities. | `engine_src/06_viewports__*.py`, `engine_src/12_*` |
| `FalseDisagreementMiner` | Find conflicts that become stable after region splits. | `engine_src/06_viewports__*.py`, `engine_src/12_*` |
| `GrokkingSentinel` | Let MLP/neural branches run longer when internal structure improves despite flat validation. | `engine_src/01_configuration.py`, `engine_src/07_skills__*.py`, `tools/fleet.py` |
| `RouteRevivalPlanner` | Generate first-layer revival candidates while penalizing over-stacked residual surgery. | `tools/route_carve.py`, `tools/telemetry_guidance.py` |
| `ValidationBudgetLedger` | Discount repeatedly reused validation surfaces and correlated public/private probes. | `tools/memory_matrices.py` |
| `ProofCarryingPath` | Emit machine-readable evidence/risk/scope certificates for shipped paths. | `engine_src/12_*`, `tools/memory_matrices.py` |
| `WorldForgeSyntheticTests` | Synthetic traps for null worlds, leakage, collinearity, hidden subgroup, label delay, and train/test shift. | `tests/`, `tools/` |

### How to work safely

Use this protocol for code changes:

```text
1. Read README.md, RESEARCH_AGENT_BRIEF.md, SOURCE_OF_TRUTH.md.
2. Identify one small module or guardrail to add.
3. Prefer a flag-gated no-op default or report-only first version.
4. If changing engine behavior, edit engine_src/ first.
5. Run python sync_engine.py after engine_src edits.
6. Add tests for synthetic failure modes, not only happy paths.
7. Do not remove existing validators, predators, courts, ledgers, or reports.
8. Do not convert leaderboard results into training features or labels.
9. Do not let grokking branches ship directly.
10. Do not make Kaggle kernels monolithic; keep logic in GitHub.
```

Recommended local checks:

```bash
python3 -m py_compile tools/fleet.py kaggle/bootstrap_kernel.py
../.venv/bin/python -m pytest \
  tests/test_adapter_kaggle.py \
  tests/test_telemetry_guidance.py \
  tests/test_memory_guardrails.py \
  tests/test_fleet_revival.py
python tools/source_audit.py
```

If engine modules are changed:

```bash
python sync_engine.py
python3 -m py_compile worldexplorer/_engine.py
```

### What not to do

Do not:

```text
delete validators because they block exciting paths
optimize directly to public/private score rows
turn old leaderboard winners into unquestioned priors
ship MLP/grokking branches without robust court promotion
mutate raw data to "move material"
hide logic in Kaggle notebooks
replace the engine with one giant monolith
drop quarantined material without preserving reversal conditions
add high-cost modules without budget guards
judge a path by one scalar score only
```

### Suggested first tasks for an external coding agent

1. Add `LatticeAtlas` as a report-only artifact:

```text
inputs: existing transformed coordinates, residuals, uncertainty, testlike score
outputs: cell ids, cell adjacency, cell density, cell residual, cell disagreement
status: report-only, no model behavior change
tests: deterministic synthetic grid with one high-residual cell
```

2. Add a `route_carve` anti-stacking penalty:

```text
penalize candidates whose route name already contains multiple residual layers
prefer first-layer revival unless direct evidence says stacking works
write stack_depth into route_carve_manifest.csv
```

3. Add `proof_carrying_paths.jsonl` details:

```text
source route
operator
public/private or forward/sealed evidence
known risks
scope
invalidation conditions
required next retest
```

4. Add grokking report fields:

```text
train-loss trend if available
validation plateau length
MLP dropout/noise settings
warm genomes used
ship_eligible=false
promotion gates required
```

5. Add synthetic tests:

```text
null world: no false promotion
hidden subgroup: false disagreement becomes carve candidate
duplicate features: false agreement detected
train/test shift: testlike gate downweights shifted feature
tiny pocket: high local gain stays branch-only
```

### What research agents should inspect

For adversarial validation and improvement work, start with:

```text
RESEARCH_AGENT_BRIEF.md
SOURCE_OF_TRUTH.md
engine_src/06_viewports__*.py
engine_src/07_skills__*.py
engine_src/11_predator_persona.py
engine_src/15_v21_forensic_regime_scienc__*.py
engine_src/16_harness.py
tools/memory_matrices.py
tools/telemetry_guidance.py
tools/source_audit.py
```

Useful external feedback should name:

```text
file/function inspected
failure mode found
artifact or metric affected
minimal reproduction or test
specific code or validation change
expected risk and expected benefit
```

---

## Install

```bash
# straight from GitHub (no clone):
pip install "git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git"
pip install "worldexplorer[full] @ git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git"   # + torch/lgbm/xgb

# or a local checkout:
pip install -e .            # core (numpy/pandas/scikit-learn/pyarrow)
pip install -e .[full]      # + torch, lightgbm, xgboost, scipy, psutil (GPU + boosting)
```

The engine (`worldexplorer/_engine.py`) is amalgamated from the modular source
by `sync_engine.py`; re-sync with `python sync_engine.py`.

## Source of truth

GitHub is the source of truth for WorldExplorer logic: engine modules, package
API, fleet strategy, memory/atlas tooling, telemetry guidance, route-carve
experiments, and the slim Kaggle bootstrap template all live in this repo.

Kaggle notebooks are launch surfaces. The preferred notebook contains only
`CONFIG`, package acquisition, `import worldexplorer as wx`, and
`wx.kaggle.run(CONFIG)`.

For external review, adversarial validation, and improvement work, send agents
[RESEARCH_AGENT_BRIEF.md](RESEARCH_AGENT_BRIEF.md).

See [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md). To check the guardrail locally:

```bash
python tools/source_audit.py
```

---

## Use on Kaggle (thin front end)

The whole notebook becomes a few lines — no giant paste:

```python
!pip install -q git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git
import pandas as pd, worldexplorer as wx
r = wx.explore("/kaggle/input/<comp>/train.parquet", target="label",
               test="/kaggle/input/<comp>/test.parquet", out="/kaggle/working", time_budget=690)
sub = pd.read_csv("/kaggle/input/<comp>/sample_submission.csv")
sub[sub.columns[-1]] = r.predictions["prediction"].to_numpy()
sub.to_csv("/kaggle/working/submission.csv", index=False)
```

See `examples/kaggle_drw.py`. **Self‑improvement across runs:** attach a run's output as an input dataset to the next run; the engine finds `world_cairn.json` / `learning_ledger.json` and stands on it (governor warm‑start, survivors, anti‑priors).

## Repo layout

```
worldexplorer/        pip package: __init__, autoconfig, adapter, cli, kaggle, _engine
engine_src/           modular engine sources; edit these before regenerating _engine
kaggle/               slim bootstrap template
tools/                fleet, memory, telemetry, publishing, smoke, and audit tools
examples/             kaggle_drw.py, quickstart.py
tests/                adapter and Kaggle wrapper tests
sync_engine.py        engine_src/*.py -> worldexplorer/_engine.py
SOURCE_OF_TRUTH.md    rule that GitHub owns the logic and Kaggle owns launch config
RESEARCH_AGENT_BRIEF.md external handoff for adversarial validation agents
```

`_engine.py` is generated (committed so installs are self‑contained); edit `engine_src/` and re‑run `sync_engine.py`.

## Roadmap (the feature‑combination framework, staged)
The "combine features without mistaking noise for signal" gates, drawn from the top **private‑LB** solutions and folded in behind the law above:
1. **Sign‑stability gate** — drop any feature whose univariate sign flips on a held‑out interior block (4th‑place private).
2. **Sign‑aligned directional‑combination** transform family (gauge‑invariant aggregation).
3. **Interior‑block CV** — train oldest∪newest, validate the middle (resists tail/leak overfit).
4. **PLS‑as‑selector** + importance‑weighted column sampling; **James‑Stein** shrinkage for group/regime encodings.
5. **Panel/era CV geometry** for cross‑sectional (Numerai‑style) data.

**Permanent guardrail:** never uses leaderboard feedback, future/lead values, test‑row ordering, or any label‑inference leak.
