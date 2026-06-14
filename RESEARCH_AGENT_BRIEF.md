# WorldExplorer Research Agent Brief

This file is for research agents reviewing, improving, or adversarially
validating WorldExplorer.

WorldExplorer is a feature-space exploration framework for low-signal tabular
prediction problems. It is not only an AutoML runner and not only a Kaggle
notebook. It is a package that:

```text
loads data
maps feature and row topology
generates many candidate representations
trains candidate paths
attacks candidates with adversarial validation
selects robust survivors
writes predictions and telemetry
persists cross-run memory
```

The core principle is:

```text
Do not trust a path because it fits.
Trust it only if it survives changed worlds.
```

Changed worlds include time folds, terrain partitions, weather/pressure
partitions, test-like partitions, feature communities, null shuffles,
perturbations, model-family disagreement, and held-out/sealed validation.

## Source Of Truth

The source of truth is the GitHub repo:

```text
https://github.com/Amarel-Taylor-Scott/worldexplorer
```

In this local workspace, the repo is:

```text
/home/username/new_algo/worldexplorer
```

Kaggle kernels should be slim launchers. They should not contain the modeling
logic. The preferred Kaggle kernel does this:

```text
install worldexplorer from GitHub
load CONFIG
import worldexplorer as wx
wx.kaggle.run(CONFIG)
```

Offline Kaggle notebooks can use an attached wheel/source dataset, but that is
a fallback mirror, not the source of truth.

See:

```text
SOURCE_OF_TRUTH.md
kaggle/bootstrap_kernel.py
tools/fleet.py
tools/source_audit.py
```

## Main Entry Points

### Package API

```text
worldexplorer/__init__.py
worldexplorer/adapter.py
worldexplorer/autoconfig.py
worldexplorer/cli.py
```

Typical Python usage:

```python
import worldexplorer as wx

result = wx.explore(
    "train.parquet",
    target="label",
    test="test.parquet",
    out="wx_out",
    time_budget=120,
)
```

### Kaggle API

```text
worldexplorer/kaggle.py
```

Typical Kaggle usage:

```python
import worldexplorer as wx
result = wx.kaggle.run(CONFIG)
```

`worldexplorer.kaggle.run` resolves train/test/sample-submission paths, sets
the harness config, runs the engine, and writes `submission.csv`.

### Engine

```text
engine_src/*.py
worldexplorer/_engine.py
sync_engine.py
```

The real engine source is modularized in `engine_src/`.

`worldexplorer/_engine.py` is the generated single-file engine committed for
installability. Edit `engine_src/`, then run:

```bash
python sync_engine.py
```

Do not make primary logic edits directly in `worldexplorer/_engine.py` unless
the same change is applied back to `engine_src/`.

## Code Map

```text
engine_src/00_preamble.py
  Version notes, design assumptions, report inventory, doctrine.

engine_src/01_configuration.py
  HarnessConfig: budget, validation, topology, model, predator, and shipping
  knobs.

engine_src/02_logging_io.py
  Logging, CSV/JSON writers, artifact helpers.

engine_src/03_metabolism.py
  Runtime budget accounting and phase-level resource control.

engine_src/04_core_math.py
  Core metrics, stable seeds, utility math, scoring helpers.

engine_src/05_data_discovery_synthetic_f.py
  Competition data discovery, synthetic fallback, profile initialization.

engine_src/06_viewports__*.py
  Row topology, feature topology, transformations, viewports, signal operators.

engine_src/07_skills__*.py
  Model/skill families such as ridge, linear, greedy OLS, PLS, ranking,
  stability-selected paths, neural/GPU paths when enabled.

engine_src/08_lessons.py
  Lesson object and candidate-path bookkeeping.

engine_src/09_phase_1.py
  First exploration phase: scouts, species, initial lesson search.

engine_src/10_phase_2.py
  Evolutionary/refinement phase: offspring, dreams, warm genomes, continuation.

engine_src/11_predator_persona.py
  Adversarial attacks against promoted paths.

engine_src/12_topography_reports__*.py
  Terrain, feature, path, and narrative reports.

engine_src/13_ensemble__*.py
  Ensemble building, many-world reports, blending helpers.

engine_src/14_harness_orchestration.py
  Harness state objects and orchestration helpers.

engine_src/15_v21_forensic_regime_scienc__*.py
  Forensic regime science, robust OOS selection, shipping court.

engine_src/16_harness.py
  ExplorerHarness.run: the main runtime sequence.

engine_src/17_main.py
  Script entrypoint.
```

Supporting tools:

```text
tools/fleet.py
  Generates slim Kaggle bootstrap kernels for sprout, grok, and breaker fleets.

tools/memory_matrices.py
  Reads run artifacts and builds durable route/memory/typology matrices.

tools/telemetry_guidance.py
  Turns submission scores and runtime artifacts into next-run guidance.

tools/route_carve.py
  Compares submissions and creates reversible output-space perturbation probes.

tools/width_elasticity.py
  Studies whether wider/narrower path mixes are helping or decaying.

tools/worldview_loop.py
  Runtime review loop for collecting artifacts and feeding future runs.

tools/publish.py
  GitHub/Kaggle publishing and submission helper.
```

## Runtime Flow

At a high level:

```text
1. Bootstrap
2. Discover data
3. Build config and budget
4. Build row-space topology
5. Build feature-space topology
6. Generate candidate viewports and skills
7. Train candidate lessons
8. Promote only paths that survive gates
9. Attack promotions with predators
10. Run forensic robust-OOS selection
11. Run shipping court and final blend
12. Write submission and artifacts
13. Write cross-run memory
```

### 1. Bootstrap

Kaggle slim kernels use:

```text
kaggle/bootstrap_kernel.py
```

Default policy:

```text
source_policy = "github_first"
enable_internet = true
repo = git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git@master
```

Offline policy:

```text
source_policy = "wheel_first"
enable_internet = false
attached dataset = worldexplorer wheel/source mirror
```

### 2. Data Discovery

The Kaggle adapter finds:

```text
train.parquet / train.csv
test.parquet / test.csv
sample_submission.csv
target column
submission target column
output directory
```

The generic adapter profiles:

```text
target type
metric
temporal vs random validation geometry
id columns
categorical columns
feature alignment
auto runtime budget
```

### 3. Row-Space Topology

WorldExplorer treats rows/examples as a terrain.

Important objects:

```text
TerrainAtlas
WeatherGauge
PressureGauge
BeaconField
TESTLIKE
```

Conceptually:

```text
Z = T(X)
P[i, r] = membership of row i in terrain/region r
density[r] = local population
loss[r] = local residual behavior
uncertainty[r] = local model uncertainty
```

The system uses row topology to find:

```text
stable plateaus
weird pockets
high-loss regions
test-like zones
regions where models disagree
regions where a local route may be justified
```

### 4. Feature-Space Topology

WorldExplorer also treats columns/features as a graph.

Important objects:

```text
FeatureGraph
feature communities
feature shift vectors
sign stability vectors
PLS/target-alignment rankings
test-like stability indicators
consensus and redundancy measures
```

Conceptually:

```text
W_feat[j, k] = relationship between feature j and feature k
community[j] = target-free feature community
signal[j] = feature-target alignment under multiple environments
shift[j] = train/test or train/forward instability
```

This is used to avoid false confidence from duplicated, collinear, unstable, or
shift-dependent features.

### 5. Viewports And Operators

A viewport is a restricted representation of the data. It may select features,
rank them, quantize them, project them, combine them, or build topology-derived
features.

Examples:

```text
identity
rank
quantized rank
sign-stability-selected features
PLS-selected features
test-like-stable features
feature-community residuals
tail/order blocks
beacon fields
terrain routers
folds and symmetry transforms
greedy OLS selected subsets
```

Research agents should inspect:

```text
engine_src/06_viewports__*.py
engine_src/07_skills__*.py
```

Key question:

```text
Does this operator expose stable signal, or does it create a fragile local fit?
```

### 6. Lessons And Promotions

A lesson is a candidate path:

```text
viewport + skill + transform + feature subset + model + validation result
```

Each lesson is measured by:

```text
OOF score
walk-forward or sealed score
width/stability
sign stability
worst-world behavior
complexity
crowding/redundancy
predator verdict
```

A path is not promoted just because it scores well once. It must pass gates that
try to measure whether it is stable across changed worlds.

### 7. Predator Attacks

Predators try to kill promoted paths.

Implemented attack families include variants of:

```text
null/permutation attacks
sub-period attacks
terrain attacks
weather/pressure attacks
perturbation attacks
time-reversal or palindrome-style attacks
dead-region checks
fading-trail checks
```

Primary file:

```text
engine_src/11_predator_persona.py
```

Useful output:

```text
predator_report.csv
```

Research-agent goal:

```text
Find promoted paths that survive current predators but should not.
```

### 8. Forensic Robust-OOS Selection

Forensic selection compares candidate blends and model members across multiple
validation worlds, not just one public score or one internal fold.

Primary files:

```text
engine_src/15_v21_forensic_regime_scienc__*.py
engine_src/13_ensemble__*.py
```

Important outputs:

```text
robust_oos_selection.csv
many_worlds_cv.csv
shipping_court_report.csv
regime_criticality.json
sealed_holdout_report.json
```

Research-agent goal:

```text
Detect whether robust selection is genuinely robust or just overfitting its own
validation suite.
```

### 9. Shipping Court

The shipping court down-weights or rejects members that look strong locally but
fragile globally.

It considers:

```text
complexity
decay
crowding
worst-world score
regime criticality
many-world behavior
member redundancy
```

Output:

```text
shipping_court_report.csv
```

Research-agent goal:

```text
Find cases where the court is too strict, too lenient, or incorrectly weighting
one validation world.
```

### 10. Cross-Run Memory

WorldExplorer writes memory artifacts so future runs do not start cold.

Important artifacts:

```text
learning_ledger.json
world_cairn.json
explorer_findings_graph.json
complexity_governor.json
feature_topology_report.csv
path_texture_report.csv
many_worlds_cv.csv
shipping_court_report.csv
```

`tools/memory_matrices.py` builds more durable, matrix-like cross-run memory
from these artifacts.

The intended direction is a computational atlas:

```text
actual tensors
operators
feature-space transforms
route strengths
checkpoint lineage
residual fields
grokking candidates
contradiction graph
```

Research-agent goal:

```text
Improve how runtime observations become reusable mathematical state instead of
only prose summaries.
```

## Main Outputs To Inspect

A completed run may write:

```text
submission.csv
explorer_run_summary.json
sealed_holdout_report.json
learning_ledger.json
world_cairn.json
explorer_findings_graph.json
terrain_atlas_report.csv
beacon_atlas_report.csv
feature_topology_report.csv
testlike_report.json
explorer_lessons.csv
explorer_journal.csv
predator_report.csv
many_worlds_cv.csv
robust_oos_selection.csv
shipping_court_report.csv
regime_criticality.json
complexity_governor.json
path_texture_report.csv
antifragility_report.csv
```

Not every run produces every artifact. Missing artifacts can mean the feature
was disabled, skipped by budget, or failed safely.

## How To Run Local Checks

Basic source and bootstrap audit:

```bash
python tools/source_audit.py
```

Compile:

```bash
python3 -m py_compile kaggle/bootstrap_kernel.py tools/fleet.py tools/publish.py tools/source_audit.py
```

Unit tests:

```bash
python -m pytest tests/test_adapter_kaggle.py
```

Sync generated engine after editing `engine_src/`:

```bash
python sync_engine.py
```

Generate a GitHub-first breaker fleet:

```bash
python tools/fleet.py breaker \
  --count 5 \
  --prefix breaker-github-master \
  --time-budget 180
```

Generate an offline wheel-backed kernel:

```bash
python tools/fleet.py bootstrap \
  --name wx-offline-wheel \
  --offline \
  --source-policy wheel_first \
  --engine-dataset /kaggle/input/worldexplorer-engine \
  --dataset taylorsamarel/worldexplorer-engine \
  --time-budget 120
```

## Adversarial Validation Checklist

Research agents should attack these areas.

### Leakage

Check for:

```text
target leakage through feature transforms
using test row ordering as signal
train/test joint encoding that leaks labels
sample_submission assumptions
future/lead features
time split contamination
ledger memory leaking leaderboard-derived choices
validation folds influenced by the same score they evaluate
```

Questions:

```text
Can any feature be derived from the target?
Can any route infer labels from test distribution artifacts?
Does a transform use y where it claims to be target-free?
Does cross-run memory encode leaderboard feedback rather than run artifacts?
```

### False Robustness

Check for:

```text
many validation worlds that are highly correlated
predator attacks that duplicate the same assumption
robust_oos_selection overfitting to its own validation menu
shipping court rewarding complexity indirectly
feature communities built on unstable correlations
testlike partitions dominating selection
```

Questions:

```text
Are the validation worlds independent enough?
Does improvement survive a newly generated holdout?
Do different seeds produce similar promoted members?
Does a model survive when its strongest feature family is removed?
```

### False Agreement

Check for:

```text
collinear features counted as independent confirmation
two models agreeing because they share a leakage source
feature communities that collapse under bootstrapping
rank transforms hiding train/test instability
local region fixes that create distant failures
```

Questions:

```text
Does agreement remain after conditioning on feature communities?
Does agreement hold inside each terrain, or only globally?
Does agreement survive feature dropout and row bootstrap?
```

### False Disagreement

Check for:

```text
mixed subpopulations
wrong validation geometry
label delay
measurement error
bad normalization
hidden regime split
```

Questions:

```text
Does a feature pair disagree globally but agree after splitting a region?
Does a local carve resolve residuals without increasing global stress?
Does the disagreement survive alternate row topology?
```

### Overfitting

Check for:

```text
tiny-region improvements
large train-CV gap
CV-forward gap
too many candidate paths relative to validation strength
submission-space tuning without robust internal justification
route-carve gains not confirmed by independent worlds
```

Questions:

```text
Does the gain survive bootstraps?
Does it survive seed changes?
Does it survive feature dropout?
Does it survive time-block reversal?
Does it improve worst-world score, not only average score?
```

### Memory Poisoning

Check for:

```text
bad runs treated as positive priors
leaderboard feedback entering durable memory without a gate
stale feature schemas reused after data changes
quarantined signals revived without evidence
failed paths forgotten and retried blindly
```

Questions:

```text
Does memory distinguish observed evidence from hypothesis?
Does each memory object have provenance?
Can a future run invalidate a prior belief?
Are negative results stored in a reusable way?
```

### Grokking Incubation

The framework supports the idea of quarantined long-horizon branches: runs that
keep training/exploring despite flat immediate validation if internal structure
is still forming.

Attack points:

```text
grokking branch consumes too much shipping budget
flat validation is mistaken for delayed generalization
latent movement is only memorization
branch ships without independent robust gates
overfit branches poison route-value memory
```

Questions:

```text
Is the grokking branch quarantined?
Does it have a compute budget?
Does it have independent promotion gates?
Does it record failed incubation as evidence?
```

## Suggested Research Improvements

High-value directions:

```text
1. Make information matrices more durable.
   Persist covariance, precision, Fisher/Hessian approximations, route values,
   residual fields, feature-label tensors, and transformation operators.

2. Improve false-agreement detection.
   Add conditional agreement tensors by terrain, feature community, seed, and
   model family.

3. Improve false-disagreement recovery.
   Add systematic hidden-subgroup tests before quarantining or dropping signal.

4. Add foundation-stress scoring.
   Measure whether a local feature-space repair creates broad topology shock.

5. Strengthen memory provenance.
   Store every prior as evidence-backed, hypothesis, contradicted, superseded,
   or invalidated.

6. Improve route-value learning.
   Treat transforms, model families, and carve actions as bandit arms with
   expected gain, cost, fragility, and transferability.

7. Add formal replay buffers.
   Keep hard examples, representative examples, failed examples, and grokking
   candidates as durable training/evaluation sets.

8. Expand adversarial validators.
   Add more environment splits, null worlds, seed-stability tests, feature
   knockout tests, and branch comparison diagnostics.
```

## What Not To Do

Do not move logic back into Kaggle notebooks.

Do not promote a path because it improves one validation score.

Do not delete or quarantine suspicious signal without storing:

```text
scope
reason
evidence
reversal condition
parent checkpoint or world state
```

Do not treat leaderboard scores as training labels.

Do not edit `worldexplorer/_engine.py` without syncing `engine_src/`.

Do not let generated fleet kernels become the maintained source.

## Review Questions For External Agents

Use these as starting prompts:

```text
Where can target leakage enter despite current gates?
Which validation worlds are redundant?
Which promoted paths are fragile under new splits?
Which feature transforms create false agreement?
Which local carves improve one region but harm the global topology?
Which memory artifacts are summaries when they should be tensors/operators?
Which failed paths should become durable anti-priors?
Which grokking branches deserve more runway, and which are just overfitting?
How can the route-value matrix learn better from previous runs?
What artifact would make the next run smarter instead of just more tuned?
```

## Expected Research-Agent Output

Useful external feedback should be concrete:

```text
file/function inspected
artifact or report inspected
failure mode found
minimal reproduction or proposed test
specific change recommendation
risk of change
expected artifact or metric that should improve
```

The best improvements should add measurable protection or reusable memory, not
only another model family.
