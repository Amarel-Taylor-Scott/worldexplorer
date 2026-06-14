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
