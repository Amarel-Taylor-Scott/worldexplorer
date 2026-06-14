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
worldexplorer/      pip package: __init__, autoconfig, adapter, cli, _engine
engine_src/         the 18 modular sources the engine is amalgamated from
sync_engine.py      engine_src/*.py  ->  worldexplorer/_engine.py
examples/           kaggle_drw.py, quickstart.py
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
