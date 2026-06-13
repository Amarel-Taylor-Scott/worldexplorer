# Publishing worldexplorer + running it from a thin Kaggle cell

The goal: the Kaggle notebook is a small **bootstrap** that carries only the
config (where the data is, the target, the budget) and downloads the engine from
a public GitHub repo. All the logic lives in this repo.

```
repo (GitHub)  ──pip install / dataset──▶  Kaggle thin cell  ──▶  submission.csv
   worldexplorer/  (engine + adapter + kaggle.run)         CONFIG only
```

## 1. Publish the repo to GitHub (one time)

This directory is already a git repo with everything committed (the vendored
`worldexplorer/_engine.py` is regenerated from `engine_src/` by `sync_engine.py`,
so a fresh clone is runnable). Create the GitHub repo and push:

```bash
# from this directory
gh repo create worldexplorer --public --source=. --remote=origin --push
# or, without gh:
git remote add origin https://github.com/<YOU>/worldexplorer.git
git push -u origin master
```

Pin releases with tags so a notebook can request an exact version:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

## 2A. Run on Kaggle WITH Internet (the simple path)

Paste `kaggle/bootstrap_kernel.py` into one cell, set `CONFIG["repo"]` to your
repo URL (add `@v0.2.0` to pin), set `data_root`/`target`, and run. The cell
`pip install`s the engine from GitHub and runs it.

## 2B. Run on Kaggle WITHOUT Internet (DRW + code competitions)

Code competitions block internet, so publish the repo as a **Kaggle Dataset**
and attach it:

```bash
# package the repo as a dataset (needs the kaggle CLI + ~/.kaggle/kaggle.json)
cd ..                       # parent of this repo dir
kaggle datasets init -p worldexplorer
# edit worldexplorer/dataset-metadata.json: set "title" + "id" = "<user>/worldexplorer-engine"
kaggle datasets create -p worldexplorer -r zip
# later updates:
kaggle datasets version -p worldexplorer -m "v0.2.0" -r zip
```

Then in the notebook: **Add Input → your `worldexplorer-engine` dataset**, paste
the bootstrap cell, and leave `CONFIG["engine_dataset"] = None` (the cell
auto-finds any attached folder containing `worldexplorer/__init__.py`) — or set
it explicitly to e.g. `/kaggle/input/worldexplorer-engine`.

## 3. The CONFIG

| key | meaning | default |
|---|---|---|
| `repo` | `git+https://github.com/<you>/worldexplorer.git[@tag]` | — |
| `engine_dataset` | offline: path to the attached repo dataset (None = auto-find) | None |
| `data_root` | competition input dir (None = auto-detect the dir holding train+test) | None |
| `target` | column to predict (None = auto-detect) | None |
| `train`/`test`/`sample_submission` | file names if non-standard (None = auto) | None |
| `submission_target_col` | submission column to fill (None = last column) | None |
| `metric` | `auto`/`pearson`/`gini`/`spearman`/`rmse` | auto |
| `geometry` | `auto`/`temporal`/`random` | auto |
| `time_budget_min` | minutes of search (or `auto`) | 120 |
| `overrides` | any `HarnessConfig` field, e.g. `{"WIDTH_BIAS_START": 0.8}` | `{}` |

## 4. The self-improvement + advisor loop still work

Attach a previous run's **output** as an input and the engine finds
`world_cairn.json` / `learning_ledger.json` (cross-run governor warm-start,
survivors, the self-tuning width's measured `width_decay_corr`) and, for the
v36 advisor loop, `advisor_instructions.json` — no config change needed. Each
run also writes `explorer_findings_graph.json`; feed it to an LLM out-of-band
(`tools/advisor_stub.py`) to produce the next run's `advisor_instructions.json`.

## 5. Local / CLI use (same engine, no Kaggle)

```bash
pip install -e .
worldexplorer train.parquet --target label --test test.parquet --out ./out
# or in Python:
python -c "import worldexplorer as wx; print(wx.explore('train.parquet', target='label', test='test.parquet'))"
```
