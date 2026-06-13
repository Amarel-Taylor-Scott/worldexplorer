"""Thin Kaggle entrypoint: a CONFIG dict -> a finished submission.

The whole notebook cell becomes

    import worldexplorer as wx
    wx.kaggle.run(CONFIG)

so the cell carries only WHERE the data is and HOW to run -- never the engine.
All the logic lives in this repo (pip-installed from GitHub, or attached as a
Kaggle dataset for Internet-OFF competitions). `run` resolves the data files
(auto-detecting the competition input dir if not told), forwards everything to
the zero-config `explore` adapter (which auto-profiles metric/geometry/ids/
encodings and produces test predictions), then maps those predictions onto the
competition's own sample_submission and writes submission.csv.

Cross-run self-improvement and the v36 advisor loop need no special handling
here: attach a previous run's OUTPUT (for world_cairn.json / learning_ledger.json)
and/or an advisor_instructions.json as inputs, and the engine finds them via its
own /kaggle/input globs.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

DEFAULTS: dict = {
    "data_root": None,             # auto-detect a /kaggle/input/* dir holding train+test if None
    "target": None,                # target column (auto-detected if obvious / from sample_submission)
    "train": None,                 # file name or path; auto if None
    "test": None,                  # file name or path; auto if None
    "sample_submission": None,     # file name or path; auto if None
    "submission_target_col": None,  # default: the LAST column of sample_submission
    "metric": "auto",              # "auto" | "pearson" | "gini" | "spearman" | "rmse"
    "geometry": "auto",            # "auto" | "temporal" | "random"
    "time_budget_min": "auto",     # minutes, or "auto"
    "out": None,                   # default: /kaggle/working if present else ./wx_out
    "overrides": None,             # any HarnessConfig field -> forwarded to the engine
    "verbose": True,
}

_TRAIN_NAMES = ("train.parquet", "train.pq", "train.csv", "train.feather")
_TEST_NAMES = ("test.parquet", "test.pq", "test.csv", "test.feather")
_SUB_NAMES = ("sample_submission.csv", "sampleSubmission.csv", "sample_submission.csv.zip")


def _first_existing(root: str, names) -> "str | None":
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


def _autodetect_root(cfg: dict) -> str:
    if cfg.get("data_root"):
        return cfg["data_root"]
    for base in ("/kaggle/input",):                 # the dir holding BOTH a train and a test file
        for d in sorted(glob.glob(base + "/*")) + sorted(glob.glob(base + "/*/*")):
            if os.path.isdir(d) and _first_existing(d, _TRAIN_NAMES) and _first_existing(d, _TEST_NAMES):
                return d
    return "."


def _resolve(root: str, explicit: "str | None", names) -> "str | None":
    if explicit:
        if os.path.isabs(explicit) or os.path.exists(explicit):
            return explicit
        return os.path.join(root, explicit)
    return _first_existing(root, names)


def _train_rows(train_path: "str | None") -> int:
    """Row count from parquet metadata (no data load); 0 if unknown."""
    try:
        if train_path and train_path.endswith((".parquet", ".pq")):
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(train_path).metadata.num_rows)
    except Exception:
        pass
    return 0


def _run_engine_native(root: str, train_path: "str | None", out: str, cfg: dict, overrides: dict) -> Any:
    """Run the engine DIRECTLY on the competition dir (no adapter round-trip).
    The engine loads train/test.parquet, does its own feature engineering, and
    writes submission.csv mapped onto the competition's sample_submission."""
    import pandas as pd

    from .adapter import Result, _load_engine

    eng = _load_engine()
    c = eng.HarnessConfig()
    c.DATA_ROOTS = (root,) + tuple(getattr(c, "DATA_ROOTS", ()))
    c.OUT_DIR = out
    c.ALLOW_SYNTHETIC_FALLBACK = False
    tb = cfg.get("time_budget_min")
    if tb not in (None, "auto"):
        c.TIME_BUDGET_MIN = float(tb)
    # SAFETY CLAMP: the engine's DRW-tuned EMBARGO_ROWS=720 (and N_SEGMENTS) are
    # correct on a 500k-row set but strip CV folds to empty on small data. Clamp
    # to a safe fraction of the actual row count unless the user overrode them
    # (a no-op on DRW: min(720, 525886//200=2629)=720, min(12, 1314)=12).
    nrows = _train_rows(train_path)
    if nrows > 0:
        if "EMBARGO_ROWS" not in overrides:
            c.EMBARGO_ROWS = max(0, min(int(c.EMBARGO_ROWS), nrows // 200))
        if "N_SEGMENTS" not in overrides:
            c.N_SEGMENTS = max(4, min(int(c.N_SEGMENTS), max(4, nrows // 400)))
    for k, v in overrides.items():
        if hasattr(c, k):
            setattr(c, k, v)
    eng.CFG = c
    eng.OUT = Path(out)
    if cfg.get("verbose"):
        print(f"[worldexplorer.kaggle] engine-native run (budget={c.TIME_BUDGET_MIN} min) on {root}")
    summary = eng.ExplorerHarness(c).run()           # writes out/submission.csv itself
    sub_file = Path(out) / "submission.csv"
    preds = pd.read_csv(sub_file) if sub_file.exists() else pd.DataFrame()
    metric = eng.PROFILE.get("metric", "pearson") if isinstance(getattr(eng, "PROFILE", None), dict) else "pearson"
    prof = type("P", (), {"metric": metric})()
    if cfg.get("verbose"):
        print(f"[worldexplorer.kaggle] engine-native done; submission rows={len(preds)} "
              f"sealed={summary.get('sealed_holdout_corr')}")
    return Result(predictions=preds, score=summary.get("sealed_holdout_corr"),
                  profile=prof, report=summary, artifacts_dir=str(out))


def run(config: "dict | None" = None) -> Any:
    """Run the world-explorer on a Kaggle competition from a CONFIG dict and
    write submission.csv. Returns the worldexplorer Result (predictions + the
    run summary + the artifacts dir)."""
    import numpy as np
    import pandas as pd

    from . import explore

    cfg = {**DEFAULTS, **(config or {})}
    root = _autodetect_root(cfg)
    train_path = _resolve(root, cfg["train"], _TRAIN_NAMES)
    test_path = _resolve(root, cfg["test"], _TEST_NAMES)
    sub_path = _resolve(root, cfg["sample_submission"], _SUB_NAMES)
    out = cfg["out"] or ("/kaggle/working" if os.path.isdir("/kaggle/working") else "wx_out")
    Path(out).mkdir(parents=True, exist_ok=True)

    if cfg["verbose"]:
        print(f"[worldexplorer.kaggle] data_root={root}")
        print(f"  train={train_path}")
        print(f"  test={test_path}")
        print(f"  sample_submission={sub_path}")
        print(f"  out={out}")
    if not train_path:
        raise SystemExit(f"no training file found under {root}; set CONFIG['data_root'] or CONFIG['train']")

    overrides = dict(cfg.get("overrides") or {})
    for key, field in (("metric", "METRIC"), ("geometry", "GEOMETRY")):
        v = cfg.get(key)
        if v and v != "auto":
            overrides[field] = v

    # ENGINE-NATIVE FAST PATH: when the data is already in the engine's native
    # layout (train/test parquet + a 'label' target + a sample_submission), run
    # the engine DIRECTLY on the competition dir -- no adapter round-trip through
    # pandas, which on a 525k x 800 set like DRW would double peak memory and risk
    # an OOM. Faithful to the single-cell kernel; the engine writes submission.csv
    # mapped onto the competition's own sample_submission. Otherwise use the
    # general zero-config adapter (auto metric/geometry/ids/encodings).
    native = cfg.get("engine_native", "auto")
    native_ok = (native is True) or (
        native == "auto" and cfg["target"] in (None, "label")
        and bool(train_path) and train_path.endswith((".parquet", ".pq"))
        and bool(test_path) and bool(sub_path))
    if native_ok:
        return _run_engine_native(root, train_path, out, cfg, overrides)

    result = explore(train_path, target=cfg["target"], test=test_path, out=out,
                     time_budget=cfg["time_budget_min"], verbose=cfg["verbose"], **overrides)

    # map predictions onto the competition's OWN sample_submission layout
    preds = np.asarray(result.predictions["prediction"].to_numpy())
    if sub_path and os.path.exists(sub_path):
        sub = pd.read_csv(sub_path)
        tcol = cfg["submission_target_col"] or sub.columns[-1]
        if len(preds) != len(sub):
            print(f"[worldexplorer.kaggle] WARNING: {len(preds)} predictions vs {len(sub)} "
                  "sample_submission rows; aligning by position")
        sub[tcol] = preds[: len(sub)]
        sub.to_csv(os.path.join(out, "submission.csv"), index=False)
        if cfg["verbose"]:
            print(f"[worldexplorer.kaggle] wrote {out}/submission.csv "
                  f"({len(sub)} rows, target column '{tcol}')")
    else:
        result.predictions.to_csv(os.path.join(out, "submission.csv"), index=False)
        print(f"[worldexplorer.kaggle] no sample_submission found under {root}; "
              f"wrote raw predictions to {out}/submission.csv")
    if cfg["verbose"]:
        print(result)
    return result
