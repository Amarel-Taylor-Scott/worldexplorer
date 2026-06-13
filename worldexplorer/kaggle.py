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
