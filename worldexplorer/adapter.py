"""Bridge a generic dataframe (+ target) to the world-explorer engine, which was
born competition-shaped (train.parquet/test.parquet/sample_submission + 'label').
The adapter does the impedance-matching: auto-profile, encode categoricals, drop
id columns, rename target, write the engine's expected layout, point the engine's
config at it, run, and return predictions (+ a score if a holdout was used)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .autoconfig import DataProfile, profile as _profile

DataLike = Union[pd.DataFrame, str, "Path"]


@dataclass
class Result:
    predictions: pd.DataFrame      # id + prediction
    score: Optional[float]         # holdout score if no test was supplied
    profile: DataProfile
    report: dict                   # the engine's run summary
    artifacts_dir: str             # folder of every CSV/JSON the run wrote

    def __repr__(self) -> str:
        s = f"{self.score:.5f}" if self.score is not None else "n/a (test supplied)"
        return (f"<worldexplorer.Result score={s} rows={len(self.predictions)} "
                f"metric={self.profile.metric} artifacts='{self.artifacts_dir}'>")


def _read(d: DataLike) -> pd.DataFrame:
    if isinstance(d, pd.DataFrame):
        return d.copy()
    p = str(d)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(p)
    if p.endswith(".csv") or p.endswith(".txt"):
        return pd.read_csv(p)
    if p.endswith(".feather"):
        return pd.read_feather(p)
    return pd.read_parquet(p)


def _load_engine():
    from . import _engine  # vendored single-file engine (built from src/ by build.py)
    return _engine


def _encode(train: pd.DataFrame, test: pd.DataFrame, cats: list) -> None:
    for c in cats:
        if c not in train.columns:
            continue
        joint = pd.Categorical(pd.concat([train[c], test[c]], axis=0, ignore_index=True).astype(str))
        train[c] = pd.Categorical(train[c].astype(str), categories=joint.categories).codes.astype(np.float32)
        if c in test.columns:
            test[c] = pd.Categorical(test[c].astype(str), categories=joint.categories).codes.astype(np.float32)


def _auto_budget(n_rows: int, n_feats: int, user: Any) -> float:
    if user not in (None, "auto"):
        return float(user)
    base = (n_rows * max(n_feats, 1)) / 2.0e7
    return float(min(max(base, 3.0), 45.0))   # 3..45 min default heuristic


def explore(data: DataLike, target: Optional[str] = None, test: Optional[DataLike] = None,
            out: Optional[str] = None, time_budget: Any = "auto",
            profile_only: bool = False, verbose: bool = True, **overrides) -> Any:
    """Zero-config entrypoint. `data` = a training dataframe/path (must contain the
    target); `test` optional (no target). If `test` is None, an honest holdout is
    carved from `data` (temporal tail or random) and the returned Result carries
    its score. Anything in **overrides matching a HarnessConfig field is forwarded."""
    train = _read(data)
    test_df = _read(test) if test is not None else None
    prof = _profile(train, target, test_df)
    target = prof.target
    if profile_only:
        return prof
    if verbose:
        print("[worldexplorer] auto-profile:", prof.summary())
        for n in prof.notes:
            print("  -", n)

    has_truth, y_true = False, None
    if test_df is None:
        n = len(train); cut = max(1, int(n * 0.85))
        if prof.geometry == "temporal":
            tr, te = train.iloc[:cut].copy(), train.iloc[cut:].copy()
        else:
            idx = np.random.default_rng(0).permutation(n)
            tr, te = train.iloc[idx[:cut]].copy(), train.iloc[idx[cut:]].copy()
        y_true = te[target].to_numpy()
        test_df = te.drop(columns=[target]); train = tr; has_truth = True

    # id columns: keep for output, remove from features
    id_present = [c for c in prof.id_cols if c in test_df.columns]
    test_ids = test_df[id_present[0]].to_numpy() if id_present else np.arange(len(test_df))
    id_name = id_present[0] if id_present else "id"
    train = train.drop(columns=[c for c in prof.id_cols if c in train.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in prof.id_cols if c in test_df.columns], errors="ignore")

    _encode(train, test_df, [c for c in prof.categorical_cols if c in train.columns])
    train = train.rename(columns={target: "label"})
    feats = [c for c in train.columns if c != "label"]
    for c in feats:                                   # align test to train features
        if c not in test_df.columns:
            test_df[c] = 0.0
    test_df = test_df[feats]

    tmp = Path(tempfile.mkdtemp(prefix="wx_data_"))
    train.to_parquet(tmp / "train.parquet")
    test_df.to_parquet(tmp / "test.parquet")
    pd.DataFrame({"id": np.arange(len(test_df)), "prediction": np.zeros(len(test_df), np.float32)}) \
        .to_csv(tmp / "sample_submission.csv", index=False)

    out_dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="wx_out_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    eng = _load_engine()
    cfg = eng.HarnessConfig()
    cfg.DATA_ROOTS = (str(tmp),)
    cfg.OUT_DIR = str(out_dir)
    cfg.ALLOW_SYNTHETIC_FALLBACK = False
    cfg.METRIC = prof.metric
    cfg.GEOMETRY = "auto"                              # let the engine's own sensor confirm/adapt
    cfg.TIME_BUDGET_MIN = _auto_budget(prof.n_rows, prof.n_features, time_budget)
    # auto-scale the DRW-shaped defaults to THIS dataset's size + geometry. The
    # engine's EMBARGO_ROWS=720 (minutes) is competition-specific; on small or
    # tabular data it strips a CV fold to empty. Embargo only matters for time-
    # ordered data, and must be a small fraction of the rows.
    cfg.EMBARGO_ROWS = 0 if prof.geometry != "temporal" else max(1, min(int(cfg.EMBARGO_ROWS), prof.n_rows // 200))
    cfg.N_SEGMENTS = int(max(4, min(int(cfg.N_SEGMENTS), max(4, prof.n_rows // 400))))
    cfg.ROBUST_MIN_TRAIN = int(min(int(cfg.ROBUST_MIN_TRAIN), max(200, prof.n_rows // 4)))
    cfg.ROBUST_MIN_TEST = int(min(int(cfg.ROBUST_MIN_TEST), max(50, prof.n_rows // 20)))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    eng.CFG = cfg                                      # engine reads module-global CFG/OUT in places
    eng.OUT = out_dir
    summary = eng.ExplorerHarness(cfg).run()

    sub = pd.read_csv(out_dir / "submission.csv")
    pred_col = "prediction" if "prediction" in sub.columns else sub.columns[-1]
    preds = sub[pred_col].to_numpy()
    score = None
    if has_truth and y_true is not None and len(preds) == len(y_true):
        try:
            score = float(eng.score_metric(np.asarray(y_true, np.float64), np.asarray(preds, np.float64)))
        except Exception:
            score = None
    pred_df = pd.DataFrame({id_name: np.asarray(test_ids)[: len(preds)], "prediction": preds})
    if verbose:
        print(f"[worldexplorer] done. score={score} artifacts={out_dir}")
    return Result(predictions=pred_df, score=score, profile=prof, report=summary, artifacts_dir=str(out_dir))
