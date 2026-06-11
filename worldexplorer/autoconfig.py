"""Auto-detection: from a dataframe (+ optional target) figure out the task,
metric, time-ordering, id columns, categoricals, group/era columns, and a compute
budget -- so the user configures almost nothing. The harness engine has its own
metric/geometry sensor too; this layer adds the dataframe-shaping decisions the
engine assumes a Kaggle competition already made (target name, ids, encodings)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# metric vocabulary the engine's score_metric understands
_METRICS = {"pearson", "gini", "spearman", "neg_rmse"}
_TARGET_NAMES = ("label", "target", "y", "Target", "Label", "Y", "outcome", "resp", "response")
_TIME_NAMES = ("time", "timestamp", "date", "datetime", "ts", "datetime64")
_GROUP_NAMES = ("era", "group", "symbol", "ticker", "id_group", "season", "fold", "asset")


@dataclass
class DataProfile:
    target: str
    task: str                  # regression | binary | multiclass
    metric: str                # pearson | gini | spearman | neg_rmse
    geometry: str              # temporal | tabular
    id_cols: list
    categorical_cols: list
    group_col: Optional[str]
    time_col: Optional[str]
    n_rows: int
    n_features: int
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"target={self.target} task={self.task} metric={self.metric} "
                f"geometry={self.geometry} rows={self.n_rows} features={self.n_features} "
                f"ids={self.id_cols or '-'} categoricals={len(self.categorical_cols)} "
                f"group={self.group_col or '-'} time={self.time_col or '-'}")


def detect_target(df: pd.DataFrame, target: Optional[str] = None,
                  test: Optional[pd.DataFrame] = None) -> str:
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"target '{target}' not in columns")
        return target
    for c in _TARGET_NAMES:
        if c in df.columns:
            return c
    # the column present in train but NOT in test is the target
    if test is not None:
        only = [c for c in df.columns if c not in test.columns]
        if len(only) == 1:
            return only[0]
    # fall back: the last column
    raise ValueError("Could not auto-detect the target column; pass target='<column>'. "
                     f"Columns: {list(df.columns)[:20]}{'...' if df.shape[1] > 20 else ''}")


def detect_task_metric(y: pd.Series) -> tuple:
    yv = pd.Series(y).dropna()
    nun = int(yv.nunique())
    if nun <= 1:
        raise ValueError("Target has <2 distinct values.")
    if nun == 2:
        return "binary", "gini"
    if pd.api.types.is_float_dtype(yv) or nun > 20:
        return "regression", "pearson"
    if pd.api.types.is_integer_dtype(yv) and nun <= 20:
        # small-cardinality int: treat as multiclass; the engine optimises a
        # rank/pearson proxy (point-biserial still ranks), flagged in notes.
        return "multiclass", "spearman"
    return "regression", "pearson"


def _is_id_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    if s.nunique(dropna=False) == n and (pd.api.types.is_integer_dtype(s)
                                         or pd.api.types.is_object_dtype(s)):
        return True
    # monotonic increasing integer index column
    if pd.api.types.is_integer_dtype(s):
        a = s.to_numpy()
        if len(a) > 2 and np.all(np.diff(a) > 0):
            return True
    return False


def detect_time(df: pd.DataFrame, feature_cols: list) -> tuple:
    for c in df.columns:
        cl = str(c).lower()
        if cl in _TIME_NAMES or "date" in cl or "time" in cl:
            return "temporal", c
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return "temporal", c
    # heuristic: mean lag-1 autocorrelation of numeric features (rows time-ordered?)
    num = df[feature_cols].select_dtypes("number") if feature_cols else df.select_dtypes("number")
    if len(num) > 200 and num.shape[1] > 0:
        s = num.iloc[: min(len(num), 50000)]
        ac = []
        for c in s.columns[: min(40, s.shape[1])]:
            x = s[c].to_numpy(np.float64)
            x = x[np.isfinite(x)]
            if len(x) > 100 and x.std() > 1e-9:
                ac.append(abs(float(np.corrcoef(x[:-1], x[1:])[0, 1])))
        if ac and np.nanmean(ac) > 0.15:
            return "temporal", None
    return "tabular", None


def detect_group(df: pd.DataFrame, exclude: set) -> Optional[str]:
    for c in df.columns:
        if c in exclude:
            continue
        if str(c).lower() in _GROUP_NAMES and 1 < df[c].nunique() < len(df) * 0.5:
            return c
    return None


def profile(train: pd.DataFrame, target: Optional[str] = None,
            test: Optional[pd.DataFrame] = None) -> DataProfile:
    """The one call that 'figures out everything'. Minimal user input = the df
    (+ target if it isn't obviously named)."""
    target = detect_target(train, target, test)
    notes = []
    feats = [c for c in train.columns if c != target]
    id_cols = [c for c in feats if _is_id_like(train[c])]
    real = [c for c in feats if c not in id_cols]
    cat_cols = [c for c in real if (pd.api.types.is_object_dtype(train[c])
                                    or pd.api.types.is_categorical_dtype(train[c])
                                    or (pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() <= 20))]
    numeric_feats = [c for c in real if c not in cat_cols]
    task, metric = detect_task_metric(train[target])
    geometry, time_col = detect_time(train, numeric_feats)
    group_col = detect_group(train, exclude={target, *id_cols})
    if task == "multiclass":
        notes.append("multiclass target: optimising a Spearman/rank proxy (engine metric set is "
                     "pearson/gini/spearman/neg_rmse).")
    if id_cols:
        notes.append(f"dropping id-like columns from features: {id_cols}")
    if cat_cols:
        notes.append(f"label-encoding {len(cat_cols)} categorical/low-cardinality columns.")
    if group_col:
        notes.append(f"detected group/era column '{group_col}' (panel CV is on the roadmap; "
                     "currently treated as a feature-free grouping hint).")
    return DataProfile(target=target, task=task, metric=metric, geometry=geometry,
                       id_cols=id_cols, categorical_cols=cat_cols, group_col=group_col,
                       time_col=time_col, n_rows=len(train), n_features=len(real), notes=notes)
