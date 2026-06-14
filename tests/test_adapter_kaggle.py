from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import worldexplorer
from worldexplorer.adapter import _encode
from worldexplorer.kaggle import run


def test_encode_tolerates_train_only_categorical_columns():
    train = pd.DataFrame({"cat": ["a", "b", "a"], "label": [1.0, 0.0, 1.0]})
    test = pd.DataFrame({"num": [1.0, 2.0]})

    _encode(train, test, ["cat"])

    assert train["cat"].dtype == np.float32
    assert "cat" not in test.columns


def test_kaggle_run_rejects_prediction_sample_submission_length_mismatch(tmp_path, monkeypatch):
    pd.DataFrame({"feature": [1.0, 2.0], "target": [0.1, 0.2]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"feature": [3.0, 4.0]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": [10, 11], "target": [0.0, 0.0]}).to_csv(
        tmp_path / "sample_submission.csv",
        index=False,
    )

    def fake_explore(*args, **kwargs):
        return SimpleNamespace(
            predictions=pd.DataFrame({"prediction": [0.5]}),
            score=None,
            profile=SimpleNamespace(metric="rmse"),
            report={},
            artifacts_dir=str(tmp_path / "out"),
        )

    monkeypatch.setattr(worldexplorer, "explore", fake_explore)

    with pytest.raises(ValueError, match="prediction length mismatch"):
        run({
            "data_root": str(tmp_path),
            "target": "target",
            "engine_native": False,
            "out": str(tmp_path / "out"),
            "verbose": False,
        })
