"""worldexplorer -- zero-config, self-improving, overfit-resistant tabular ML.

    import worldexplorer as wx
    result = wx.explore(df, target="label")        # that's the whole API
    result.predictions   # id + prediction
    result.score         # honest holdout score (if no test supplied)
    result.profile       # what it auto-detected
    result.artifacts_dir # every report it wrote

The engine is a civilization of bounded explorers that move through compressed
feature/latent worlds; promotes only paths that are wide, stable, unique, cheap,
and hard to kill; measures complexity-vs-decay at runtime and ships at the
complexity the data rewards; and accumulates cross-run learnings. You supply the
data + target; it figures out the metric, the CV geometry, the ids, the encodings,
and the compute budget. See README.md for the full framework.
"""
from .adapter import Result, explore
from .autoconfig import DataProfile, profile

__version__ = "0.1.0"
__all__ = ["explore", "profile", "Result", "DataProfile", "AutoExplorer", "__version__"]


class AutoExplorer:
    """Stateful sklearn-ish wrapper: ae = AutoExplorer(); ae.fit(df, target); ae.predict(test)."""

    def __init__(self, time_budget="auto", out=None, **overrides):
        self.time_budget = time_budget
        self.out = out
        self.overrides = overrides
        self.result_ = None
        self.profile_ = None

    def fit(self, data, target=None, test=None, verbose=True):
        self.result_ = explore(data, target=target, test=test, out=self.out,
                               time_budget=self.time_budget, verbose=verbose, **self.overrides)
        self.profile_ = self.result_.profile
        return self

    def predict(self, test=None):
        if self.result_ is None:
            raise RuntimeError("call .fit(...) first")
        return self.result_.predictions

    def profile(self, data, target=None):
        return profile(data if not isinstance(data, str) else __import__("pandas").read_parquet(data), target)
