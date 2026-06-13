#!/usr/bin/env python3
"""Flag-ON smoke for the v35 feature-topology + agreement + anti-fragility build.

Part A (one tiny e2e run): the feature topology graph + per-feature train->test
shift are built; the testlike_stable + consensus ranker families are surveyed;
the agreement_weighted strategy competes in the tournament; the topology +
anti-fragility reports are written.

Part B (direct micro-tests of the two new rankers): testlike_stable demotes a
HIGH-corr HIGH-shift feature below a lower-corr LOW-shift one; consensus lifts a
feature embedded in an AGREEING topology community above a lone stronger
correlate. These prove the mechanism, not just that it ran.

Usage: python tools/smoke_v35.py worldexplorer/_engine.py
"""
import sys, io, importlib.util, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v35", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v35"] = m
spec.loader.exec_module(m)

out = Path(tempfile.mkdtemp(prefix="smoke_v35_"))
m.OUT = out
c = m.CFG
c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
c.SENSORY_ROSTER = False; c.WIDTH_SELF_TUNE = False     # v34 features (already proven) off => fast roster
c.TIME_BUDGET_MIN = 0.0; c.AUDITION_ALL_SKILLS = False; c.HETERO_PAIRING = False
c.N_EXPLORERS = 3; c.LESSON_BUDGET = 2
c.EVOLUTION_BUDGET = 6; c.EVOLUTION_MAX_GENERATIONS = 1; c.EVOLUTION_POP = 5; c.EVOLUTION_OFFSPRING = 3
c.PREDATOR_BUDGET = 2; c.PREDATOR_MAX_TARGETS = 2; c.DIVE_BUDGET = 0; c.ABLATION_BUDGET = 0
c.PROBE_MAX_ROWS = 6000; c.N_SEGMENTS = 8; c.N_SPLITS = 3; c.WF_FOLDS = 2
c.MLP_MAX_ITER = 2; c.MLP_MAX_ROWS = 2000; c.DREAM_REPLAYS = 10
c.ROBUST_BOOT = 15; c.ROBUST_SAMPLE_ROWS = 6000; c.CPCV_MAX_PATHS = 3
c.FORENSIC_ACTIONS = False; c.MAX_FAMILY_MEMBERS = 3; c.MAX_MEMBERS = 8; c.STABSEL_BOOT = 5

buf = io.StringIO()


class Tee(io.TextIOBase):
    def write(self, s):
        buf.write(s)
        return sys.__stdout__.write(s)


with redirect_stdout(Tee()):
    s = m.ExplorerHarness(c).run()
logtext = buf.getvalue()

# ---- Part A: e2e ------------------------------------------------------------
assert "feature_topology_built" in logtext, "feature topology graph not built"
assert "feature_shift_built" in logtext, "per-feature train->test shift not built"
assert m.FEATURE_GRAPH is not None and m.FEATURE_GRAPH.n_communities >= 1, "no feature communities"
print(f"OK feature topology: {m.FEATURE_GRAPH.n_communities} communities; "
      f"shift vector len={len(m.FEATURE_SHIFT)}")

survey = pd.read_csv(out / "survey_map.csv")
fams = set(survey["family"])
assert "testlike_stable" in fams and "consensus" in fams, f"new families not surveyed: {sorted(fams)}"
print("OK testlike_stable + consensus families surveyed")

assert "agreement_weighted" in s["honest_scores"], "agreement_weighted strategy did not compete"
print(f"OK agreement_weighted competed (honest={s['honest_scores']['agreement_weighted']:.4f}; "
      f"tournament winner={s['ensemble_winner']})")

topo = pd.read_csv(out / "feature_topology_report.csv")
assert {"community", "topology_coherence", "signal_consensus", "mean_train_test_shift"} <= set(topo.columns)
af = pd.read_csv(out / "antifragility_report.csv")
assert {"member", "world_floor", "antifragility"} <= set(af.columns) and len(af) >= 1
assert af["antifragility"].is_monotonic_decreasing, "antifragility report not sorted"
print(f"OK reports: feature_topology_report ({len(topo)} communities), "
      f"antifragility_report ({len(af)} members, top={af.iloc[0]['member']})")

# ---- Part B: direct ranker micro-tests --------------------------------------
rng = np.random.default_rng(0)
n = 2000
y = rng.standard_normal(n).astype(np.float32)
sp = m.ViewportSpec(name="x", family="top", k=4, transform="identity")
seg = np.zeros(n, np.int32)

# testlike_stable: f0 highest corr but HIGHEST shift; f1 lower corr, LOW shift
X = np.empty((n, 3), np.float32)
X[:, 0] = 0.9 * y + 0.2 * rng.standard_normal(n)     # strongest correlate
X[:, 1] = 0.6 * y + 0.2 * rng.standard_normal(n)     # weaker correlate
X[:, 2] = 0.3 * y + 0.5 * rng.standard_normal(n)
m.FEATURE_SHIFT = np.array([10.0, 0.1, 0.1], np.float64)   # f0 moves hard train->test
m.CFG.SHIFT_PENALTY = 0.5
ranked = m._rank_testlike_stable(sp, X, y, seg, [0, 1, 2], ())
assert ranked.index(1) < ranked.index(0), \
    f"testlike_stable did not demote the high-shift strongest feature (got {ranked})"
print(f"OK testlike_stable: strongest-but-shifting feature 0 demoted below stable feature 1 -> {ranked}")

# consensus: f0,f1,f2 agree in a community; lone f3 has higher raw corr
X2 = np.empty((n, 4), np.float32)
X2[:, 0] = 0.40 * y + 0.3 * rng.standard_normal(n)
X2[:, 1] = 0.40 * y + 0.3 * rng.standard_normal(n)
X2[:, 2] = 0.40 * y + 0.3 * rng.standard_normal(n)
X2[:, 3] = 0.52 * y + 0.3 * rng.standard_normal(n)   # lone, slightly stronger correlate


class _FG:
    community = np.array([0, 0, 0, 1], np.int32)      # 0,1,2 one community; 3 alone
    n_communities = 2
    coherence = np.array([1.0, 0.0], np.float64)


m.FEATURE_GRAPH = _FG()
m.CFG.CONSENSUS_BONUS = 0.5
ranked2 = m._rank_consensus(sp, X2, y, seg, [0, 1, 2, 3], ())
assert ranked2[0] in (0, 1, 2), \
    f"consensus did not lift an agreeing-community feature above the lone correlate (got {ranked2})"
print(f"OK consensus: agreeing-community feature lifted above lone stronger correlate -> {ranked2}")

print(f"shipped: winner={s.get('ensemble_winner')} members={len(s.get('members') or [])}")
print("SMOKE OK")
