#!/usr/bin/env python3
"""Flag-ON smoke for the v29 pls_weight ranker family.

UNIT: y is driven by two latents -- s (strong, present as FOUR collinear
copies) and u (weaker, ONE independent column). corr-ranking stacks the four
s-copies on top (univariate double-counting); pls_weight must rank the
independent u-column ABOVE the redundant copies (multivariate coefficient
mass splits across duplicates).

E2E: tiny smoke harness run; pls_weight must be satellite-surveyed.

Usage: python tools/smoke_v29.py worldexplorer/_engine.py
"""
import sys, importlib.util, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v29", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v29"] = m
spec.loader.exec_module(m)

rng = np.random.default_rng(17)
n = 4800
seg = (np.arange(n) * 8 // n).astype(np.int32)
s = rng.normal(size=n).astype(np.float32)
u = rng.normal(size=n).astype(np.float32)
y = (s + 0.6 * u + 0.5 * rng.normal(size=n)).astype(np.float32)
X = rng.normal(size=(n, 10)).astype(np.float32) * 0.1
for j in (2, 4, 5, 6):                                   # four collinear copies of s
    X[:, j] = s + 0.05 * rng.normal(size=n).astype(np.float32)
X[:, 8] = u + 0.05 * rng.normal(size=n).astype(np.float32)   # one independent u column
cols = [f"X{j}" for j in range(10)]

assert "pls_weight" in m.FAMILIES, "pls_weight missing from FAMILIES"
m._RANK_CACHE.clear()
r_top = m._ranked_for(m.ViewportSpec("top10_identity", "top", 10, "identity", 8), X, y, seg, cols)
m._RANK_CACHE.clear()
r_pls = m._ranked_for(m.ViewportSpec("pls_weight10_identity", "pls_weight", 10, "identity", 8),
                      X, y, seg, cols)
pos_u_top, pos_u_pls = r_top.index(8), r_pls.index(8)
print(f"corr ranking:  u-column @ {pos_u_top} (top-4 = {r_top[:4]})")
print(f"pls_weight:    u-column @ {pos_u_pls} (top-4 = {r_pls[:4]})")
assert pos_u_top >= 4, "fixture broken: corr should stack the 4 s-copies above u"
assert pos_u_pls < pos_u_top, "pls_weight failed to promote the independent column"
assert pos_u_pls <= 2, "pls_weight should put the independent column near the top"
print("UNIT OK: independent signal promoted above collinear duplicates")

out = Path(tempfile.mkdtemp(prefix="smoke_v29_"))
m.OUT = out
c = m.CFG
c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
c.TIME_BUDGET_MIN = 0.0; c.AUDITION_ALL_SKILLS = False; c.HETERO_PAIRING = False
c.N_EXPLORERS = 2; c.LESSON_BUDGET = 3
c.EVOLUTION_BUDGET = 5; c.EVOLUTION_MAX_GENERATIONS = 1; c.EVOLUTION_POP = 5; c.EVOLUTION_OFFSPRING = 3
c.PREDATOR_BUDGET = 2; c.PREDATOR_MAX_TARGETS = 2; c.DIVE_BUDGET = 0; c.ABLATION_BUDGET = 0
c.PROBE_MAX_ROWS = 6000; c.N_SEGMENTS = 8; c.N_SPLITS = 3; c.WF_FOLDS = 2
c.MLP_MAX_ITER = 2; c.MLP_MAX_ROWS = 2000; c.DREAM_REPLAYS = 10
c.ROBUST_BOOT = 15; c.ROBUST_SAMPLE_ROWS = 6000; c.CPCV_MAX_PATHS = 3
c.FORENSIC_ACTIONS = False; c.MAX_FAMILY_MEMBERS = 3; c.MAX_MEMBERS = 8; c.STABSEL_BOOT = 5
s_run = m.ExplorerHarness(c).run()
survey = pd.read_csv(out / "survey_map.csv")
assert "pls_weight" in set(survey["family"]), "pls_weight not satellite-surveyed"
row = survey[survey["family"] == "pls_weight"].iloc[0]
print(f"E2E: pls_weight surveyed (survey_corr={row['survey_corr']:.4f}); "
      f"winner={s_run.get('ensemble_winner')} members={len(s_run.get('members') or [])}")
print("SMOKE OK")
