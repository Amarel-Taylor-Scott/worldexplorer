#!/usr/bin/env python3
"""Flag-ON smoke for the v31 layers (IDEAS_ZOO B1+B2).

E2E on the tiny smoke config:
  - testlike sensor fires: testlike_report.json with AUC (≈0.5 on the
    stationary synthetic = correct: train and test share one distribution);
  - the robust selector scores configs over +2 testlike partitions
    (spy proves flag-on vs flag-off partition counts);
  - redundancy_factor_report.csv exists with new_info in [0,1] and factor
    exposure columns + crowding cosine.

Usage: python tools/smoke_v31.py worldexplorer/_engine.py
"""
import sys, importlib.util, json, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v31", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v31"] = m
spec.loader.exec_module(m)

out = Path(tempfile.mkdtemp(prefix="smoke_v31_"))
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

seen = {}
_orig = m.robust_oos_select
def _spy(cand_weights, members, member_lessons, spec_lookup, X_full, y_full,
         seg_full, n_work, cols, cfg):
    on = _orig(cand_weights, members, member_lessons, spec_lookup,
               X_full, y_full, seg_full, n_work, cols, cfg)
    cfg.TESTLIKE_PARTITIONS = False
    off = _orig(cand_weights, members, member_lessons, spec_lookup,
                X_full, y_full, seg_full, n_work, cols, cfg)
    cfg.TESTLIKE_PARTITIONS = True
    seen["on"], seen["off"] = on.get("partitions", 0), off.get("partitions", 0)
    return on
m.robust_oos_select = _spy

s = m.ExplorerHarness(c).run()

tl = json.loads((out / "testlike_report.json").read_text())
print(f"E2E: testlike holdout AUC={tl['auc_holdout']} (synthetic HAS designed train->test "
      f"drift: fresh factor loadings per block, so AUC>0.5 is honest), "
      f"q10/q50/q90={tl['score_q10']}/{tl['score_q50']}/{tl['score_q90']}")
assert 0.5 < tl["auc_holdout"] <= 1.0, "drifted synthetic should be separable on holdout"
assert seen.get("on", 0) - seen.get("off", 0) == 2, f"testlike partitions missing: {seen}"
print(f"E2E: robust partitions flag-on={seen['on']} vs flag-off={seen['off']} (+2 testlike)")
rf = pd.read_csv(out / "redundancy_factor_report.csv")
assert rf["new_info"].between(0, 1).all() and "crowding_cos" in rf.columns
print(f"E2E: redundancy_factor_report rows={len(rf)} | new_info min/max="
      f"{rf['new_info'].min():.3f}/{rf['new_info'].max():.3f} | max crowding_cos="
      f"{rf['crowding_cos'].max():.3f}")
print(rf[["member", "new_info", "crowding_partner", "crowding_cos"]].head(4).to_string(index=False))
print(f"E2E: shipped winner={s.get('ensemble_winner')} members={len(s.get('members') or [])}")
print("SMOKE OK")
