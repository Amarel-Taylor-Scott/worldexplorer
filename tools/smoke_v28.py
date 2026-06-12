#!/usr/bin/env python3
"""Flag-ON smoke for the v28 selection hardeners.

1. UNIT: plant a sign-FLIPPING feature (strong pooled |corr|, sign alternating
   across segments) and a weaker but sign-STABLE feature; assert the
   sign_stability ranker demotes the flipper behind the stable one (a plain
   corr ranking would order them the other way).
2. E2E: run the tiny smoke harness; assert sign_stability was satellite-
   surveyed (it is a real family now) and the robust selector scored configs
   over MORE partitions including the interior blocks.

Usage: python tools/smoke_v28.py worldexplorer/_engine.py
"""
import sys, importlib.util, json, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v28", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v28"] = m
spec.loader.exec_module(m)

# ---- 1. unit: flip-demotion --------------------------------------------------
rng = np.random.default_rng(5)
n = 4800
seg = (np.arange(n) * 8 // n).astype(np.int32)
y = rng.normal(size=n).astype(np.float32)
X = rng.normal(size=(n, 10)).astype(np.float32) * 0.1
flip_sign = np.where(np.isin(seg, [1, 4, 6]), -1.0, 1.0).astype(np.float32)   # negative in 3/8 segments
X[:, 3] = 1.5 * flip_sign * y + 0.3 * rng.normal(size=n).astype(np.float32)   # strong POOLED corr, sign flips
X[:, 7] = 0.2 * y + 1.0 * rng.normal(size=n).astype(np.float32)               # weaker pooled corr, sign-stable
cols = [f"X{j}" for j in range(10)]

assert "sign_stability" in m.FAMILIES, "sign_stability missing from FAMILIES"
m._RANK_CACHE.clear()
sp = m.ViewportSpec(name="sign_stability8_identity", family="sign_stability", k=8,
                    transform="identity", proj_dim=8)
ranked = m._ranked_for(sp, X, y, seg, cols)
m._RANK_CACHE.clear()
sp_top = m.ViewportSpec(name="top8_identity", family="top", k=8, transform="identity", proj_dim=8)
ranked_top = m._ranked_for(sp_top, X, y, seg, cols)
pos_flip, pos_stab = ranked.index(3), ranked.index(7)
print(f"corr ranking puts flipper first: {ranked_top.index(3) < ranked_top.index(7)} "
      f"(top order: flipper@{ranked_top.index(3)}, stable@{ranked_top.index(7)})")
print(f"sign_stability order: flipper@{pos_flip}, stable@{pos_stab}")
assert ranked_top.index(3) < ranked_top.index(7), "fixture broken: flipper should win pooled corr"
assert pos_stab < pos_flip, "sign_stability failed to demote the flipper"
print("UNIT OK: sign-flipper demoted behind sign-stable feature")

# ---- 2. e2e: family surveyed + interior partitions present -------------------
out = Path(tempfile.mkdtemp(prefix="smoke_v28_"))
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

# spy: call the robust selector twice (flag on / off) to PROVE the interior
# partitions are present and counted; the harness proceeds with the on-result
seen = {}
_orig_robust = m.robust_oos_select
def _spy(cand_weights, members, member_lessons, spec_lookup, X_full, y_full,
         seg_full, n_work, cols, cfg):
    on = _orig_robust(cand_weights, members, member_lessons, spec_lookup,
                      X_full, y_full, seg_full, n_work, cols, cfg)
    cfg.ROBUST_INTERIOR = False
    off = _orig_robust(cand_weights, members, member_lessons, spec_lookup,
                       X_full, y_full, seg_full, n_work, cols, cfg)
    cfg.ROBUST_INTERIOR = True
    seen["on"], seen["off"] = on.get("partitions", 0), off.get("partitions", 0)
    return on
m.robust_oos_select = _spy

s = m.ExplorerHarness(c).run()

survey = pd.read_csv(out / "survey_map.csv")
assert "sign_stability" in set(survey["family"]), "sign_stability not satellite-surveyed"
row = survey[survey["family"] == "sign_stability"].iloc[0]
print(f"E2E: sign_stability surveyed (survey_corr={row['survey_corr']:.4f})")
assert seen.get("on", 0) - seen.get("off", 0) == 2, f"interior partitions missing: {seen}"
print(f"E2E: robust partitions flag-on={seen['on']} vs flag-off={seen['off']} (+2 interior blocks)")
print(f"E2E: shipped decision: winner={s.get('ensemble_winner')} "
      f"members={len(s.get('members') or [])} sealed={s.get('sealed_holdout_corr')}")
print("SMOKE OK")
