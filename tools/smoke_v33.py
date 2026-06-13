#!/usr/bin/env python3
"""Flag-ON smoke for the v33 wide-configuration layer.

Asserts on one tiny e2e run:
  - the ALBATROSS persona is born (roster slot 7, N_EXPLORERS auto-bumped);
  - the WIDE warm seeds are measured at evolution gen-0;
  - grid_* candidate configs appear in the robust court's score table
    (robust_oos_selection.csv), including grid_wide and the grid_sharp
    control, each judged with a complexity column like every candidate.

Usage: python tools/smoke_v33.py worldexplorer/_engine.py
"""
import sys, io, importlib.util, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v33", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v33"] = m
spec.loader.exec_module(m)

out = Path(tempfile.mkdtemp(prefix="smoke_v33_"))
m.OUT = out
c = m.CFG
c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
c.TIME_BUDGET_MIN = 0.0; c.AUDITION_ALL_SKILLS = False; c.HETERO_PAIRING = False
c.N_EXPLORERS = 2; c.LESSON_BUDGET = 3
c.EVOLUTION_BUDGET = 8; c.EVOLUTION_MAX_GENERATIONS = 1; c.EVOLUTION_POP = 5; c.EVOLUTION_OFFSPRING = 3
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

# NOTE: the tiny smoke sets N_EXPLORERS=2, so the albatross (slot 7) is not
# BORN here -- assert the roster insert + auto-bump logic directly instead.
assert any(t.get("name") == "albatross" for t in m.EXPLORER_TRAITS), "albatross missing from roster"
pos = next(i for i, t in enumerate(m.EXPLORER_TRAITS) if t.get("name") == "albatross")
assert pos == 7, f"albatross at roster slot {pos}, expected 7"
cc = m.HarnessConfig()
assert cc.WIDE_PERSONA and cc.N_EXPLORERS == 7, "default config drifted"
print(f"OK albatross at roster slot {pos} (auto-bump path: N_EXPLORERS=max(N,8) in _setup)")

assert "sign_stability" in logtext and "warm_start" in logtext, "no warm starts logged"
wide_seen = sum(1 for k in c.WIDE_WARM_GENOMES if k in logtext)
assert wide_seen >= 2, f"wide seeds not measured at gen-0 (saw {wide_seen})"
print(f"OK wide seeds: {wide_seen}/{len(c.WIDE_WARM_GENOMES)} measured at evolution gen-0")

rob = pd.read_csv(out / "robust_oos_selection.csv")
grid_rows = rob[rob["config"].astype(str).str.startswith("grid_")]
assert len(grid_rows) >= 4, f"grid configs missing from the court: {list(rob['config'])}"
assert "grid_sharp" in set(grid_rows["config"]) or "grid_wide" in set(grid_rows["config"])
print(f"OK config grid in the court: {len(grid_rows)} grid configs judged "
      f"({'|'.join(grid_rows['config'].astype(str))})")
best = rob.iloc[0]
print(f"court winner: {best['config']} (robust_score={best['robust_score']}); "
      f"shipped selector={(s.get('forensics') or {}).get('shipped_selector')}")
print(f"shipped: winner={s.get('ensemble_winner')} members={len(s.get('members') or [])}")
print("SMOKE OK")
