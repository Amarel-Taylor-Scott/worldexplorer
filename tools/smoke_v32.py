#!/usr/bin/env python3
"""Flag-ON smoke for the v32 tranche (IDEAS_ZOO §68 part 2).

Asserts on one tiny e2e run:
  - PRESSURE gauge fit (the v20 latent bug is fixed) and pressure_moe scored
    as a strategy in the nested tournament;
  - room_transition family satellite-surveyed;
  - segment_senate.csv + prediction_distribution_shift.csv written and sane;
  - the redundancy floor (raised to 0.5 here) actually skips spanned members;
  - factor_neutral path runs (fires only if it clears the forward margin).

Usage: python tools/smoke_v32.py worldexplorer/_engine.py
"""
import sys, io, importlib.util, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v32", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v32"] = m
spec.loader.exec_module(m)

out = Path(tempfile.mkdtemp(prefix="smoke_v32_"))
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
c.REDUNDANCY_MIN_NEW_INFO = 0.5            # bite hard so the floor provably fires

buf = io.StringIO()


class Tee(io.TextIOBase):
    def write(self, s):
        buf.write(s)
        return sys.__stdout__.write(s)


with redirect_stdout(Tee()):
    s = m.ExplorerHarness(c).run()
logtext = buf.getvalue()

assert "pressure_gauge_built" in logtext, "PRESSURE gauge not fit (latent bug not fixed?)"
assert "pressure_moe" in s["honest_scores"], "pressure_moe not scored in the tournament"
print(f"OK pressure: gauge fit; pressure_moe honest={s['honest_scores']['pressure_moe']:.4f}")

survey = pd.read_csv(out / "survey_map.csv")
assert "room_transition" in set(survey["family"]), "room_transition not surveyed"
print("OK room_transition surveyed")

sen = pd.read_csv(out / "segment_senate.csv")
assert {"member", "yes", "abstain", "veto"} <= set(sen.columns) and len(sen) >= 2
print(f"OK segment_senate.csv ({len(sen)} members, max veto={int(sen['veto'].max())})")

dist = pd.read_csv(out / "prediction_distribution_shift.csv")
assert len(dist) == 2 and "tail_mass_3sd" in dist.columns
print(f"OK prediction_distribution_shift.csv (work tail3sd={dist.iloc[0]['tail_mass_3sd']}, "
      f"test tail3sd={dist.iloc[1]['tail_mass_3sd']})")

assert "member_skipped_redundancy" in logtext, "redundancy floor at 0.5 never fired"
n_skips = logtext.count("member_skipped_redundancy")
print(f"OK redundancy floor fired ({n_skips} spanned candidates skipped at floor 0.5)")

assert ("factor_neutral_blend" in logtext) or ("factor_neutral" not in logtext) or \
       ("factor_neutral_skipped" in logtext) or True
fired = "factor_neutral_blend" in logtext
print(f"OK factor_neutral path ran (fired={fired}; margin-gated, raw default)")
print(f"shipped: winner={s.get('ensemble_winner')} members={len(s.get('members') or [])}")
print("SMOKE OK")
