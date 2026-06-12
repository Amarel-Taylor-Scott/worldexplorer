#!/usr/bin/env python3
"""Flag-ON smoke for the v30 initial wide-path bias.

UNIT: with WIDTH_BIAS_START=0.8,
  - width_share anneals 0.8 -> 0.65 (one half-life) -> ~0.5 (many lessons);
  - a NARROW high-corr lesson vs a WIDE lower-corr lesson: EARLY fitness must
    prefer the wide path, the annealed (late) fitness must prefer raw corr;
  - width_bias_beta starts 0.6 and anneals to 0.
E2E: tiny smoke harness; complexity_governor.json must carry the measured
width_decay_corr + width_share_now (the evidence loop for recalibration).

Usage: python tools/smoke_v30.py worldexplorer/_engine.py
"""
import sys, importlib.util, json, tempfile
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("eng_v30", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v30"] = m
spec.loader.exec_module(m)

c = m.CFG
assert abs(c.WIDTH_BIAS_START - 0.8) < 1e-9, "expected default WIDTH_BIAS_START=0.8"
hl = int(c.WIDTH_BIAS_HALFLIFE)

m.WIDTH_BIAS["n"] = 0
w0, b0 = m.width_share(), m.width_bias_beta()
m.WIDTH_BIAS["n"] = hl
w1, b1 = m.width_share(), m.width_bias_beta()
m.WIDTH_BIAS["n"] = 50 * hl
w2, b2 = m.width_share(), m.width_bias_beta()
print(f"width_share anneal: n=0 -> {w0:.3f} (beta {b0:.3f}); n={hl} -> {w1:.3f} "
      f"(beta {b1:.3f}); n={50*hl} -> {w2:.3f} (beta {b2:.3f})")
assert abs(w0 - 0.8) < 1e-9 and abs(b0 - 0.6) < 1e-9
assert abs(w1 - 0.65) < 1e-9 and abs(b1 - 0.3) < 1e-9
assert abs(w2 - 0.5) < 1e-6 and b2 < 1e-6

rng = np.random.default_rng(1)
oof = rng.normal(size=100).astype(np.float32)


def mk(name, corr, width):
    return m.Lesson("ut", "phase1", "linear_assoc", "top8_identity", "top", "identity",
                    name, 7, oof, [corr], corr, width, 0.0, 0.05, corr * 1.2, 1.2,
                    0.5, 2, "promote", "unit", wf_corr=corr, wf_width=width, k=8)


narrow = mk("narrow", 0.14, 0.02)   # high corr, narrow lucky ridgeline
wide = mk("wide", 0.06, 0.06)       # lower corr, wide robust path
m.WIDTH_BIAS["n"] = 0
f_n0, f_w0 = m.lesson_fitness(narrow), m.lesson_fitness(wide)
m.WIDTH_BIAS["n"] = 50 * hl
f_n1, f_w1 = m.lesson_fitness(narrow), m.lesson_fitness(wide)
print(f"EARLY  fitness: narrow={f_n0:.4f} wide={f_w0:.4f}  -> wide preferred: {f_w0 > f_n0}")
print(f"LATE   fitness: narrow={f_n1:.4f} wide={f_w1:.4f}  -> narrow preferred: {f_n1 > f_w1}")
assert f_w0 > f_n0, "early fitness must prefer the wide path"
assert f_n1 > f_w1, "annealed fitness must return to the corr-led v4 balance"
print("UNIT OK: initial wide-path bias anneals exactly as configured")

out = Path(tempfile.mkdtemp(prefix="smoke_v30_"))
m.OUT = out
c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
c.TIME_BUDGET_MIN = 0.0; c.AUDITION_ALL_SKILLS = False; c.HETERO_PAIRING = False
c.N_EXPLORERS = 2; c.LESSON_BUDGET = 3
c.EVOLUTION_BUDGET = 5; c.EVOLUTION_MAX_GENERATIONS = 1; c.EVOLUTION_POP = 5; c.EVOLUTION_OFFSPRING = 3
c.PREDATOR_BUDGET = 2; c.PREDATOR_MAX_TARGETS = 2; c.DIVE_BUDGET = 0; c.ABLATION_BUDGET = 0
c.PROBE_MAX_ROWS = 6000; c.N_SEGMENTS = 8; c.N_SPLITS = 3; c.WF_FOLDS = 2
c.MLP_MAX_ITER = 2; c.MLP_MAX_ROWS = 2000; c.DREAM_REPLAYS = 10
c.ROBUST_BOOT = 15; c.ROBUST_SAMPLE_ROWS = 6000; c.CPCV_MAX_PATHS = 3
c.FORENSIC_ACTIONS = False; c.MAX_FAMILY_MEMBERS = 3; c.MAX_MEMBERS = 8; c.STABSEL_BOOT = 5
s = m.ExplorerHarness(c).run()
gov = json.loads((out / "complexity_governor.json").read_text())
print(f"E2E: width_decay_corr={gov.get('width_decay_corr')} width_share_now={gov.get('width_share_now')}")
assert "width_decay_corr" in gov and "width_share_now" in gov
led = json.loads((out / "learning_ledger.json").read_text())
assert "width_decay_corr" in led["governor"], "ledger missing the wide-path evidence"
print(f"E2E: ledger governor carries width_decay_corr={led['governor']['width_decay_corr']}")
print(f"E2E: shipped winner={s.get('ensemble_winner')} members={len(s.get('members') or [])}")
print("SMOKE OK")
