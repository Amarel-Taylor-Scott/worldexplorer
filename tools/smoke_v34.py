#!/usr/bin/env python3
"""Flag-ON smoke for the v34 self-tuning-width + sensory-menagerie build.

Asserts on one tiny e2e run (with a PRIOR ledger injected so the self-tune
fires):
  - the 6 sensory personas are in the roster, spliced right after the albatross
    (wide<->narrow spread ahead of the older v11 menagerie tail);
  - per-explorer WIDTH_PREF is present (albatross 0.9 wide, kestrel 0.1 narrow);
  - WIDTH_SELF_TUNE reads the ledger's measured width_decay_corr (+0.26 here,
    the sign the v33 run measured) and LOWERS the width target below 0.5 -- the
    population self-tunes a sharper LATE-run lean from evidence, no human revert;
  - a narrow sensory persona (kestrel) is actually BORN (roster + N_EXPLORERS
    bump put it on the field);
  - nothing from v33 is removed: grid_* configs still reach the robust court.

Usage: python tools/smoke_v34.py worldexplorer/_engine.py
"""
import sys, io, json, importlib.util, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v34", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v34"] = m
spec.loader.exec_module(m)

out = Path(tempfile.mkdtemp(prefix="smoke_v34_"))
m.OUT = out

# inject a PRIOR run's cairn whose ledger carries a POSITIVE width_decay_corr
# (the sign v33 measured: wide paths decayed more) so the self-tune must lean
# the target sharper. count=2 => evidence weight 2/3.
prior_cairn = out / "world_cairn_prev.json"
prior_cairn.write_text(json.dumps({
    "version": "v33", "data_source": "SYNTHETIC", "seed_bank": [],
    "ledger": {"version": "v33",
               "governor": {"beta": 0.02, "lambda": 0.01, "width_decay_corr": 0.26, "count": 2},
               "family_decay": {}, "skill_decay": {}, "survivors": [], "decayers": []}}))

c = m.CFG
c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
c.CAIRN_PATHS = (str(prior_cairn),)        # the self-tune evidence source
c.TIME_BUDGET_MIN = 0.0; c.AUDITION_ALL_SKILLS = False; c.HETERO_PAIRING = False
c.LESSON_BUDGET = 2                          # tiny per-explorer budget (17 explorers run under TIME_BUDGET=0)
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

# --- roster: the 6 sensory personas, spliced right after the albatross --------
names = [t["name"] for t in m.EXPLORER_TRAITS]
sensory = ["kestrel", "mantis_shrimp", "owl", "bloodhound", "spider", "octopus"]
for nm in sensory:
    assert nm in names, f"sensory persona {nm} missing from roster"
ai = names.index("albatross")
assert names[ai + 1] == "kestrel", f"kestrel not spliced right after albatross (got {names[ai+1:ai+3]})"
print(f"OK sensory roster: {len(names)} personas; albatross@{ai} then {names[ai+1:ai+7]}")

# --- per-explorer width_pref: wide albatross vs narrow kestrel ----------------
pref = {t["name"]: t.get("width_pref") for t in m.EXPLORER_TRAITS}
assert pref.get("albatross") == 0.9 and pref.get("kestrel") == 0.1, f"width_prefs wrong: {pref}"
assert all(pref.get(p) is not None for p in sensory), "a sensory persona lacks width_pref"
print(f"OK width_pref mix: albatross={pref['albatross']} (wide) .. kestrel={pref['kestrel']} (narrow); "
      f"spread={sorted(v for v in pref.values() if v is not None)}")

# --- self-tuning width target from the measured (ledger) width_decay_corr -----
assert "width_self_tune" in logtext, "WIDTH_SELF_TUNE did not fire (no prior evidence consumed?)"
assert m.WIDTH_BIAS["target"] < 0.5 - 1e-6, \
    f"width target not leaned sharp from +0.26 evidence (target={m.WIDTH_BIAS['target']})"
assert c.WIDTH_TARGET_MIN <= m.WIDTH_BIAS["target"] <= c.WIDTH_TARGET_MAX, "target out of clip bounds"
print(f"OK self-tune: measured width_decay=+0.26 -> width target={m.WIDTH_BIAS['target']:.4f} "
      f"(<0.5 = late-run sharper lean; START {c.WIDTH_BIAS_START} stays wide, no revert)")

# --- a narrow sensory persona actually took the field -------------------------
assert "explorer=kestrel" in logtext, "kestrel never born (roster bump did not reach it)"
print("OK kestrel born (wide+narrow personas BOTH on the field in one run)")

# --- nothing removed: the v33 grid layer still reaches the court ---------------
rob = pd.read_csv(out / "robust_oos_selection.csv")
grid_rows = rob[rob["config"].astype(str).str.startswith("grid_")]
assert len(grid_rows) >= 4, f"v33 grid configs lost: {list(rob['config'])}"
print(f"OK v33 grid layer intact: {len(grid_rows)} grid configs still judged")

print(f"shipped: winner={s.get('ensemble_winner')} members={len(s.get('members') or [])} "
      f"selector={(s.get('forensics') or {}).get('shipped_selector')}")
print("SMOKE OK")
