#!/usr/bin/env python3
"""Flag-ON smoke for the v36 EXPLORER ADVISOR LOOP.

One tiny e2e run with a pre-written advisor_instructions.json attached proves
the whole loop, plus the sacred-rule guard and the advisor_stub round-trip:
  - INGEST: the advisor file becomes ADVISOR_PRIORS (family/transform/skill) +
    warm genomes in the seed bank; a FORBIDDEN key (labels) is IGNORED.
  - EXPORT: explorer_findings_graph.json is written with the expected sections
    (measured laws, research nodes, feature topology, open questions, advice
    schema), ready to hand to an external model.
  - advisor_stub.heuristic_advice + sanitize on that findings graph returns ONLY
    sacred-rule-safe keys (the out-of-band LLM bridge round-trips).

Usage: python tools/smoke_v36.py worldexplorer/_engine.py
"""
import sys, io, json, importlib.util, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("eng_v36", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["eng_v36"] = m
spec.loader.exec_module(m)

out = Path(tempfile.mkdtemp(prefix="smoke_v36_"))
m.OUT = out

# a pre-written advisor file -- as an external LLM would return it, plus a
# FORBIDDEN 'labels' key that the engine must ignore (the sacred rule).
adv = {"family_priors": {"consensus": 1.0, "shadow": -0.5},
       "transform_priors": {"identity": 0.3},
       "skill_priors": {"linear_ols": 0.5},
       "warm_genomes": ["majority_vote|consensus24_identity", "pls|consensus64_identity"],
       "notes": "favor agreeing low-shift communities; keep wide/narrow mix",
       "labels": [1, 2, 3]}                       # <- forbidden; must be ignored
adv_path = out / "advisor_instructions.json"
adv_path.write_text(json.dumps(adv))

c = m.CFG
c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
c.SENSORY_ROSTER = False; c.WIDTH_SELF_TUNE = False
c.ADVISOR_PATHS = (str(adv_path),)
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

# ---- INGEST -----------------------------------------------------------------
assert "advisor_ingested" in logtext, "advisor file not ingested"
assert m.ADVISOR_PRIORS.get("family", {}).get("consensus") == 1.0, "family prior not loaded"
assert m.ADVISOR_PRIORS.get("skill", {}).get("linear_ols") == 0.5, "skill prior not loaded"
assert "labels" not in m.ADVISOR_PRIORS, "forbidden key leaked into ADVISOR_PRIORS (sacred rule!)"
seed_keys = [g.key for g in m.SEEDBANK]
assert any("consensus" in k for k in seed_keys), f"advisor warm genome not germinated: {seed_keys}"
print(f"OK ingest: family/skill priors loaded, forbidden 'labels' ignored, "
      f"warm genomes in seed bank ({[k for k in seed_keys if 'consensus' in k]})")

# ---- EXPORT -----------------------------------------------------------------
fg = json.loads((out / "explorer_findings_graph.json").read_text())
for sec in ("measured_laws", "research_nodes", "feature_topology", "open_questions", "advice_schema"):
    assert sec in fg, f"findings graph missing section {sec}"
assert "_note" in fg["advice_schema"] and "labels" not in str(fg["advice_schema"]).lower() \
    or "do not use labels" in str(fg["advice_schema"]).lower(), "advice schema missing sacred-rule note"
print(f"OK export: explorer_findings_graph.json ({len(fg['research_nodes'])} research nodes, "
      f"{len(fg['feature_topology'])} feature communities, {len(fg['open_questions'])} open questions)")

# ---- advisor_stub round-trip (the out-of-band LLM bridge) --------------------
sspec = importlib.util.spec_from_file_location("advisor_stub",
                                               str(Path(sys.argv[0]).parent / "advisor_stub.py"))
adv_mod = importlib.util.module_from_spec(sspec)
sspec.loader.exec_module(adv_mod)
advice = adv_mod.sanitize(adv_mod.heuristic_advice(fg))
assert set(advice) <= set(adv_mod.ALLOWED), f"advisor produced non-allowed keys: {set(advice)}"
assert "labels" not in advice and "leaderboard" not in advice, "sacred rule violated by advisor"
assert advice.get("family_priors"), "advisor returned no family priors"
print(f"OK advisor_stub round-trip: keys={sorted(advice)} (all sacred-rule-safe); "
      f"family_priors={len(advice['family_priors'])}")

print(f"shipped: winner={s.get('ensemble_winner')} members={len(s.get('members') or [])}")
print("SMOKE OK")
