#!/usr/bin/env python3
"""Behavior-preservation GATE for engine refactors. Runs an engine build on a
tiny deterministic synthetic seed (hetero off, cpu) and asserts the shipped
DECISION matches a baseline. Every refactor step must pass this gate.

Usage:
  python tools/check_equivalence.py --save  <engine.py> <baseline.json>
  python tools/check_equivalence.py --check <engine.py> <baseline.json>
  python tools/check_equivalence.py <engineA.py> <engineB.py>     # full A/B
"""
import sys, importlib.util, json, tempfile
from pathlib import Path


def run(path: str, tag: str) -> dict:
    spec = importlib.util.spec_from_file_location(f"eng_{tag}", path)
    k = importlib.util.module_from_spec(spec)
    sys.modules[f"eng_{tag}"] = k
    spec.loader.exec_module(k)
    out = Path(tempfile.mkdtemp(prefix=f"eq_{tag}_"))
    k.OUT = out
    c = k.CFG
    c.OUT_DIR = str(out); c.ALLOW_SYNTHETIC_FALLBACK = True; c.SYN_ROWS = 8000; c.SYN_ANON = 32; c.SEED = 42
    c.TIME_BUDGET_MIN = 0.0; c.AUDITION_ALL_SKILLS = False; c.HETERO_PAIRING = False
    c.N_EXPLORERS = 2; c.LESSON_BUDGET = 3
    c.EVOLUTION_BUDGET = 5; c.EVOLUTION_MAX_GENERATIONS = 1; c.EVOLUTION_POP = 5; c.EVOLUTION_OFFSPRING = 3
    c.PREDATOR_BUDGET = 2; c.PREDATOR_MAX_TARGETS = 2; c.DIVE_BUDGET = 0; c.ABLATION_BUDGET = 0
    c.PROBE_MAX_ROWS = 6000; c.N_SEGMENTS = 8; c.N_SPLITS = 3; c.WF_FOLDS = 2
    c.MLP_MAX_ITER = 2; c.MLP_MAX_ROWS = 2000; c.DREAM_REPLAYS = 10
    c.ROBUST_BOOT = 15; c.ROBUST_SAMPLE_ROWS = 6000; c.CPCV_MAX_PATHS = 3
    c.FORENSIC_ACTIONS = False; c.MAX_FAMILY_MEMBERS = 3; c.MAX_MEMBERS = 8; c.STABSEL_BOOT = 5
    s = k.ExplorerHarness(c).run()
    return {"winner": s.get("ensemble_winner"), "members": s.get("members"),
            "shipped_weights": {kk: round(vv, 6) for kk, vv in (s.get("shipped_weights") or {}).items()},
            "shipped_selector": (s.get("forensics") or {}).get("shipped_selector"),
            "sealed": round(s.get("sealed_holdout_corr") or 0.0, 6),
            "forward": round(s.get("forward_blend_corr") or 0.0, 6)}


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in ("--save", "--check"):
        mode, eng, base = args[0], args[1], Path(args[2])
        dec = run(eng, "X")
        print("decision:", json.dumps(dec, default=str))
        if mode == "--save":
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_text(json.dumps(dec, indent=2, default=str))
            print(f"baseline saved -> {base}")
            return 0
        ref = json.loads(base.read_text())
        ok = dec == ref
        if not ok:
            print("baseline:", json.dumps(ref, default=str))
            for key in sorted(set(dec) | set(ref)):
                if dec.get(key) != ref.get(key):
                    print(f"  MISMATCH {key}: {dec.get(key)!r} != {ref.get(key)!r}")
        print("EQUIVALENT" if ok else "MISMATCH")
        return 0 if ok else 1
    a, b = run(args[0], "A"), run(args[1], "B")
    print("A:", json.dumps(a, default=str))
    print("B:", json.dumps(b, default=str))
    ok = a == b
    print("EQUIVALENT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
