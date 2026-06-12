#!/usr/bin/env python3
"""Build the single-cell Kaggle kernel from the vendored engine.

kernel.py = worldexplorer/_engine.py (the synced engine_src concat) + a run
banner + the per-RUN config overrides injected right after CFG/OUT creation.
The engine's __main__ guard at the tail starts the run when the cell executes.

Usage: python tools/make_kaggle_kernel.py [out_path]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "worldexplorer" / "_engine.py"
DEFAULT_OUT = ROOT.parent / "kaggle" / "drw_world_explorer_v30" / "kernel.py"

BANNER = '''\
# =============================================================================
# DRW world-explorer v30 -- single-cell Kaggle kernel
# (engine built from worldexplorer/engine_src; do not edit here, edit the repo)
#
# HOW TO RUN
#   1. Kaggle -> New Notebook -> paste this whole file into ONE code cell.
#   2. Add Input -> competition "DRW Crypto Market Prediction".
#   3. Settings -> Accelerator -> GPU T4 x2.  Internet OFF is fine.
#   4. Run (or "Save & Run All" to also produce a late submission file).
#      Budget: ~90 min of search + reserved shipping time (~2 h wall total).
#   5. OPTIONAL self-improvement: Add Input -> your PREVIOUS run's notebook
#      OUTPUT. The engine reads its world_cairn.json / learning_ledger.json
#      (cross-run governor-beta blend, survivor warm starts, decayer taboos).
#
# WHAT THIS RUN TESTS (vs v25's 0.07827 and the v11 high-water 0.08969)
#   - v27 runtime complexity GOVERNOR: beta = measured decay~complexity slope;
#     lambda = clip(beta*0.5, 0, 0.04) penalizes capacity at SELECTION only if
#     THIS data punishes capacity. Expect beta > 0 on DRW; see
#     complexity_governor.json + complexity_generalization_curve.csv.
#   - v27 anti-overfit SHIPPING COURT + cross-run LEARNING LEDGER.
#   - v28 sign_stability family (4th-place sign-flip gate) + interior-block
#     robust-CV partitions (train oldest+newest, validate middle).
#   - v29 pls_weight family (PLS-as-selector, the private-0.099 recipe):
#     multivariate |coef| ranking that spends k on DISTINCT signals instead of
#     collinear copies.
#   - v30 INITIAL WIDE-PATH BIAS: the search currency starts width-heavy
#     (0.8 -> 0.5 anneal over lessons) so early exploration prefers wide
#     robust trails; corr(width, decay) is measured into the ledger.
#   - LIGHT-SEARCH budget (the measured sealed-cliff lever: the 3 best private
#     runs were 41-70 min; every 200+ min run regressed).
# =============================================================================
'''

OVERRIDES = '''\

# ---- v30 KAGGLE RUN OVERRIDES (the only knobs changed vs library defaults) --
CFG.TIME_BUDGET_MIN = 90.0   # LIGHT SEARCH: v8/v9/v11 (41-70 min) are the 3 best
                             # private runs; every 200+ min run fell off the
                             # sealed cliff. The governor handles the rest.
'''


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    src = ENGINE.read_text()
    hdr_end = src.index("\n", src.index("re-run: python sync_engine.py")) + 1
    body = src[hdr_end:]
    anchor = "OUT.mkdir(parents=True, exist_ok=True)\n"
    i = body.index(anchor) + len(anchor)
    kernel = BANNER + body[:i] + OVERRIDES + body[i:]
    compile(kernel, str(out), "exec")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(kernel)
    n = len(kernel.splitlines())
    print(f"wrote {out} ({n} lines, {len(kernel)} bytes)")
    assert 'if __name__ == "__main__":' in kernel, "missing __main__ runner"
    print("kernel compiles; __main__ runner present; overrides injected after OUT creation")


if __name__ == "__main__":
    main()
