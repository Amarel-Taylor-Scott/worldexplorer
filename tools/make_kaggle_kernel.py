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
DEFAULT_OUT = ROOT.parent / "kaggle" / "drw_world_explorer_v34" / "kernel.py"

BANNER = '''\
# =============================================================================
# DRW world-explorer v34 -- single-cell Kaggle kernel
# (engine built from worldexplorer/engine_src; do not edit here, edit the repo)
#
# HOW TO RUN
#   1. Kaggle -> New Notebook -> paste this whole file into ONE code cell.
#   2. Add Input -> competition "DRW Crypto Market Prediction".
#   3. Settings -> Accelerator -> GPU T4 x2.  Internet OFF is fine.
#   4. Run (or "Save & Run All" to also produce a late submission file).
#      Budget: ~120 min of search + reserved shipping (~2.5 h wall) -- a touch
#      longer than v33 so the new wide+narrow menagerie gets airtime.
#   5. **STRONGLY RECOMMENDED for v34** -- Add Input -> the v33 run's notebook
#      OUTPUT. The engine reads its learning_ledger.json, whose governor block
#      carries the MEASURED width_decay_corr=+0.2574; v34's self-tuning width
#      then leans the late-run population sharper ON ITS OWN EVIDENCE (no human
#      revert). Without it, the width target stays 0.5 (= v33 wide behaviour).
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
#   - v31 TEST-LIKENESS COURT: a target-free working-vs-test classifier (X
#     only, no labels/LB) adds validate-on-most-test-like partitions to the
#     robust selector; see testlike_report.json (holdout AUC = drift gauge).
#   - v31 REDUNDANCY/CROWDING + winner-network reports: new_info per member,
#     latent-factor crowding, prediction-community graph.
#   - v32 PRESSURE ACTIVATED (v20 gauge was never fit -- latent bug): the
#     pressure family ranks for real + pressure_moe competes in the
#     tournament. Plus: segment senate, prediction-distribution shift,
#     redundancy floor (new_info >= 0.05 at admission), forward-chosen
#     factor-neutral blend (margin-gated), room_transition family.
#   - v33 WIDE-CONFIGURATION GRID + albatross wide persona (RAN: private
#     0.08159, recovered over v25/v24/v19; grid_stable shipped as half the
#     winning hedge; governor beta=+0.0208 confirmed DRW punishes capacity).
#
# v34 (THIS BUILD) -- the run TUNES ITS OWN wide/narrow lean and MIXES both;
# nothing from v30/v33 is removed (max flexibility), per the user directive:
#   - SELF-TUNING WIDTH: the v33 run measured width_decay_corr=+0.2574 (wider
#     paths decayed MORE on DRW). v34 does NOT hard-revert the wide bias -- it
#     reads that measured number from the attached ledger and lowers the width
#     TARGET the share anneals toward (here ~0.24), so the population stays wide
#     EARLY (START 0.8) and self-corrects sharper LATE. beta<0 datasets lean
#     wider instead. No prior ledger => target 0.5 => exact v33.
#   - WIDE+NARROW MIX in one run: per-explorer WIDTH_PREF -- albatross leans
#     wide (0.9), the new 'kestrel' leans narrow (0.1) -- so robust wide trails
#     and sharp narrow ridgelines are explored simultaneously.
#   - SENSORY MENAGERIE: 6 new explorer primitives, each a different way of
#     looking at the world AND a different point on the wide<->narrow axis --
#     kestrel (sharp stoop), mantis_shrimp (spectral multichannel), owl (quiet
#     regions), bloodhound (faint persistent scent), spider (feature-selection
#     web), octopus (independent multi-transform arms). Spliced after the
#     albatross; the metabolism gates how many run (raise TIME_BUDGET for more).
# =============================================================================
'''

OVERRIDES = '''\

# ---- v34 KAGGLE RUN OVERRIDES (the only knobs changed vs library defaults) --
CFG.TIME_BUDGET_MIN = 120.0  # a touch above v33's 90 so the new wide+narrow
                             # menagerie (albatross slot 7, kestrel slot 8, then
                             # the sensory primitives) actually gets airtime past
                             # the ~40-min audition parade. Still far below the
                             # heavy-search regime; the SELF-TUNING WIDTH leans
                             # the late-run population sharper from the attached
                             # v33 evidence (width_decay_corr +0.2574) to counter
                             # the sealed cliff. Drop to 90 to stay in the proven
                             # light-search zone; raise it to run the full menagerie.
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
