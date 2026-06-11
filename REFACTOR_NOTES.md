# Refactor notes — manageability roadmap

The engine is rebuilt by concatenating `engine_src/*.py` in **filename order** into
`worldexplorer/_engine.py` (see `sync_engine.py`). So any split must be byte‑preserving
to be provably behavior‑neutral.

## Done (byte‑preserving, zero behavior change)
`split_module.py` cuts a module at top‑level AST boundaries into ordered `__NN`
sub‑files; concatenation is byte‑identical (verified by `diff` against the prior
`_engine.py`). 18 → 26 modules:

| was | now |
|---|---|
| `06_viewports.py` (1056) | `__00` rank caches/helpers · `__01` family rankers (`_ranked_for`) · `__02` `build_viewport` + transforms |
| `07_skills.py` (1031) | `__00` registry + lane/gpu helpers · `__01` `fit_skill` · `__02` predict |
| `12_topography_reports.py` (392) | `__00` · `__01` |
| `13_ensemble.py` (390) | `__00` · `__01` |
| `15_…forensic….py` (837) | `__00` governor+ledger+court · `__01` robust selector · `__02` forensic orchestrator |

## The two TRUE monoliths — single AST nodes, need a STRUCTURAL refactor (not a byte split)
Each is one giant construct, so it can't be byte‑split. Decompose in small commits,
**each gated by a smoke run** (the synthetic‑seed decision must stay identical; no
byte‑identity is possible once structure changes).

### 1. `fit_skill` (~600 lines, ~28 `if kind == …` branches) → a `SKILLS` registry
- `_FIT: dict[str, callable] = {}`; extract each branch to `def _fit_<kind>(Z, y_tr, in_tr, in_va, state, ctx) -> state`, where `ctx` bundles the shared inputs `(X_tr, seg_tr, cols, idx, transform, rng, cfg, seed, spec)`.
- `fit_skill` becomes: build viewport → `Z` → inner split → `return _FIT[kind](Z, y_tr, in_tr, in_va, state, ctx)`.
- Mirror `_predict_core` → `_PREDICT`. Adding a skill = one small function + one registry line (not a 3‑place edit).

### 2. `ExplorerHarness.run` (1481 lines, one method) → phase methods on a `RunState`
- A `RunState` carries cross‑phase locals (`cfg, library, X_full/y_full/seg_full, Xp/yp/segp, ATLAS/GAUGE, members, weights, …`).
- `run()` → slim orchestrator: `_setup → _load_data → _build_atlas → _pre_scans → _phase1_explore → _raid1 → _phase2_evolve → _raid2 → _ablation_dive → _select_members → _ensemble → _forward_gate → _governor → _forensics → _shipping_court → _shrink_chorus_shape → _sealed_audit → _final_refit_submit → _reports_cairn_ledger → _summary`.
- Each extraction is its own commit + smoke. Afterwards the phase methods can move into mixin files (`16a_…`, `16b_…`), each individually valid.

### Also (same registry pattern)
`_ranked_for` (24‑branch family dispatch) → `RANKERS`; `build_viewport` transforms → `TRANSFORMS`.

## Source of truth
`worldexplorer/engine_src/` is now **the** source. `kaggle/drw_world_explorer_v26/src/` is a
frozen R&D snapshot. Workflow: edit `engine_src/` → `python sync_engine.py` → validate → commit.
