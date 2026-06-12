# Refactor notes — manageability roadmap

The engine is rebuilt by concatenating `engine_src/*.py` in **filename order** into
`worldexplorer/_engine.py` (see `sync_engine.py`). Workflow: edit `engine_src/` →
`python sync_engine.py` → gate → commit.

## Gates (run them for ANY engine change)

- `python tools/check_units.py --check worldexplorer/_engine.py <goldens.json>` —
  83 per-branch golden fingerprints: every skill kind (fit+predict), every ranker
  family, every transform, on a fixed synthetic fixture. Catches per-branch breakage
  the e2e smoke can't see. Capture goldens from a known-good engine with `--save`.
- `python tools/check_equivalence.py --check worldexplorer/_engine.py <baseline.json>` —
  end-to-end tiny-synthetic run; asserts the SHIPPED DECISION (winner / members /
  weights / selector / sealed / forward) matches. `--save` to capture, or
  `<engineA.py> <engineB.py>` for a full A/B. Both gates verified deterministic
  (double-run identical) before first use.

Baselines for the 2026-06 refactor live in `/tmp/wx_gate/` (recapture anytime from
a committed engine via `git show <rev>:worldexplorer/_engine.py`).

## Done — byte-preserving splits (zero behavior change)

`split_module.py` cuts a module at top-level AST boundaries into ordered `__NN`
sub-files; concatenation proved byte-identical by `diff`. 18 → 26 modules (see
git history for the table).

## Done — structural refactors (2026-06-11, all decision-identical via both gates)

### 1. `fit_skill` / `_predict_core` → `_FIT` / `_PREDICT` registries
`engine_src/07_skills__01.py`: each of the 32 `if kind == …` fit branches is now
`_fit_<kind>(Z, y_tr, in_tr, in_va, state, ctx)` with a `FitCtx` NamedTuple
(spec, X_tr, seg_tr, cols, rng, cfg, seed); `fit_skill` = build viewport → Z →
inner split → `_FIT[kind](…)`. `07_skills__02.py`: `_PREDICT` registry with
`_predict_default` (= plain `state["model"].predict`) as the fall-through.
**Adding a skill** = one `_fit_*` fn + `_FIT` row + `SKILL_REGISTRY` row
(+ `_PREDICT` row only if predict isn't the default).

### 2. `_ranked_for` → `RANKERS`; `build_viewport` transforms → `TRANSFORMS`
`engine_src/06_viewports__01.py`: 20-family elif chain → `RANKERS` registry
(`_rank_<family>(spec, X_tr, y_tr, seg_tr, pool, sig)`), corr-ranking default;
cache + red-pheromone/trap post-pass stay in `_ranked_for`.
`engine_src/06_viewports__02.py`: `TRANSFORMS` = name → `(setup, apply)`;
setup returns fitted params (None = identity fall-through, matching the old
chain), pure transforms have `setup=None`; only the ACTIVE transform's setup
runs, so rng draw order is unchanged. **Adding a family/transform** = one or two
functions + one registry row.

### 3. `ExplorerHarness.run` (1651-line method) → `_RunState` + 26 phase methods
Two mechanical, individually-gated passes:
- **Pass A** (`tools/method_object_rewrite.py`): symtable-exact rewrite of all
  296 run-locals to `rs.<name>` attributes (2097 references; CPython scope rules
  honored incl. comprehension first-iter; inner defs exported onto `rs`; the one
  `nonlocal` dropped; `except` binders kept local; every edit substring-verified
  + missed-rewrite resolver check).
- **Pass B** (`tools/split_run_phases.py`): byte-exact slice of the body into
  phase methods at the section-comment boundaries; splitter verifies reassembly
  and that every phase assigning a module global declares it.

`run()` is now: `_setup → _load_data → _quarantine_probe → _build_atlas →
_beacons → _pre_scans → _phase1_explore → _raid1 → _phase2_evolve → _raid2 →
_ablation_dive → _trail_reports → _select_members → _ensemble →
_forward_holdout → _governor → _forensics → _forward_gate → _shipping_court →
_shrink_chorus_shape → _health_alarms → _sealed_audit → _final_refit_submit →
_cairn_ledger → _chronicle → _summarize` (returns `rs.summary`).

## Optional follow-ups (not blocking anything)

- Move groups of phase methods into mixin files (`16a_…`, `16b_…` before
  `16_harness.py`, `class ExplorerHarness(_PhasesA, _PhasesB)`): only worth it
  if 16_harness.py keeps growing; the monolith problem itself is solved.
- Inert `global ATLAS, GAUGE` line at the top of `_setup` (its assignments
  moved to `_build_atlas`, which has its own declaration) — cosmetic.
- P4 from the v26 roadmap (consolidate the layered shipping overrides into the
  robust selector as the single shipping authority) is a BEHAVIOR change —
  deliberately not part of this behavior-preserving refactor.

## Source of truth
`worldexplorer/engine_src/` is **the** source. `kaggle/drw_world_explorer_v26/src/`
is a frozen R&D snapshot.
