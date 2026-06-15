#!/usr/bin/env python3
"""Launch a FLEET of parallel Kaggle runs (mix of T4-GPU and CPU) for faster,
more diverse learning.

Each fleet member is a self-contained copy of the engine kernel with a different
config injected -- seed, time budget, accelerator, wide/narrow lean, governor
strength -- so the runs explore different regions of the search space AT ONCE.
Every run writes learning_ledger.json + explorer_findings_graph.json; attach one
run's output to the next (or merge the ledgers) and the colony learns across the
whole fleet, not one run at a time. GPU members exercise the capacity-heavy
skills; CPU members run the ridge-family search for free (no GPU quota) -- the
shipped blends are ridge-family anyway, so CPU members are first-class.

Subcommands (build is local + safe; push/collect/submit hit Kaggle):
  python tools/fleet.py build   [--kernel K] [--out DIR]   # write one kernel dir per member
  python tools/fleet.py bootstrap --name NAME               # write a slim GitHub/dataset kernel
  python tools/fleet.py sprout  --count 8                  # stochastic, bounded experiment members
  python tools/fleet.py revive  --count 6                  # old motifs + MLP grokking retests
  python tools/fleet.py push    [--out DIR]                # `kaggle kernels push` every member
  python tools/fleet.py status                             # poll every member's run
  python tools/fleet.py collect [--out DIR] [--submit]     # download outputs (+ submit each CSV)
  python tools/fleet.py harvest-grok --manifest M          # collect completed incubators; never submit

The push/collect-submit steps export code/predictions to Kaggle; the agent is
blocked from that, so run them yourself or hand AUTOMATION.md's prompt to a tool
with shell access. Credentials come from ~/.config/worldexplorer/.env.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KERNEL = ROOT.parent / "kaggle" / "drw_world_explorer_v36" / "kernel.py"
DEFAULT_BOOTSTRAP = ROOT / "kaggle" / "bootstrap_kernel.py"
DEFAULT_OUT = ROOT.parent / "kaggle" / "fleet"
ENV_PATH = Path.home() / ".config" / "worldexplorer" / ".env"
COMP = "drw-crypto-market-prediction"

# A diverse sweep -- 3 GPU + 3 CPU -- varying the wide<->narrow lean (the open
# question after v33 measured width_decay_corr=+0.26), governor strength, seed,
# and budget. All in the proven light-to-mid search zone (the sealed cliff).
FLEET = [
    {"name": "gpu-wide",     "gpu": True,  "ov": {"SEED": 7,   "TIME_BUDGET_MIN": 60, "WIDTH_BIAS_START": 0.8}},
    {"name": "gpu-sharp",    "gpu": True,  "ov": {"SEED": 123, "TIME_BUDGET_MIN": 90, "WIDTH_BIAS_START": 0.3}},
    {"name": "gpu-governor", "gpu": True,  "ov": {"SEED": 5,   "TIME_BUDGET_MIN": 90, "GOV_LAMBDA_SCALE": 1.0, "GOV_LAMBDA_MAX": 0.08}},
    {"name": "cpu-balanced", "gpu": False, "ov": {"SEED": 42,  "TIME_BUDGET_MIN": 60, "WIDTH_BIAS_START": 0.5}},
    {"name": "cpu-sharp",    "gpu": False, "ov": {"SEED": 99,  "TIME_BUDGET_MIN": 60, "WIDTH_BIAS_START": 0.3}},
    {"name": "cpu-light",    "gpu": False, "ov": {"SEED": 777, "TIME_BUDGET_MIN": 45}},
    # Extrema-reconciliation probes. These use fresh slugs so Kaggle treats them
    # as new notebooks, and they explicitly test the chaotic-market topology
    # layer: can forward-gated spike/mound surgery help without hurting path
    # width or shipping discipline?
    {"name": "cpu-extrema",   "gpu": False, "ov": {"SEED": 2026, "TIME_BUDGET_MIN": 45, "WIDTH_BIAS_START": 0.65,
                                                   "EXTREMA_RECONCILE": True, "EXTREMA_MARGIN": 0.0015}},
    {"name": "gpu-sharp-ext", "gpu": True,  "ov": {"SEED": 123,  "TIME_BUDGET_MIN": 90, "WIDTH_BIAS_START": 0.3,
                                                   "EXTREMA_RECONCILE": True, "EXTREMA_MARGIN": 0.0015}},
    {"name": "gpu-gov-ext",   "gpu": True,  "ov": {"SEED": 5,    "TIME_BUDGET_MIN": 90, "GOV_LAMBDA_SCALE": 1.0,
                                                   "GOV_LAMBDA_MAX": 0.08, "EXTREMA_RECONCILE": True,
                                                   "EXTREMA_MARGIN": 0.0015}},
]

SPROUT_PARENTS = [
    {"name": "wide", "ov": {"WIDTH_BIAS_START": 0.80, "TIME_BUDGET_MIN": 60, "CONFIG_GRID_TOPK": 10}},
    {"name": "sharp", "ov": {"WIDTH_BIAS_START": 0.30, "TIME_BUDGET_MIN": 75, "CONFIG_GRID_TOPK": 6}},
    {"name": "governor", "ov": {"GOV_LAMBDA_SCALE": 1.0, "GOV_LAMBDA_MAX": 0.08, "TIME_BUDGET_MIN": 75}},
    {"name": "extrema", "ov": {"EXTREMA_RECONCILE": True, "EXTREMA_MARGIN": 0.0015, "WIDTH_BIAS_START": 0.60}},
    {"name": "testlike", "ov": {"TESTLIKE_PARTITIONS": True, "TESTLIKE_FRACS": (0.20, 0.35), "TESTLIKE_COLS": 320}},
    {"name": "stable", "ov": {"STABSEL_BOOT": 16, "STABSEL_POOL": 320, "PLSRANK_POOL": 256}},
    {"name": "foundry", "ov": {"CONFIG_GRID_TOPK": 10, "PLSRANK_POOL": 320, "STABSEL_POOL": 320,
                               "REDUNDANCY_MIN_NEW_INFO": 0.08, "WIDTH_BIAS_START": 0.65}},
]

SPROUT_AXES = {
    "TIME_BUDGET_MIN": (35, 45, 60, 75, 90),
    "WIDTH_BIAS_START": (0.20, 0.30, 0.45, 0.60, 0.75, 0.85),
    "GOV_LAMBDA_SCALE": (0.25, 0.50, 0.75, 1.00, 1.25),
    "GOV_LAMBDA_MAX": (0.03, 0.04, 0.06, 0.08),
    "CONFIG_GRID_TOPK": (6, 8, 10, 12),
    "ROBUST_HEDGE_BAND": (0.002, 0.004, 0.006),
    "FORENSIC_MARGIN": (0.0010, 0.0015, 0.0025),
    "EXTREMA_MARGIN": (0.0010, 0.0015, 0.0020, 0.0030),
    "TESTLIKE_FRACS": ((0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.30, 0.50)),
    "MAX_MEMBERS": (8, 10, 12),
    "REDUNDANCY_MIN_NEW_INFO": (0.03, 0.05, 0.08),
    "PLSRANK_POOL": (128, 192, 256, 320),
    "PLSRANK_COMPONENTS": (6, 8, 10, 12),
    "STABSEL_BOOT": (8, 12, 16),
    "STABSEL_POOL": (192, 256, 320),
}

SPROUT_MODES = ("forager", "warper", "governor", "stable", "testlike", "foundry", "random")

SPROUT_DOCTRINE = (
    "Sprouts are signal-reconciliation probes, not leaderboard-driven "
    "hyperparameter tuning. A run is useful only if it helps find, explain, "
    "merge, route, reinterpret, reduce, carve, terraform, subtract, or only "
    "eventually quarantine/retire regions of signal."
)

SPROUT_HYPOTHESES = {
    "forager": "Does a mixed, low-commitment probe find a wider path in underexplored terrain?",
    "warper": "Can output-space extrema be softened or reconciled without losing forward stability?",
    "governor": "Does stricter complexity pressure improve survival across hostile worlds?",
    "stable": "Do stability/PLS-biased feature views turn sharp peaks into broader basins?",
    "testlike": "Do target-free test-like partitions expose the terrain that normal CV misses?",
    "foundry": "Can compatible signal regions be refined into a reusable representation hub?",
    "random": "Does a bounded random shoot reveal an unexpected but verifiable topology direction?",
}

SPROUT_TRANSLATION_GOALS = {
    "forager": "Translate any gain into a repeatable viewport, feature family, or submodel jurisdiction.",
    "warper": "Explain whether conflicting extrema can become one wider mound under a prediction/feature-space morph, or whether they need route carving.",
    "governor": "Explain whether complexity was amplifying false peaks or exposing real curved structure.",
    "stable": "Translate a win into stable feature communities, PLS factors, or low-noise causal witnesses.",
    "testlike": "Explain which target-free shift directions make normal validation lie, then carve or downweight those route parts.",
    "foundry": "Translate raw local signals into reusable factories: feature communities, PLS factors, residual witnesses, or ensemble priors.",
    "random": "Treat any win as unexplained until a later transform, ablation, or independent traveler reproduces it.",
}

SPROUT_SIGNAL_ACTIONS = {
    "forager": "find_or_route",
    "warper": "merge_morph_or_carve",
    "governor": "retire_fragile_or_keep_simple",
    "stable": "compose_wider_basin",
    "testlike": "carve_or_downweight_shift_noise",
    "foundry": "industrialize_signal_hub",
    "random": "find_then_explain",
}

GROK_DOCTRINE = (
    "Grokking incubation is a quarantined research lane. It may spend extra "
    "budget on weird topology/surface moves despite flat validation, but it "
    "cannot ship directly; any result must pass the same robust OOS court."
)

BREAKER_DOCTRINE = (
    "WorldExplorer .1-breaker runs are framework-native attempts to leave the "
    "0.085 route-carve basin. They focus budget on the internal 0.111-class "
    "mechanisms already in the engine: greedy OLS, tail/order blocks, stability "
    "selection, PLS selectors, testlike partitions, and robust OOS promotion."
)

BREAKER_WARM_GENOMES = [
    "greedy_ols|tail32_identity",
    "greedy_ols|tail48_identity",
    "greedy_ols|tail64_identity",
    "greedy_ols|tail96_identity",
    "greedy_ols|tail120_identity",
    "greedy_ols|tail64_rank",
    "greedy_ols|tail64_quantize4",
    "linear_ols|tail50_identity",
    "linear_ols|tail80_identity",
    "pls|tail120_identity",
    "bayes_ridge|tail120_identity",
    "ard_linear|tail80_identity",
    "greedy_ols|sign_stability64_identity",
    "greedy_ols|pls_weight64_identity",
    "greedy_ols|testlike_stable64_identity",
    "pls|sign_stability96_identity",
    "bayes_ridge|pls_weight120_identity",
]

REVIVAL_DOCTRINE = (
    "Revival runs deliberately re-test old useful material through the current "
    "WorldExplorer courts instead of treating old branches as dead. Old motifs "
    "earn a fresh audition; neural/MLP grokking lanes are quarantined research "
    "branches and cannot become main-world paths without current robust support."
)

REVIVAL_OLD_WARM_GENOMES = [
    "greedy_ols|tail32_identity",
    "greedy_ols|tail64_rank",
    "greedy_ols|tail96_identity",
    "linear_ols|tail80_identity",
    "linear_pearson|tail96_identity",
    "gpu_ridge_swarm|tail96_rank",
    "gpu_ridge_swarm|medoid64_pair_aug",
    "pls|tail120_identity",
    "bayes_ridge|tail120_identity",
    "ard_linear|medoid64_identity",
    "bagged_linear|testlike_stable96_rank",
    "bagged_linear|sign_stability96_rank",
    "pls|pls_weight96_identity",
    "bayes_ridge|pls_weight120_identity",
    "swell_rider|mycelium30_quantize4",
]

REVIVAL_MLP_WARM_GENOMES = [
    "mlp_assoc|top64_rank",
    "mlp_assoc|stable64_rank",
    "mlp_assoc|tail64_identity",
    "mlp_assoc|tail96_rank",
    "mlp_assoc|medoid64_pair_aug",
    "mlp_assoc|sign_stability64_rank",
    "mlp_assoc|pls_weight64_rank",
    "mlp_assoc|testlike_stable64_rank",
    "mlp_assoc|consensus64_rank",
    "mlp_assoc|mycelium30_quantize4",
    "mlp_assoc|invariant64_rank",
    "mlp_assoc|both_clocks48_rank",
]

REVIVAL_MODES = (
    "old-conflict-new-lens",
    "old-tail-new-lens",
    "old-stability-new-lens",
    "mlp-dropout-grok",
    "mlp-low-noise-grok",
    "mlp-tail-grok",
)


def breaker_members(prefix: str, *, count: int, seed: int, gpu_frac: float, time_budget: float) -> list[dict]:
    rng = random.Random(seed)
    modes = [
        ("tail-sharp", {
            "WIDTH_BIAS_START": 0.25,
            "POSITIONAL_BLOCK": 96,
            "GREEDY_OLS_CAP": 64,
            "GREEDY_OLS_MIN_DELTA": 0.00005,
            "GOV_LAMBDA_SCALE": 0.9,
            "GOV_LAMBDA_MAX": 0.06,
        }),
        ("tail-wide", {
            "WIDTH_BIAS_START": 0.55,
            "POSITIONAL_BLOCK": 160,
            "GREEDY_OLS_CAP": 96,
            "GREEDY_OLS_MIN_DELTA": 0.00002,
            "GOV_LAMBDA_SCALE": 0.7,
            "GOV_LAMBDA_MAX": 0.05,
        }),
        ("stabsel-greedy", {
            "WIDTH_BIAS_START": 0.45,
            "POSITIONAL_BLOCK": 120,
            "GREEDY_OLS_CAP": 80,
            "STABSEL_POOL": 480,
            "STABSEL_BOOT": 20,
            "SIGNSTAB_MAX_FLIP": 0.20,
            "GOV_LAMBDA_SCALE": 1.0,
            "GOV_LAMBDA_MAX": 0.07,
        }),
        ("pls-testlike", {
            "WIDTH_BIAS_START": 0.40,
            "POSITIONAL_BLOCK": 160,
            "GREEDY_OLS_CAP": 96,
            "PLSRANK_POOL": 480,
            "PLSRANK_COMPONENTS": 16,
            "TESTLIKE_FRACS": [0.15, 0.30, 0.45],
            "SHIFT_PENALTY": 0.75,
            "GOV_LAMBDA_SCALE": 0.85,
        }),
        ("governed-simple", {
            "WIDTH_BIAS_START": 0.30,
            "POSITIONAL_BLOCK": 80,
            "GREEDY_OLS_CAP": 48,
            "GREEDY_OLS_MIN_DELTA": 0.00010,
            "GOV_LAMBDA_SCALE": 1.35,
            "GOV_LAMBDA_MAX": 0.09,
            "MAX_FAMILY_MEMBERS": 2,
            "MIN_BLEND_MEMBERS": 4,
        }),
    ]
    base = {
        "TIME_BUDGET_MIN": float(time_budget),
        "RESERVE_MIN": 30.0,
        "N_EXPLORERS": 14,
        "LESSON_BUDGET": 36,
        "MAX_SEASONS": 14,
        "MAX_EPOCHS": 12,
        "EVOLUTION_BUDGET": 96,
        "EVOLUTION_OFFSPRING": 8,
        "EVOLUTION_PATIENCE": 6,
        "ATTENTION_POOL": 24,
        "DIVE_BUDGET": 12,
        "DREAM_REPLAYS": 220,
        "CONFIG_GRID_TOPK": 12,
        "ROBUST_HEDGE_BAND": 0.006,
        "ROBUST_MAX_MEMBERS": 18,
        "MAX_MEMBERS": 14,
        "WIDTH_BIAS_HALFLIFE": 90,
        "GREEDY_OLS_TR_ROWS": 12000,
        "GREEDY_OLS_VA_ROWS": 8000,
        "GREEDY_OLS_FALLBACK_K": 36,
        "WIDE_PERSONA": True,
        "SENSORY_ROSTER": True,
        "WIDE_SEEDS": True,
        "TESTLIKE_PARTITIONS": True,
        "TESTLIKE_FEATURE_GATE": True,
        "PLSRANK_FAMILY": True,
        "SIGNSTAB_FAMILY": True,
        "CONSENSUS_FAMILY": True,
        "CONSENSUS_ENSEMBLE": True,
        "FORENSIC_ENABLED": True,
        "ROBUST_OOS_SELECT": True,
        "SHIPPING_COURT": True,
        "WARM_GENOMES": BREAKER_WARM_GENOMES,
        "WIDE_WARM_GENOMES": BREAKER_WARM_GENOMES[:10],
    }
    members: list[dict] = []
    for i in range(count):
        mode, mode_ov = modes[i % len(modes)]
        ov = dict(base)
        ov.update(mode_ov)
        ov["SEED"] = rng.randrange(1_000, 2_000_000_000)
        gpu = rng.random() < max(0.0, min(1.0, gpu_frac))
        members.append({
            "name": f"{prefix}-{i + 1:02d}-{mode}",
            "gpu": bool(gpu),
            "ov": ov,
            "mode": "breaker",
            "breaker_mode": mode,
            "hypothesis": (
                "Can WorldExplorer's own greedy-OLS/tail/stability/PLS/testlike "
                "families escape the 0.085 route-carve basin under robust OOS court?"
            ),
            "translation_goal": (
                "Promote only if a framework-native low-complexity path survives "
                "forward, testlike, many-world, and shipping-court checks."
            ),
            "signal_action": "framework_native_breakpoint_search",
            "trust_policy": "normal_submission_candidate_after_robust_court",
        })
    return members


def revival_members(prefix: str, *, count: int, seed: int, gpu_frac: float,
                    time_budget: float) -> list[dict]:
    """Build evidence-gated old-branch revival and neural grokking members.

    These are not leaderboard-memory shortcuts. They only seed hypotheses that
    the engine re-measures from scratch: old linear/tail/stability material is
    tried through current testlike/topology courts, and MLP routes are marked as
    incubators so they can run long without being allowed to ship directly.
    """
    rng = random.Random(seed)
    gpu_frac = max(0.0, min(1.0, float(gpu_frac)))
    old_modes = {
        "old-conflict-new-lens": {
            "WIDTH_BIAS_START": 0.42,
            "POSITIONAL_BLOCK": 160,
            "GREEDY_OLS_CAP": 80,
            "PLSRANK_POOL": 512,
            "STABSEL_POOL": 480,
            "TESTLIKE_FRACS": [0.15, 0.30, 0.45],
            "SHIFT_PENALTY": 0.80,
            "WARM_GENOMES": REVIVAL_OLD_WARM_GENOMES,
            "WIDE_WARM_GENOMES": REVIVAL_OLD_WARM_GENOMES[:8],
        },
        "old-tail-new-lens": {
            "WIDTH_BIAS_START": 0.30,
            "POSITIONAL_BLOCK": 192,
            "GREEDY_OLS_CAP": 96,
            "GREEDY_OLS_MIN_DELTA": 0.00004,
            "GOV_LAMBDA_SCALE": 0.95,
            "GOV_LAMBDA_MAX": 0.06,
            "WARM_GENOMES": REVIVAL_OLD_WARM_GENOMES,
            "WIDE_WARM_GENOMES": REVIVAL_OLD_WARM_GENOMES[:6],
        },
        "old-stability-new-lens": {
            "WIDTH_BIAS_START": 0.62,
            "POSITIONAL_BLOCK": 120,
            "STABSEL_BOOT": 24,
            "STABSEL_POOL": 512,
            "PLSRANK_POOL": 512,
            "CONSENSUS_BONUS": 0.75,
            "REDUNDANCY_MIN_NEW_INFO": 0.08,
            "WARM_GENOMES": REVIVAL_OLD_WARM_GENOMES,
            "WIDE_WARM_GENOMES": REVIVAL_OLD_WARM_GENOMES[:10],
        },
    }
    neural_modes = {
        "mlp-dropout-grok": {
            "WIDTH_BIAS_START": 0.72,
            "MLP_DROPOUT": 0.18,
            "MLP_PATIENCE": 22,
            "MLP_MAX_ITER": 70,
            "MLP_MAX_ROWS": 140_000,
            "GPU_MLP_MAX_ITER": 80,
            "GPU_MLP_MAX_ROWS": 180_000,
            "GPU_MLP_HIDDEN": [192, 96, 48],
            "MLP_PEARSON_LOSS_W": 0.55,
            "WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES + REVIVAL_OLD_WARM_GENOMES[:5],
            "WIDE_WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES[:8],
        },
        "mlp-low-noise-grok": {
            "WIDTH_BIAS_START": 0.58,
            "MLP_DROPOUT": 0.06,
            "MLP_PATIENCE": 24,
            "MLP_MAX_ITER": 80,
            "MLP_MAX_ROWS": 150_000,
            "GPU_MLP_MAX_ITER": 90,
            "GPU_MLP_MAX_ROWS": 200_000,
            "GPU_MLP_HIDDEN": [128, 64, 32],
            "MLP_PEARSON_LOSS_W": 0.45,
            "WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES + REVIVAL_OLD_WARM_GENOMES[:4],
            "WIDE_WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES[:8],
        },
        "mlp-tail-grok": {
            "WIDTH_BIAS_START": 0.36,
            "POSITIONAL_BLOCK": 192,
            "MLP_DROPOUT": 0.12,
            "MLP_PATIENCE": 24,
            "MLP_MAX_ITER": 75,
            "MLP_MAX_ROWS": 150_000,
            "GPU_MLP_MAX_ITER": 85,
            "GPU_MLP_MAX_ROWS": 200_000,
            "GPU_MLP_HIDDEN": [160, 80, 40],
            "MLP_PEARSON_LOSS_W": 0.50,
            "WARM_GENOMES": [
                "mlp_assoc|tail64_identity",
                "mlp_assoc|tail96_identity",
                "mlp_assoc|tail120_rank",
                "mlp_assoc|tail64_pair_aug",
                "mlp_assoc|testlike_stable64_rank",
                "mlp_assoc|sign_stability64_rank",
            ] + REVIVAL_OLD_WARM_GENOMES[:6],
            "WIDE_WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES[:6],
        },
    }
    base = {
        "TIME_BUDGET_MIN": float(time_budget),
        "RESERVE_MIN": 30.0,
        "N_EXPLORERS": 14,
        "LESSON_BUDGET": 38,
        "MAX_SEASONS": 14,
        "MAX_EPOCHS": 12,
        "EVOLUTION_BUDGET": 104,
        "EVOLUTION_OFFSPRING": 8,
        "EVOLUTION_PATIENCE": 7,
        "ATTENTION_POOL": 24,
        "DIVE_BUDGET": 14,
        "DREAM_REPLAYS": 240,
        "CONFIG_GRID_TOPK": 12,
        "ROBUST_HEDGE_BAND": 0.006,
        "ROBUST_MAX_MEMBERS": 18,
        "MAX_MEMBERS": 14,
        "MAX_FAMILY_MEMBERS": 3,
        "WIDTH_BIAS_HALFLIFE": 100,
        "GREEDY_OLS_TR_ROWS": 12000,
        "GREEDY_OLS_VA_ROWS": 8000,
        "GREEDY_OLS_FALLBACK_K": 36,
        "AUDITION_ALL_SKILLS": True,
        "WIDE_PERSONA": True,
        "SENSORY_ROSTER": True,
        "WIDE_SEEDS": True,
        "TESTLIKE_PARTITIONS": True,
        "TESTLIKE_FEATURE_GATE": True,
        "PLSRANK_FAMILY": True,
        "SIGNSTAB_FAMILY": True,
        "CONSENSUS_FAMILY": True,
        "CONSENSUS_ENSEMBLE": True,
        "FORENSIC_ENABLED": True,
        "ROBUST_OOS_SELECT": True,
        "SHIPPING_COURT": True,
        "REVIVAL_BRANCH_MODE": True,
        "REVIVAL_REQUIRE_CURRENT_EVIDENCE": True,
    }
    members: list[dict] = []
    modes = list(REVIVAL_MODES)
    for i in range(count):
        mode = modes[i % len(modes)]
        is_neural = mode in neural_modes
        ov = dict(base)
        ov.update(neural_modes[mode] if is_neural else old_modes[mode])
        ov["SEED"] = rng.randrange(1_000, 2_000_000_000)
        if is_neural:
            ov.update({
                "GROK_INCUBATION": True,
                "GROK_BRANCH_MODE": mode,
                "GROK_SHIP_ELIGIBLE": False,
                "GROK_EXPECTED_DELAY": rng.choice((3, 4, 5)),
                "GROK_BUDGET_SHARE": rng.choice((0.10, 0.12, 0.15)),
                "STABILITY_NOISE": rng.choice((0.03, 0.05, 0.08)),
            })
        gpu = (is_neural and gpu_frac > 0.0) or (rng.random() < gpu_frac)
        members.append({
            "name": f"{prefix}-{i + 1:02d}-{mode}",
            "gpu": bool(gpu),
            "ov": ov,
            "mode": "revival",
            "revival_mode": mode,
            "hypothesis": (
                "Can old supported material become useful again when remeasured "
                "through current testlike/topology/robust courts?"
                if not is_neural else
                "Can MLP/dropout routes grok delayed structure when seeded as "
                "branch-only warm genomes and judged by current courts?"
            ),
            "translation_goal": (
                "Revive only as fresh evidence: compare local/global impact, "
                "public/private gap, false-agreement risk, and foundation stress."
            ),
            "signal_action": (
                "retest_old_material_with_new_lenses"
                if not is_neural else
                "incubate_mlp_branch_then_retest"
            ),
            "trust_policy": (
                "candidate_only_after_current_robust_court"
                if not is_neural else
                "branch_only_until_independent_current_evidence"
            ),
        })
    return members


def grok_members(prefix: str, *, count: int, seed: int, gpu_frac: float) -> list[dict]:
    rng = random.Random(seed)
    members: list[dict] = []
    for i in range(count):
        gpu = rng.random() < max(0.0, min(1.0, gpu_frac))
        mode = rng.choice(("incubator", "surface", "weird", "long-horizon"))
        ov = {
            "SEED": rng.randrange(1_000, 2_000_000_000),
            "TIME_BUDGET_MIN": rng.choice((90, 120, 150)),
            "RESERVE_MIN": 30.0,
            "MAX_SEASONS": rng.choice((10, 12, 14)),
            "MAX_EPOCHS": rng.choice((10, 12, 14)),
            "EVOLUTION_PATIENCE": rng.choice((6, 8, 10)),
            "EVOLUTION_BUDGET": rng.choice((56, 72, 88)),
            "EVOLUTION_OFFSPRING": rng.choice((6, 8)),
            "ATTENTION_POOL": rng.choice((16, 20, 24)),
            "DIVE_BUDGET": rng.choice((10, 12, 16)),
            "DREAM_REPLAYS": rng.choice((180, 220)),
            "MLP_PATIENCE": rng.choice((14, 18)),
            "MLP_MAX_ITER": rng.choice((40, 55)),
            "WIDTH_BIAS_START": rng.choice((0.65, 0.75, 0.85)),
            "WIDTH_BIAS_HALFLIFE": rng.choice((80, 100)),
            "CONFIG_GRID_TOPK": rng.choice((10, 12)),
            "ROBUST_HEDGE_BAND": rng.choice((0.004, 0.006)),
            "PLSRANK_POOL": rng.choice((256, 320)),
            "PLSRANK_COMPONENTS": rng.choice((8, 10, 12)),
            "STABSEL_BOOT": rng.choice((12, 16)),
            "STABSEL_POOL": rng.choice((256, 320)),
            "REDUNDANCY_MIN_NEW_INFO": rng.choice((0.05, 0.08)),
            "EXTREMA_RECONCILE": True,
            "EXTREMA_MARGIN": rng.choice((0.0015, 0.0020, 0.0030)),
            "TESTLIKE_PARTITIONS": True,
            "TESTLIKE_FRACS": rng.choice(SPROUT_AXES["TESTLIKE_FRACS"]),
            "GROK_INCUBATION": True,
            "GROK_BRANCH_MODE": mode,
            "GROK_SHIP_ELIGIBLE": False,
            "GROK_EXPECTED_DELAY": rng.choice((2, 3, 4)),
            "GROK_BUDGET_SHARE": rng.choice((0.08, 0.10, 0.12)),
            "WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES + REVIVAL_OLD_WARM_GENOMES[:5],
            "WIDE_WARM_GENOMES": REVIVAL_MLP_WARM_GENOMES[:8],
        }
        members.append({
            "name": f"{prefix}-{i + 1:02d}-{mode}",
            "gpu": bool(gpu),
            "ov": ov,
            "mode": "grokking",
            "hypothesis": (
                "Can a quarantined long-horizon branch reveal delayed structure "
                "through topology/surface motion even without early validation gain?"
            ),
            "translation_goal": (
                "Collect grokking evidence: plateau, internal movement proxies, "
                "surface stress, route strengths, and whether delayed value appears."
            ),
            "signal_action": "incubate_branch_then_challenge",
            "trust_policy": "cannot_ship_directly_collect_only_until_robust_court_passes",
        })
    return members


def slug(name: str) -> str:
    return f"drw-wx-{name}"


def load_env() -> dict:
    env = dict(os.environ)
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def kaggle_user() -> str:
    user = load_env().get("KAGGLE_USERNAME")
    if not user:
        sys.exit("missing KAGGLE_USERNAME (set it in ~/.config/worldexplorer/.env or env)")
    return user


def load_manifest(path: str | None) -> list[dict]:
    if not path:
        return list(FLEET)
    p = Path(path)
    data = json.loads(p.read_text())
    members = data.get("members", data if isinstance(data, list) else [])
    if not isinstance(members, list):
        sys.exit(f"bad manifest: {p}")
    return members


def member_pool(a) -> list[dict]:
    return load_manifest(getattr(a, "manifest", None))


def selected_members(a) -> list[dict]:
    pool = member_pool(a)
    wanted = set(a.members or [])
    if not wanted:
        return list(pool)
    names = {m["name"] for m in pool}
    unknown = sorted(wanted - names)
    if unknown:
        sys.exit(f"unknown fleet member(s): {', '.join(unknown)}")
    return [m for m in pool if m["name"] in wanted]


def _merge_parent_ovs(rng: random.Random, parents: list[dict]) -> dict:
    ov: dict = {}
    keys = sorted({k for p in parents for k in p["ov"]})
    for k in keys:
        choices = [p["ov"][k] for p in parents if k in p["ov"]]
        ov[k] = rng.choice(choices)
    return ov


def sprout_members(count: int, *, seed: int, prefix: str, gpu_frac: float) -> list[dict]:
    rng = random.Random(seed)
    gpu_frac = max(0.0, min(1.0, float(gpu_frac)))
    out: list[dict] = []
    for i in range(count):
        mode = rng.choice(SPROUT_MODES)
        n_parents = 2 if rng.random() < 0.55 else 1
        parents = rng.sample(SPROUT_PARENTS, n_parents)
        ov = _merge_parent_ovs(rng, parents)
        # Mutate a small random subset of safe knobs. This is deliberately
        # not a grid: sprouts should bias and wander, while the harness verifies.
        n_mut = rng.randint(4, 8)
        for k in rng.sample(list(SPROUT_AXES), n_mut):
            ov[k] = rng.choice(SPROUT_AXES[k])
        ov["SEED"] = rng.randrange(1_000, 2_000_000_000)
        if mode == "warper":
            ov["EXTREMA_RECONCILE"] = True
            ov.setdefault("EXTREMA_MARGIN", 0.0015)
        elif mode == "governor":
            ov.setdefault("GOV_LAMBDA_SCALE", rng.choice((0.75, 1.0, 1.25)))
            ov.setdefault("GOV_LAMBDA_MAX", rng.choice((0.06, 0.08)))
        elif mode == "testlike":
            ov["TESTLIKE_PARTITIONS"] = True
            ov.setdefault("TESTLIKE_FRACS", rng.choice(SPROUT_AXES["TESTLIKE_FRACS"]))
        elif mode == "stable":
            ov.setdefault("STABSEL_BOOT", rng.choice((12, 16)))
            ov.setdefault("PLSRANK_POOL", rng.choice((256, 320)))
        elif mode == "foundry":
            ov.setdefault("CONFIG_GRID_TOPK", rng.choice((10, 12)))
            ov.setdefault("PLSRANK_POOL", rng.choice((256, 320)))
            ov.setdefault("STABSEL_POOL", rng.choice((256, 320)))
            ov.setdefault("REDUNDANCY_MIN_NEW_INFO", rng.choice((0.05, 0.08)))
            ov.setdefault("WIDTH_BIAS_START", rng.choice((0.60, 0.75)))
        elif mode == "random":
            ov["WIDTH_BIAS_START"] = round(rng.uniform(0.18, 0.88), 2)
            ov["EXTREMA_RECONCILE"] = rng.random() < 0.70
        # Keep sprouts in the light-to-mid zone unless the caller later edits the
        # manifest. This preserves throughput and avoids rigid marathon bets.
        ov["TIME_BUDGET_MIN"] = min(float(ov.get("TIME_BUDGET_MIN", 60)), 90.0)
        gpu = rng.random() < gpu_frac
        name = f"{prefix}-{i + 1:02d}-{mode}"
        out.append({"name": name, "gpu": bool(gpu), "ov": ov,
                    "mode": mode, "parents": [p["name"] for p in parents],
                    "hypothesis": SPROUT_HYPOTHESES[mode],
                    "translation_goal": SPROUT_TRANSLATION_GOALS[mode],
                    "signal_action": SPROUT_SIGNAL_ACTIONS[mode],
                    "trust_policy": "sensor_only_until_translated_and_verified"})
    return out


def _inject(kernel_text: str, ov: dict, *, gpu: bool = False) -> str:
    """Append `CFG.<k> = <v>` overrides as the LAST thing before the run guard,
    so they win over the kernel's own built-in override block."""
    lines = "\n".join(f"CFG.{k} = {v!r}" for k, v in ov.items())
    guard = ""
    if gpu:
        guard = """
# Kaggle sometimes assigns older GPUs (for example P100/K80) to GPU-enabled
# batch kernels. The installed torch build may not contain kernels for those
# architectures; degrade those runs to the CPU-safe schedule instead of crashing.
try:
    _fleet_gpu_names = "|".join(_gpu_names()).lower()
    if any(name in _fleet_gpu_names for name in ("p100", "k80")):
        HAVE_TORCH = False
        N_GPUS = 0
        log("fleet_gpu_degraded", reason="unsupported_torch_gpu", gpu_names=_fleet_gpu_names)
except Exception as _fleet_exc:
    HAVE_TORCH = False
    N_GPUS = 0
    log("fleet_gpu_degraded", reason=f"gpu_probe_failed:{_fleet_exc}")
"""
    block = f"\n# ---- FLEET OVERRIDES (parallel-learning member) ----\n{lines}\n{guard}\n"
    if ov.get("GROK_INCUBATION"):
        block += """
# ---- GROKKING INCUBATION REPORT WRAPPER ----
# Quarantined research lane: this records the long-horizon intent and whatever
# evidence the current engine exposes. It does not alter shipping eligibility.
CFG.GROK_SHIP_ELIGIBLE = False
_wx_orig_run = ExplorerHarness.run
def _wx_grok_run(self):
    summary = _wx_orig_run(self)
    try:
        report = {
            "schema_version": 1,
            "mode": "grokking_incubation",
            "branch_mode": getattr(CFG, "GROK_BRANCH_MODE", "incubator"),
            "ship_eligible": bool(getattr(CFG, "GROK_SHIP_ELIGIBLE", False)),
            "doctrine": "quarantined research lane; cannot ship without robust OOS court",
            "config": {
                "time_budget_min": CFG.TIME_BUDGET_MIN,
                "max_seasons": CFG.MAX_SEASONS,
                "max_epochs": CFG.MAX_EPOCHS,
                "evolution_patience": CFG.EVOLUTION_PATIENCE,
                "evolution_budget": CFG.EVOLUTION_BUDGET,
                "attention_pool": CFG.ATTENTION_POOL,
                "dive_budget": CFG.DIVE_BUDGET,
                "dream_replays": CFG.DREAM_REPLAYS,
                "mlp_patience": CFG.MLP_PATIENCE,
                "mlp_max_iter": CFG.MLP_MAX_ITER,
                "expected_delay": getattr(CFG, "GROK_EXPECTED_DELAY", None),
                "budget_share": getattr(CFG, "GROK_BUDGET_SHARE", None),
            },
            "observed": {
                "forward_blend_corr": summary.get("forward_blend_corr"),
                "sealed_holdout_corr": summary.get("sealed_holdout_corr"),
                "ensemble_winner": summary.get("ensemble_winner"),
                "metabolism": summary.get("metabolism"),
                "anti_decay": summary.get("anti_decay"),
                "feature_clusters": (summary.get("forensics") or {}).get("feature_clusters"),
            },
            "required_promotion_gates": [
                "robust_score_beats_parent",
                "worst_world_nonnegative",
                "overfit_ratio_controlled",
                "false_agreement_risk_checked",
                "private_public_or_forward_gap_not_a_trap",
            ],
            "next_atlas_ingestion": [
                "surface_surgery_matrix",
                "impact_field_matrix",
                "foundation_stress_matrix",
                "route_strength_matrix",
                "contradiction_graph",
            ],
        }
        write_json(report, "grokking_incubation_report.json")
        log("grokking_incubation_report", mode=report["branch_mode"],
            ship_eligible=report["ship_eligible"],
            time_budget_min=report["config"]["time_budget_min"])
    except Exception as _grok_exc:
        log("grokking_incubation_report_skipped", err=str(_grok_exc)[:120])
    return summary
ExplorerHarness.run = _wx_grok_run
"""
    anchor = 'if __name__ == "__main__":'
    i = kernel_text.rindex(anchor)
    return kernel_text[:i] + block + kernel_text[i:]


def _json_kv(items: list[str] | None, *, kind: str) -> dict:
    out: dict = {}
    for raw in items or []:
        if "=" not in raw:
            sys.exit(f"{kind} must be KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            sys.exit(f"{kind} has an empty key: {raw!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _repo_with_ref(repo: str, ref: str | None) -> str:
    if not ref:
        return repo
    if ".git@" in repo:
        return repo
    return f"{repo}@{ref}"


def _inject_bootstrap_config(template: str, config: dict, overrides: dict) -> str:
    marker = "# ---- acquire worldexplorer"
    if marker not in template:
        sys.exit(f"bootstrap template is missing marker: {marker!r}")
    block = (
        "\n# ---- FLEET BOOTSTRAP CONFIG (generated by tools/fleet.py bootstrap) ----\n"
        "import json as _wx_bootstrap_json\n"
        "CONFIG.update(_wx_bootstrap_json.loads('''\n"
        f"{json.dumps(config, indent=4, sort_keys=True)}\n"
        "'''))\n"
        "CONFIG.setdefault('overrides', {}).update(_wx_bootstrap_json.loads('''\n"
        f"{json.dumps(overrides, indent=4, sort_keys=True)}\n"
        "'''))\n\n"
    )
    return template.replace(marker, block + marker, 1)


def _kernel_metadata(
    *,
    user: str,
    name: str,
    title: str,
    gpu: bool,
    internet: bool,
    competition: str | None,
    datasets: list[str],
) -> dict:
    meta = {
        "id": f"{user}/{slug(name)}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(gpu),
        "enable_internet": bool(internet),
    }
    if competition:
        meta["competition_sources"] = [competition]
    if datasets:
        meta["dataset_sources"] = datasets
    return meta


def cmd_bootstrap(a) -> None:
    """Build a small Kaggle kernel that fetches the package from GitHub or an
    attached dataset. This is the non-monolith path; it does not embed the
    generated engine in the Kaggle script."""
    template = Path(a.template).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)

    source_policy = a.source_policy or ("github_first" if a.internet else "wheel_first")
    config = {
        "repo": _repo_with_ref(a.repo, a.repo_ref),
        "source_policy": source_policy,
        "engine_dataset": a.engine_dataset,
        "data_root": a.data_root,
        "target": a.target,
        "train": a.train,
        "test": a.test,
        "sample_submission": a.sample_submission,
        "submission_target_col": a.submission_target_col,
        "metric": a.metric,
        "geometry": a.geometry,
        "time_budget_min": a.time_budget,
        "out": a.working_dir,
    }
    config.update(_json_kv(a.set, kind="--set"))
    overrides = _json_kv(a.override, kind="--override")
    text = _inject_bootstrap_config(template, config, overrides)

    d = out / a.name
    d.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(d / "__pycache__", ignore_errors=True)
    (d / "kernel.py").write_text(text, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(
        json.dumps(
            _kernel_metadata(
                user=user,
                name=a.name,
                title=a.title or f"DRW WX {a.name}",
                gpu=bool(a.gpu),
                internet=bool(a.internet),
                competition=a.competition,
                datasets=list(a.dataset or []),
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    member = {
        "name": a.name,
        "gpu": bool(a.gpu),
        "mode": "bootstrap",
        "source_mode": source_policy,
        "repo": config["repo"],
        "engine_dataset": a.engine_dataset,
        "ov": overrides,
        "trust_policy": (
            "slim kernel; source must be reachable and pinned for reproducible runs"
        ),
    }
    manifest = Path(a.manifest) if a.manifest else out / f"{a.name}_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "doctrine": (
                    "Slim bootstrap fleet member: Kaggle script contains only "
                    "CONFIG plus package acquisition; engine logic lives in GitHub "
                    "or an attached dataset."
                ),
                "members": [member],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"bootstrap {a.name} -> {d}")
    print(f"manifest: {manifest}")
    print(f"push:     python tools/fleet.py push --out {out} --manifest {manifest}")


def cmd_breaker(a) -> None:
    """Build slim framework-native .1-breaker kernels.

    Unlike route_carve, this does not transform existing submissions. It asks
    WorldExplorer to spend runtime on the internal mechanisms most likely to
    produce a new basin: greedy OLS over positional/stability/PLS/testlike
    spaces, then normal robust promotion.
    """
    template = Path(a.template).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)
    members = breaker_members(
        a.prefix,
        count=a.count,
        seed=a.seed,
        gpu_frac=a.gpu_frac,
        time_budget=a.time_budget,
    )
    manifest = Path(a.manifest) if a.manifest else out / f"{a.prefix}_manifest.json"
    repo = _repo_with_ref(a.repo, a.repo_ref)
    dataset_sources = list(a.dataset or [])
    source_policy = a.source_policy or ("github_first" if a.internet else "wheel_first")
    for m in members:
        d = out / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(d / "__pycache__", ignore_errors=True)
        config = {
            "repo": repo,
            "source_policy": source_policy,
            "engine_dataset": a.engine_dataset,
            "data_root": a.data_root,
            "target": a.target,
            "train": None,
            "test": None,
            "sample_submission": None,
            "submission_target_col": None,
            "metric": "auto",
            "geometry": "auto",
            "time_budget_min": a.time_budget,
            "out": "/kaggle/working",
        }
        text = _inject_bootstrap_config(template, config, m["ov"])
        (d / "kernel.py").write_text(text, encoding="utf-8")
        (d / "kernel-metadata.json").write_text(
            json.dumps(
                _kernel_metadata(
                    user=user,
                    name=m["name"],
                    title=f"DRW WX {m['name']}",
                    gpu=bool(m["gpu"]),
                    internet=bool(a.internet),
                    competition=a.competition,
                    datasets=dataset_sources,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        m["repo"] = repo
        m["engine_dataset"] = a.engine_dataset
        m["source_mode"] = source_policy
        print(f"breaker {m['name']:26s} {'GPU' if m['gpu'] else 'CPU'} mode={m['breaker_mode']} repo={repo}")
    manifest.write_text(
        json.dumps(
            {"version": 1, "seed": a.seed, "doctrine": BREAKER_DOCTRINE, "members": members},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nmanifest: {manifest}\n"
          f"push:    python tools/fleet.py push --out {out} --manifest {manifest}\n"
          f"status:  python tools/fleet.py status --manifest {manifest}\n"
          f"collect: python tools/fleet.py collect --out {out} --manifest {manifest} --submit")


def cmd_revival(a) -> None:
    """Build slim revival kernels that remeasure old material and MLP branches.

    This is a GitHub-first counterpart to route-carve probing: the generated
    kernels carry only config overrides, while the engine logic comes from the
    package source. Neural revival members write incubation metadata and are
    intentionally branch-only until an independent current-world retest passes.
    """
    template = Path(a.template).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)
    members = revival_members(
        a.prefix,
        count=a.count,
        seed=a.seed,
        gpu_frac=a.gpu_frac,
        time_budget=a.time_budget,
    )
    manifest = Path(a.manifest) if a.manifest else out / f"{a.prefix}_manifest.json"
    repo = _repo_with_ref(a.repo, a.repo_ref)
    dataset_sources = list(a.dataset or [])
    source_policy = a.source_policy or ("github_first" if a.internet else "wheel_first")
    for m in members:
        d = out / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(d / "__pycache__", ignore_errors=True)
        config = {
            "repo": repo,
            "source_policy": source_policy,
            "engine_dataset": a.engine_dataset,
            "data_root": a.data_root,
            "target": a.target,
            "train": None,
            "test": None,
            "sample_submission": None,
            "submission_target_col": None,
            "metric": "auto",
            "geometry": "auto",
            "time_budget_min": a.time_budget,
            "out": "/kaggle/working",
        }
        text = _inject_bootstrap_config(template, config, m["ov"])
        (d / "kernel.py").write_text(text, encoding="utf-8")
        (d / "kernel-metadata.json").write_text(
            json.dumps(
                _kernel_metadata(
                    user=user,
                    name=m["name"],
                    title=f"DRW WX {m['name']}",
                    gpu=bool(m["gpu"]),
                    internet=bool(a.internet),
                    competition=a.competition,
                    datasets=dataset_sources,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        m["repo"] = repo
        m["engine_dataset"] = a.engine_dataset
        m["source_mode"] = source_policy
        print(
            f"revival {m['name']:34s} {'GPU' if m['gpu'] else 'CPU'} "
            f"mode={m['revival_mode']} trust={m['trust_policy']} repo={repo}"
        )
    manifest.write_text(
        json.dumps(
            {"version": 1, "seed": a.seed, "doctrine": REVIVAL_DOCTRINE, "members": members},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nmanifest: {manifest}\n"
          f"push:    python tools/fleet.py push --out {out} --manifest {manifest}\n"
          f"status:  python tools/fleet.py status --manifest {manifest}\n"
          f"collect: python tools/fleet.py collect --out {out} --manifest {manifest} --submit\n"
          f"grok:    python tools/fleet.py harvest-grok --out {out} --manifest {manifest}")


def cmd_build(a) -> None:
    base = Path(a.kernel).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)
    members = selected_members(a)
    for m in members:
        d = out / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(_inject(base, m["ov"], gpu=bool(m["gpu"])))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug(m['name'])}", "title": f"DRW WX {m['name']}",
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(m["gpu"]), "enable_internet": False,
            "competition_sources": [COMP]}, indent=2))
        print(f"built {m['name']:14s} {'GPU' if m['gpu'] else 'CPU'}  {m['ov']}")
    print(f"\n{len(members)} members -> {out}\n"
          f"next: python tools/fleet.py push --out {out}")


def cmd_sprout(a) -> None:
    base = Path(a.kernel).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)
    members = sprout_members(a.count, seed=a.seed, prefix=a.prefix, gpu_frac=a.gpu_frac)
    manifest = Path(a.manifest) if a.manifest else out / f"{a.prefix}_manifest.json"
    for m in members:
        d = out / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(_inject(base, m["ov"], gpu=bool(m["gpu"])))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug(m['name'])}", "title": f"DRW WX {m['name']}",
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(m["gpu"]), "enable_internet": False,
            "competition_sources": [COMP]}, indent=2))
        parent_s = "+".join(m.get("parents", []))
        print(f"sprouted {m['name']:18s} {'GPU' if m['gpu'] else 'CPU'} "
              f"mode={m.get('mode')} parents={parent_s} hypothesis={m.get('hypothesis')} "
              f"action={m.get('signal_action')} translation={m.get('translation_goal')} ov={m['ov']}")
    manifest.write_text(json.dumps({"version": 1, "seed": a.seed,
                                    "doctrine": SPROUT_DOCTRINE,
                                    "members": members},
                                   indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nmanifest: {manifest}\n"
          f"push:    python tools/fleet.py push --manifest {manifest}\n"
          f"status:  python tools/fleet.py status --manifest {manifest}\n"
          f"collect: python tools/fleet.py collect --manifest {manifest} --submit")


def cmd_grok(a) -> None:
    base = Path(a.kernel).read_text()
    out = Path(a.out)
    user = kaggle_user()
    out.mkdir(parents=True, exist_ok=True)
    members = grok_members(a.prefix, count=a.count, seed=a.seed, gpu_frac=a.gpu_frac)
    manifest = Path(a.manifest) if a.manifest else out / f"{a.prefix}_manifest.json"
    for m in members:
        d = out / m["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(_inject(base, m["ov"], gpu=bool(m["gpu"])))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug(m['name'])}", "title": f"DRW WX {m['name']}",
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(m["gpu"]), "enable_internet": False,
            "competition_sources": [COMP]}, indent=2))
        print(f"grok {m['name']:24s} {'GPU' if m['gpu'] else 'CPU'} "
              f"mode={m['ov']['GROK_BRANCH_MODE']} ov={m['ov']}")
    manifest.write_text(json.dumps({"version": 1, "seed": a.seed,
                                    "doctrine": GROK_DOCTRINE,
                                    "members": members},
                                   indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nmanifest: {manifest}\n"
          f"push:    python tools/fleet.py push --manifest {manifest}\n"
          f"status:  python tools/fleet.py status --manifest {manifest}\n"
          f"collect: python tools/fleet.py collect --manifest {manifest}")


def cmd_push(a) -> None:
    out = Path(a.out)
    failures = 0
    for m in selected_members(a):
        d = out / m["name"]
        if not (d / "kernel.py").exists():
            print(f"skip {m['name']} (not built)"); continue
        print(f"+ kaggle kernels push -p {d}")
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=False)
        failures += int(r.returncode != 0)
    if failures:
        print(f"\npush finished with {failures} failed member(s); retry with --members when slots free")
    print("poll with: python tools/fleet.py status")


def cmd_status(a) -> None:
    user = kaggle_user()
    for m in selected_members(a):
        r = subprocess.run(["kaggle", "kernels", "status", f"{user}/{slug(m['name'])}"],
                           capture_output=True, text=True)
        print(f"{m['name']:14s} {(r.stdout + r.stderr).strip()[:90]}")


def cmd_collect(a) -> None:
    out = Path(a.out)
    user = kaggle_user()
    for m in selected_members(a):
        dest = out / m["name"] / "output"
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["kaggle", "kernels", "output", f"{user}/{slug(m['name'])}",
                           "-p", str(dest)], check=False)
        if r.returncode != 0:
            print(f"{m['name']:14s} output unavailable")
            continue
        sub = dest / "submission.csv"
        if sub.exists():
            print(f"{m['name']:14s} submission.csv ready ({sub})")
            if a.submit:
                subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                                "-f", str(sub), "-m", f"fleet {m['name']}"], check=False)
        else:
            print(f"{m['name']:14s} no submission.csv yet")


def _kernel_status(user: str, name: str) -> str:
    r = subprocess.run(
        ["kaggle", "kernels", "status", f"{user}/{slug(name)}"],
        capture_output=True,
        text=True,
    )
    return (r.stdout + r.stderr).strip()


def _status_is_complete(text: str) -> bool:
    upper = text.upper()
    return "COMPLETE" in upper or "SUCCEEDED" in upper


def _status_is_terminal_failure(text: str) -> bool:
    upper = text.upper()
    return any(token in upper for token in ("ERROR", "FAILED", "CANCEL", "TIMEOUT"))


def cmd_harvest_grok(a) -> None:
    out = Path(a.out)
    user = kaggle_user()
    collected = 0
    pending = 0
    failed = 0
    for m in selected_members(a):
        ov = m.get("ov") if isinstance(m.get("ov"), dict) else {}
        if not ov.get("GROK_INCUBATION"):
            continue
        status = _kernel_status(user, m["name"])
        print(f"{m['name']:24s} {status[:120]}")
        if _status_is_terminal_failure(status):
            failed += 1
            continue
        if not _status_is_complete(status):
            pending += 1
            continue
        dest = out / m["name"] / "output"
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["kaggle", "kernels", "output", f"{user}/{slug(m['name'])}", "-p", str(dest)],
            check=False,
        )
        if r.returncode != 0:
            print(f"{m['name']:24s} output unavailable")
            pending += 1
            continue
        report = dest / "grokking_incubation_report.json"
        if report.exists():
            print(f"{m['name']:24s} collected {report}")
        else:
            print(f"{m['name']:24s} collected output, but no grokking_incubation_report.json")
        collected += 1
    print(
        f"grok harvest: collected={collected} pending={pending} failed={failed}; "
        "rerun tools/memory_matrices.py after collection to ingest observed reports"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="parallel Kaggle fleet launcher")
    sub = ap.add_subparsers(dest="action", required=True)
    bs = sub.add_parser("bootstrap")
    bs.add_argument("--template", default=str(DEFAULT_BOOTSTRAP))
    bs.add_argument("--out", default=str(DEFAULT_OUT))
    bs.add_argument("--name", default="github-bootstrap")
    bs.add_argument("--title", default=None)
    bs.add_argument("--repo", default="git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git")
    bs.add_argument("--repo-ref", default=None, help="commit SHA/tag/branch appended as @ref")
    bs.add_argument("--engine-dataset", default=None,
                    help="offline attached dataset path, e.g. /kaggle/input/worldexplorer-engine")
    bs.add_argument("--internet", dest="internet", action="store_true",
                    help="set Kaggle enable_internet=true so pip can install from GitHub")
    bs.add_argument("--offline", dest="internet", action="store_false",
                    help="set Kaggle enable_internet=false and prefer attached wheel/source")
    bs.add_argument("--source-policy", choices=["github_first", "wheel_first", "github_only", "wheel_only"],
                    default=None,
                    help="override package acquisition order injected into CONFIG")
    bs.add_argument("--gpu", action="store_true")
    bs.add_argument("--competition", default=COMP)
    bs.add_argument("--dataset", nargs="*", default=None,
                    help="Kaggle dataset sources to attach, e.g. user/worldexplorer-engine")
    bs.add_argument("--data-root", default=None)
    bs.add_argument("--target", default="label")
    bs.add_argument("--train", default=None)
    bs.add_argument("--test", default=None)
    bs.add_argument("--sample-submission", default=None)
    bs.add_argument("--submission-target-col", default=None)
    bs.add_argument("--metric", default="auto")
    bs.add_argument("--geometry", default="auto")
    bs.add_argument("--time-budget", type=float, default=120.0)
    bs.add_argument("--working-dir", default="/kaggle/working")
    bs.add_argument("--set", action="append", default=[],
                    help="CONFIG override as KEY=JSON_VALUE, e.g. --set verbose=false")
    bs.add_argument("--override", action="append", default=[],
                    help="HarnessConfig override as KEY=JSON_VALUE, e.g. --override SEED=7")
    bs.add_argument("--manifest", default=None)
    bs.set_defaults(internet=True)
    bs.set_defaults(fn=cmd_bootstrap)
    b = sub.add_parser("build"); b.add_argument("--kernel", default=str(DEFAULT_KERNEL))
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--manifest", default=None)
    b.add_argument("--members", nargs="*", default=None); b.set_defaults(fn=cmd_build)
    sp = sub.add_parser("sprout"); sp.add_argument("--kernel", default=str(DEFAULT_KERNEL))
    sp.add_argument("--out", default=str(DEFAULT_OUT))
    sp.add_argument("--count", type=int, default=6)
    sp.add_argument("--seed", type=int, default=20260613)
    sp.add_argument("--prefix", default="sprout")
    sp.add_argument("--gpu-frac", type=float, default=0.25)
    sp.add_argument("--manifest", default=None); sp.set_defaults(fn=cmd_sprout)
    gp = sub.add_parser("grok"); gp.add_argument("--kernel", default=str(DEFAULT_KERNEL))
    gp.add_argument("--out", default=str(DEFAULT_OUT))
    gp.add_argument("--count", type=int, default=1)
    gp.add_argument("--seed", type=int, default=20260614)
    gp.add_argument("--prefix", default="grok-atlas")
    gp.add_argument("--gpu-frac", type=float, default=0.0)
    gp.add_argument("--manifest", default=None); gp.set_defaults(fn=cmd_grok)
    br = sub.add_parser("breaker")
    br.add_argument("--template", default=str(DEFAULT_BOOTSTRAP))
    br.add_argument("--out", default=str(DEFAULT_OUT))
    br.add_argument("--count", type=int, default=5)
    br.add_argument("--seed", type=int, default=20260614)
    br.add_argument("--prefix", default="breaker")
    br.add_argument("--gpu-frac", type=float, default=0.0)
    br.add_argument("--time-budget", type=float, default=180.0)
    br.add_argument("--repo", default="git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git")
    br.add_argument("--repo-ref", default="master")
    br.add_argument("--internet", dest="internet", action="store_true")
    br.add_argument("--offline", dest="internet", action="store_false")
    br.add_argument("--source-policy", choices=["github_first", "wheel_first", "github_only", "wheel_only"],
                    default=None)
    br.add_argument("--engine-dataset", default=None)
    br.add_argument("--dataset", nargs="*", default=None)
    br.add_argument("--competition", default=COMP)
    br.add_argument("--data-root", default=None)
    br.add_argument("--target", default="label")
    br.add_argument("--manifest", default=None)
    br.set_defaults(internet=True)
    br.set_defaults(fn=cmd_breaker)
    rv = sub.add_parser("revive")
    rv.add_argument("--template", default=str(DEFAULT_BOOTSTRAP))
    rv.add_argument("--out", default=str(DEFAULT_OUT))
    rv.add_argument("--count", type=int, default=6)
    rv.add_argument("--seed", type=int, default=20260615)
    rv.add_argument("--prefix", default="revival")
    rv.add_argument("--gpu-frac", type=float, default=0.50)
    rv.add_argument("--time-budget", type=float, default=210.0)
    rv.add_argument("--repo", default="git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git")
    rv.add_argument("--repo-ref", default="master")
    rv.add_argument("--internet", dest="internet", action="store_true")
    rv.add_argument("--offline", dest="internet", action="store_false")
    rv.add_argument("--source-policy", choices=["github_first", "wheel_first", "github_only", "wheel_only"],
                    default=None)
    rv.add_argument("--engine-dataset", default=None)
    rv.add_argument("--dataset", nargs="*", default=None)
    rv.add_argument("--competition", default=COMP)
    rv.add_argument("--data-root", default=None)
    rv.add_argument("--target", default="label")
    rv.add_argument("--manifest", default=None)
    rv.set_defaults(internet=True)
    rv.set_defaults(fn=cmd_revival)
    p = sub.add_parser("push"); p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--manifest", default=None)
    p.add_argument("--members", nargs="*", default=None); p.set_defaults(fn=cmd_push)
    s = sub.add_parser("status"); s.add_argument("--manifest", default=None)
    s.add_argument("--members", nargs="*", default=None); s.set_defaults(fn=cmd_status)
    c = sub.add_parser("collect"); c.add_argument("--out", default=str(DEFAULT_OUT))
    c.add_argument("--manifest", default=None)
    c.add_argument("--members", nargs="*", default=None)
    c.add_argument("--submit", action="store_true"); c.set_defaults(fn=cmd_collect)
    hg = sub.add_parser("harvest-grok"); hg.add_argument("--out", default=str(DEFAULT_OUT))
    hg.add_argument("--manifest", default=None)
    hg.add_argument("--members", nargs="*", default=None); hg.set_defaults(fn=cmd_harvest_grok)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
