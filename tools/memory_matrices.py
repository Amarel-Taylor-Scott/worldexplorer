#!/usr/bin/env python3
"""Compile multi-run learning memory matrices from worldexplorer artifacts.

This is deliberately not a blend-weight generator. It builds provenance-bearing
memory stores:

- path_memory_matrix.csv: every observed path/member with typology + evidence.
- typology_memory_matrix.csv: skill/family/transform aggregates with shrinkage.
- typology_coverage_matrix.csv: tried and structurally adjacent untried cells.
- typology_vector_field.csv: directed local gradients between typology cells.
- feature_space_memory_matrix.csv: micro feature/topology/field observations.
- surface_surgery_matrix.csv: local/global agreement, disagreement, and edit plans.
- impact_field_matrix.csv: local/global ripple effects from candidate moves.
- foundation_stress_matrix.csv: pressure that the current surface foundation is wrong.
- route_strength_matrix.csv: learned value estimates for reusable action types.
- validation_budget_ledger.csv: validation-world reuse pressure and trust discounts.
- evidence_gate_matrix.csv: branch admission and promotion gate decisions.
- proof_carrying_paths.jsonl: machine-readable evidence/risk certificates for candidates.
- contradiction_graph.csv: supports/contradicts/revives claim edges.
- grokking_incubation_matrix.csv: quarantined long-horizon branch records.
- projection_memory_matrix.csv: dimensionality-reduction/transform memory.
- collinearity_memory_matrix.csv: feature-community redundancy memory.
- operation_memory_matrix.csv: route/blend/path actions with weights/biases.
- run_memory_matrix.csv: runtime, hardware, score, and artifact health.
- relation_edges.csv: run/path/typology/submission relationships.
- numeric_memory_bundle.npz: loadable vectors/matrices/weights for runtimes.
- tensor_artifact_manifest.json: actual loadable array catalog.
- checkpoint_graph.json: versioned learning-state nodes and branches.
- operator_graph_edges.csv: replayable state transitions/actions.
- computational_atlas_manifest.json: top-level atlas contract.
- attention_inputs.json: short/medium/long-term memory views for the next run.
- next_runtime_policy.json: machine-consumable budget, branch, grokking, and guard policy.

Scores are optional. When supplied, they are attached as external observations
instead of replacing artifact-derived evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import telemetry_guidance as tg

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FLEET_DIR = WORKSPACE / "kaggle" / "fleet"
DEFAULT_OUT = FLEET_DIR / "memory_matrices"

PATH_NUMERIC_COLUMNS = [
    "oof_corr",
    "wf_corr",
    "decay",
    "width",
    "wf_width",
    "stability",
    "fit_corr",
    "overfit_ratio",
    "uniqueness",
    "cost",
    "k",
    "worst3_corr",
    "terrain_min_corr",
    "weather_min_corr",
    "beacon_min_corr",
    "world_floor",
    "world_frac_positive",
    "escape_velocity",
    "crowd_load",
    "complexity",
    "roughness",
    "wake_ac1",
    "side_asym",
    "robust_memory_score",
    "promoted",
    "shipped",
]

TYPOLOGY_NUMERIC_COLUMNS = [
    "n_observations",
    "n_runs",
    "promote_rate",
    "ship_rate",
    "robust_score_mean",
    "robust_score_std",
    "robust_score_shrunk",
    "decay_mean",
    "world_floor_mean",
]

FEATURE_NUMERIC_COLUMNS = [
    "size",
    "rows",
    "signal",
    "own_corr",
    "foreign_corr",
    "stability_or_coherence",
    "shift_or_noise",
    "hazard",
    "vector_x",
    "vector_y",
    "vector_z",
    "micro_priority",
]

SURFACE_NUMERIC_COLUMNS = [
    "local_signal",
    "global_stability",
    "agreement_score",
    "disagreement_score",
    "uncertainty",
    "false_agreement_risk",
    "false_disagreement_risk",
    "overfit_risk",
    "rearrangement_gain",
    "grokking_priority",
]

IMPACT_NUMERIC_COLUMNS = [
    "local_effect",
    "global_effect",
    "stability_effect",
    "agreement_delta",
    "disagreement_delta",
    "uncertainty_delta",
    "false_agreement_risk",
    "false_disagreement_risk",
    "overfit_risk",
    "ripple_radius",
    "side_effect_load",
    "foundation_stress_delta",
    "move_quality",
    "branch_priority",
]

FOUNDATION_STRESS_NUMERIC_COLUMNS = [
    "n_surface_records",
    "stress_score",
    "mean_uncertainty",
    "mean_false_agreement",
    "mean_false_disagreement",
    "mean_overfit",
    "mean_disagreement",
    "high_risk_fraction",
    "branch_pressure",
    "foundation_rethink_priority",
]

ROUTE_STRENGTH_NUMERIC_COLUMNS = [
    "n_observations",
    "expected_gain",
    "uncertainty",
    "success_probability",
    "overfit_risk",
    "transferability",
    "fragility",
    "complementarity",
    "branch_value",
]

VALIDATION_BUDGET_NUMERIC_COLUMNS = [
    "candidate_count_seen",
    "selection_use_count",
    "reporting_use_count",
    "reuse_pressure",
    "redundancy_pressure",
    "trust_discount",
    "remaining_trust_budget",
]

EVIDENCE_GATE_NUMERIC_COLUMNS = [
    "local_effect",
    "global_effect",
    "worst_world_proxy",
    "support_count",
    "independent_support",
    "false_agreement_risk",
    "false_disagreement_risk",
    "overfit_risk",
    "foundation_stress_delta",
    "branch_priority",
    "evidence_score",
    "drift_risk",
    "promotion_allowed",
    "branch_allowed",
    "grokking_allowed",
]

GROKKING_NUMERIC_COLUMNS = [
    "time_budget_min",
    "max_seasons",
    "max_epochs",
    "evolution_patience",
    "evolution_budget",
    "attention_pool",
    "dive_budget",
    "dream_replays",
    "mlp_patience",
    "mlp_max_iter",
    "expected_delay",
    "budget_share",
    "forward_blend_corr",
    "sealed_holdout_corr",
    "feature_clusters",
    "seasons_observed",
    "epochs_observed",
    "ship_eligible",
]

PROJECTION_NUMERIC_COLUMNS = [
    "n_observations",
    "n_runs",
    "robust_score_mean",
    "robust_score_shrunk",
    "downstream_path_gain",
    "decay_mean",
    "width_mean",
    "world_floor_mean",
    "oof_corr_mean",
    "wf_corr_mean",
    "promotion_rate",
    "shipping_rate",
    "grokking_potential",
    "hazard_bias",
]

COLLINEARITY_NUMERIC_COLUMNS = [
    "community_size",
    "topology_coherence",
    "mean_abs_corr_y",
    "max_abs_corr_y",
    "signal_consensus",
    "mean_train_test_shift",
    "redundancy_score",
    "compression_bias",
    "shift_hazard",
]

OPERATION_NUMERIC_COLUMNS = [
    "alpha",
    "operation_strength",
    "external_private",
    "external_public",
    "weight_path_strength",
    "weight_route_information_gain",
    "weight_component_abs_sum",
    "weight_component_l2",
    "weight_component_signed_sum",
    "bias_corr_prev_best",
    "bias_corr_gpu",
    "bias_corr_balanced",
    "bias_corr_sharp",
    "bias_corr_light",
    "bias_corr_champion_route",
    "bias_corr_candidate_champion",
    "bias_corr_candidate_route",
    "bias_novelty_from_champion",
    "bias_conflict_tail",
    "bias_agree_tail",
    "bias_route_tail",
    "bias_decay",
    "bias_world_floor",
    "bias_width",
    "cost",
    "k",
    "promoted",
    "shipped",
]


def f(value: Any) -> float | None:
    return tg.as_float(value)


def safe_float(row: dict[str, Any], key: str) -> float | None:
    return f(row.get(key))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except Exception:
        return []


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stable_id(*parts: Any) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def mean(vals: list[float | None]) -> float | None:
    good = [x for x in vals if x is not None and math.isfinite(x)]
    if not good:
        return None
    return sum(good) / len(good)


def stdev(vals: list[float | None]) -> float | None:
    good = [x for x in vals if x is not None and math.isfinite(x)]
    if len(good) < 2:
        return None
    m = sum(good) / len(good)
    return math.sqrt(sum((x - m) ** 2 for x in good) / len(good))


def shrink_mean(vals: list[float | None], prior: float, prior_n: float = 4.0) -> float | None:
    good = [x for x in vals if x is not None and math.isfinite(x)]
    if not good:
        return None
    return (sum(good) + prior * prior_n) / (len(good) + prior_n)


def bounded_corr(value: float | None) -> float | None:
    if value is None:
        return None
    return max(-1.0, min(1.0, value))


def numeric_value(value: Any) -> float:
    val = f(value)
    if val is None or not math.isfinite(val):
        return np.nan
    return float(val)


def numeric_matrix(
    rows: list[dict[str, Any]],
    id_key: str,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([str(row.get(id_key, "")) for row in rows], dtype=np.str_)
    matrix = np.empty((len(rows), len(columns)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            matrix[i, j] = numeric_value(row.get(column))
    return ids, matrix


def nan_to_zero(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        out[finite] = 1.0 if hi > 0.0 else 0.0
        return out
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


def clip01(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def softmax_weights(values: np.ndarray, temperature: float = 40.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    scaled = np.clip((arr[finite] - np.nanmax(arr[finite])) * temperature, -80.0, 0.0)
    exp = np.exp(scaled)
    denom = float(exp.sum())
    if denom > 0:
        out[finite] = exp / denom
    return out


def standardized_information_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a compact X'X-style matrix with missing values imputed by column mean."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.size == 0:
        return np.zeros((x.shape[1] if x.ndim == 2 else 0, x.shape[1] if x.ndim == 2 else 0))
    filled = x.copy()
    for j in range(filled.shape[1]):
        col = filled[:, j]
        finite = np.isfinite(col)
        fill = float(np.nanmean(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        filled[:, j] = col
    centered = filled - filled.mean(axis=0, keepdims=True)
    scale = filled.std(axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    z = centered / scale
    denom = max(1, z.shape[0] - 1)
    return (z.T @ z) / denom


def scores_by_filename(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for score in scores:
        filename = Path(str(score.get("filename", ""))).name
        if filename:
            out[filename] = score
    return out


def path_memory_score(
    *,
    oof_corr: float | None,
    wf_corr: float | None,
    decay: float | None,
    world_floor: float | None,
    terrain_min: float | None,
    weather_min: float | None,
    worst3_corr: float | None,
    width: float | None,
    overfit_ratio: float | None,
    promoted: bool,
    shipped: bool,
    escape_velocity: float | None,
) -> float | None:
    # Keep the memory score in correlation-like units. Fields such as
    # "stability" can be diagnostic but are not necessarily bounded, so they
    # stay in the matrix and out of this scalar summary.
    components = [
        bounded_corr(wf_corr),
        None if oof_corr is None else 0.5 * bounded_corr(oof_corr),
        bounded_corr(world_floor),
        bounded_corr(terrain_min),
        bounded_corr(weather_min),
        bounded_corr(worst3_corr),
    ]
    base = mean(components)
    if base is None:
        return None
    score = base
    if decay is not None:
        score -= 0.55 * max(0.0, decay)
        score += 0.20 * max(0.0, -decay)
    if overfit_ratio is not None:
        score -= min(0.025, 0.010 * max(0.0, overfit_ratio - 1.0))
    if width is not None and width < 0.0:
        score -= 0.035
    if promoted:
        score += 0.010
    if shipped:
        score += 0.020
    if escape_velocity is not None:
        score += 0.25 * max(-0.05, min(0.08, escape_velocity))
    return score


def parse_key(key: str) -> dict[str, str | None]:
    skill, rest = (key.split("|", 1) + [""])[:2] if "|" in key else (key, "")
    family = None
    width = None
    transform = None
    if rest:
        bits = rest.rsplit("_", 1)
        transform = bits[-1] if len(bits) == 2 else None
        fam_width = bits[0] if len(bits) == 2 else rest
        match = re.match(r"(.+?)(\d+)$", fam_width)
        if match:
            family = match.group(1)
            width = match.group(2)
        else:
            family = fam_width
    return {"skill_from_key": skill, "family_from_key": family, "width_from_key": width,
            "transform_from_key": transform}


def run_dirs(fleet_dir: Path) -> list[Path]:
    return sorted(p for p in fleet_dir.iterdir() if p.is_dir() and (p / "output").exists())


def log_tail(path: Path | None, n: int = 8) -> str:
    if path is None or not path.exists():
        return ""
    lines = path.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def artifact_map(out: Path) -> dict[str, str]:
    wanted = {
        "lessons": "explorer_lessons.csv",
        "shipping": "shipping_court_report.csv",
        "worlds": "many_worlds_cv.csv",
        "path_texture": "path_texture_report.csv",
        "shared_library": "shared_library.csv",
        "trap_map": "trap_map.csv",
        "partial_dynamics": "partial_dynamics_tensor.csv",
        "support_sheaf": "support_sheaf_consistency.csv",
        "feature_topology": "feature_topology_report.csv",
        "summary": "explorer_run_summary.json",
        "ledger": "learning_ledger.json",
        "cairn": "world_cairn.json",
    }
    return {k: str(out / v) for k, v in wanted.items() if (out / v).exists()}


def run_record(run_dir: Path, scores_by_run: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = run_dir / "output"
    runs = tg.load_runs(run_dir.parent)
    base = runs.get(run_dir.name, {"run": run_dir.name, "output_dir": str(out)})
    scores = scores_by_run.get(run_dir.name, [])
    best_private = max((f(s.get("private")) for s in scores if f(s.get("private")) is not None), default=None)
    best_public = max((f(s.get("public")) for s in scores if f(s.get("public")) is not None), default=None)
    logs = list(out.glob("*.log"))
    rec = dict(base)
    rec.update({
        "run_id": stable_id("run", run_dir.name),
        "best_external_private": best_private,
        "best_external_public": best_public,
        "external_observations": len(scores),
        "artifact_count": len(list(out.iterdir())) if out.exists() else 0,
        "artifact_manifest": json.dumps(artifact_map(out), sort_keys=True),
        "log_tail": log_tail(logs[0] if logs else None),
    })
    return rec


def attach_worlds(out: Path) -> dict[str, dict[str, str]]:
    return {row.get("member", ""): row for row in read_csv(out / "many_worlds_cv.csv") if row.get("member")}


def attach_shipping(out: Path) -> dict[str, dict[str, str]]:
    return {row.get("member", ""): row for row in read_csv(out / "shipping_court_report.csv") if row.get("member")}


def attach_texture(out: Path) -> dict[str, dict[str, str]]:
    return {row.get("key", ""): row for row in read_csv(out / "path_texture_report.csv") if row.get("key")}


def path_records(run_dir: Path) -> list[dict[str, Any]]:
    out = run_dir / "output"
    worlds = attach_worlds(out)
    shipping = attach_shipping(out)
    texture = attach_texture(out)
    rows: list[dict[str, Any]] = []
    for lesson in read_csv(out / "explorer_lessons.csv"):
        key = lesson.get("key") or ""
        if not key:
            continue
        shipped = shipping.get(key, {})
        world = worlds.get(key, {})
        tex = texture.get(key, {})
        parsed = parse_key(key)
        decay = safe_float(shipped, "decay")
        if decay is None:
            oof = safe_float(lesson, "oof_corr")
            wf = safe_float(lesson, "wf_corr")
            decay = None if oof is None or wf is None else oof - wf
        world_floor = safe_float(world, "world_survival_min")
        world_frac = safe_float(world, "world_frac_positive")
        width = safe_float(lesson, "width")
        wf_width = safe_float(lesson, "wf_width")
        terrain_min = safe_float(lesson, "terrain_min_corr") or safe_float(tex, "terrain_min")
        weather_min = safe_float(lesson, "weather_min_corr") or safe_float(tex, "weather_min")
        promoted = "promote" in str(lesson.get("decision", ""))
        shipped_flag = bool(shipped)
        oof_corr = safe_float(lesson, "oof_corr")
        wf_corr = safe_float(lesson, "wf_corr")
        stability = safe_float(lesson, "stability")
        overfit_ratio = safe_float(lesson, "overfit_ratio")
        escape_velocity = safe_float(shipped, "escape_velocity")
        robust_score = path_memory_score(
            oof_corr=oof_corr,
            wf_corr=wf_corr,
            decay=decay,
            world_floor=world_floor,
            terrain_min=terrain_min,
            weather_min=weather_min,
            worst3_corr=safe_float(lesson, "worst3_corr"),
            width=width,
            overfit_ratio=overfit_ratio,
            promoted=promoted,
            shipped=shipped_flag,
            escape_velocity=escape_velocity,
        )
        rows.append({
            "path_id": stable_id(run_dir.name, key),
            "run": run_dir.name,
            "run_id": stable_id("run", run_dir.name),
            "key": key,
            "explorer": lesson.get("explorer"),
            "stage": lesson.get("stage"),
            "skill": lesson.get("skill") or parsed["skill_from_key"],
            "viewport": lesson.get("viewport"),
            "family": lesson.get("family") or parsed["family_from_key"],
            "transform": lesson.get("transform") or parsed["transform_from_key"],
            "width_from_key": parsed["width_from_key"],
            "decision": lesson.get("decision"),
            "reason": lesson.get("reason"),
            "promoted": int(promoted),
            "shipped": int(shipped_flag),
            "oof_corr": oof_corr,
            "wf_corr": wf_corr,
            "decay": decay,
            "width": width,
            "wf_width": wf_width,
            "stability": stability,
            "fit_corr": safe_float(lesson, "fit_corr"),
            "overfit_ratio": overfit_ratio,
            "uniqueness": safe_float(lesson, "uniqueness"),
            "cost": safe_float(lesson, "cost"),
            "k": safe_float(lesson, "k"),
            "worst3_corr": safe_float(lesson, "worst3_corr"),
            "terrain_min_corr": terrain_min,
            "weather_min_corr": weather_min,
            "beacon_min_corr": safe_float(lesson, "beacon_min_corr"),
            "predator_verdict": lesson.get("predator_verdict"),
            "world_floor": world_floor,
            "world_frac_positive": world_frac,
            "escape_velocity": escape_velocity,
            "crowd_load": safe_float(shipped, "crowd_load"),
            "complexity": safe_float(shipped, "complexity"),
            "roughness": safe_float(tex, "roughness"),
            "wake_ac1": safe_float(tex, "wake_ac1"),
            "side_asym": safe_float(tex, "side_asym"),
            "trail_family": tex.get("trail_family"),
            "robust_memory_score": robust_score,
            "artifact_source": str(out / "explorer_lessons.csv"),
        })
    return rows


def relation_edges(
    run_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    route_index: dict[str, dict[str, Any]],
    operation_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for row in run_rows:
        edges.append({
            "src_type": "run",
            "src": row["run_id"],
            "relation": "has_artifacts",
            "dst_type": "artifact_manifest",
            "dst": row.get("artifact_manifest"),
            "weight": 1.0,
            "evidence": row.get("output_dir"),
        })
    for row in path_rows:
        path_id = row["path_id"]
        run_id = row["run_id"]
        typology = stable_id("typology", row.get("skill"), row.get("family"), row.get("transform"))
        edges.extend([
            {"src_type": "run", "src": run_id, "relation": "observed_path", "dst_type": "path",
             "dst": path_id, "weight": 1.0, "evidence": row.get("artifact_source")},
            {"src_type": "path", "src": path_id, "relation": "has_typology", "dst_type": "typology",
             "dst": typology, "weight": 1.0, "evidence": f"{row.get('skill')}|{row.get('family')}|{row.get('transform')}"},
        ])
        if row.get("shipped"):
            edges.append({"src_type": "path", "src": path_id, "relation": "shipped", "dst_type": "run",
                          "dst": run_id, "weight": row.get("escape_velocity") or 1.0,
                          "evidence": "shipping_court_report.csv"})
    for score in score_rows:
        filename = Path(str(score.get("filename", ""))).name
        manifest = route_index.get(filename)
        if manifest:
            route = manifest.get("route")
            edges.append({
                "src_type": "submission",
                "src": filename,
                "relation": "route_carved_from",
                "dst_type": "run",
                "dst": stable_id("run", route),
                "weight": score.get("private") or 0.0,
                "evidence": manifest.get("manifest_path"),
            })
    for op in operation_rows or []:
        op_id = op.get("operation_id")
        if not op_id:
            continue
        source_run_id = op.get("source_run_id")
        if source_run_id:
            edges.append({
                "src_type": "run",
                "src": source_run_id,
                "relation": "performed_operation",
                "dst_type": "operation",
                "dst": op_id,
                "weight": op.get("operation_strength") or op.get("weight_path_strength") or 0.0,
                "evidence": op.get("artifact_source"),
            })
        if op.get("path_id"):
            edges.append({
                "src_type": "operation",
                "src": op_id,
                "relation": "materialized_path",
                "dst_type": "path",
                "dst": op.get("path_id"),
                "weight": op.get("weight_path_strength") or 0.0,
                "evidence": op.get("artifact_source"),
            })
        if op.get("candidate_filename"):
            edges.append({
                "src_type": "operation",
                "src": op_id,
                "relation": "materialized_submission",
                "dst_type": "submission",
                "dst": op.get("candidate_filename"),
                "weight": op.get("operation_strength") or 0.0,
                "evidence": op.get("artifact_source"),
            })
    return edges


def aggregate_typology(path_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        groups[(row.get("skill"), row.get("family"), row.get("transform"))].append(row)

    all_scores = [f(r.get("robust_memory_score")) for r in path_rows]
    prior = mean(all_scores) or 0.0
    out: list[dict[str, Any]] = []
    for (skill, family, transform), rows in groups.items():
        robust = [f(r.get("robust_memory_score")) for r in rows]
        decay = [f(r.get("decay")) for r in rows]
        world = [f(r.get("world_floor")) for r in rows]
        promoted = [f(r.get("promoted")) for r in rows]
        shipped = [f(r.get("shipped")) for r in rows]
        out.append({
            "typology_id": stable_id("typology", skill, family, transform),
            "skill": skill,
            "family": family,
            "transform": transform,
            "n_observations": len(rows),
            "n_runs": len({r.get("run") for r in rows}),
            "promote_rate": mean(promoted),
            "ship_rate": mean(shipped),
            "robust_score_mean": mean(robust),
            "robust_score_std": stdev(robust),
            "robust_score_shrunk": shrink_mean(robust, prior),
            "decay_mean": mean(decay),
            "world_floor_mean": mean(world),
            "best_path_key": max(rows, key=lambda r: f(r.get("robust_memory_score")) or -999).get("key"),
            "memory_role": memory_role(mean(robust), mean(decay), mean(world), mean(promoted), len(rows)),
        })
    return sorted(out, key=lambda r: f(r.get("robust_score_shrunk")) or -999, reverse=True)


def feature_space_records(run_dir: Path) -> list[dict[str, Any]]:
    out = run_dir / "output"
    rows: list[dict[str, Any]] = []
    run = run_dir.name
    run_id = stable_id("run", run)

    for row in read_csv(out / "feature_topology_report.csv"):
        signal = safe_float(row, "mean_abs_corr_y")
        shift = safe_float(row, "mean_train_test_shift")
        coherence = safe_float(row, "topology_coherence")
        rows.append({
            "feature_memory_id": stable_id(run, "feature_community", row.get("community")),
            "run": run,
            "run_id": run_id,
            "record_type": "feature_community",
            "coordinate_kind": "community",
            "coordinate": row.get("community"),
            "secondary_coordinate": None,
            "size": safe_float(row, "size"),
            "signal": signal,
            "stability_or_coherence": coherence,
            "shift_or_noise": shift,
            "hazard": None if shift is None else int(shift > 0.30),
            "vector_x": signal,
            "vector_y": None if shift is None else -shift,
            "vector_z": coherence,
            "micro_priority": mean([signal, coherence, None if shift is None else -shift]),
            "verdict": "shift_hazard" if shift is not None and shift > 0.30 else "feature_region",
            "artifact_source": str(out / "feature_topology_report.csv"),
        })

    for row in read_csv(out / "trap_map.csv"):
        signal = safe_float(row, "full_abs_corr")
        flip = safe_float(row, "fold_flip_rate")
        verdict = row.get("verdict")
        rows.append({
            "feature_memory_id": stable_id(run, "trap", row.get("col_idx")),
            "run": run,
            "run_id": run_id,
            "record_type": "feature_trap",
            "coordinate_kind": "feature_col",
            "coordinate": row.get("col_idx"),
            "secondary_coordinate": None,
            "size": None,
            "signal": signal,
            "stability_or_coherence": None if flip is None else 1.0 - flip,
            "shift_or_noise": flip,
            "hazard": int(str(verdict).upper() == "MIRAGE") if verdict else None,
            "vector_x": signal,
            "vector_y": None if flip is None else -flip,
            "vector_z": None,
            "micro_priority": mean([signal, None if flip is None else -flip]),
            "verdict": verdict,
            "artifact_source": str(out / "trap_map.csv"),
        })

    for row in read_csv(out / "partial_dynamics_tensor.csv"):
        corr = safe_float(row, "corr")
        rows.append({
            "feature_memory_id": stable_id(run, "partial_dynamics", row.get("feature_cluster"), row.get("time_block")),
            "run": run,
            "run_id": run_id,
            "record_type": "partial_dynamics_cell",
            "coordinate_kind": "feature_cluster",
            "coordinate": row.get("feature_cluster"),
            "secondary_coordinate": row.get("time_block"),
            "size": safe_float(row, "n_cols"),
            "rows": safe_float(row, "rows"),
            "signal": corr,
            "stability_or_coherence": None,
            "shift_or_noise": None,
            "hazard": None,
            "vector_x": corr,
            "vector_y": row.get("time_block"),
            "vector_z": row.get("feature_cluster"),
            "micro_priority": corr,
            "verdict": "local_time_field",
            "artifact_source": str(out / "partial_dynamics_tensor.csv"),
        })

    for row in read_csv(out / "support_sheaf_consistency.csv"):
        own = safe_float(row, "own_corr")
        foreign = safe_float(row, "foreign_corr")
        defect = safe_float(row, "sheaf_defect")
        rows.append({
            "feature_memory_id": stable_id(run, "support_sheaf", row.get("home_terrain"), row.get("applied_model_from")),
            "run": run,
            "run_id": run_id,
            "record_type": "support_sheaf_edge",
            "coordinate_kind": "terrain",
            "coordinate": row.get("home_terrain"),
            "secondary_coordinate": row.get("applied_model_from"),
            "size": None,
            "signal": foreign,
            "own_corr": own,
            "foreign_corr": foreign,
            "stability_or_coherence": None if defect is None else 1.0 - defect,
            "shift_or_noise": defect,
            "hazard": int(defect > 0.15) if defect is not None else None,
            "vector_x": None if own is None or foreign is None else foreign - own,
            "vector_y": defect,
            "vector_z": own,
            "micro_priority": mean([foreign, None if defect is None else -defect]),
            "verdict": "portable_model" if defect is not None and defect < 0.08 else "terrain_specific",
            "artifact_source": str(out / "support_sheaf_consistency.csv"),
        })
    return rows


def typology_coverage_matrix(typology_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = {
        (row.get("skill"), row.get("family"), row.get("transform")): row
        for row in typology_rows
    }
    skills = sorted({row.get("skill") for row in typology_rows if row.get("skill")})
    families = sorted({row.get("family") for row in typology_rows if row.get("family")})
    transforms = sorted({row.get("transform") for row in typology_rows if row.get("transform")})

    by_skill_family: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    by_skill_transform: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    by_family_transform: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in typology_rows:
        by_skill_family[(row.get("skill"), row.get("family"))].append(row)
        by_skill_transform[(row.get("skill"), row.get("transform"))].append(row)
        by_family_transform[(row.get("family"), row.get("transform"))].append(row)

    rows: list[dict[str, Any]] = []
    for skill in skills:
        for family in families:
            for transform in transforms:
                key = (skill, family, transform)
                tried = key in observed
                neighbors = (
                    by_skill_family.get((skill, family), [])
                    + by_skill_transform.get((skill, transform), [])
                    + by_family_transform.get((family, transform), [])
                )
                # Keep structurally meaningful untried cells, not a blind full
                # Cartesian dump. A cell is meaningful when at least one pair of
                # axes has already been observed.
                if not tried and not neighbors:
                    continue
                neighbor_scores = [f(r.get("robust_score_shrunk")) for r in neighbors]
                neighbor_promote = [f(r.get("promote_rate")) for r in neighbors]
                obs = observed.get(key, {})
                neighbor_score = mean(neighbor_scores)
                gap_priority = None if tried else neighbor_score
                if gap_priority is not None:
                    gap_priority += 0.01 * len({(r.get("skill"), r.get("family"), r.get("transform")) for r in neighbors})
                    gap_priority += 0.02 * (mean(neighbor_promote) or 0.0)
                if tried:
                    status = "tried"
                elif gap_priority is not None and gap_priority > 0.065:
                    status = "untried_high_support_gap"
                else:
                    status = "untried_adjacent_gap"
                rows.append({
                    "coverage_id": stable_id("coverage", skill, family, transform),
                    "skill": skill,
                    "family": family,
                    "transform": transform,
                    "tried": int(tried),
                    "coverage_status": status,
                    "n_observations": obs.get("n_observations"),
                    "n_runs": obs.get("n_runs"),
                    "promote_rate": obs.get("promote_rate"),
                    "ship_rate": obs.get("ship_rate"),
                    "robust_score_shrunk": obs.get("robust_score_shrunk"),
                    "memory_role": obs.get("memory_role"),
                    "neighbor_count": len(neighbors),
                    "neighbor_score_mean": neighbor_score,
                    "neighbor_promote_rate": mean(neighbor_promote),
                    "gap_priority": gap_priority,
                    "nearest_evidence": "|".join(
                        sorted({str(r.get("typology_id")) for r in neighbors if r.get("typology_id")})[:12]
                    ),
                })
    return sorted(
        rows,
        key=lambda r: (int(r.get("tried") or 0), f(r.get("gap_priority")) or f(r.get("robust_score_shrunk")) or -999),
        reverse=True,
    )


def typology_vector_field(typology_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in typology_rows:
        for dst in typology_rows:
            if src is dst:
                continue
            same = [
                src.get("skill") == dst.get("skill"),
                src.get("family") == dst.get("family"),
                src.get("transform") == dst.get("transform"),
            ]
            if sum(same) != 2:
                continue
            if not same[0]:
                axis = "skill"
                src_value, dst_value = src.get("skill"), dst.get("skill")
            elif not same[1]:
                axis = "family"
                src_value, dst_value = src.get("family"), dst.get("family")
            else:
                axis = "transform"
                src_value, dst_value = src.get("transform"), dst.get("transform")
            src_score = f(src.get("robust_score_shrunk"))
            dst_score = f(dst.get("robust_score_shrunk"))
            if src_score is None or dst_score is None:
                continue
            delta = dst_score - src_score
            rows.append({
                "vector_id": stable_id("typology_vector", src.get("typology_id"), dst.get("typology_id")),
                "src_typology_id": src.get("typology_id"),
                "dst_typology_id": dst.get("typology_id"),
                "changed_axis": axis,
                "src_axis_value": src_value,
                "dst_axis_value": dst_value,
                "skill": src.get("skill") if axis != "skill" else None,
                "family": src.get("family") if axis != "family" else None,
                "transform": src.get("transform") if axis != "transform" else None,
                "src_score": src_score,
                "dst_score": dst_score,
                "delta_score": delta,
                "magnitude": abs(delta),
                "direction": "improves" if delta > 0 else "degrades" if delta < 0 else "flat",
                "src_role": src.get("memory_role"),
                "dst_role": dst.get("memory_role"),
            })
    return sorted(rows, key=lambda r: abs(f(r.get("delta_score")) or 0.0), reverse=True)


def projection_method(row: dict[str, Any]) -> tuple[str, str, str]:
    skill = str(row.get("skill") or "").lower()
    family = str(row.get("family") or "").lower()
    transform = str(row.get("transform") or "").lower()
    key = str(row.get("key") or "").lower()
    text = "|".join([skill, family, transform, key])
    if "pca" in text:
        return "pca", "transform", "pca"
    if "svd" in text:
        return "svd", "transform", "svd"
    if "pls" in text:
        return "pls", "transform", "pls"
    if "decor" in text:
        return "decorrelation", "family", row.get("family") or "decor"
    if "medoid" in text:
        return "medoid_compression", "family", row.get("family") or "medoid"
    if "top" in family:
        return "topk_selection", "family", row.get("family") or "top"
    if "lastn" in family:
        return "recency_projection", "family", row.get("family") or "lastN"
    if "market" in family or "beacon" in family:
        return "semantic_route_projection", "family", row.get("family") or family
    if "quantize" in transform:
        return "quantized_projection", "transform", row.get("transform") or "quantize"
    if "rank" in transform:
        return "rank_projection", "transform", row.get("transform") or "rank"
    if "sign" in transform:
        return "sign_projection", "transform", row.get("transform") or "sign"
    if "fold" in transform:
        return "folded_projection", "transform", row.get("transform") or "fold"
    return "raw_or_identity", "transform", row.get("transform") or "identity"


def projection_memory_matrix(path_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        groups[projection_method(row)].append(row)
    prior = mean([f(row.get("robust_memory_score")) for row in path_rows]) or 0.0
    rows: list[dict[str, Any]] = []
    for (method, source_axis, source_value), members in groups.items():
        robust = [f(row.get("robust_memory_score")) for row in members]
        decay = [f(row.get("decay")) for row in members]
        width = [f(row.get("width")) for row in members]
        world = [f(row.get("world_floor")) for row in members]
        oof = [f(row.get("oof_corr")) for row in members]
        wf = [f(row.get("wf_corr")) for row in members]
        promoted = [f(row.get("promoted")) for row in members]
        shipped = [f(row.get("shipped")) for row in members]
        shrunk = shrink_mean(robust, prior)
        downstream_gain = None if shrunk is None else shrunk - prior
        decay_mean = mean(decay)
        world_mean = mean(world)
        width_mean = mean(width)
        hazard = 0.0
        if decay_mean is not None:
            hazard += max(0.0, decay_mean)
        if world_mean is not None:
            hazard += max(0.0, -world_mean)
        grokking = mean([
            None if downstream_gain is None else max(0.0, downstream_gain),
            width_mean,
            None if decay_mean is None else max(0.0, -decay_mean),
            mean(promoted),
        ])
        if downstream_gain is not None and downstream_gain > 0.015 and hazard < 0.02:
            recommendation = "warm_prior"
        elif downstream_gain is not None and downstream_gain > 0.0:
            recommendation = "retest_or_specialize"
        elif hazard > 0.03:
            recommendation = "hazard_or_anti_prior"
        else:
            recommendation = "low_priority"
        rows.append({
            "projection_id": stable_id("projection", method, source_axis, source_value),
            "method": method,
            "source_axis": source_axis,
            "source_value": source_value,
            "n_observations": len(members),
            "n_runs": len({row.get("run") for row in members}),
            "robust_score_mean": mean(robust),
            "robust_score_shrunk": shrunk,
            "downstream_path_gain": downstream_gain,
            "decay_mean": decay_mean,
            "width_mean": width_mean,
            "world_floor_mean": world_mean,
            "oof_corr_mean": mean(oof),
            "wf_corr_mean": mean(wf),
            "promotion_rate": mean(promoted),
            "shipping_rate": mean(shipped),
            "grokking_potential": grokking,
            "hazard_bias": hazard,
            "recommendation": recommendation,
            "example_paths": "|".join(
                str(row.get("path_id")) for row in sorted(
                    members, key=lambda r: f(r.get("robust_memory_score")) or -999, reverse=True
                )[:12]
            ),
        })
    return sorted(rows, key=lambda r: f(r.get("grokking_potential")) or -999, reverse=True)


def collinearity_memory_matrix(fleet_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs(fleet_dir):
        out = run_dir / "output"
        for row in read_csv(out / "feature_topology_report.csv"):
            size = safe_float(row, "size")
            coherence = safe_float(row, "topology_coherence")
            signal = safe_float(row, "mean_abs_corr_y")
            max_signal = safe_float(row, "max_abs_corr_y")
            consensus = safe_float(row, "signal_consensus")
            shift = safe_float(row, "mean_train_test_shift")
            redundancy = 0.0
            if size is not None and size > 1:
                redundancy += math.log1p(size)
            if coherence is not None:
                redundancy *= max(0.0, coherence)
            compression_bias = mean([
                signal,
                max_signal,
                consensus,
                None if shift is None else 1.0 - shift,
                redundancy,
            ])
            shift_hazard = int(shift > 0.30) if shift is not None else None
            rows.append({
                "collinearity_id": stable_id(run_dir.name, "collinearity", row.get("community")),
                "run": run_dir.name,
                "run_id": stable_id("run", run_dir.name),
                "community": row.get("community"),
                "community_size": size,
                "topology_coherence": coherence,
                "mean_abs_corr_y": signal,
                "max_abs_corr_y": max_signal,
                "signal_consensus": consensus,
                "mean_train_test_shift": shift,
                "redundancy_score": redundancy,
                "compression_bias": compression_bias,
                "shift_hazard": shift_hazard,
                "recommendation": (
                    "compress_or_project" if redundancy > 0.15 and not shift_hazard else
                    "shift_hazard" if shift_hazard else
                    "single_feature_or_low_redundancy"
                ),
                "artifact_source": str(out / "feature_topology_report.csv"),
            })
    return sorted(rows, key=lambda r: f(r.get("compression_bias")) or -999, reverse=True)


def surface_action(
    *,
    local_signal: float | None,
    disagreement: float | None,
    false_agreement: float | None,
    false_disagreement: float | None,
    overfit: float | None,
    default: str,
) -> str:
    signal = local_signal or 0.0
    dis = disagreement or 0.0
    fa = false_agreement or 0.0
    fd = false_disagreement or 0.0
    ov = overfit or 0.0
    if signal > 0.06 and dis > 0.25 and ov < 0.60:
        return "carve_signal_then_retest_global_surface"
    if fa > 0.20:
        return "challenge_false_agreement_with_adversarial_split"
    if fd > 0.12:
        return "revive_or_recombine_suspect_signal"
    if ov > 0.60:
        return "quarantine_but_keep_reversal_path"
    return default


def surface_surgery_matrix(
    feature_rows: list[dict[str, Any]],
    collinearity_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in feature_rows:
        signal = abs(f(row.get("signal")) or 0.0)
        stability = f(row.get("stability_or_coherence"))
        shift = f(row.get("shift_or_noise"))
        if stability is None and shift is not None:
            stability = 1.0 - shift
        hazard = f(row.get("hazard")) or 0.0
        disagreement = clip01((shift or 0.0) + 0.25 * hazard)
        agreement = clip01(mean([signal, stability]) or 0.0)
        false_agreement = clip01((agreement or 0.0) * (disagreement or 0.0))
        false_disagreement = clip01(signal * (disagreement or 0.0) * (1.0 - min(1.0, hazard)))
        overfit = clip01((disagreement or 0.0) + 0.25 * hazard)
        rearrange = clip01(mean([signal, stability, false_disagreement]) or 0.0)
        grok = clip01(mean([signal, false_disagreement, 1.0 - (overfit or 0.0), rearrange]) or 0.0)
        action = surface_action(
            local_signal=signal,
            disagreement=disagreement,
            false_agreement=false_agreement,
            false_disagreement=false_disagreement,
            overfit=overfit,
            default="keep_as_local_surface_observation",
        )
        rows.append({
            "surface_id": stable_id("surface", row.get("feature_memory_id")),
            "source_kind": row.get("record_type"),
            "source_id": row.get("feature_memory_id"),
            "run": row.get("run"),
            "coordinate_kind": row.get("coordinate_kind"),
            "coordinate": row.get("coordinate"),
            "secondary_coordinate": row.get("secondary_coordinate"),
            "local_signal": signal,
            "global_stability": stability,
            "agreement_score": agreement,
            "disagreement_score": disagreement,
            "uncertainty": clip01(mean([disagreement, None if agreement is None else 1.0 - agreement]) or 0.0),
            "false_agreement_risk": false_agreement,
            "false_disagreement_risk": false_disagreement,
            "overfit_risk": overfit,
            "rearrangement_gain": rearrange,
            "grokking_priority": grok,
            "suggested_surgery": action,
            "reversal_plan": "retain source feature/region and compare before/after surface edit",
            "artifact_source": row.get("artifact_source"),
        })

    for row in collinearity_rows:
        signal = f(row.get("mean_abs_corr_y")) or 0.0
        max_signal = f(row.get("max_abs_corr_y")) or signal
        coherence = f(row.get("topology_coherence"))
        consensus = f(row.get("signal_consensus"))
        shift = f(row.get("mean_train_test_shift")) or 0.0
        compression = f(row.get("compression_bias")) or 0.0
        agreement = clip01(mean([coherence, consensus]) or 0.0)
        disagreement = clip01(shift)
        false_agreement = clip01((agreement or 0.0) * disagreement)
        false_disagreement = clip01(max(0.0, max_signal - signal) + signal * disagreement)
        overfit = clip01(disagreement + 0.10 * max(0.0, (f(row.get("community_size")) or 0.0) - 50.0) / 50.0)
        grok = clip01(mean([compression, false_disagreement, signal, 1.0 - overfit]) or 0.0)
        rows.append({
            "surface_id": stable_id("surface", row.get("collinearity_id")),
            "source_kind": "collinearity_community",
            "source_id": row.get("collinearity_id"),
            "run": row.get("run"),
            "coordinate_kind": "feature_community",
            "coordinate": row.get("community"),
            "secondary_coordinate": None,
            "local_signal": signal,
            "global_stability": None if shift is None else 1.0 - shift,
            "agreement_score": agreement,
            "disagreement_score": disagreement,
            "uncertainty": clip01(mean([disagreement, None if agreement is None else 1.0 - agreement]) or 0.0),
            "false_agreement_risk": false_agreement,
            "false_disagreement_risk": false_disagreement,
            "overfit_risk": overfit,
            "rearrangement_gain": compression,
            "grokking_priority": grok,
            "suggested_surgery": surface_action(
                local_signal=signal,
                disagreement=disagreement,
                false_agreement=false_agreement,
                false_disagreement=false_disagreement,
                overfit=overfit,
                default=row.get("recommendation") or "compress_or_project",
            ),
            "reversal_plan": "keep original community features; compare compressed, masked, and restored variants",
            "artifact_source": row.get("artifact_source"),
        })

    for row in projection_rows:
        gain = f(row.get("downstream_path_gain")) or 0.0
        hazard = f(row.get("hazard_bias")) or 0.0
        promo = f(row.get("promotion_rate"))
        ship = f(row.get("shipping_rate"))
        grok = f(row.get("grokking_potential")) or 0.0
        signal = max(0.0, gain)
        stability = clip01(1.0 - hazard)
        agreement = clip01(mean([signal, stability, promo, ship]) or 0.0)
        disagreement = clip01(hazard)
        false_agreement = clip01((agreement or 0.0) * disagreement)
        false_disagreement = clip01(max(0.0, -gain) * (promo or 0.0))
        rows.append({
            "surface_id": stable_id("surface", row.get("projection_id")),
            "source_kind": "projection_operator_family",
            "source_id": row.get("projection_id"),
            "run": None,
            "coordinate_kind": row.get("source_axis"),
            "coordinate": row.get("source_value"),
            "secondary_coordinate": row.get("method"),
            "local_signal": signal,
            "global_stability": stability,
            "agreement_score": agreement,
            "disagreement_score": disagreement,
            "uncertainty": clip01(mean([disagreement, 1.0 - (agreement or 0.0)]) or 0.0),
            "false_agreement_risk": false_agreement,
            "false_disagreement_risk": false_disagreement,
            "overfit_risk": disagreement,
            "rearrangement_gain": max(0.0, gain),
            "grokking_priority": grok,
            "suggested_surgery": row.get("recommendation"),
            "reversal_plan": "branch before projection and compare no-projection, lower-rank, and alternate-fold variants",
            "artifact_source": "projection_memory_matrix.csv",
        })

    for row in operation_rows:
        private = f(row.get("external_private"))
        public = f(row.get("external_public"))
        strength = f(row.get("operation_strength")) or 0.0
        score_signal = private if private is not None else min(1.0, strength / 1.1)
        gap = abs(private - public) if private is not None and public is not None else None
        novelty = f(row.get("bias_novelty_from_champion")) or 0.0
        conflict = f(row.get("bias_conflict_tail")) or 0.0
        route_gain = f(row.get("weight_route_information_gain")) or 0.0
        disagreement = clip01((gap or 0.0) + conflict + max(0.0, -route_gain))
        stability = None if gap is None else clip01(1.0 - gap)
        agreement = clip01(mean([score_signal, stability, max(0.0, route_gain)]) or 0.0)
        false_agreement = clip01((agreement or 0.0) * (disagreement or 0.0))
        false_disagreement = clip01(max(0.0, route_gain) + novelty)
        overfit = clip01(disagreement)
        grok = clip01(mean([score_signal, false_disagreement, 1.0 - overfit]) or 0.0)
        rows.append({
            "surface_id": stable_id("surface", row.get("operation_id")),
            "source_kind": f"operation:{row.get('operation_type')}",
            "source_id": row.get("operation_id"),
            "run": row.get("source_run"),
            "coordinate_kind": "operation",
            "coordinate": row.get("operation_name"),
            "secondary_coordinate": operation_operator_type(row),
            "local_signal": score_signal,
            "global_stability": stability,
            "agreement_score": agreement,
            "disagreement_score": disagreement,
            "uncertainty": clip01(mean([disagreement, 1.0 - (agreement or 0.0)]) or 0.0),
            "false_agreement_risk": false_agreement,
            "false_disagreement_risk": false_disagreement,
            "overfit_risk": overfit,
            "rearrangement_gain": max(0.0, route_gain) + novelty,
            "grokking_priority": grok,
            "suggested_surgery": challenge_plan(row),
            "reversal_plan": "branch before operation; compare parent, sibling operation, and ablation",
            "artifact_source": row.get("artifact_source"),
        })

    return sorted(rows, key=lambda r: f(r.get("grokking_priority")) or -999, reverse=True)


def move_status(
    local_effect: float | None,
    global_effect: float | None,
    stress: float | None,
    overfit: float | None,
    branch_priority: float | None,
) -> str:
    local = local_effect or 0.0
    global_ = global_effect or 0.0
    stress_v = stress or 0.0
    overfit_v = overfit or 0.0
    branch = branch_priority or 0.0
    if stress_v > 0.55:
        return "foundation_rethink_signal"
    if local > 0.55 and global_ < 0.15:
        return "route_limited"
    if branch > 0.55 and overfit_v < 0.35:
        return "kept_as_branch"
    if global_ > 0.40 and overfit_v < 0.25 and stress_v < 0.30:
        return "accepted_candidate_needs_retest"
    if overfit_v > 0.60:
        return "rejected_but_remembered"
    return "speculative_branch"


def impact_field_matrix(surface_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in surface_rows:
        local = f(row.get("local_signal")) or 0.0
        stability = f(row.get("global_stability"))
        agreement = f(row.get("agreement_score")) or 0.0
        disagreement = f(row.get("disagreement_score")) or 0.0
        uncertainty = f(row.get("uncertainty")) or 0.0
        false_agree = f(row.get("false_agreement_risk")) or 0.0
        false_disagree = f(row.get("false_disagreement_risk")) or 0.0
        overfit = f(row.get("overfit_risk")) or 0.0
        rearrange = f(row.get("rearrangement_gain")) or 0.0
        grok = f(row.get("grokking_priority")) or 0.0
        stability_effect = (stability if stability is not None else 1.0 - overfit)
        global_effect = clip01(mean([stability_effect, agreement, 1.0 - disagreement, 1.0 - overfit]) or 0.0)
        ripple = clip01(mean([disagreement, uncertainty, false_agree, overfit]) or 0.0)
        side_effect = clip01(mean([false_agree, overfit, uncertainty]) or 0.0)
        stress_delta = clip01(mean([ripple, side_effect, max(0.0, disagreement - agreement)]) or 0.0)
        move_quality = clip01(mean([local, global_effect, rearrange, 1.0 - overfit, 1.0 - false_agree]) or 0.0)
        branch_priority = clip01(mean([grok, local, false_disagree, 1.0 - false_agree, 1.0 - overfit]) or 0.0)
        rows.append({
            "impact_id": stable_id("impact", row.get("surface_id")),
            "surface_id": row.get("surface_id"),
            "source_kind": row.get("source_kind"),
            "source_id": row.get("source_id"),
            "run": row.get("run"),
            "coordinate_kind": row.get("coordinate_kind"),
            "coordinate": row.get("coordinate"),
            "secondary_coordinate": row.get("secondary_coordinate"),
            "operator_move": row.get("suggested_surgery"),
            "local_effect": local,
            "global_effect": global_effect,
            "stability_effect": stability_effect,
            "agreement_delta": agreement,
            "disagreement_delta": disagreement,
            "uncertainty_delta": uncertainty,
            "false_agreement_risk": false_agree,
            "false_disagreement_risk": false_disagree,
            "overfit_risk": overfit,
            "ripple_radius": ripple,
            "side_effect_load": side_effect,
            "foundation_stress_delta": stress_delta,
            "move_quality": move_quality,
            "branch_priority": branch_priority,
            "move_status": move_status(local, global_effect, stress_delta, overfit, branch_priority),
            "followup_action": row.get("suggested_surgery"),
            "reversal_plan": row.get("reversal_plan"),
            "artifact_source": row.get("artifact_source"),
        })
    return sorted(rows, key=lambda r: f(r.get("branch_priority")) or -999, reverse=True)


def foundation_stress_matrix(
    surface_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in surface_rows:
        run = str(row.get("run") or "global")
        source_group = str(row.get("source_kind") or "unknown").split(":", 1)[0]
        groups[(run, source_group)].append(row)
        groups[("global", source_group)].append(row)
        groups[("global", "all")].append(row)

    impact_by_surface = {row.get("surface_id"): row for row in impact_rows}
    out: list[dict[str, Any]] = []
    for (run, source_group), rows in groups.items():
        if not rows:
            continue
        uncertainty = [f(r.get("uncertainty")) for r in rows]
        false_agree = [f(r.get("false_agreement_risk")) for r in rows]
        false_disagree = [f(r.get("false_disagreement_risk")) for r in rows]
        overfit = [f(r.get("overfit_risk")) for r in rows]
        disagreement = [f(r.get("disagreement_score")) for r in rows]
        branch = [f(r.get("grokking_priority")) for r in rows]
        high_risk = [
            1.0 if max(
                f(r.get("uncertainty")) or 0.0,
                f(r.get("false_agreement_risk")) or 0.0,
                f(r.get("overfit_risk")) or 0.0,
                f(r.get("disagreement_score")) or 0.0,
            ) > 0.35 else 0.0
            for r in rows
        ]
        impact_stress = [
            f(impact_by_surface.get(r.get("surface_id"), {}).get("foundation_stress_delta"))
            for r in rows
        ]
        stress = clip01(mean([
            mean(uncertainty),
            mean(false_agree),
            mean(overfit),
            mean(disagreement),
            mean(impact_stress),
            mean(high_risk),
        ]) or 0.0)
        rethink = clip01(mean([stress, mean(branch), mean(false_disagree)]) or 0.0)
        top_rows = sorted(
            rows,
            key=lambda r: max(
                f(r.get("uncertainty")) or 0.0,
                f(r.get("false_agreement_risk")) or 0.0,
                f(r.get("overfit_risk")) or 0.0,
                f(r.get("grokking_priority")) or 0.0,
            ),
            reverse=True,
        )[:12]
        if stress is not None and stress > 0.45:
            recommendation = "do_not_stack_more_local_fixes; branch_to_alternate_foundation"
        elif rethink is not None and rethink > 0.35:
            recommendation = "keep_branch_pressure_high; require independent retest"
        else:
            recommendation = "continue_local_surface_surgery_with_reversal_paths"
        out.append({
            "foundation_stress_id": stable_id("foundation_stress", run, source_group),
            "run": run,
            "source_group": source_group,
            "n_surface_records": len(rows),
            "stress_score": stress,
            "mean_uncertainty": mean(uncertainty),
            "mean_false_agreement": mean(false_agree),
            "mean_false_disagreement": mean(false_disagree),
            "mean_overfit": mean(overfit),
            "mean_disagreement": mean(disagreement),
            "high_risk_fraction": mean(high_risk),
            "branch_pressure": mean(branch),
            "foundation_rethink_priority": rethink,
            "recommendation": recommendation,
            "top_stress_sources": "|".join(str(r.get("surface_id")) for r in top_rows),
        })
    return sorted(out, key=lambda r: f(r.get("foundation_rethink_priority")) or -999, reverse=True)


def route_strength_matrix(
    operation_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    impact_by_source = {row.get("source_id"): row for row in impact_rows}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in operation_rows:
        action_type = str(row.get("operation_type") or "unknown")
        operator = operation_operator_type(row)
        terrain = str(row.get("source_run") or row.get("operation_batch") or "global")
        groups[(terrain, action_type, operator)].append(row)
        groups[("global", action_type, operator)].append(row)

    out: list[dict[str, Any]] = []
    for (terrain, action_type, operator), rows in groups.items():
        scores = []
        risks = []
        branches = []
        novelty = []
        for row in rows:
            private = f(row.get("external_private"))
            strength = f(row.get("operation_strength"))
            score = private if private is not None else strength
            scores.append(score)
            impact = impact_by_source.get(row.get("operation_id"), {})
            risks.append(f(impact.get("overfit_risk")) or f(row.get("bias_decay")))
            branches.append(f(impact.get("branch_priority")))
            novelty.append(f(row.get("bias_novelty_from_champion")) or (1.0 - (f(row.get("bias_corr_prev_best")) or 1.0)))
        expected = mean(scores)
        uncertainty = stdev(scores) or 0.0
        risk = mean(risks) or 0.0
        complementarity = clip01(mean(novelty) or 0.0)
        success_probability = clip01(0.5 + 5.0 * (expected or 0.0) - risk)
        transferability = clip01(1.0 - uncertainty - risk)
        fragility = clip01(mean([uncertainty, risk]) or 0.0)
        branch_value = clip01(mean([mean(branches), expected, complementarity, 1.0 - fragility]) or 0.0)
        if branch_value > 0.55 and fragility < 0.35:
            status = "preferred_reusable_move"
        elif branch_value > 0.45:
            status = "promising_but_needs_retest"
        elif fragility > 0.50:
            status = "fragile_or_context_limited"
        else:
            status = "background_prior"
        out.append({
            "route_strength_id": stable_id("route_strength", terrain, action_type, operator),
            "terrain_signature": terrain,
            "action_type": action_type,
            "operator_type": operator,
            "n_observations": len(rows),
            "expected_gain": expected,
            "uncertainty": uncertainty,
            "success_probability": success_probability,
            "overfit_risk": risk,
            "transferability": transferability,
            "fragility": fragility,
            "complementarity": complementarity,
            "branch_value": branch_value,
            "status": status,
            "evidence_operations": "|".join(str(r.get("operation_id")) for r in rows[:12]),
        })
    return sorted(out, key=lambda r: f(r.get("branch_value")) or -999, reverse=True)


def validation_budget_ledger(
    run_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Estimate validation-world reuse pressure.

    This is intentionally conservative. It does not pretend to know every fold
    used by the engine; it records visible selection surfaces so the next run
    can discount worlds that are being optimized repeatedly.
    """

    eligible_scores = [row for row in score_rows if tg.is_weight_eligible(row)]
    external_refs = [row for row in score_rows if not tg.is_weight_eligible(row)]
    route_carves = [row for row in operation_rows if row.get("operation_type") == "route_carve"]
    blends = [row for row in operation_rows if row.get("operation_type") == "blend"]
    path_transforms = [row for row in operation_rows if row.get("operation_type") == "path_transform"]
    robust_paths = [row for row in path_rows if f(row.get("robust_memory_score")) is not None]
    promoted_paths = [row for row in path_rows if int(row.get("promoted") or 0)]
    shipped_paths = [row for row in path_rows if int(row.get("shipped") or 0)]

    specs = [
        {
            "validation_world": "external_private_score_context",
            "world_kind": "external_score",
            "rows": eligible_scores,
            "selection_use_count": len(eligible_scores),
            "reporting_use_count": len(score_rows),
            "guard": "use as telemetry only; never as training label",
        },
        {
            "validation_world": "external_public_score_context",
            "world_kind": "external_score",
            "rows": eligible_scores,
            "selection_use_count": max(0, len(eligible_scores) // 2),
            "reporting_use_count": len(score_rows),
            "guard": "watch private/public gap; public-positive private-negative is a trap signal",
        },
        {
            "validation_world": "external_reference_only",
            "world_kind": "excluded_reference",
            "rows": external_refs,
            "selection_use_count": 0,
            "reporting_use_count": len(external_refs),
            "guard": "excluded from weight learning; comparative reference only",
        },
        {
            "validation_world": "route_carve_review_surface",
            "world_kind": "operation_review",
            "rows": route_carves,
            "selection_use_count": len(route_carves),
            "reporting_use_count": len(route_carves),
            "guard": "route-carves need parent/sibling ablation before main-world promotion",
        },
        {
            "validation_world": "blend_review_surface",
            "world_kind": "operation_review",
            "rows": blends,
            "selection_use_count": len(blends),
            "reporting_use_count": len(blends),
            "guard": "blend gains need leave-one-route-out and correlation-crowding checks",
        },
        {
            "validation_world": "path_transform_review_surface",
            "world_kind": "operation_review",
            "rows": path_transforms,
            "selection_use_count": len(path_transforms),
            "reporting_use_count": len(path_transforms),
            "guard": "transform gains need raw-parent and alternate-transform comparison",
        },
        {
            "validation_world": "robust_memory_score_surface",
            "world_kind": "internal_robust_oos",
            "rows": robust_paths,
            "selection_use_count": len(promoted_paths) + len(shipped_paths),
            "reporting_use_count": len(robust_paths),
            "guard": "discount if many candidates are searched against the same robust menu",
        },
        {
            "validation_world": "runtime_artifact_surface",
            "world_kind": "run_artifacts",
            "rows": run_rows,
            "selection_use_count": len(run_rows),
            "reporting_use_count": len(run_rows),
            "guard": "use for provenance and budget priors, not direct promotion",
        },
    ]

    rows: list[dict[str, Any]] = []
    total_ops = max(1, len(operation_rows))
    for spec in specs:
        count = len(spec["rows"])
        selection = int(spec["selection_use_count"])
        reporting = int(spec["reporting_use_count"])
        reuse_pressure = clip01(selection / 25.0) or 0.0
        redundancy_pressure = clip01(reporting / max(10.0, total_ops)) or 0.0
        trust_discount = clip01(1.0 / (1.0 + reuse_pressure + 0.5 * redundancy_pressure)) or 0.0
        remaining = clip01(1.0 - reuse_pressure) or 0.0
        if spec["world_kind"] == "excluded_reference":
            policy = "reference_only_never_weight"
        elif reuse_pressure > 0.70:
            policy = "mutate_or_hold_back; do_not_treat_as_independent"
        elif reuse_pressure > 0.35:
            policy = "discount_for_selection; require independent world"
        elif count == 0:
            policy = "inactive"
        else:
            policy = "usable_with_standard_guard"
        rows.append({
            "validation_world_id": stable_id("validation_world", spec["validation_world"]),
            "validation_world": spec["validation_world"],
            "world_kind": spec["world_kind"],
            "candidate_count_seen": count,
            "selection_use_count": selection,
            "reporting_use_count": reporting,
            "reuse_pressure": reuse_pressure,
            "redundancy_pressure": redundancy_pressure,
            "trust_discount": trust_discount,
            "remaining_trust_budget": remaining,
            "policy": policy,
            "guard": spec["guard"],
        })
    return rows


def evidence_grade(score: float, drift: float, support: float) -> str:
    if score >= 0.72 and drift <= 0.22 and support >= 0.50:
        return "A_supported_candidate"
    if score >= 0.58 and drift <= 0.35:
        return "B_branch_with_retest"
    if score >= 0.42 and drift <= 0.50:
        return "C_route_limited_or_quarantine"
    return "D_reject_or_hazard_memory"


def evidence_gate_matrix(
    impact_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    route_strength_rows: list[dict[str, Any]],
    contradiction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    op_by_id = {row.get("operation_id"): row for row in operation_rows}
    route_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in route_strength_rows:
        route_by_operator[str(row.get("operator_type"))].append(row)
    contradictions_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contradiction_rows:
        contradictions_by_scope[str(row.get("scope") or "")].append(row)

    rows: list[dict[str, Any]] = []
    for row in impact_rows:
        source_id = row.get("source_id")
        op = op_by_id.get(source_id, {})
        operator = operation_operator_type(op) if op else str(row.get("operator_move") or "unknown")
        route_support = route_by_operator.get(operator, [])
        local = f(row.get("local_effect")) or 0.0
        global_ = f(row.get("global_effect")) or 0.0
        false_agree = f(row.get("false_agreement_risk")) or 0.0
        false_disagree = f(row.get("false_disagreement_risk")) or 0.0
        overfit = f(row.get("overfit_risk")) or 0.0
        stress = f(row.get("foundation_stress_delta")) or 0.0
        branch = f(row.get("branch_priority")) or 0.0
        private = f(op.get("external_private"))
        public = f(op.get("external_public"))
        has_external = 1.0 if private is not None or public is not None else 0.0
        public_private_gap = abs(private - public) if private is not None and public is not None else 0.0
        support_count = len(route_support) + int(has_external)
        independent_support = clip01(0.20 * support_count + 0.30 * has_external) or 0.0
        contradiction_count = len(contradictions_by_scope.get(str(row.get("coordinate") or ""), []))
        worst_world_proxy = clip01(min(global_, 1.0 - overfit, 1.0 - stress, 1.0 - public_private_gap)) or 0.0
        drift = clip01(mean([false_agree, overfit, stress, public_private_gap, min(1.0, contradiction_count / 5.0)]) or 0.0) or 0.0
        evidence = clip01(mean([
            local,
            global_,
            worst_world_proxy,
            branch * 0.75,
            independent_support,
            1.0 - drift,
        ]) or 0.0) or 0.0
        hard_hazard = stress > 0.55 or false_agree > 0.55 or overfit > 0.60
        grade = "D_reject_or_hazard_memory" if hard_hazard else evidence_grade(evidence, drift, independent_support)
        if not has_external and grade == "A_supported_candidate":
            grade = "B_branch_with_retest"
        promotion_allowed = int(
            has_external
            and grade == "A_supported_candidate"
            and stress < 0.25
            and false_agree < 0.18
        )
        branch_allowed = int(
            grade in {"A_supported_candidate", "B_branch_with_retest"}
            and drift < 0.45
            and not hard_hazard
        )
        grokking_allowed = int(branch_allowed and branch > 0.55 and overfit < 0.35 and false_agree < 0.22)
        if promotion_allowed:
            decision = "candidate_retest_for_main_world"
        elif branch_allowed:
            decision = "branch_only_requires_independent_retest"
        elif grade.startswith("C"):
            decision = "route_limited_or_quarantine"
        else:
            decision = "reject_but_remember_as_hazard"
        rows.append({
            "evidence_gate_id": stable_id("evidence_gate", row.get("impact_id")),
            "impact_id": row.get("impact_id"),
            "surface_id": row.get("surface_id"),
            "source_kind": row.get("source_kind"),
            "source_id": source_id,
            "coordinate": row.get("coordinate"),
            "operator_move": row.get("operator_move"),
            "local_effect": local,
            "global_effect": global_,
            "worst_world_proxy": worst_world_proxy,
            "support_count": support_count,
            "independent_support": independent_support,
            "false_agreement_risk": false_agree,
            "false_disagreement_risk": false_disagree,
            "overfit_risk": overfit,
            "foundation_stress_delta": stress,
            "branch_priority": branch,
            "evidence_score": evidence,
            "drift_risk": drift,
            "evidence_grade": grade,
            "promotion_allowed": promotion_allowed,
            "branch_allowed": branch_allowed,
            "grokking_allowed": grokking_allowed,
            "decision": decision,
            "required_next_evidence": (
                "independent seed/split plus parent/sibling ablation before promotion"
                if not promotion_allowed else
                "promotion still requires shipping court and sealed/forward confirmation"
            ),
            "artifact_source": row.get("artifact_source"),
        })
    return sorted(rows, key=lambda r: f(r.get("evidence_score")) or -999, reverse=True)


def proof_carrying_paths(
    path_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_source = {row.get("source_id"): row for row in evidence_rows}
    proofs: list[dict[str, Any]] = []
    for row in sorted(path_rows, key=lambda r: f(r.get("robust_memory_score")) or -999, reverse=True)[:80]:
        robust = f(row.get("robust_memory_score"))
        if robust is None:
            continue
        proof = {
            "proof_id": stable_id("proof", "path", row.get("path_id")),
            "candidate_id": row.get("path_id"),
            "candidate_kind": "path",
            "claim": "stable_path_candidate",
            "evidence": {
                "skill": row.get("skill"),
                "family": row.get("family"),
                "transform": row.get("transform"),
                "oof_corr": row.get("oof_corr"),
                "wf_corr": row.get("wf_corr"),
                "robust_memory_score": robust,
                "world_floor": row.get("world_floor"),
                "world_frac_positive": row.get("world_frac_positive"),
                "promoted": row.get("promoted"),
                "shipped": row.get("shipped"),
            },
            "risks": {
                "decay": row.get("decay"),
                "overfit_ratio": row.get("overfit_ratio"),
                "complexity": row.get("complexity"),
                "roughness": row.get("roughness"),
                "crowd_load": row.get("crowd_load"),
            },
            "scope": {
                "run": row.get("run"),
                "stage": row.get("stage"),
                "artifact_source": row.get("artifact_source"),
            },
            "decision": (
                "candidate_retest_for_main_world" if robust > 0.08
                else "warm_prior_or_branch_only"
            ),
            "must_not_ship_until": [
                "independent split/seed confirmation",
                "worst-world floor checked",
                "overfit and complexity gates checked",
            ],
        }
        proofs.append(proof)

    for row in sorted(operation_rows, key=lambda r: f(r.get("external_private")) or f(r.get("operation_strength")) or -999, reverse=True)[:120]:
        evidence = evidence_by_source.get(row.get("operation_id"), {})
        proof = {
            "proof_id": stable_id("proof", "operation", row.get("operation_id")),
            "candidate_id": row.get("operation_id"),
            "candidate_kind": row.get("operation_type"),
            "claim": "operation_may_improve_surface",
            "evidence": {
                "operation_name": row.get("operation_name"),
                "operator_type": operation_operator_type(row),
                "operation_strength": row.get("operation_strength"),
                "external_private": row.get("external_private"),
                "external_public": row.get("external_public"),
                "route_information_gain": row.get("weight_route_information_gain"),
                "evidence_grade": evidence.get("evidence_grade"),
                "evidence_score": evidence.get("evidence_score"),
                "independent_support": evidence.get("independent_support"),
            },
            "risks": {
                "false_agreement_risk": evidence.get("false_agreement_risk"),
                "overfit_risk": evidence.get("overfit_risk"),
                "foundation_stress_delta": evidence.get("foundation_stress_delta"),
                "drift_risk": evidence.get("drift_risk"),
            },
            "scope": {
                "source_run": row.get("source_run"),
                "operation_batch": row.get("operation_batch"),
                "artifact_source": row.get("artifact_source"),
            },
            "decision": evidence.get("decision") or "unscored_requires_retest",
            "must_not_ship_until": [
                "parent/sibling ablation",
                "independent seed or validation world",
                "private-public or forward gap checked",
                "false agreement and foundation stress below guardrails",
            ],
        }
        proofs.append(proof)
    return proofs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def contradiction_graph(
    surface_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in surface_rows[:600]:
        agreement = f(row.get("agreement_score")) or 0.0
        false_agree = f(row.get("false_agreement_risk")) or 0.0
        false_disagree = f(row.get("false_disagreement_risk")) or 0.0
        source = row.get("source_id")
        if agreement > 0.25 and false_agree > 0.02:
            rows.append({
                "claim_edge_id": stable_id("claim_edge", source, "false_agreement"),
                "src_claim": f"{source}:local_agreement_is_real",
                "relation": "weakened_by",
                "dst_claim": f"{source}:false_agreement_risk",
                "confidence": false_agree,
                "scope": row.get("coordinate"),
                "recommended_test": "adversarial split, bootstrap, and parent-surface comparison",
                "evidence": row.get("artifact_source"),
            })
        if false_disagree > 0.05:
            rows.append({
                "claim_edge_id": stable_id("claim_edge", source, "false_disagreement"),
                "src_claim": f"{source}:signal_is_noise_or_should_be_quarantined",
                "relation": "contradicted_by",
                "dst_claim": f"{source}:false_disagreement_risk",
                "confidence": false_disagree,
                "scope": row.get("coordinate"),
                "recommended_test": "revive signal in branch, recombine with adjacent surface, retest globally",
                "evidence": row.get("artifact_source"),
            })
    for row in projection_rows:
        hazard = f(row.get("hazard_bias")) or 0.0
        gain = f(row.get("downstream_path_gain")) or 0.0
        if gain > 0.0 and hazard > 0.01:
            rows.append({
                "claim_edge_id": stable_id("claim_edge", row.get("projection_id"), "projection_hazard"),
                "src_claim": f"{row.get('projection_id')}:projection_should_be_prior",
                "relation": "weakened_by",
                "dst_claim": f"{row.get('projection_id')}:projection_has_hazard",
                "confidence": hazard,
                "scope": row.get("method"),
                "recommended_test": "branch before projection and compare lower-rank/no-projection variants",
                "evidence": "projection_memory_matrix.csv",
            })
    for row in operation_rows:
        private = f(row.get("external_private"))
        public = f(row.get("external_public"))
        route_gain = f(row.get("weight_route_information_gain"))
        if private is not None and route_gain is not None and route_gain < 0.0:
            rows.append({
                "claim_edge_id": stable_id("claim_edge", row.get("operation_id"), "score_gain_conflict"),
                "src_claim": f"{row.get('operation_id')}:external_score_supports_operation",
                "relation": "contradicted_by",
                "dst_claim": f"{row.get('operation_id')}:route_information_gain_negative",
                "confidence": min(1.0, abs(route_gain) * 50.0),
                "scope": row.get("operation_name"),
                "recommended_test": "route ablation and private-public gap replay",
                "evidence": row.get("artifact_source"),
            })
        if private is not None and public is not None and private - public > 0.018:
            rows.append({
                "claim_edge_id": stable_id("claim_edge", row.get("operation_id"), "public_private_gap"),
                "src_claim": f"{row.get('operation_id')}:public_score_generalizes",
                "relation": "weakened_by",
                "dst_claim": f"{row.get('operation_id')}:private_public_gap",
                "confidence": min(1.0, private - public),
                "scope": row.get("operation_name"),
                "recommended_test": "check public-specific trap and hidden split sensitivity",
                "evidence": row.get("artifact_source"),
            })
    for row in impact_rows[:400]:
        if (f(row.get("foundation_stress_delta")) or 0.0) > 0.30:
            rows.append({
                "claim_edge_id": stable_id("claim_edge", row.get("impact_id"), "foundation_stress"),
                "src_claim": f"{row.get('surface_id')}:local_move_is_safe",
                "relation": "contradicted_by",
                "dst_claim": f"{row.get('impact_id')}:foundation_stress_delta_high",
                "confidence": row.get("foundation_stress_delta"),
                "scope": row.get("coordinate"),
                "recommended_test": "stop stacking local fixes; branch to alternate normalization or partition",
                "evidence": row.get("artifact_source"),
            })
    return sorted(rows, key=lambda r: f(r.get("confidence")) or -999, reverse=True)


def grokking_rows_from_manifest(path: Path) -> list[dict[str, Any]]:
    data = load_json(path, {})
    members = data.get("members", []) if isinstance(data, dict) else []
    rows: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, dict):
            continue
        ov = member.get("ov") if isinstance(member.get("ov"), dict) else {}
        if not ov.get("GROK_INCUBATION"):
            continue
        name = member.get("name")
        rows.append({
            "grokking_id": stable_id("grok", path, name),
            "run": name,
            "run_id": stable_id("run", name),
            "record_status": "planned_or_running",
            "mode": member.get("mode"),
            "branch_mode": ov.get("GROK_BRANCH_MODE"),
            "hypothesis": member.get("hypothesis"),
            "translation_goal": member.get("translation_goal"),
            "trust_policy": member.get("trust_policy"),
            "time_budget_min": f(ov.get("TIME_BUDGET_MIN")),
            "max_seasons": f(ov.get("MAX_SEASONS")),
            "max_epochs": f(ov.get("MAX_EPOCHS")),
            "evolution_patience": f(ov.get("EVOLUTION_PATIENCE")),
            "evolution_budget": f(ov.get("EVOLUTION_BUDGET")),
            "attention_pool": f(ov.get("ATTENTION_POOL")),
            "dive_budget": f(ov.get("DIVE_BUDGET")),
            "dream_replays": f(ov.get("DREAM_REPLAYS")),
            "mlp_patience": f(ov.get("MLP_PATIENCE")),
            "mlp_max_iter": f(ov.get("MLP_MAX_ITER")),
            "expected_delay": f(ov.get("GROK_EXPECTED_DELAY")),
            "budget_share": f(ov.get("GROK_BUDGET_SHARE")),
            "ship_eligible": int(bool(ov.get("GROK_SHIP_ELIGIBLE"))),
            "artifact_source": str(path),
        })
    return rows


def grokking_incubation_matrix(fleet_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(fleet_dir.glob("*grok*manifest.json")):
        for row in grokking_rows_from_manifest(path):
            rows.append(row)
            if row.get("run"):
                seen.add(str(row.get("run")))
    for run_dir in run_dirs(fleet_dir):
        report = load_json(run_dir / "output" / "grokking_incubation_report.json", {})
        if not isinstance(report, dict) or not report:
            continue
        config = report.get("config") if isinstance(report.get("config"), dict) else {}
        observed = report.get("observed") if isinstance(report.get("observed"), dict) else {}
        metabolism = observed.get("metabolism") if isinstance(observed.get("metabolism"), dict) else {}
        row = {
            "grokking_id": stable_id("grok", run_dir.name, "observed"),
            "run": run_dir.name,
            "run_id": stable_id("run", run_dir.name),
            "record_status": "observed",
            "mode": report.get("mode"),
            "branch_mode": report.get("branch_mode"),
            "hypothesis": "observed_grokking_incubation_report",
            "translation_goal": "ingest observed long-horizon branch evidence into atlas",
            "trust_policy": report.get("doctrine"),
            "time_budget_min": f(config.get("time_budget_min")),
            "max_seasons": f(config.get("max_seasons")),
            "max_epochs": f(config.get("max_epochs")),
            "evolution_patience": f(config.get("evolution_patience")),
            "evolution_budget": f(config.get("evolution_budget")),
            "attention_pool": f(config.get("attention_pool")),
            "dive_budget": f(config.get("dive_budget")),
            "dream_replays": f(config.get("dream_replays")),
            "mlp_patience": f(config.get("mlp_patience")),
            "mlp_max_iter": f(config.get("mlp_max_iter")),
            "expected_delay": f(config.get("expected_delay")),
            "budget_share": f(config.get("budget_share")),
            "forward_blend_corr": f(observed.get("forward_blend_corr")),
            "sealed_holdout_corr": f(observed.get("sealed_holdout_corr")),
            "feature_clusters": f(observed.get("feature_clusters")),
            "seasons_observed": f(metabolism.get("seasons")),
            "epochs_observed": f(metabolism.get("epochs")),
            "ship_eligible": int(bool(report.get("ship_eligible"))),
            "promotion_gates": json.dumps(report.get("required_promotion_gates", []), sort_keys=True),
            "atlas_ingestion": json.dumps(report.get("next_atlas_ingestion", []), sort_keys=True),
            "artifact_source": str(run_dir / "output" / "grokking_incubation_report.json"),
        }
        rows = [r for r in rows if r.get("run") != run_dir.name]
        rows.append(row)
        seen.add(run_dir.name)
    return sorted(rows, key=lambda r: (str(r.get("record_status")) != "observed", str(r.get("run"))))


def external_score_for_path(score_by_file: dict[str, dict[str, Any]], path_value: Any, name: str) -> dict[str, Any]:
    filename = Path(str(path_value or "")).name
    candidates = [filename, f"{name}.csv", f"submission_{name}.csv"]
    for candidate in candidates:
        if candidate in score_by_file:
            return score_by_file[candidate]
    return {}


def operation_records(
    fleet_dir: Path,
    path_rows: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_by_file = scores_by_filename(scores)
    rows: list[dict[str, Any]] = []

    for manifest_path in sorted(fleet_dir.glob("route_carves*/route_carve_manifest.csv")):
        batch = manifest_path.parent.name
        for row in read_csv(manifest_path):
            name = row.get("candidate") or Path(row.get("path", "")).stem
            score = external_score_for_path(score_by_file, row.get("path"), name)
            review = safe_float(row, "review_score")
            gain = safe_float(row, "route_information_gain")
            novelty = safe_float(row, "novelty_from_champion")
            rows.append({
                "operation_id": stable_id("operation", "route_carve", manifest_path, name),
                "operation_type": "route_carve",
                "operation_batch": batch,
                "operation_name": name,
                "source_run": row.get("route"),
                "source_run_id": stable_id("run", row.get("route")),
                "candidate_path": row.get("path"),
                "candidate_filename": Path(str(row.get("path", ""))).name,
                "rehab_stage": row.get("rehab_stage"),
                "lens": row.get("lens"),
                "alpha": safe_float(row, "alpha"),
                "operation_strength": review,
                "external_private": f(score.get("private")),
                "external_public": f(score.get("public")),
                "weight_route_information_gain": gain,
                "bias_corr_champion_route": safe_float(row, "corr_champion_route"),
                "bias_corr_candidate_champion": safe_float(row, "corr_candidate_champion"),
                "bias_corr_candidate_route": safe_float(row, "corr_candidate_route"),
                "bias_novelty_from_champion": novelty,
                "bias_conflict_tail": safe_float(row, "conflict_tail_frac"),
                "bias_agree_tail": safe_float(row, "agree_tail_frac"),
                "bias_route_tail": safe_float(row, "route_tail_frac"),
                "intent": row.get("intent"),
                "weights_json": json.dumps({
                    "alpha": safe_float(row, "alpha"),
                    "route_information_gain": gain,
                    "review_score": review,
                    "novelty_from_champion": novelty,
                }, sort_keys=True),
                "artifact_source": str(manifest_path),
            })

    for blend_path in sorted((fleet_dir / "blends").glob("*blend*.csv")):
        for row in read_csv(blend_path):
            weight_cols = [key for key in row if key.startswith("w_")]
            if not weight_cols:
                continue
            name = row.get("name") or Path(row.get("path", "")).stem
            weights = {key: safe_float(row, key) for key in weight_cols}
            good_weights = [val for val in weights.values() if val is not None]
            abs_sum = sum(abs(val) for val in good_weights)
            l2 = math.sqrt(sum(val * val for val in good_weights)) if good_weights else None
            signed_sum = sum(good_weights) if good_weights else None
            score = external_score_for_path(score_by_file, row.get("path"), name)
            novelty = None
            corr_prev = safe_float(row, "corr_prev_best")
            if corr_prev is not None:
                novelty = 1.0 - corr_prev
            op = {
                "operation_id": stable_id("operation", "blend", blend_path, name),
                "operation_type": "blend",
                "operation_batch": blend_path.parent.name,
                "operation_name": name,
                "candidate_path": row.get("path"),
                "candidate_filename": Path(str(row.get("path", ""))).name,
                "operation_strength": novelty,
                "external_private": f(score.get("private")),
                "external_public": f(score.get("public")),
                "weight_component_abs_sum": abs_sum,
                "weight_component_l2": l2,
                "weight_component_signed_sum": signed_sum,
                "bias_corr_prev_best": corr_prev,
                "bias_corr_gpu": safe_float(row, "corr_gpu"),
                "bias_corr_balanced": safe_float(row, "corr_balanced"),
                "bias_corr_sharp": safe_float(row, "corr_sharp"),
                "bias_corr_light": safe_float(row, "corr_light"),
                "bias_tail3sd": safe_float(row, "tail3sd"),
                "std": safe_float(row, "std"),
                "weights_json": json.dumps(weights, sort_keys=True),
                "artifact_source": str(blend_path),
            }
            op.update(weights)
            rows.append(op)

    for path in path_rows:
        rows.append({
            "operation_id": stable_id("operation", "path_transform", path.get("path_id")),
            "operation_type": "path_transform",
            "operation_batch": path.get("run"),
            "operation_name": path.get("key"),
            "source_run": path.get("run"),
            "source_run_id": path.get("run_id"),
            "path_id": path.get("path_id"),
            "skill": path.get("skill"),
            "family": path.get("family"),
            "transform": path.get("transform"),
            "stage": path.get("stage"),
            "weight_path_strength": path.get("robust_memory_score"),
            "operation_strength": path.get("robust_memory_score"),
            "bias_decay": path.get("decay"),
            "bias_world_floor": path.get("world_floor"),
            "bias_width": path.get("width"),
            "cost": path.get("cost"),
            "k": path.get("k"),
            "promoted": path.get("promoted"),
            "shipped": path.get("shipped"),
            "artifact_source": path.get("artifact_source"),
        })

    return sorted(rows, key=lambda r: f(r.get("operation_strength")) or -999, reverse=True)


def memory_role(
    robust: float | None,
    decay: float | None,
    world_floor: float | None,
    promote_rate: float | None,
    n: int,
) -> str:
    if robust is None:
        return "unknown"
    if n < 3 and robust > 0.07:
        return "short_term_candidate_needs_retest"
    if decay is not None and decay > 0.03:
        return "hazard_memory_width_or_decay"
    if world_floor is not None and world_floor < 0.0:
        return "anti_prior_world_fragile"
    if promote_rate is not None and promote_rate > 0.45 and robust > 0.065:
        return "long_term_prior"
    if robust > 0.055:
        return "medium_term_candidate"
    return "anti_prior_low_signal"


def score_rows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_s in args.scores_csv or []:
        rows.extend(tg.load_scores_csv(Path(path_s)))
    for item in args.score or []:
        rows.append(tg.score_arg(item))
    return rows


def scores_by_run(
    scores: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    route_index: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        filename = Path(score.get("filename", "")).name
        manifest = route_index.get(filename)
        if manifest and manifest.get("route"):
            out[str(manifest["route"])].append(score)
            continue
        for run in tg.source_runs_from_description(score.get("description", ""), runs):
            out[run].append(score)
    return out


def numeric_column(matrix: np.ndarray, columns: list[str], name: str) -> np.ndarray:
    if name not in columns or matrix.shape[0] == 0:
        return np.full((matrix.shape[0],), np.nan, dtype=np.float64)
    return matrix[:, columns.index(name)]


def path_control_vectors(path_x: np.ndarray) -> dict[str, np.ndarray]:
    score = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "robust_memory_score")
    decay = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "decay")
    overfit = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "overfit_ratio")
    width = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "width")
    wf_width = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "wf_width")
    world_floor = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "world_floor")
    uniqueness = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "uniqueness")
    stability = numeric_column(path_x, PATH_NUMERIC_COLUMNS, "stability")
    promoted = nan_to_zero(numeric_column(path_x, PATH_NUMERIC_COLUMNS, "promoted"))
    shipped = nan_to_zero(numeric_column(path_x, PATH_NUMERIC_COLUMNS, "shipped"))

    hazard_raw = (
        np.maximum(0.0, nan_to_zero(decay))
        + 0.015 * np.maximum(0.0, nan_to_zero(overfit) - 1.0)
        + 0.035 * (nan_to_zero(width) < 0.0).astype(np.float64)
        + np.maximum(0.0, -nan_to_zero(world_floor))
    )
    hazard_bias = normalize01(hazard_raw)
    strength_weight = softmax_weights(score)
    exploration_raw = (
        4.0 * np.maximum(0.0, nan_to_zero(score))
        + 0.20 * np.maximum(0.0, nan_to_zero(uniqueness))
        + 0.15 * (1.0 - np.clip(promoted + shipped, 0.0, 1.0))
        + 0.10 * hazard_bias
    )
    exploration_bias = normalize01(exploration_raw)
    grok_raw = (
        4.5 * np.maximum(0.0, nan_to_zero(score))
        + 0.60 * np.maximum(0.0, nan_to_zero(wf_width))
        + 0.15 * np.maximum(0.0, nan_to_zero(stability))
        + 0.20 * np.maximum(0.0, -nan_to_zero(decay))
        + 0.10 * (1.0 - np.clip(shipped, 0.0, 1.0))
    )
    grokking_potential = normalize01(grok_raw)
    return {
        "path_strength_weight": strength_weight,
        "path_hazard_bias": hazard_bias,
        "path_exploration_bias": exploration_bias,
        "path_grokking_potential": grokking_potential,
    }


def operation_control_vectors(operation_x: np.ndarray) -> dict[str, np.ndarray]:
    strength = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "operation_strength")
    private = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "external_private")
    public = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "external_public")
    novelty = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "bias_novelty_from_champion")
    gain = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "weight_route_information_gain")
    decay = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "bias_decay")
    corr_prev = numeric_column(operation_x, OPERATION_NUMERIC_COLUMNS, "bias_corr_prev_best")
    score_signal = np.where(np.isfinite(private), private, strength)
    exploration_raw = (
        3.0 * np.maximum(0.0, nan_to_zero(score_signal))
        + 10.0 * np.maximum(0.0, nan_to_zero(gain))
        + np.maximum(0.0, nan_to_zero(novelty))
        + np.maximum(0.0, 1.0 - nan_to_zero(corr_prev))
    )
    hazard_raw = np.maximum(0.0, nan_to_zero(decay))
    generalization_gap = np.where(np.isfinite(private) & np.isfinite(public), private - public, np.nan)
    return {
        "operation_strength_weight": softmax_weights(score_signal),
        "operation_exploration_bias": normalize01(exploration_raw),
        "operation_hazard_bias": normalize01(hazard_raw),
        "operation_private_public_gap": generalization_gap,
    }


def typology_edge_matrices(
    typology_ids: np.ndarray,
    vector_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(typology_ids)
    adjacency = np.zeros((n, n), dtype=np.float64)
    delta = np.full((n, n), np.nan, dtype=np.float64)
    index = {str(value): i for i, value in enumerate(typology_ids)}
    for row in vector_rows:
        src = index.get(str(row.get("src_typology_id")))
        dst = index.get(str(row.get("dst_typology_id")))
        if src is None or dst is None:
            continue
        adjacency[src, dst] = 1.0
        delta[src, dst] = numeric_value(row.get("delta_score"))
    return adjacency, delta


def write_numeric_bundle(
    out_dir: Path,
    path_rows: list[dict[str, Any]],
    typology_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    foundation_rows: list[dict[str, Any]],
    route_strength_rows: list[dict[str, Any]],
    grokking_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    collinearity_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    validation_budget_rows: list[dict[str, Any]],
    evidence_gate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path_ids, path_x = numeric_matrix(path_rows, "path_id", PATH_NUMERIC_COLUMNS)
    typology_ids, typology_x = numeric_matrix(typology_rows, "typology_id", TYPOLOGY_NUMERIC_COLUMNS)
    feature_ids, feature_x = numeric_matrix(feature_rows, "feature_memory_id", FEATURE_NUMERIC_COLUMNS)
    surface_ids, surface_x = numeric_matrix(surface_rows, "surface_id", SURFACE_NUMERIC_COLUMNS)
    impact_ids, impact_x = numeric_matrix(impact_rows, "impact_id", IMPACT_NUMERIC_COLUMNS)
    foundation_ids, foundation_x = numeric_matrix(
        foundation_rows, "foundation_stress_id", FOUNDATION_STRESS_NUMERIC_COLUMNS
    )
    route_strength_ids, route_strength_x = numeric_matrix(
        route_strength_rows, "route_strength_id", ROUTE_STRENGTH_NUMERIC_COLUMNS
    )
    grokking_ids, grokking_x = numeric_matrix(grokking_rows, "grokking_id", GROKKING_NUMERIC_COLUMNS)
    projection_ids, projection_x = numeric_matrix(projection_rows, "projection_id", PROJECTION_NUMERIC_COLUMNS)
    collinearity_ids, collinearity_x = numeric_matrix(
        collinearity_rows, "collinearity_id", COLLINEARITY_NUMERIC_COLUMNS
    )
    operation_ids, operation_x = numeric_matrix(operation_rows, "operation_id", OPERATION_NUMERIC_COLUMNS)
    validation_ids, validation_x = numeric_matrix(
        validation_budget_rows, "validation_world_id", VALIDATION_BUDGET_NUMERIC_COLUMNS
    )
    evidence_gate_ids, evidence_gate_x = numeric_matrix(
        evidence_gate_rows, "evidence_gate_id", EVIDENCE_GATE_NUMERIC_COLUMNS
    )
    typology_adjacency, typology_delta = typology_edge_matrices(typology_ids, vector_rows)
    path_controls = path_control_vectors(path_x)
    operation_controls = operation_control_vectors(operation_x)

    bundle_path = out_dir / "numeric_memory_bundle.npz"
    np.savez_compressed(
        bundle_path,
        path_ids=path_ids,
        path_X=path_x,
        path_information_matrix=standardized_information_matrix(path_x),
        **path_controls,
        typology_ids=typology_ids,
        typology_X=typology_x,
        typology_information_matrix=standardized_information_matrix(typology_x),
        typology_adjacency=typology_adjacency,
        typology_delta=typology_delta,
        feature_ids=feature_ids,
        feature_X=feature_x,
        feature_information_matrix=standardized_information_matrix(feature_x),
        surface_surgery_ids=surface_ids,
        surface_surgery_X=surface_x,
        surface_surgery_information_matrix=standardized_information_matrix(surface_x),
        impact_ids=impact_ids,
        impact_X=impact_x,
        impact_information_matrix=standardized_information_matrix(impact_x),
        foundation_stress_ids=foundation_ids,
        foundation_stress_X=foundation_x,
        foundation_stress_information_matrix=standardized_information_matrix(foundation_x),
        route_strength_ids=route_strength_ids,
        route_strength_X=route_strength_x,
        route_strength_information_matrix=standardized_information_matrix(route_strength_x),
        grokking_incubation_ids=grokking_ids,
        grokking_incubation_X=grokking_x,
        grokking_incubation_information_matrix=standardized_information_matrix(grokking_x),
        projection_ids=projection_ids,
        projection_X=projection_x,
        projection_information_matrix=standardized_information_matrix(projection_x),
        collinearity_ids=collinearity_ids,
        collinearity_X=collinearity_x,
        collinearity_information_matrix=standardized_information_matrix(collinearity_x),
        operation_ids=operation_ids,
        operation_types=np.asarray([str(row.get("operation_type", "")) for row in operation_rows], dtype=np.str_),
        operation_names=np.asarray([str(row.get("operation_name", "")) for row in operation_rows], dtype=np.str_),
        operation_X=operation_x,
        operation_information_matrix=standardized_information_matrix(operation_x),
        validation_budget_ids=validation_ids,
        validation_budget_X=validation_x,
        validation_budget_information_matrix=standardized_information_matrix(validation_x),
        evidence_gate_ids=evidence_gate_ids,
        evidence_gate_X=evidence_gate_x,
        evidence_gate_information_matrix=standardized_information_matrix(evidence_gate_x),
        **operation_controls,
    )

    schema = {
        "bundle": str(bundle_path),
        "format": "numpy savez compressed",
        "purpose": (
            "Loadable numeric long-memory: observed vectors, action weights, "
            "hazard/exploration/grokking biases, and X'X-style information matrices."
        ),
        "arrays": {
            "path_X": PATH_NUMERIC_COLUMNS,
            "path_information_matrix": PATH_NUMERIC_COLUMNS,
            "path_strength_weight": "softmax over robust path memory score",
            "path_hazard_bias": "normalized decay/overfit/negative-width/world-fragility penalty",
            "path_exploration_bias": "normalized signal plus uncertainty/retest bias",
            "path_grokking_potential": "normalized signal-width-stability retest potential",
            "typology_X": TYPOLOGY_NUMERIC_COLUMNS,
            "typology_information_matrix": TYPOLOGY_NUMERIC_COLUMNS,
            "typology_adjacency": "directed edge exists for one-axis typology moves",
            "typology_delta": "delta robust_score_shrunk for directed typology moves",
            "feature_X": FEATURE_NUMERIC_COLUMNS,
            "feature_information_matrix": FEATURE_NUMERIC_COLUMNS,
            "surface_surgery_X": SURFACE_NUMERIC_COLUMNS,
            "surface_surgery_information_matrix": SURFACE_NUMERIC_COLUMNS,
            "impact_X": IMPACT_NUMERIC_COLUMNS,
            "impact_information_matrix": IMPACT_NUMERIC_COLUMNS,
            "foundation_stress_X": FOUNDATION_STRESS_NUMERIC_COLUMNS,
            "foundation_stress_information_matrix": FOUNDATION_STRESS_NUMERIC_COLUMNS,
            "route_strength_X": ROUTE_STRENGTH_NUMERIC_COLUMNS,
            "route_strength_information_matrix": ROUTE_STRENGTH_NUMERIC_COLUMNS,
            "grokking_incubation_X": GROKKING_NUMERIC_COLUMNS,
            "grokking_incubation_information_matrix": GROKKING_NUMERIC_COLUMNS,
            "projection_X": PROJECTION_NUMERIC_COLUMNS,
            "projection_information_matrix": PROJECTION_NUMERIC_COLUMNS,
            "collinearity_X": COLLINEARITY_NUMERIC_COLUMNS,
            "collinearity_information_matrix": COLLINEARITY_NUMERIC_COLUMNS,
            "operation_X": OPERATION_NUMERIC_COLUMNS,
            "operation_information_matrix": OPERATION_NUMERIC_COLUMNS,
            "validation_budget_X": VALIDATION_BUDGET_NUMERIC_COLUMNS,
            "validation_budget_information_matrix": VALIDATION_BUDGET_NUMERIC_COLUMNS,
            "evidence_gate_X": EVIDENCE_GATE_NUMERIC_COLUMNS,
            "evidence_gate_information_matrix": EVIDENCE_GATE_NUMERIC_COLUMNS,
            "operation_strength_weight": "softmax over external/private score when present, else operation strength",
            "operation_exploration_bias": "normalized action gain/novelty/retest bias",
            "operation_hazard_bias": "normalized operation-level decay hazard",
            "operation_private_public_gap": "private minus public when external scores are present",
        },
        "row_ids": {
            "path_ids": "matches path_memory_matrix.csv path_id",
            "typology_ids": "matches typology_memory_matrix.csv typology_id",
            "feature_ids": "matches feature_space_memory_matrix.csv feature_memory_id",
            "surface_surgery_ids": "matches surface_surgery_matrix.csv surface_id",
            "impact_ids": "matches impact_field_matrix.csv impact_id",
            "foundation_stress_ids": "matches foundation_stress_matrix.csv foundation_stress_id",
            "route_strength_ids": "matches route_strength_matrix.csv route_strength_id",
            "grokking_incubation_ids": "matches grokking_incubation_matrix.csv grokking_id",
            "projection_ids": "matches projection_memory_matrix.csv projection_id",
            "collinearity_ids": "matches collinearity_memory_matrix.csv collinearity_id",
            "operation_ids": "matches operation_memory_matrix.csv operation_id",
            "validation_budget_ids": "matches validation_budget_ledger.csv validation_world_id",
            "evidence_gate_ids": "matches evidence_gate_matrix.csv evidence_gate_id",
        },
    }
    (out_dir / "numeric_memory_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return schema


def json_ref(filename: str) -> str:
    return f"artifact://{filename}"


def npz_ref(array_name: str) -> str:
    return f"npz://numeric_memory_bundle.npz#{array_name}"


def tensor_category(name: str) -> str:
    if name.endswith("_ids") or name in {"operation_names", "operation_types"}:
        return "row_index"
    if name.startswith("path_"):
        return "path_state"
    if name.startswith("typology_"):
        return "typology_state"
    if name.startswith("feature_"):
        return "feature_space"
    if name.startswith("surface_surgery_"):
        return "surface_operation_state"
    if name.startswith("impact_"):
        return "impact_field"
    if name.startswith("foundation_stress_"):
        return "foundation_stress"
    if name.startswith("route_strength_"):
        return "strategy_value_state"
    if name.startswith("grokking_incubation_"):
        return "grokking_incubation_state"
    if name.startswith("projection_"):
        return "transformation_space"
    if name.startswith("collinearity_"):
        return "feature_space"
    if name.startswith("operation_"):
        return "operator_state"
    if name.startswith("validation_budget_"):
        return "validation_budget_state"
    if name.startswith("evidence_gate_"):
        return "evidence_gate_state"
    return "misc"


def matrix_semantics(name: str) -> str:
    if name.endswith("_X"):
        return "row-aligned numeric state vectors"
    if name.endswith("_information_matrix"):
        return "standardized X^T X style information matrix over columns"
    if name.endswith("_weight"):
        return "normalized route/path attention weight"
    if name.endswith("_bias"):
        return "normalized guidance or hazard bias"
    if name.endswith("_potential"):
        return "normalized grokking or exploration potential"
    if name == "typology_adjacency":
        return "directed adjacency for one-axis moves in typology space"
    if name == "typology_delta":
        return "directed robust-score delta for typology moves"
    if name == "operation_private_public_gap":
        return "external private minus public score where observed"
    if name.endswith("_ids"):
        return "row-id alignment vector"
    return "supporting tensor or vector"


def tensor_artifact_manifest(out_dir: Path, schema: dict[str, Any]) -> dict[str, Any]:
    bundle_path = out_dir / "numeric_memory_bundle.npz"
    tensors: list[dict[str, Any]] = []
    with np.load(bundle_path) as bundle:
        for name in sorted(bundle.files):
            arr = bundle[name]
            tensors.append({
                "artifact_id": stable_id("tensor", name, arr.shape, str(arr.dtype)),
                "name": name,
                "uri": npz_ref(name),
                "category": tensor_category(name),
                "semantics": matrix_semantics(name),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "schema": schema.get("arrays", {}).get(name),
            })
    return {
        "schema_version": 1,
        "bundle": str(bundle_path),
        "tensors": tensors,
        "model_checkpoint_artifacts_detected": [],
        "model_checkpoint_note": (
            "No .pt/.pkl/.joblib/.safetensors/.npy/.npz model checkpoint artifacts were "
            "present in the current fleet outputs besides numeric_memory_bundle.npz. "
            "Future runtimes should emit weights, optimizer state, replay buffers, "
            "Fisher/Hessian approximations, split stats, and router/adaptor weights "
            "as artifact-store entries referenced from this manifest."
        ),
    }


def action_parameters(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "alpha",
        "lens",
        "rehab_stage",
        "skill",
        "family",
        "transform",
        "stage",
        "cost",
        "k",
        "weights_json",
    ]
    out = {key: row.get(key) for key in keys if row.get(key) not in (None, "")}
    if isinstance(out.get("weights_json"), str):
        try:
            out["weights"] = json.loads(str(out.pop("weights_json")))
        except json.JSONDecodeError:
            pass
    return out


def effect_vector(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "operation_strength",
        "external_private",
        "external_public",
        "weight_path_strength",
        "weight_route_information_gain",
        "weight_component_abs_sum",
        "weight_component_l2",
        "weight_component_signed_sum",
        "bias_corr_prev_best",
        "bias_corr_champion_route",
        "bias_corr_candidate_champion",
        "bias_corr_candidate_route",
        "bias_novelty_from_champion",
        "bias_conflict_tail",
        "bias_agree_tail",
        "bias_route_tail",
        "bias_decay",
        "bias_world_floor",
        "bias_width",
        "promoted",
        "shipped",
    ]
    return {key: row.get(key) for key in keys if row.get(key) not in (None, "")}


def operation_operator_type(row: dict[str, Any]) -> str:
    op_type = str(row.get("operation_type") or "")
    if op_type == "route_carve":
        stage = str(row.get("rehab_stage") or "route_carve")
        lens = str(row.get("lens") or "raw")
        return f"{stage}:{lens}"
    if op_type == "blend":
        return "linear_prediction_blend"
    if op_type == "path_transform":
        family = str(row.get("family") or "unknown")
        transform = str(row.get("transform") or "identity")
        return f"path_transform:{family}:{transform}"
    return op_type or "unknown_operator"


def operator_graph_edges(operation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in operation_rows:
        op_id = op.get("operation_id")
        if not op_id:
            continue
        parent = f"checkpoint:{op.get('source_run_id') or 'atlas_root'}"
        child_token = op.get("path_id") or op.get("candidate_filename") or op_id
        child = f"checkpoint:{stable_id('child_state', child_token)}"
        rows.append({
            "edge_id": stable_id("operator_edge", op_id),
            "parent_checkpoint": parent,
            "child_checkpoint": child,
            "operation_id": op_id,
            "action_type": op.get("operation_type"),
            "operator_type": operation_operator_type(op),
            "operator_name": op.get("operation_name"),
            "input_space": op.get("source_run") or op.get("operation_batch") or "atlas_root",
            "output_space": op.get("candidate_filename") or op.get("path_id") or op.get("operation_name"),
            "learned_artifacts": json.dumps({
                "operation_row": "operation_memory_matrix.csv",
                "operation_vector": npz_ref("operation_X"),
                "operation_information_matrix": npz_ref("operation_information_matrix"),
            }, sort_keys=True),
            "action_parameters_json": json.dumps(action_parameters(op), sort_keys=True),
            "effect_vector_json": json.dumps(effect_vector(op), sort_keys=True),
            "challenge_plan": challenge_plan(op),
            "artifact_source": op.get("artifact_source"),
        })
    return rows


def challenge_plan(row: dict[str, Any]) -> str:
    op_type = row.get("operation_type")
    if op_type == "route_carve":
        return "branch_before_carve; ablate lens/alpha; compare residual and private-public gap"
    if op_type == "blend":
        return "reweight components; leave-one-route-out; compare correlation and tail risk"
    if op_type == "path_transform":
        return "branch from raw feature space; ablate transform/family; retest across seed and terrain"
    return "compare against parent and sibling checkpoints"


def checkpoint_graph(
    run_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    atlas_id: str,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [{
        "checkpoint": "checkpoint:atlas_root",
        "kind": "root",
        "atlas_id": atlas_id,
        "state_components": ["raw_runtime_artifacts", "matrix_compiler"],
    }]
    edges: list[dict[str, Any]] = []
    for row in run_rows:
        checkpoint = f"checkpoint:{row.get('run_id')}"
        nodes.append({
            "checkpoint": checkpoint,
            "kind": "runtime_checkpoint",
            "run": row.get("run"),
            "best_external_private": row.get("best_external_private"),
            "best_external_public": row.get("best_external_public"),
            "artifact_manifest": row.get("artifact_manifest"),
            "state_components": ["run_artifacts", "path_matrix", "feature_space_observations"],
        })
        edges.append({
            "edge_id": stable_id("checkpoint_edge", "atlas_root", row.get("run_id")),
            "from": "checkpoint:atlas_root",
            "to": checkpoint,
            "action_type": "runtime_observation",
            "operator": "append_artifacts",
        })
    for row in operation_rows:
        child_token = row.get("path_id") or row.get("candidate_filename") or row.get("operation_id")
        child = f"checkpoint:{stable_id('child_state', child_token)}"
        parent = f"checkpoint:{row.get('source_run_id') or 'atlas_root'}"
        nodes.append({
            "checkpoint": child,
            "kind": "operation_result",
            "operation_id": row.get("operation_id"),
            "operation_type": row.get("operation_type"),
            "operation_name": row.get("operation_name"),
            "candidate": row.get("candidate_filename") or row.get("path_id"),
            "state_components": ["operator_state", "evaluation_effects"],
        })
        edges.append({
            "edge_id": stable_id("checkpoint_edge", row.get("operation_id")),
            "from": parent,
            "to": child,
            "action_type": row.get("operation_type"),
            "operator": operation_operator_type(row),
            "operation_id": row.get("operation_id"),
        })
    return {
        "schema_version": 1,
        "atlas_id": atlas_id,
        "formalism": "S_{t+1} = a_t(S_t); nodes are states, edges are operators/actions",
        "nodes": nodes,
        "edges": edges,
    }


def open_hypotheses(attn: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in attn.get("coverage_gaps", [])[:25]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "coverage", row.get("coverage_id")),
            "type": "untried_typology_gap",
            "priority": row.get("gap_priority"),
            "statement": f"Try {row.get('skill')}|{row.get('family')}|{row.get('transform')} because neighbors have support.",
            "support": row.get("nearest_evidence"),
            "recommended_action": "allocate sprout budget with independent seed/terrain retest",
            "invalidation_conditions": ["fails across independent seed", "private/public gap widens", "world floor turns negative"],
        })
    for row in attn.get("projection_memory", [])[:20]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "projection", row.get("projection_id")),
            "type": "projection_or_reduction",
            "priority": row.get("grokking_potential"),
            "statement": f"Use or challenge {row.get('method')} via {row.get('source_axis')}={row.get('source_value')}.",
            "support": row.get("example_paths"),
            "recommended_action": row.get("recommendation"),
            "invalidation_conditions": ["downstream gain disappears", "width narrows", "decay increases", "tree route regresses"],
        })
    for row in attn.get("collinearity_memory", [])[:20]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "collinearity", row.get("collinearity_id")),
            "type": "feature_space_compression",
            "priority": row.get("compression_bias"),
            "statement": f"Community {row.get('community')} in run {row.get('run')} may be compressible.",
            "support": row.get("artifact_source"),
            "recommended_action": row.get("recommendation"),
            "invalidation_conditions": ["compression loses target alignment", "shift hazard rises", "restored features improve residuals"],
        })
    for row in attn.get("operation_attention", [])[:25]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "operation", row.get("operation_id")),
            "type": "action_branch",
            "priority": row.get("operation_strength"),
            "statement": f"Revisit operation {row.get('operation_type')}:{row.get('operation_name')}.",
            "support": row.get("artifact_source"),
            "recommended_action": challenge_plan(row),
            "invalidation_conditions": ["ablation has equal score", "route novelty is low", "private-public gap suggests trap"],
        })
    for row in attn.get("surface_surgery_attention", [])[:25]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "surface", row.get("surface_id")),
            "type": "surface_surgery",
            "priority": row.get("grokking_priority"),
            "statement": (
                f"Surface {row.get('source_kind')} at {row.get('coordinate')} may need "
                f"{row.get('suggested_surgery')}."
            ),
            "support": row.get("artifact_source"),
            "recommended_action": row.get("suggested_surgery"),
            "invalidation_conditions": [
                "global stability falls after local edit",
                "false-agreement risk rises",
                "restoring the previous surface wins",
            ],
        })
    for row in attn.get("impact_field_attention", [])[:25]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "impact", row.get("impact_id")),
            "type": "ripple_field",
            "priority": row.get("branch_priority"),
            "statement": (
                f"Move {row.get('operator_move')} at {row.get('coordinate')} has "
                f"status {row.get('move_status')}."
            ),
            "support": row.get("artifact_source"),
            "recommended_action": row.get("followup_action"),
            "invalidation_conditions": [
                "ripple radius shrinks under independent retest",
                "global effect improves without side effects",
                "local effect disappears under bootstrap",
            ],
        })
    for row in attn.get("foundation_stress_attention", [])[:20]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "foundation", row.get("foundation_stress_id")),
            "type": "foundation_stress",
            "priority": row.get("foundation_rethink_priority"),
            "statement": (
                f"Foundation stress for run={row.get('run')} source={row.get('source_group')} "
                f"is {row.get('stress_score')}."
            ),
            "support": row.get("top_stress_sources"),
            "recommended_action": row.get("recommendation"),
            "invalidation_conditions": [
                "alternate foundation does not reduce stress",
                "stress is localized to one route-limited branch",
                "new data confirms current topology",
            ],
        })
    for row in attn.get("contradiction_attention", [])[:20]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "contradiction", row.get("claim_edge_id")),
            "type": "contradiction",
            "priority": row.get("confidence"),
            "statement": f"{row.get('src_claim')} {row.get('relation')} {row.get('dst_claim')}",
            "support": row.get("evidence"),
            "recommended_action": row.get("recommended_test"),
            "invalidation_conditions": [
                "supporting claim is superseded",
                "contradiction vanishes under independent split",
                "scope is narrowed to a safe branch",
            ],
        })
    for row in attn.get("grokking_incubation_attention", [])[:20]:
        items.append({
            "hypothesis_id": stable_id("hypothesis", "grok", row.get("grokking_id")),
            "type": "grokking_incubation",
            "priority": row.get("budget_share") or row.get("time_budget_min"),
            "statement": (
                f"Grokking incubator {row.get('run')} is {row.get('record_status')} "
                f"with branch mode {row.get('branch_mode')}."
            ),
            "support": row.get("artifact_source"),
            "recommended_action": (
                "collect Kaggle output and ingest report" if row.get("record_status") != "observed"
                else "compare delayed evidence against impact/foundation/route matrices"
            ),
            "invalidation_conditions": [
                "no internal movement evidence after budget",
                "overfit or false-agreement risk dominates",
                "robust OOS court rejects delayed jump",
            ],
        })
    return {
        "schema_version": 1,
        "purpose": "Unfinished, challengeable hypotheses for the next runtime; not accepted truth.",
        "hypotheses": items,
    }


def compact_row(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            out[key] = value
    return out


def policy_budget(attn: dict[str, Any]) -> dict[str, Any]:
    foundation_peak = max(
        [f(row.get("foundation_rethink_priority")) or 0.0 for row in attn.get("foundation_stress_attention", [])]
        or [0.0]
    )
    contradiction_peak = max(
        [f(row.get("confidence")) or 0.0 for row in attn.get("contradiction_attention", [])]
        or [0.0]
    )
    open_grok = sum(
        1 for row in attn.get("grokking_incubation_attention", [])
        if str(row.get("record_status")) != "observed"
    )
    branch_pressure = max(foundation_peak, contradiction_peak * 0.75)
    grok_share = 0.08 + min(0.06, open_grok * 0.02)
    repair_share = 0.12 + min(0.10, branch_pressure * 0.20)
    random_share = 0.08
    consolidation_share = 0.12
    normal_share = 1.0 - grok_share - repair_share - random_share - consolidation_share
    if normal_share < 0.50:
        spill = 0.50 - normal_share
        random_share = max(0.04, random_share - spill * 0.5)
        consolidation_share = max(0.08, consolidation_share - spill * 0.5)
        normal_share = 1.0 - grok_share - repair_share - random_share - consolidation_share
    return {
        "normal_search": round(max(0.0, normal_share), 4),
        "safe_repair_and_consolidation": round(repair_share + consolidation_share, 4),
        "grokking_incubation": round(grok_share, 4),
        "random_sprouts": round(random_share, 4),
        "signals": {
            "foundation_peak": foundation_peak,
            "contradiction_peak": contradiction_peak,
            "open_grokking_incubators": open_grok,
        },
    }


def next_runtime_policy(attn: dict[str, Any]) -> dict[str, Any]:
    protected_priors = [
        compact_row(row, ["typology_id", "skill", "family", "transform", "robust_score_shrunk", "memory_role"])
        for row in attn.get("long_term_attention", [])[:12]
    ]
    hazard_exclusions = [
        compact_row(row, ["typology_id", "skill", "family", "transform", "robust_score_shrunk", "memory_role"])
        for row in attn.get("hazard_attention", [])[:16]
    ]
    branch_queue = []
    for row in attn.get("impact_field_attention", [])[:12]:
        branch_queue.append({
            "kind": "impact_field",
            "priority": row.get("branch_priority"),
            "status": row.get("move_status"),
            "action": row.get("followup_action"),
            "source": compact_row(
                row,
                ["impact_id", "surface_id", "source_kind", "run", "coordinate", "operator_move", "artifact_source"],
            ),
            "guard": "branch-only unless independent retest improves global effect without raising false agreement",
        })
    for row in attn.get("surface_surgery_attention", [])[:12]:
        branch_queue.append({
            "kind": "surface_surgery",
            "priority": row.get("grokking_priority"),
            "status": "reversible_surface_edit",
            "action": row.get("suggested_surgery"),
            "source": compact_row(
                row,
                ["surface_id", "source_kind", "run", "coordinate", "secondary_coordinate", "artifact_source"],
            ),
            "guard": row.get("reversal_plan"),
        })
    branch_queue = sorted(branch_queue, key=lambda r: f(r.get("priority")) or -999, reverse=True)[:20]

    grokking_queue = []
    for row in attn.get("grokking_incubation_attention", [])[:20]:
        status = str(row.get("record_status"))
        observed = status == "observed"
        grokking_queue.append({
            "run": row.get("run"),
            "branch_mode": row.get("branch_mode"),
            "status": status,
            "priority": f(row.get("budget_share")) or f(row.get("time_budget_min")) or 0.0,
            "ship_eligible": 0,
            "action": (
                "collect_output_then_recompile_atlas" if not observed
                else "score_delayed_evidence_against_promotion_gates"
            ),
            "manifest_or_report": row.get("artifact_source"),
            "promotion_gates": [
                "beats_parent_on_robust_score",
                "worst_world_nonnegative",
                "false_agreement_risk_not_high",
                "overfit_ratio_controlled",
                "private_public_or_forward_gap_not_a_trap",
            ],
        })

    retest_queue = [
        {
            "kind": "typology_gap",
            "priority": row.get("gap_priority"),
            "action": "sprout_independent_seed",
            "source": compact_row(row, ["coverage_id", "skill", "family", "transform", "nearest_evidence"]),
        }
        for row in attn.get("coverage_gaps", [])[:12]
    ]
    reduction_queue = [
        {
            "kind": "projection_or_compression",
            "priority": row.get("grokking_potential") or row.get("compression_bias"),
            "action": row.get("recommendation"),
            "source": compact_row(
                row,
                ["projection_id", "collinearity_id", "method", "community", "run", "source_axis", "source_value", "artifact_source"],
            ),
        }
        for row in (attn.get("projection_memory", [])[:8] + attn.get("collinearity_memory", [])[:8])
    ]
    contradiction_queue = [
        {
            "priority": row.get("confidence"),
            "action": row.get("recommended_test"),
            "source": compact_row(row, ["claim_edge_id", "src_claim", "relation", "dst_claim", "scope", "evidence"]),
        }
        for row in attn.get("contradiction_attention", [])[:16]
    ]
    evidence_queue = [
        {
            "priority": row.get("evidence_score"),
            "grade": row.get("evidence_grade"),
            "decision": row.get("decision"),
            "required_next_evidence": row.get("required_next_evidence"),
            "source": compact_row(
                row,
                [
                    "evidence_gate_id",
                    "impact_id",
                    "source_kind",
                    "coordinate",
                    "operator_move",
                    "artifact_source",
                ],
            ),
        }
        for row in attn.get("evidence_gate_attention", [])[:20]
    ]
    validation_budget = [
        compact_row(
            row,
            [
                "validation_world",
                "world_kind",
                "candidate_count_seen",
                "reuse_pressure",
                "trust_discount",
                "remaining_trust_budget",
                "policy",
                "guard",
            ],
        )
        for row in attn.get("validation_budget_attention", [])[:12]
    ]
    collect_by_command: dict[str, dict[str, Any]] = {}
    for row in attn.get("grokking_incubation_attention", []):
        if str(row.get("record_status")) == "observed":
            continue
        source = str(row.get("artifact_source") or "")
        command = (
            f"python3 tools/fleet.py harvest-grok --manifest {source}"
            if source.endswith("manifest.json")
            else "python3 tools/fleet.py harvest-grok --manifest /home/username/new_algo/kaggle/fleet/grok-atlas_manifest.json"
        )
        entry = collect_by_command.setdefault(command, {
            "runs": [],
            "reason": "grokking incubator is still planned/running; collect report when Kaggle completes",
            "artifact_source": source,
            "command": command,
            "post_collect": "rerun tools/memory_matrices.py with score context to ingest observed grokking reports",
        })
        entry["runs"].append(row.get("run"))
    collect_actions = list(collect_by_command.values())
    return {
        "schema_version": 1,
        "purpose": "Operational next-runtime policy derived from the computational atlas; guidance, not accepted truth.",
        "budget_allocation": policy_budget(attn),
        "hard_guards": {
            "grokking_branches_cannot_ship_directly": True,
            "local_surface_edits_need_global_ripple_check": True,
            "contradictions_require_ablation_before_promotion": True,
            "hazards_are_masks_or_penalties_not_deletions": True,
            "unsupported_material_operations_stay_branch_only": True,
            "overused_validation_worlds_are_discounted": True,
        },
        "protected_priors": protected_priors,
        "hazard_exclusions": hazard_exclusions,
        "branch_queue": branch_queue,
        "grokking_queue": grokking_queue,
        "retest_queue": retest_queue,
        "reduction_queue": sorted(reduction_queue, key=lambda r: f(r.get("priority")) or -999, reverse=True)[:16],
        "contradiction_queue": contradiction_queue,
        "evidence_gate_queue": evidence_queue,
        "validation_budget": validation_budget,
        "collect_actions": collect_actions,
        "promotion_gates": [
            "independent_seed_or_split_reproduction",
            "global_score_improves_or_route_is_explicitly_limited",
            "worst_world_nonnegative",
            "false_agreement_and_false_disagreement_checked",
            "foundation_stress_not_increased_without_branch_reason",
            "private_public_or_forward_gap_not_a_trap",
        ],
    }


def atlas_manifest(
    out_dir: Path,
    atlas_id: str,
    run_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    typology_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    foundation_rows: list[dict[str, Any]],
    route_strength_rows: list[dict[str, Any]],
    contradiction_rows: list[dict[str, Any]],
    grokking_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    collinearity_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    validation_budget_rows: list[dict[str, Any]],
    evidence_gate_rows: list[dict[str, Any]],
    proof_rows: list[dict[str, Any]],
    tensor_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "atlas_id": atlas_id,
        "name": "versioned_computational_atlas",
        "definition": (
            "Durable learning memory = actual tensors + actual operators + checkpoint graph "
            "+ transformation graph + information matrices + route-strength estimates "
            "+ replayable evidence + evaluation-gated lineage."
        ),
        "state_components": {
            "theta": "model parameters; currently empty until runtimes emit model checkpoint artifacts",
            "omega": [
                npz_ref("path_information_matrix"),
                npz_ref("typology_information_matrix"),
                npz_ref("feature_information_matrix"),
                npz_ref("projection_information_matrix"),
                npz_ref("collinearity_information_matrix"),
                npz_ref("impact_information_matrix"),
                npz_ref("foundation_stress_information_matrix"),
                npz_ref("route_strength_information_matrix"),
                npz_ref("grokking_incubation_information_matrix"),
                npz_ref("operation_information_matrix"),
                npz_ref("validation_budget_information_matrix"),
                npz_ref("evidence_gate_information_matrix"),
            ],
            "phi": [
                json_ref("projection_memory_matrix.csv"),
                json_ref("collinearity_memory_matrix.csv"),
                json_ref("surface_surgery_matrix.csv"),
                json_ref("impact_field_matrix.csv"),
                npz_ref("projection_X"),
                npz_ref("collinearity_X"),
                npz_ref("surface_surgery_X"),
                npz_ref("impact_X"),
            ],
            "pi": [
                json_ref("typology_vector_field.csv"),
                npz_ref("typology_adjacency"),
                npz_ref("typology_delta"),
            ],
            "lambda": [
                json_ref("run_memory_matrix.csv"),
                json_ref("path_memory_matrix.csv"),
            ],
            "replay": "not yet emitted; future runs should store replay/core/hard/regression refs",
            "evaluation": [
                json_ref("operation_memory_matrix.csv"),
                json_ref("evidence_gate_matrix.csv"),
                json_ref("proof_carrying_paths.jsonl"),
                json_ref("attention_inputs.json"),
            ],
            "policy": [
                json_ref("next_runtime_policy.json"),
            ],
            "history": [
                json_ref("checkpoint_graph.json"),
                json_ref("operator_graph_edges.csv"),
                json_ref("relation_edges.csv"),
                json_ref("contradiction_graph.csv"),
            ],
            "uncertainty": [
                json_ref("open_hypotheses.json"),
                json_ref("foundation_stress_matrix.csv"),
                npz_ref("path_exploration_bias"),
                npz_ref("path_grokking_potential"),
                npz_ref("surface_surgery_X"),
                npz_ref("foundation_stress_X"),
                npz_ref("grokking_incubation_X"),
                npz_ref("operation_exploration_bias"),
                npz_ref("validation_budget_X"),
                npz_ref("evidence_gate_X"),
            ],
        },
        "stores": {
            "tensor_store": json_ref("numeric_memory_bundle.npz"),
            "tensor_manifest": json_ref("tensor_artifact_manifest.json"),
            "operator_graph": json_ref("operator_graph_edges.csv"),
            "checkpoint_graph": json_ref("checkpoint_graph.json"),
            "evaluation_store": json_ref("attention_inputs.json"),
            "strategy_store": json_ref("open_hypotheses.json"),
            "policy_store": json_ref("next_runtime_policy.json"),
            "impact_store": json_ref("impact_field_matrix.csv"),
            "validation_budget_store": json_ref("validation_budget_ledger.csv"),
            "evidence_gate_store": json_ref("evidence_gate_matrix.csv"),
            "proof_store": json_ref("proof_carrying_paths.jsonl"),
            "contradiction_store": json_ref("contradiction_graph.csv"),
            "grokking_store": json_ref("grokking_incubation_matrix.csv"),
        },
        "counts": {
            "runs": len(run_rows),
            "paths": len(path_rows),
            "typologies": len(typology_rows),
            "feature_space_records": len(feature_rows),
            "surface_surgery_records": len(surface_rows),
            "impact_field_records": len(impact_rows),
            "foundation_stress_records": len(foundation_rows),
            "route_strength_records": len(route_strength_rows),
            "contradiction_edges": len(contradiction_rows),
            "grokking_incubation_records": len(grokking_rows),
            "projection_records": len(projection_rows),
            "collinearity_records": len(collinearity_rows),
            "operations": len(operation_rows),
            "validation_worlds": len(validation_budget_rows),
            "evidence_gates": len(evidence_gate_rows),
            "proof_carrying_paths": len(proof_rows),
            "tensors": len(tensor_manifest.get("tensors", [])),
        },
        "branching_policy": {
            "resume": "load checkpoint_graph and tensor_store row ids",
            "challenge": "select an operator edge, branch from parent_checkpoint, replay with altered parameters",
            "compare": "compare child checkpoints through operation/evaluation tensors and relation edges",
            "reverse": "return to parent_checkpoint and restore pre-operator feature/path state",
        },
    }


def write_atlas_files(
    out_dir: Path,
    run_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    typology_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    foundation_rows: list[dict[str, Any]],
    route_strength_rows: list[dict[str, Any]],
    contradiction_rows: list[dict[str, Any]],
    grokking_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    collinearity_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    validation_budget_rows: list[dict[str, Any]],
    evidence_gate_rows: list[dict[str, Any]],
    proof_rows: list[dict[str, Any]],
    attn: dict[str, Any],
    numeric_schema: dict[str, Any],
) -> None:
    atlas_id = stable_id(
        "atlas",
        len(run_rows),
        len(path_rows),
        len(typology_rows),
        len(operation_rows),
        len(evidence_gate_rows),
        json.dumps(attn.get("external_score_context", []), sort_keys=True, default=str),
    )
    tensor_manifest = tensor_artifact_manifest(out_dir, numeric_schema)
    operators = operator_graph_edges(operation_rows)
    checkpoint = checkpoint_graph(run_rows, operation_rows, atlas_id)
    hypotheses = open_hypotheses(attn)
    policy = next_runtime_policy(attn)
    manifest = atlas_manifest(
        out_dir,
        atlas_id,
        run_rows,
        path_rows,
        typology_rows,
        feature_rows,
        surface_rows,
        impact_rows,
        foundation_rows,
        route_strength_rows,
        contradiction_rows,
        grokking_rows,
        projection_rows,
        collinearity_rows,
        operation_rows,
        validation_budget_rows,
        evidence_gate_rows,
        proof_rows,
        tensor_manifest,
    )
    (out_dir / "tensor_artifact_manifest.json").write_text(
        json.dumps(tensor_manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "operator_graph_edges.csv", operators)
    (out_dir / "checkpoint_graph.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "open_hypotheses.json").write_text(
        json.dumps(hypotheses, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "next_runtime_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "computational_atlas_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def attention_inputs(
    run_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    typology_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    foundation_rows: list[dict[str, Any]],
    route_strength_rows: list[dict[str, Any]],
    contradiction_rows: list[dict[str, Any]],
    grokking_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    collinearity_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    validation_budget_rows: list[dict[str, Any]],
    evidence_gate_rows: list[dict[str, Any]],
    proof_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def top(rows: list[dict[str, Any]], key: str, n: int = 12) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: f(r.get(key)) or -999, reverse=True)[:n]

    short = top([r for r in typology_rows if str(r.get("memory_role", "")).startswith("short")],
                "robust_score_shrunk", 10)
    medium = top([r for r in typology_rows if "medium" in str(r.get("memory_role"))],
                 "robust_score_shrunk", 10)
    long = top([r for r in typology_rows if str(r.get("memory_role")) == "long_term_prior"],
               "robust_score_shrunk", 10)
    hazards = top([r for r in typology_rows if "hazard" in str(r.get("memory_role")) or "anti" in str(r.get("memory_role"))],
                  "robust_score_shrunk", 10)
    frontier_paths = top([r for r in path_rows if f(r.get("robust_memory_score")) is not None],
                         "robust_memory_score", 20)
    coverage_gaps = top([r for r in coverage_rows if not int(r.get("tried") or 0)],
                        "gap_priority", 20)
    improving_vectors = top([r for r in vector_rows if (f(r.get("delta_score")) or 0.0) > 0.0],
                            "delta_score", 20)
    feature_hotspots = top([r for r in feature_rows if f(r.get("micro_priority")) is not None],
                           "micro_priority", 30)
    surface_surgery = top([r for r in surface_rows if f(r.get("grokking_priority")) is not None],
                          "grokking_priority", 30)
    impact_candidates = top([r for r in impact_rows if f(r.get("branch_priority")) is not None],
                            "branch_priority", 30)
    foundation_stress = top([r for r in foundation_rows if f(r.get("foundation_rethink_priority")) is not None],
                            "foundation_rethink_priority", 20)
    route_strength = top([r for r in route_strength_rows if f(r.get("branch_value")) is not None],
                         "branch_value", 20)
    contradictions = top([r for r in contradiction_rows if f(r.get("confidence")) is not None],
                         "confidence", 30)
    grokking = sorted(
        grokking_rows,
        key=lambda r: (str(r.get("record_status")) == "observed", f(r.get("time_budget_min")) or 0.0),
        reverse=True,
    )[:20]
    projection_candidates = top([r for r in projection_rows if f(r.get("grokking_potential")) is not None],
                                "grokking_potential", 20)
    collinearity_candidates = top(
        [r for r in collinearity_rows if f(r.get("compression_bias")) is not None],
        "compression_bias",
        20,
    )
    operation_candidates = top(
        [r for r in operation_rows if f(r.get("operation_strength")) is not None],
        "operation_strength",
        30,
    )
    validation_budget = top(
        [r for r in validation_budget_rows if f(r.get("reuse_pressure")) is not None],
        "reuse_pressure",
        16,
    )
    evidence_gates = top(
        [r for r in evidence_gate_rows if f(r.get("evidence_score")) is not None],
        "evidence_score",
        30,
    )
    proof_candidates = proof_rows[:40]

    score_context = sorted(score_rows, key=lambda r: f(r.get("private")) or -999, reverse=True)
    return {
        "schema": {
            "short_term": "fresh high-signal typologies with low observation count; retest, do not trust blindly",
            "medium_term": "repeatable but not yet structural; use as search bias",
            "long_term": "stable promoted priors with shrinkage; use as warm memory",
            "hazard": "decay/world-fragile or low-signal memories; keep as anti-priors",
        },
        "short_term_attention": short,
        "medium_term_attention": medium,
        "long_term_attention": long,
        "hazard_attention": hazards,
        "frontier_paths": frontier_paths,
        "coverage_gaps": coverage_gaps,
        "typology_vector_fields": improving_vectors,
        "feature_space_hotspots": feature_hotspots,
        "surface_surgery_attention": surface_surgery,
        "impact_field_attention": impact_candidates,
        "foundation_stress_attention": foundation_stress,
        "route_strength_attention": route_strength,
        "contradiction_attention": contradictions,
        "grokking_incubation_attention": grokking,
        "projection_memory": projection_candidates,
        "collinearity_memory": collinearity_candidates,
        "operation_attention": operation_candidates,
        "validation_budget_attention": validation_budget,
        "evidence_gate_attention": evidence_gates,
        "proof_carrying_paths": proof_candidates,
        "external_score_context": score_context,
        "numeric_memory_bundle": "numeric_memory_bundle.npz",
        "numeric_memory_schema": "numeric_memory_schema.json",
        "tensor_artifact_manifest": "tensor_artifact_manifest.json",
        "checkpoint_graph": "checkpoint_graph.json",
        "operator_graph": "operator_graph_edges.csv",
        "surface_surgery_matrix": "surface_surgery_matrix.csv",
        "impact_field_matrix": "impact_field_matrix.csv",
        "foundation_stress_matrix": "foundation_stress_matrix.csv",
        "route_strength_matrix": "route_strength_matrix.csv",
        "validation_budget_ledger": "validation_budget_ledger.csv",
        "evidence_gate_matrix": "evidence_gate_matrix.csv",
        "proof_carrying_paths_jsonl": "proof_carrying_paths.jsonl",
        "contradiction_graph": "contradiction_graph.csv",
        "grokking_incubation_matrix": "grokking_incubation_matrix.csv",
        "open_hypotheses": "open_hypotheses.json",
        "computational_atlas_manifest": "computational_atlas_manifest.json",
        "next_runtime_questions": [
            "Which short-term typologies reproduce in a different seed and terrain split?",
            "Which long-term priors still survive when route-carved against the current champion?",
            "Which high-support untried typology cells deserve random-sprout budget?",
            "Which feature-space vector fields identify local regions worth specialized models?",
            "Which local surface edits improve one region while creating global disagreement elsewhere?",
            "Which false agreements or false disagreements should be challenged before quarantine?",
            "Which high-ripple edits should remain branch-only instead of becoming foundation?",
            "Which stress signals indicate the current coordinate system or topology is wrong?",
            "Which route-strength priors are reusable and which are context-limited?",
            "Which accepted claims are contradicted, weakened, or revived by newer surface evidence?",
            "Which grokking incubators deserve more budget despite flat early validation?",
            "Which projection/reduction families widen useful signal instead of only sharpening spikes?",
            "Which collinear communities should be compressed, residualized, masked, or challenged?",
            "Which route/blend/path actions should become priors, anti-priors, or controlled ablations?",
            "Which validation worlds have been reused enough to require discounting or mutation?",
            "Which branches have enough evidence for A/B retest status, and which remain C/D hazards?",
            "Which proof-carrying paths lack the evidence needed for promotion?",
            "Which public-positive/private-negative candidates identify public-specific traps?",
            "Which decay hazards should become negative features, masks, or route exclusions?",
        ],
    }


def markdown_summary(attn: dict[str, Any], out_dir: Path) -> str:
    lines = [
        "# Memory Matrices",
        "",
        "This directory stores provenance-bearing memory, not just blend weights.",
        "",
        "Files:",
        f"- `{(out_dir / 'run_memory_matrix.csv').name}`: run/runtime/artifact evidence.",
        f"- `{(out_dir / 'path_memory_matrix.csv').name}`: path-level typology and survival evidence.",
        f"- `{(out_dir / 'typology_memory_matrix.csv').name}`: shrunk skill/family/transform memory.",
        f"- `{(out_dir / 'typology_coverage_matrix.csv').name}`: tried and untried typology cells.",
        f"- `{(out_dir / 'typology_vector_field.csv').name}`: local gradients over typology axes.",
        f"- `{(out_dir / 'feature_space_memory_matrix.csv').name}`: feature/topology micro observations.",
        f"- `{(out_dir / 'surface_surgery_matrix.csv').name}`: local/global agreement, disagreement, and reversible edits.",
        f"- `{(out_dir / 'impact_field_matrix.csv').name}`: ripple effects from each candidate surface/action move.",
        f"- `{(out_dir / 'foundation_stress_matrix.csv').name}`: foundation stress and rethink pressure.",
        f"- `{(out_dir / 'route_strength_matrix.csv').name}`: learned route/action value estimates.",
        f"- `{(out_dir / 'validation_budget_ledger.csv').name}`: validation-world reuse pressure and trust discounts.",
        f"- `{(out_dir / 'evidence_gate_matrix.csv').name}`: branch admission and promotion gate decisions.",
        f"- `{(out_dir / 'proof_carrying_paths.jsonl').name}`: evidence/risk certificates for paths and operations.",
        f"- `{(out_dir / 'contradiction_graph.csv').name}`: supports/contradicts/revives claim edges.",
        f"- `{(out_dir / 'grokking_incubation_matrix.csv').name}`: quarantined long-horizon branches allowed to continue without early shipping.",
        f"- `{(out_dir / 'projection_memory_matrix.csv').name}`: dimensionality-reduction/transform evidence.",
        f"- `{(out_dir / 'collinearity_memory_matrix.csv').name}`: feature redundancy and compression evidence.",
        f"- `{(out_dir / 'operation_memory_matrix.csv').name}`: route, blend, and path actions with weights/biases.",
        f"- `{(out_dir / 'relation_edges.csv').name}`: graph edges tying runs, paths, typologies, operations, and submissions.",
        f"- `{(out_dir / 'numeric_memory_bundle.npz').name}`: loadable numeric vectors, weights, biases, and information matrices.",
        f"- `{(out_dir / 'numeric_memory_schema.json').name}`: array names, row-id contracts, and column schema.",
        f"- `{(out_dir / 'tensor_artifact_manifest.json').name}`: loadable tensor catalog with shapes and semantics.",
        f"- `{(out_dir / 'operator_graph_edges.csv').name}`: replayable operators/actions as state transitions.",
        f"- `{(out_dir / 'checkpoint_graph.json').name}`: branchable runtime and operation checkpoints.",
        f"- `{(out_dir / 'open_hypotheses.json').name}`: challengeable next-run hypotheses.",
        f"- `{(out_dir / 'next_runtime_policy.json').name}`: operational budget, branch, grokking, and guard policy.",
        f"- `{(out_dir / 'computational_atlas_manifest.json').name}`: top-level computational atlas contract.",
        f"- `{(out_dir / 'attention_inputs.json').name}`: short/medium/long/hazard memory views.",
        "",
        "Top long-term priors:",
    ]
    for row in attn.get("long_term_attention", [])[:8]:
        lines.append(
            f"- {row.get('skill')} | {row.get('family')} | {row.get('transform')}: "
            f"{row.get('robust_score_shrunk')}"
        )
    lines.append("")
    lines.append("Top short-term candidates:")
    for row in attn.get("short_term_attention", [])[:8]:
        lines.append(
            f"- {row.get('skill')} | {row.get('family')} | {row.get('transform')}: "
            f"{row.get('robust_score_shrunk')}"
        )
    lines.append("")
    lines.append("Hazards / anti-priors:")
    for row in attn.get("hazard_attention", [])[:8]:
        lines.append(
            f"- {row.get('skill')} | {row.get('family')} | {row.get('transform')}: "
            f"{row.get('memory_role')}"
        )
    lines.append("")
    lines.append("High-support untried typology gaps:")
    for row in attn.get("coverage_gaps", [])[:8]:
        lines.append(
            f"- {row.get('skill')} | {row.get('family')} | {row.get('transform')}: "
            f"gap_priority={row.get('gap_priority')}"
        )
    lines.append("")
    lines.append("Strong typology vector fields:")
    for row in attn.get("typology_vector_fields", [])[:8]:
        lines.append(
            f"- change {row.get('changed_axis')} {row.get('src_axis_value')} -> {row.get('dst_axis_value')}: "
            f"delta={row.get('delta_score')}"
        )
    lines.append("")
    lines.append("Surface surgery attention:")
    for row in attn.get("surface_surgery_attention", [])[:8]:
        lines.append(
            f"- {row.get('source_kind')} {row.get('coordinate')}: "
            f"grokking={row.get('grokking_priority')} action={row.get('suggested_surgery')}"
        )
    lines.append("")
    lines.append("High-ripple impact fields:")
    for row in attn.get("impact_field_attention", [])[:8]:
        lines.append(
            f"- {row.get('source_kind')} {row.get('coordinate')}: "
            f"branch={row.get('branch_priority')} status={row.get('move_status')}"
        )
    lines.append("")
    lines.append("Foundation stress:")
    for row in attn.get("foundation_stress_attention", [])[:8]:
        lines.append(
            f"- run={row.get('run')} source={row.get('source_group')}: "
            f"stress={row.get('stress_score')} rec={row.get('recommendation')}"
        )
    lines.append("")
    lines.append("Reusable route strengths:")
    for row in attn.get("route_strength_attention", [])[:8]:
        lines.append(
            f"- {row.get('terrain_signature')} {row.get('operator_type')}: "
            f"value={row.get('branch_value')} status={row.get('status')}"
        )
    lines.append("")
    lines.append("Validation budget pressure:")
    for row in attn.get("validation_budget_attention", [])[:8]:
        lines.append(
            f"- {row.get('validation_world')}: reuse={row.get('reuse_pressure')} "
            f"discount={row.get('trust_discount')} policy={row.get('policy')}"
        )
    lines.append("")
    lines.append("Evidence gates:")
    for row in attn.get("evidence_gate_attention", [])[:8]:
        lines.append(
            f"- {row.get('source_kind')} {row.get('coordinate')}: "
            f"grade={row.get('evidence_grade')} decision={row.get('decision')} "
            f"score={row.get('evidence_score')}"
        )
    lines.append("")
    lines.append("Grokking incubators:")
    for row in attn.get("grokking_incubation_attention", [])[:8]:
        lines.append(
            f"- {row.get('run')} {row.get('branch_mode')}: "
            f"status={row.get('record_status')} budget={row.get('time_budget_min')} ship={row.get('ship_eligible')}"
        )
    lines.append("")
    lines.append("Projection / reduction candidates:")
    for row in attn.get("projection_memory", [])[:8]:
        lines.append(
            f"- {row.get('method')} via {row.get('source_axis')}={row.get('source_value')}: "
            f"grokking={row.get('grokking_potential')} rec={row.get('recommendation')}"
        )
    lines.append("")
    lines.append("Collinearity / compression candidates:")
    for row in attn.get("collinearity_memory", [])[:8]:
        lines.append(
            f"- run={row.get('run')} community={row.get('community')}: "
            f"compression_bias={row.get('compression_bias')} rec={row.get('recommendation')}"
        )
    lines.append("")
    lines.append("Top operation attention:")
    for row in attn.get("operation_attention", [])[:8]:
        lines.append(
            f"- {row.get('operation_type')} {row.get('operation_name')}: "
            f"strength={row.get('operation_strength')}"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="compile long/medium/short-term memory matrices")
    ap.add_argument("--fleet-dir", default=str(FLEET_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--score", action="append", default=[], help="filename|private|public|description")
    ap.add_argument("--scores-csv", action="append", default=[])
    args = ap.parse_args(argv)

    fleet_dir = Path(args.fleet_dir)
    out_dir = Path(args.out)
    route_index = tg.load_route_manifests(fleet_dir)
    runs = tg.load_runs(fleet_dir)
    scores = score_rows_from_args(args)
    score_by_run = scores_by_run(scores, runs, route_index)

    run_rows = [run_record(d, score_by_run) for d in run_dirs(fleet_dir)]
    path_rows: list[dict[str, Any]] = []
    for d in run_dirs(fleet_dir):
        path_rows.extend(path_records(d))
    typology_rows = aggregate_typology(path_rows)
    coverage_rows = typology_coverage_matrix(typology_rows)
    vector_rows = typology_vector_field(typology_rows)
    feature_rows: list[dict[str, Any]] = []
    for d in run_dirs(fleet_dir):
        feature_rows.extend(feature_space_records(d))
    projection_rows = projection_memory_matrix(path_rows)
    collinearity_rows = collinearity_memory_matrix(fleet_dir)
    operation_rows = operation_records(fleet_dir, path_rows, scores)
    surface_rows = surface_surgery_matrix(feature_rows, collinearity_rows, projection_rows, operation_rows)
    impact_rows = impact_field_matrix(surface_rows)
    foundation_rows = foundation_stress_matrix(surface_rows, impact_rows)
    route_strength_rows = route_strength_matrix(operation_rows, impact_rows)
    contradiction_rows = contradiction_graph(surface_rows, impact_rows, projection_rows, operation_rows)
    validation_budget_rows = validation_budget_ledger(run_rows, path_rows, operation_rows, scores)
    evidence_gate_rows = evidence_gate_matrix(impact_rows, operation_rows, route_strength_rows, contradiction_rows)
    proof_rows = proof_carrying_paths(path_rows, operation_rows, evidence_gate_rows)
    grokking_rows = grokking_incubation_matrix(fleet_dir)
    edges = relation_edges(run_rows, path_rows, scores, route_index, operation_rows)
    attn = attention_inputs(
        run_rows,
        path_rows,
        typology_rows,
        coverage_rows,
        vector_rows,
        feature_rows,
        surface_rows,
        impact_rows,
        foundation_rows,
        route_strength_rows,
        contradiction_rows,
        grokking_rows,
        projection_rows,
        collinearity_rows,
        operation_rows,
        validation_budget_rows,
        evidence_gate_rows,
        proof_rows,
        scores,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "run_memory_matrix.csv", run_rows)
    write_csv(out_dir / "path_memory_matrix.csv", path_rows)
    write_csv(out_dir / "typology_memory_matrix.csv", typology_rows)
    write_csv(out_dir / "typology_coverage_matrix.csv", coverage_rows)
    write_csv(out_dir / "typology_vector_field.csv", vector_rows)
    write_csv(out_dir / "feature_space_memory_matrix.csv", feature_rows)
    write_csv(out_dir / "surface_surgery_matrix.csv", surface_rows)
    write_csv(out_dir / "impact_field_matrix.csv", impact_rows)
    write_csv(out_dir / "foundation_stress_matrix.csv", foundation_rows)
    write_csv(out_dir / "route_strength_matrix.csv", route_strength_rows)
    write_csv(out_dir / "validation_budget_ledger.csv", validation_budget_rows)
    write_csv(out_dir / "evidence_gate_matrix.csv", evidence_gate_rows)
    write_jsonl(out_dir / "proof_carrying_paths.jsonl", proof_rows)
    write_csv(out_dir / "contradiction_graph.csv", contradiction_rows)
    write_csv(out_dir / "grokking_incubation_matrix.csv", grokking_rows)
    write_csv(out_dir / "projection_memory_matrix.csv", projection_rows)
    write_csv(out_dir / "collinearity_memory_matrix.csv", collinearity_rows)
    write_csv(out_dir / "operation_memory_matrix.csv", operation_rows)
    write_csv(out_dir / "relation_edges.csv", edges)
    numeric_schema = write_numeric_bundle(
        out_dir,
        path_rows,
        typology_rows,
        vector_rows,
        feature_rows,
        surface_rows,
        impact_rows,
        foundation_rows,
        route_strength_rows,
        grokking_rows,
        projection_rows,
        collinearity_rows,
        operation_rows,
        validation_budget_rows,
        evidence_gate_rows,
    )
    write_atlas_files(
        out_dir,
        run_rows,
        path_rows,
        typology_rows,
        feature_rows,
        surface_rows,
        impact_rows,
        foundation_rows,
        route_strength_rows,
        contradiction_rows,
        grokking_rows,
        projection_rows,
        collinearity_rows,
        operation_rows,
        validation_budget_rows,
        evidence_gate_rows,
        proof_rows,
        attn,
        numeric_schema,
    )
    (out_dir / "attention_inputs.json").write_text(
        json.dumps(attn, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "memory_summary.md").write_text(markdown_summary(attn, out_dir), encoding="utf-8")

    print(markdown_summary(attn, out_dir))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
