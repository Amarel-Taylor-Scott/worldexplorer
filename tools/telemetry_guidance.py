#!/usr/bin/env python3
"""Turn leaderboard outcomes plus local artifacts into next-iteration guidance.

The scoreboard says what won. The local telemetry explains what should be
copied, downweighted, carved, or used as a negative witness in the next batch.

Input scores are intentionally explicit so this can be run after deadline from
the visible Kaggle rows without needing to contact Kaggle:

  python tools/telemetry_guidance.py \
    --score "submission_x.csv|0.08577|0.06599|route carve x" \
    --score "submission_y.csv|0.08400|0.06650|route carve y"

The score format is filename|private|public|description. A CSV with columns
filename, private, public, description is also accepted via --scores-csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FLEET_DIR = WORKSPACE / "kaggle" / "fleet"
DEFAULT_OUT = FLEET_DIR / "telemetry_guidance"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def score_arg(value: str) -> dict[str, Any]:
    parts = value.split("|", 3)
    if len(parts) < 3:
        sys.exit("--score must be filename|private|public|description")
    filename, private, public = parts[:3]
    description = parts[3] if len(parts) == 4 else ""
    return {
        "filename": Path(filename.strip()).name,
        "private": as_float(private.strip()),
        "public": as_float(public.strip()),
        "description": description.strip(),
    }


def is_external_reference(score: dict[str, Any]) -> bool:
    """True for leaderboard context that should not drive WE source weights."""
    haystack = f"{score.get('filename', '')} {score.get('description', '')}".lower()
    markers = (
        "external reference",
        "not worldexplorer",
        "not world explorer",
        "arunemble",
    )
    return any(marker in haystack for marker in markers)


def is_weight_eligible(score: dict[str, Any]) -> bool:
    return not is_external_reference(score)


def load_scores_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        filename = row.get("filename") or row.get("file") or row.get("submission")
        if not filename:
            continue
        out.append({
            "filename": Path(filename).name,
            "private": as_float(row.get("private")),
            "public": as_float(row.get("public")),
            "description": row.get("description") or row.get("message") or "",
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def load_route_manifests(fleet_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest in fleet_dir.glob("route_carves*/route_carve_manifest.csv"):
        for row in read_csv_rows(manifest):
            candidate = row.get("candidate") or ""
            path = row.get("path") or ""
            names = {Path(path).name} if path else set()
            if candidate:
                names.add(f"submission_{candidate}.csv")
                names.add(candidate)
            for name in names:
                if not name:
                    continue
                item = dict(row)
                item["manifest_path"] = str(manifest)
                index[Path(name).name] = item
    return index


def extract_seed(kernel: Path) -> int | None:
    if not kernel.exists():
        return None
    text = kernel.read_text(errors="ignore")
    hits = re.findall(r"CFG\.SEED\s*=\s*(\d+)", text)
    return int(hits[-1]) if hits else None


def log_health(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"log_path": None, "log_bytes": None, "log_runtime_sec": None,
                "log_stderr_events": None, "log_tracebacks": None}
    text = path.read_text(errors="ignore")
    times = [as_float(x) for x in re.findall(r'"time"\s*:\s*([0-9.]+)', text)]
    times_f = [x for x in times if x is not None]
    return {
        "log_path": str(path),
        "log_bytes": path.stat().st_size,
        "log_runtime_sec": max(times_f) if times_f else None,
        "log_stderr_events": text.count('"stream_name":"stderr"'),
        "log_tracebacks": text.count("Traceback"),
    }


def load_runs(fleet_dir: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(p for p in fleet_dir.iterdir() if p.is_dir()):
        out = run_dir / "output"
        if not out.exists():
            continue
        summary = load_json(out / "explorer_run_summary.json", {})
        governor = load_json(out / "complexity_governor.json", {})
        criticality = load_json(out / "regime_criticality.json", {})
        contract = load_json(out / "submission_contract.json", {})
        forensic = load_json(out / "forensic_selection_decision.json", {})
        extrema = load_json(out / "extrema_reconciliation.json", {})
        metabolism = summary.get("metabolism") or {}
        hardware = summary.get("hardware") or {}
        log_path = next(out.glob("*.log"), None)
        runs[run_dir.name] = {
            "run": run_dir.name,
            "seed": extract_seed(run_dir / "kernel.py"),
            "output_dir": str(out),
            "local_submission": str(out / "submission.csv") if (out / "submission.csv").exists() else None,
            "forward_blend_corr": summary.get("forward_blend_corr"),
            "sealed_holdout_corr": summary.get("sealed_holdout_corr"),
            "selector": (summary.get("forensics") or {}).get("shipped_selector") or forensic.get("winner"),
            "width_decay_corr": governor.get("width_decay_corr"),
            "criticality": criticality.get("criticality"),
            "members_down_weighted": criticality.get("members_down_weighted"),
            "extrema_method": (extrema.get("selected") or {}).get("method"),
            "elapsed_min": metabolism.get("elapsed_min"),
            "time_budget_min": metabolism.get("time_budget_min"),
            "hardware_schedule": hardware.get("schedule"),
            "gpus": hardware.get("gpus"),
            "rows": contract.get("rows"),
            "pred_std": contract.get("pred_std"),
            **log_health(log_path),
        }
    return runs


def run_from_description(description: str, runs: dict[str, dict[str, Any]]) -> str | None:
    desc = description.lower()
    for name in sorted(runs, key=len, reverse=True):
        if name.lower() in desc:
            return name
    seed_hits = re.findall(r"\bs(\d+)(?:\b|w)", desc)
    if seed_hits:
        seed_to_run = {str(info.get("seed")): name for name, info in runs.items() if info.get("seed") is not None}
        for seed in seed_hits:
            if seed in seed_to_run:
                return seed_to_run[seed]
    return None


def source_runs_from_description(description: str, runs: dict[str, dict[str, Any]]) -> list[str]:
    desc = description.lower()
    found = [name for name in runs if name.lower() in desc]
    seed_hits = re.findall(r"\bs(\d+)(?:\b|w)", desc)
    if seed_hits:
        seed_to_runs: dict[str, list[str]] = defaultdict(list)
        for name, info in runs.items():
            seed = info.get("seed")
            if seed is not None:
                seed_to_runs[str(seed)].append(name)
        for seed in seed_hits:
            found.extend(seed_to_runs.get(seed, []))
    out: list[str] = []
    seen: set[str] = set()
    for name in found:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def avg_run_value(runs: dict[str, dict[str, Any]], names: list[str], key: str) -> float | None:
    vals = [as_float(runs.get(name, {}).get(key)) for name in names]
    vals_f = [x for x in vals if x is not None]
    if not vals_f:
        return None
    return sum(vals_f) / len(vals_f)


def classify(row: dict[str, Any], champion_private: float, champion_public: float) -> str:
    if row.get("is_external_reference"):
        return "external_reference_only"
    private = row.get("private")
    public = row.get("public")
    if private is None:
        return "missing_score"
    if private >= champion_private - 0.00008:
        return "promote_as_teacher"
    if private >= champion_private - 0.0006:
        return "near_champion_neighbor"
    if public is not None and public > champion_public and private < champion_private:
        return "public_stabilizer_private_lag"
    if private < champion_private - 0.006:
        return "negative_witness_or_demote"
    if private < champion_private - 0.002:
        return "weak_route_rehabilitate_only"
    return "useful_witness"


def softmax(rows: list[dict[str, Any]], key: str, temperature: float) -> dict[int, float]:
    vals: list[float] = []
    for row in rows:
        val = as_float(row.get(key))
        vals.append(val if val is not None else -1e9)
    finite = [v for v in vals if v > -1e8]
    if not finite:
        return {i: 0.0 for i in range(len(rows))}
    top = max(finite)
    raw = [math.exp(max(-60.0, (v - top) / max(temperature, 1e-9))) if v > -1e8 else 0.0 for v in vals]
    total = sum(raw) or 1.0
    return {i: raw[i] / total for i in range(len(rows))}


def add_bucket(bucket: dict[str, float], key: Any, value: float) -> None:
    if key is None or key == "":
        return
    bucket[str(key)] += float(value)


def normalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in bucket.values()) or 1.0
    return {k: round(max(0.0, v) / total, 6) for k, v in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)}


def annotate_scores(
    scores: list[dict[str, Any]],
    route_index: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    champion_private: float,
    champion_public: float,
    public_penalty: float,
    width_decay_penalty: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score in scores:
        filename = Path(score["filename"]).name
        manifest = route_index.get(filename)
        source_runs: list[str] = []
        if manifest:
            route = manifest.get("route")
            if route:
                source_runs = [route]
        elif filename == "submission.csv":
            run_name = run_from_description(score.get("description", ""), runs)
            if run_name:
                source_runs = [run_name]
        if not source_runs:
            source_runs = source_runs_from_description(score.get("description", ""), runs)

        private = score.get("private")
        public = score.get("public")
        private_delta = None if private is None else private - champion_private
        public_delta = None if public is None else public - champion_public
        width_decay = avg_run_value(runs, source_runs, "width_decay_corr")
        sealed = avg_run_value(runs, source_runs, "sealed_holdout_corr")
        forward = avg_run_value(runs, source_runs, "forward_blend_corr")
        generalization_score = private
        if generalization_score is not None and public is not None:
            generalization_score -= public_penalty * max(0.0, champion_public - public)
        if generalization_score is not None and width_decay is not None:
            generalization_score -= width_decay_penalty * max(0.0, width_decay - 0.14)

        first_run = runs.get(source_runs[0], {}) if source_runs else {}
        row: dict[str, Any] = {
            "filename": filename,
            "description": score.get("description", ""),
            "is_external_reference": is_external_reference(score),
            "weight_eligible": is_weight_eligible(score),
            "private": private,
            "public": public,
            "private_delta_vs_champion": private_delta,
            "public_delta_vs_champion": public_delta,
            "public_private_gap": None if private is None or public is None else private - public,
            "candidate": manifest.get("candidate") if manifest else None,
            "source_runs": ",".join(source_runs),
            "source_run_count": len(source_runs),
            "rehab_stage": manifest.get("rehab_stage") if manifest else ("blend" if "blend" in filename.lower() else "direct_run"),
            "lens": manifest.get("lens") if manifest else None,
            "alpha": as_float(manifest.get("alpha")) if manifest else None,
            "corr_champion_route": as_float(manifest.get("corr_champion_route")) if manifest else None,
            "corr_candidate_champion": as_float(manifest.get("corr_candidate_champion")) if manifest else None,
            "route_information_gain": as_float(manifest.get("route_information_gain")) if manifest else None,
            "route_manifest": manifest.get("manifest_path") if manifest else None,
            "local_path": manifest.get("path") if manifest else first_run.get("local_submission"),
            "source_forward_blend_corr": forward,
            "source_sealed_holdout_corr": sealed,
            "source_width_decay_corr": width_decay,
            "source_selector": first_run.get("selector"),
            "source_criticality": avg_run_value(runs, source_runs, "criticality"),
            "source_elapsed_min": avg_run_value(runs, source_runs, "elapsed_min"),
            "source_log_runtime_sec": avg_run_value(runs, source_runs, "log_runtime_sec"),
            "source_log_stderr_events": avg_run_value(runs, source_runs, "log_stderr_events"),
            "source_log_tracebacks": avg_run_value(runs, source_runs, "log_tracebacks"),
            "generalization_score": generalization_score,
        }
        row["action"] = classify(row, champion_private, champion_public)
        rows.append(row)

    private_rows = [row for row in rows if row.get("weight_eligible")]
    private_weights = softmax(private_rows, "private", temperature=0.0014)
    gen_weights = softmax(private_rows, "generalization_score", temperature=0.0016)
    private_by_id = {id(row): private_weights[i] for i, row in enumerate(private_rows)}
    gen_by_id = {id(row): gen_weights[i] for i, row in enumerate(private_rows)}
    for i, row in enumerate(rows):
        row["private_replay_weight"] = round(private_by_id.get(id(row), 0.0), 6)
        row["generalization_weight"] = round(gen_by_id.get(id(row), 0.0), 6)
    return rows


def build_guidance(rows: list[dict[str, Any]], champion: dict[str, Any]) -> dict[str, Any]:
    source_private: dict[str, float] = defaultdict(float)
    source_general: dict[str, float] = defaultdict(float)
    stage_bias: dict[str, float] = defaultdict(float)
    lens_bias: dict[str, float] = defaultdict(float)
    alpha_bias: dict[str, float] = defaultdict(float)
    avoid: list[str] = []
    teachers: list[str] = []

    for row in rows:
        private_w = float(row.get("private_replay_weight") or 0.0)
        general_w = float(row.get("generalization_weight") or 0.0)
        sources = [x for x in str(row.get("source_runs") or "").split(",") if x]
        per_source = 1.0 / max(1, len(sources))
        for source in sources:
            source_private[source] += private_w * per_source
            source_general[source] += general_w * per_source
        add_bucket(stage_bias, row.get("rehab_stage"), private_w)
        add_bucket(lens_bias, row.get("lens"), private_w)
        add_bucket(alpha_bias, row.get("alpha"), private_w)
        if row.get("action") == "negative_witness_or_demote":
            avoid.append(str(row["filename"]))
        if row.get("action") in ("promote_as_teacher", "near_champion_neighbor"):
            teachers.append(str(row["filename"]))

    source_general_norm = normalize_bucket(source_general)
    source_private_norm = normalize_bucket(source_private)
    stage_norm = normalize_bucket(stage_bias)
    lens_norm = normalize_bucket(lens_bias)
    alpha_norm = normalize_bucket(alpha_bias)
    best_stage = next(iter(stage_norm), None)
    best_lens = next(iter(lens_norm), None)

    width_values = [
        as_float(row.get("source_width_decay_corr"))
        for row in rows
        if float(row.get("generalization_weight") or 0.0) > 0.05
    ]
    width_values = [x for x in width_values if x is not None]
    weighted_width_decay = sum(width_values) / len(width_values) if width_values else None

    gov_scale = 1.0 if weighted_width_decay is not None and weighted_width_decay > 0.13 else 0.75
    gov_max = 0.08 if weighted_width_decay is not None and weighted_width_decay > 0.15 else 0.06

    return {
        "champion": champion,
        "candidate_weights": {
            row["filename"]: {
                "private_replay": row["private_replay_weight"],
                "generalization": row["generalization_weight"],
                "action": row["action"],
                "source_runs": row.get("source_runs"),
            }
            for row in rows
        },
        "source_run_weights": {
            "private_replay": source_private_norm,
            "generalization": source_general_norm,
        },
        "route_carve_biases": {
            "rehab_stage": stage_norm,
            "lens": lens_norm,
            "alpha": alpha_norm,
            "preferred_stage": best_stage,
            "preferred_lens": best_lens,
        },
        "sprout_config_biases": {
            "GOV_LAMBDA_SCALE": {
                "value_hint": gov_scale,
                "reason": "increase complexity pressure when winning routes come from high width-decay sources",
            },
            "GOV_LAMBDA_MAX": {
                "value_hint": gov_max,
                "reason": "cap pressure high enough to punish brittle width without erasing useful route novelty",
            },
            "EXTREMA_RECONCILE": {
                "value_hint": True,
                "reason": "route-carve winners are prediction-space shape edits; keep extrema reconciliation active",
            },
            "WIDTH_BIAS_START": {
                "value_hint": [0.55, 0.80],
                "reason": "bias toward wide teachers but let conflict/residual carving handle local sharpness",
            },
        },
        "teachers": teachers,
        "negative_witnesses": avoid,
        "notes": [
            "Use private_replay weights when mining this completed competition for lessons.",
            "Use generalization weights when choosing the next live submission blend.",
            "External reference rows are kept as aspiration/context and receive zero direct source weight.",
            "Do not give raw blend failures zero value: keep them as anti-priors for naive rank averaging.",
        ],
    }


def markdown_report(rows: list[dict[str, Any]], guidance: dict[str, Any]) -> str:
    lines = [
        "# Telemetry Guidance",
        "",
        "## Next Weights",
        "",
        "Source weights for generalization:",
    ]
    for source, weight in guidance["source_run_weights"]["generalization"].items():
        lines.append(f"- {source}: {weight:.6f}")
    lines.extend(["", "Route-carve biases:"])
    for group in ("rehab_stage", "lens", "alpha"):
        vals = guidance["route_carve_biases"].get(group, {})
        top = ", ".join(f"{k}={v:.4f}" for k, v in list(vals.items())[:5])
        lines.append(f"- {group}: {top or 'none'}")
    lines.extend(["", "## Candidate Actions", ""])
    for row in sorted(rows, key=lambda r: float(r.get("private") or -1), reverse=True):
        lines.append(
            f"- {row['filename']}: {row['action']}; "
            f"private={row.get('private')}, public={row.get('public')}, "
            f"generalization_weight={row.get('generalization_weight')}, "
            f"source={row.get('source_runs') or 'unknown'}, "
            f"eligible={row.get('weight_eligible')}"
        )
    lines.extend(["", "## Config Biases", ""])
    for key, item in guidance["sprout_config_biases"].items():
        lines.append(f"- {key}: {item['value_hint']} - {item['reason']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build next-iteration guidance from scores and telemetry")
    ap.add_argument("--score", action="append", default=[], help="filename|private|public|description")
    ap.add_argument("--scores-csv", action="append", default=[], help="CSV with filename,private,public,description")
    ap.add_argument("--fleet-dir", default=str(FLEET_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--champion-json", default=str(FLEET_DIR / "champion.json"))
    ap.add_argument("--champion-private", type=float, default=None)
    ap.add_argument("--champion-public", type=float, default=None)
    ap.add_argument("--public-penalty", type=float, default=0.25)
    ap.add_argument("--width-decay-penalty", type=float, default=0.003)
    args = ap.parse_args(argv)

    scores: list[dict[str, Any]] = []
    for path_s in args.scores_csv:
        scores.extend(load_scores_csv(Path(path_s)))
    scores.extend(score_arg(s) for s in args.score)
    if not scores:
        sys.exit("provide --score or --scores-csv")

    fleet_dir = Path(args.fleet_dir)
    champion = load_json(Path(args.champion_json), {})
    champion_private = args.champion_private
    champion_public = args.champion_public
    if champion_private is None:
        champion_private = as_float(champion.get("private"))
    if champion_public is None:
        champion_public = as_float(champion.get("public"))
    eligible_scores = [s for s in scores if is_weight_eligible(s)]
    champion_pool = eligible_scores or scores
    if champion_private is None:
        champion_private = max((s["private"] for s in champion_pool if s.get("private") is not None), default=None)
    if champion_public is None:
        champion_public = max((s["public"] for s in champion_pool if s.get("public") is not None), default=None)
    if champion_private is None or champion_public is None:
        sys.exit("could not infer champion scores")
    champion = {
        "private": champion_private,
        "public": champion_public,
        "path": champion.get("path"),
        "name": champion.get("name"),
    }

    route_index = load_route_manifests(fleet_dir)
    runs = load_runs(fleet_dir)
    rows = annotate_scores(
        scores,
        route_index,
        runs,
        champion_private,
        champion_public,
        args.public_penalty,
        args.width_decay_penalty,
    )
    guidance = build_guidance(rows, champion)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "telemetry_guidance_report.csv", rows)
    (out_dir / "next_iteration_weights.json").write_text(
        json.dumps(guidance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "telemetry_guidance.md").write_text(
        markdown_report(rows, guidance), encoding="utf-8"
    )
    print(markdown_report(rows, guidance))
    print(f"wrote {out_dir / 'telemetry_guidance_report.csv'}")
    print(f"wrote {out_dir / 'next_iteration_weights.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
