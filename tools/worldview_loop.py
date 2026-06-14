#!/usr/bin/env python3
"""Evidence-driven DRW worldview loop.

This is the project-local analogue of a Ralph-style loop, but for Kaggle
research instead of coding stories. It persists state, polls/collects runs,
reads artifacts, compares leaderboard scores to the champion, and writes a
small next-action plan. It is deliberately conservative: weak probes are first
reinterpreted, downweighted, carved, or morphed. Full quarantine/retirement is
reserved for repeated contamination.

Typical use:
  python tools/worldview_loop.py once
  python tools/worldview_loop.py once --collect
  python tools/worldview_loop.py status

The loop does not launch new runs by default. Use --allow-launch only after the
plan says a launch is justified; this avoids blind hyperparameter churn.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FLEET_DIR = WORKSPACE / "kaggle" / "fleet"
STATE_PATH = FLEET_DIR / "worldview_loop_state.json"
COMP = "drw-crypto-market-prediction"

CHAMPION = {
    "name": "route-carve-forager-conflict",
    "private": 0.08577,
    "public": 0.06599,
    "path": str(
        FLEET_DIR / "route_carves_hierarchy2"
        / "submission_bio-sprout-03-forager_pow15_conflict_sub_a0.12.csv"
    ),
}

DEFAULT_MANIFESTS = [FLEET_DIR / "bio-sprout_manifest.json"]
DEFAULT_MEMBERS = ["cpu-extrema", "gpu-governor-v2"]


@dataclass
class RunScore:
    ref: str
    filename: str
    description: str
    public: float | None
    private: float | None


def run(cmd: list[str], *, cwd: Path = ROOT, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=check)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = load_json(path, {})
    members = data.get("members", []) if isinstance(data, dict) else data
    return members if isinstance(members, list) else []


def tracked_runs(manifests: list[Path], members: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in manifests:
        for m in load_manifest(path):
            if isinstance(m, dict) and m.get("name"):
                r = dict(m)
                r["manifest"] = str(path)
                runs.append(r)
    for name in members:
        if name.endswith("-v2"):
            # Manually staged older runs may not be in tools/fleet.py.
            runs.append({"name": name, "slug": f"drw-wx-{name}", "manual": True})
        else:
            runs.append({"name": name, "slug": f"drw-wx-{name}", "manual": True})
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in runs:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        r.setdefault("slug", f"drw-wx-{r['name']}")
        out.append(r)
    return out


def kaggle_user() -> str:
    env_path = Path.home() / ".config" / "worldexplorer" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("KAGGLE_USERNAME="):
                return line.split("=", 1)[1].strip()
    return "taylorsamarel"


def status_one(user: str, slug: str) -> str:
    res = run(["kaggle", "kernels", "status", f"{user}/{slug}"])
    txt = (res.stdout + res.stderr).strip()
    if "KernelWorkerStatus." in txt:
        return txt.split("KernelWorkerStatus.", 1)[1].split('"', 1)[0]
    return "UNKNOWN"


def collect_one(user: str, r: dict[str, Any], *, submit: bool) -> str:
    name = r["name"]
    dest = FLEET_DIR / name / "output"
    dest.mkdir(parents=True, exist_ok=True)
    res = run(["kaggle", "kernels", "output", f"{user}/{r['slug']}", "-p", str(dest)])
    if res.returncode != 0:
        return "collect_failed"
    sub = dest / "submission.csv"
    if submit and sub.exists():
        msg = f"loop {name}"
        run(["kaggle", "competitions", "submit", "-c", COMP, "-f", str(sub), "-m", msg])
        return "collected_submitted"
    return "collected" if sub.exists() else "collected_no_submission"


def parse_submissions(text: str) -> list[RunScore]:
    rows: list[RunScore] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        ref = parts[0]
        status_i = next((i for i, p in enumerate(parts) if p.startswith("SubmissionStatus.")), None)
        if status_i is None or len(parts) <= status_i + 2:
            continue
        filename = parts[1]
        desc = " ".join(parts[4:status_i])
        try:
            public = float(parts[status_i + 1])
            private = float(parts[status_i + 2])
        except Exception:
            public = private = None
        rows.append(RunScore(ref, filename, desc, public, private))
    return rows


def latest_scores() -> list[RunScore]:
    res = run(["kaggle", "competitions", "submissions", COMP])
    return parse_submissions(res.stdout + res.stderr)


def score_for_run(name: str, scores: list[RunScore]) -> RunScore | None:
    exact_desc = {
        f"fleet {name}".lower(),
        f"loop {name}".lower(),
        f"drw wx {name}".lower(),
        f"drw wx {name} raw".lower(),
    }
    for s in scores:
        if s.description.lower() in exact_desc:
            return s
    needles = [name, name.replace("-", " ")]
    for s in scores:
        blob = f"{s.filename} {s.description}".lower()
        desc = s.description.lower()
        if "route carve" in desc:
            continue
        if any(n.lower() in blob for n in needles):
            return s
    return None


def read_artifacts(name: str) -> dict[str, Any]:
    out = FLEET_DIR / name / "output"
    summary = load_json(out / "explorer_run_summary.json", {})
    extrema = load_json(out / "extrema_reconciliation.json", {})
    governor = load_json(out / "complexity_governor.json", {})
    criticality = load_json(out / "regime_criticality.json", {})
    forensic = load_json(out / "forensic_selection_decision.json", {})
    return {
        "output_dir": str(out),
        "has_output": out.exists(),
        "forward": summary.get("forward_blend_corr"),
        "sealed": summary.get("sealed_holdout_corr"),
        "selector": (summary.get("forensics") or {}).get("shipped_selector") or forensic.get("winner"),
        "extrema": extrema.get("selected"),
        "extrema_forward": extrema.get("forward_corr"),
        "width_decay_corr": governor.get("width_decay_corr"),
        "criticality": criticality.get("criticality"),
        "members_down_weighted": criticality.get("members_down_weighted"),
        "width_elasticity": width_elasticity(out),
    }


def _float_list_from_csv(path: Path, column: str, *, decision_contains: str | None = None) -> list[float]:
    if not path.exists():
        return []
    vals: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if decision_contains is not None and decision_contains not in str(row.get("decision", "")):
                    continue
                raw = row.get(column)
                if raw is None or raw == "":
                    continue
                try:
                    x = float(raw)
                except Exception:
                    continue
                if math.isfinite(x):
                    vals.append(x)
    except Exception:
        return []
    return vals


def _cv(vals: list[float]) -> float | None:
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return math.sqrt(var) / (abs(mean) + 1e-12)


def width_elasticity(out: Path) -> dict[str, Any]:
    lessons = out / "explorer_lessons.csv"
    shipping = out / "shipping_court_report.csv"
    promoted_width = _float_list_from_csv(lessons, "width", decision_contains="promote")
    lesson_wf_width = _float_list_from_csv(lessons, "wf_width")
    shipped_width = _float_list_from_csv(shipping, "width")
    cvs = [x for x in (_cv(promoted_width), _cv(lesson_wf_width), _cv(shipped_width)) if x is not None]
    idx = sum(cvs) / len(cvs) if cvs else None
    if idx is None:
        interpretation = "missing_width_evidence"
    elif idx < 0.20:
        interpretation = "too_fixed_check_if_route_is_overconstrained"
    elif idx > 1.50:
        interpretation = "very_elastic_check_if_width_changes_are_explained"
    else:
        interpretation = "elastic_width_healthy_measure_by_region"
    return {
        "elastic_width_index": idx,
        "promoted_width_cv": _cv(promoted_width),
        "lesson_wf_width_cv": _cv(lesson_wf_width),
        "shipped_width_cv": _cv(shipped_width),
        "interpretation": interpretation,
    }


def verdict(run_obj: dict[str, Any], score: RunScore | None, artifacts: dict[str, Any]) -> dict[str, Any]:
    name = run_obj["name"]
    action = "wait_for_score"
    reason = "no leaderboard score yet"
    private = score.private if score else None
    public = score.public if score else None
    delta = private - CHAMPION["private"] if private is not None else None
    signal_action = run_obj.get("signal_action", "unknown")

    if private is not None:
        if private >= CHAMPION["private"] + 0.0005:
            action = "promote_candidate"
            reason = "beats champion by margin"
        elif private >= CHAMPION["private"] - 0.0005:
            action = "blend_or_retest"
            reason = "near champion; test diversity/translation before reuse"
        elif private < 0.074:
            action = "rehabilitate_route"
            reason = "well below champion; reinterpret, reduce weight, carve residual, or use as negative witness before quarantine"
        else:
            action = "witness_only"
            reason = "below champion; keep only if artifacts explain a useful residual"
    if artifacts.get("extrema") and artifacts.get("extrema", {}).get("method") not in (None, "raw"):
        reason += "; extrema transform selected internally"
    if public is not None and private is not None and public > CHAMPION["public"] and private < CHAMPION["private"]:
        action = "reinterpret_public_trap"
        reason = "public improved while private fell; carve the public-specific signal rather than trusting it globally"
    return {
        "name": name,
        "signal_action": signal_action,
        "public": public,
        "private": private,
        "delta_vs_champion": delta,
        "action": action,
        "reason": reason,
        "artifacts": artifacts,
    }


def decide_next(verdicts: list[dict[str, Any]], active: list[dict[str, Any]], *, allow_launch: bool) -> dict[str, Any]:
    if active:
        return {
            "decision": "wait",
            "reason": f"{len(active)} run(s) still active; do not launch more before reading them",
            "active": [r["name"] for r in active],
        }
    promoted = [v for v in verdicts if v["action"] in ("promote_candidate", "blend_or_retest")]
    if promoted:
        return {
            "decision": "blend_retest",
            "reason": "at least one run is near/above champion; test diversity against gpu-wide before new search",
            "candidates": [v["name"] for v in promoted],
        }
    weak = [v for v in verdicts if v["action"] in ("rehabilitate_route", "reinterpret_public_trap", "witness_only")]
    if len(weak) >= max(1, len(verdicts)):
        return {
            "decision": "route_rehabilitation",
            "reason": "recent probes underperformed; carve/reweight/morph their information against the champion before launching more",
            "launch_allowed": bool(allow_launch),
            "suggested_command": "python tools/route_carve.py --routes <weak submission.csv> --submit-top 0",
            "telemetry_guidance_command": (
                "python tools/telemetry_guidance.py --scores-csv <leaderboard_scores.csv>"
            ),
            "fallback_sprout_command": (
                "python tools/fleet.py sprout --count 2 --prefix loop-foundry "
                "--gpu-frac 0 --seed " + str(int(time.time()) % 10_000_000)
            ),
        }
    return {
        "decision": "hold",
        "reason": "mixed/insufficient evidence; update score labels or inspect artifacts manually",
    }


def cmd_once(a: argparse.Namespace) -> int:
    state = load_json(Path(a.state), {"champion": CHAMPION, "history": []})
    manifests = [Path(p) for p in (a.manifest or [])] or DEFAULT_MANIFESTS
    members = a.members or DEFAULT_MEMBERS
    user = kaggle_user()
    runs = tracked_runs(manifests, members)

    statuses = []
    active = []
    for r in runs:
        st = status_one(user, r["slug"])
        r["status"] = st
        statuses.append({"name": r["name"], "slug": r["slug"], "status": st})
        if st not in ("COMPLETE", "CANCELED", "ERROR", "UNKNOWN"):
            active.append(r)
        if a.collect and st == "COMPLETE":
            r["collect"] = collect_one(user, r, submit=a.submit)

    scores = latest_scores()
    verdicts = []
    for r in runs:
        sc = score_for_run(r["name"], scores)
        verdicts.append(verdict(r, sc, read_artifacts(r["name"])))

    plan = decide_next(verdicts, active, allow_launch=a.allow_launch)
    snapshot = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "champion": CHAMPION,
        "statuses": statuses,
        "verdicts": verdicts,
        "plan": plan,
    }
    state["latest"] = snapshot
    state.setdefault("history", []).append(snapshot)
    state["history"] = state["history"][-50:]
    write_json(Path(a.state), state)
    print(json.dumps(snapshot, indent=2, sort_keys=True, default=str))
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    state = load_json(Path(a.state), {})
    latest = state.get("latest", {})
    print(json.dumps(latest or state, indent=2, sort_keys=True, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evidence-driven Kaggle worldview loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("once")
    o.add_argument("--state", default=str(STATE_PATH))
    o.add_argument("--manifest", action="append", default=None)
    o.add_argument("--members", nargs="*", default=None)
    o.add_argument("--collect", action="store_true")
    o.add_argument("--submit", action="store_true")
    o.add_argument("--allow-launch", action="store_true")
    o.set_defaults(fn=cmd_once)
    s = sub.add_parser("status")
    s.add_argument("--state", default=str(STATE_PATH))
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
