#!/usr/bin/env python3
"""Measure route width elasticity from worldexplorer run artifacts.

Routes should not be forced to a fixed width. A useful path can widen in a
stable basin, narrow around a hostile pass, and widen again after a transform.
This report measures that behavior from existing artifacts so the loop can tell
the difference between flexible breadth and brittle, unexplained width.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FLEET_DIR = Path("/home/username/new_algo/kaggle/fleet")


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def stats(values: pd.Series, prefix: str) -> dict[str, float | int | None]:
    v = num(values).dropna()
    if v.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_cv": None,
            f"{prefix}_q10": None,
            f"{prefix}_q50": None,
            f"{prefix}_q90": None,
        }
    mean = float(v.mean())
    std = float(v.std(ddof=0))
    return {
        f"{prefix}_n": int(len(v)),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": float(std / (abs(mean) + 1e-12)),
        f"{prefix}_q10": float(v.quantile(0.10)),
        f"{prefix}_q50": float(v.quantile(0.50)),
        f"{prefix}_q90": float(v.quantile(0.90)),
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    x = num(a)
    y = num(b)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return None
    xv = x[mask].to_numpy(np.float64)
    yv = y[mask].to_numpy(np.float64)
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def run_name(path: Path) -> str:
    if path.name == "output":
        return path.parent.name
    return path.name


def analyze_run(path: Path, out_dir: Path) -> tuple[dict, pd.DataFrame]:
    base = path if path.name == "output" else path / "output"
    name = run_name(base)
    lessons_path = base / "explorer_lessons.csv"
    shipping_path = base / "shipping_court_report.csv"
    worlds_path = base / "many_worlds_cv.csv"
    summary = load_json(base / "explorer_run_summary.json")
    governor = load_json(base / "complexity_governor.json")

    row: dict[str, object] = {
        "run": name,
        "output_dir": str(base),
        "forward_blend_corr": summary.get("forward_blend_corr"),
        "sealed_holdout_corr": summary.get("sealed_holdout_corr"),
        "width_decay_corr": governor.get("width_decay_corr"),
    }
    member_rows: list[dict] = []

    lessons = pd.DataFrame()
    if lessons_path.exists():
        lessons = pd.read_csv(lessons_path)
        row.update(stats(lessons.get("width", pd.Series(dtype=float)), "lesson_width"))
        row.update(stats(lessons.get("wf_width", pd.Series(dtype=float)), "lesson_wf_width"))
        row.update(stats(lessons.get("k", pd.Series(dtype=float)), "lesson_k"))
        if "decision" in lessons:
            promoted = lessons[lessons["decision"].astype(str).str.contains("promote", na=False)]
        else:
            promoted = lessons
        row.update(stats(promoted.get("width", pd.Series(dtype=float)), "promoted_width"))
        row.update(stats(promoted.get("k", pd.Series(dtype=float)), "promoted_k"))
        row["lesson_width_vs_wf_width_corr"] = safe_corr(lessons.get("width", pd.Series(dtype=float)),
                                                         lessons.get("wf_width", pd.Series(dtype=float)))
        row["lesson_width_vs_k_corr"] = safe_corr(lessons.get("width", pd.Series(dtype=float)),
                                                  lessons.get("k", pd.Series(dtype=float)))
    else:
        row.update(stats(pd.Series(dtype=float), "lesson_width"))
        row.update(stats(pd.Series(dtype=float), "lesson_wf_width"))
        row.update(stats(pd.Series(dtype=float), "lesson_k"))
        row.update(stats(pd.Series(dtype=float), "promoted_width"))
        row.update(stats(pd.Series(dtype=float), "promoted_k"))

    shipping = pd.DataFrame()
    if shipping_path.exists():
        shipping = pd.read_csv(shipping_path)
        row.update(stats(shipping.get("width", pd.Series(dtype=float)), "shipped_width"))
        row["shipped_width_vs_decay_corr"] = safe_corr(shipping.get("width", pd.Series(dtype=float)),
                                                       shipping.get("decay", pd.Series(dtype=float)))
        row["shipped_width_vs_escape_corr"] = safe_corr(shipping.get("width", pd.Series(dtype=float)),
                                                        shipping.get("escape_velocity", pd.Series(dtype=float)))
    else:
        row.update(stats(pd.Series(dtype=float), "shipped_width"))

    worlds = pd.DataFrame()
    if worlds_path.exists():
        worlds = pd.read_csv(worlds_path)

    if not shipping.empty:
        members = shipping.copy()
        if not worlds.empty and "member" in worlds.columns:
            members = members.merge(worlds, on="member", how="left", suffixes=("", "_world"))
        if not lessons.empty and "key" in lessons.columns:
            keep_cols = [c for c in ["key", "k", "family", "transform", "skill", "decision", "wf_width"] if c in lessons]
            members = members.merge(lessons[keep_cols].drop_duplicates("key"),
                                    left_on="member", right_on="key", how="left")
        members.insert(0, "run", name)
        member_rows = members.to_dict("records")
        row["shipped_width_vs_world_floor_corr"] = safe_corr(members.get("width", pd.Series(dtype=float)),
                                                             members.get("world_survival_min", pd.Series(dtype=float)))
        row["shipped_width_vs_world_frac_corr"] = safe_corr(members.get("width", pd.Series(dtype=float)),
                                                            members.get("world_frac_positive", pd.Series(dtype=float)))

    width_cvs = [
        row.get("promoted_width_cv"),
        row.get("shipped_width_cv"),
        row.get("lesson_wf_width_cv"),
    ]
    width_cvs_f = [float(x) for x in width_cvs if x is not None and np.isfinite(float(x))]
    row["elastic_width_index"] = float(np.mean(width_cvs_f)) if width_cvs_f else None
    row["fixed_width_risk"] = bool(row["elastic_width_index"] is not None and row["elastic_width_index"] < 0.20)
    row["chaotic_width_risk"] = bool(row["elastic_width_index"] is not None and row["elastic_width_index"] > 1.50)
    row["interpretation"] = interpret(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    member_df = pd.DataFrame(member_rows)
    if not member_df.empty:
        member_df.to_csv(out_dir / f"{name}_width_members.csv", index=False)
    return row, member_df


def interpret(row: dict[str, object]) -> str:
    idx = row.get("elastic_width_index")
    decay = row.get("width_decay_corr")
    if idx is None:
        return "missing_width_evidence"
    idxf = float(idx)
    decayf = None if decay is None else float(decay)
    if idxf < 0.20:
        return "too_fixed_check_if_route_is_overconstrained"
    if idxf > 1.50:
        return "very_elastic_check_if_width_changes_are_explained"
    if decayf is not None and decayf > 0.12:
        return "elastic_but_width_decay_positive_use_width_as_hazard_signal"
    return "elastic_width_healthy_measure_by_region"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure width variability across run artifacts")
    ap.add_argument("runs", nargs="+", help="run names under kaggle/fleet, output dirs, or run dirs")
    ap.add_argument("--out", default=str(FLEET_DIR / "width_elasticity"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    rows: list[dict] = []
    members: list[pd.DataFrame] = []
    for item in args.runs:
        p = Path(item)
        if not p.exists():
            p = FLEET_DIR / item
        row, member_df = analyze_run(p, out_dir)
        rows.append(row)
        if not member_df.empty:
            members.append(member_df)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "width_elasticity_report.csv", index=False)
    (out_dir / "width_elasticity_report.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if members:
        pd.concat(members, ignore_index=True).to_csv(out_dir / "width_member_reconciliation.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {out_dir / 'width_elasticity_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
