#!/usr/bin/env python3
"""Carve useful information out of weak routes.

Given a champion submission and one or more weaker route submissions, this tool
creates conservative prediction-space variants across a rehabilitation hierarchy:

- reinterpret route through rank/tanh/sqrt/clip lenses
- reduce route to a small direct component
- extract residual not explained by the champion
- carve agreement/disagreement masks
- terraform tails through local route blending or contrastive subtraction

It does not claim these are better. It creates reviewable artifacts for the next
Kaggle test, preserving the route-rehabilitation ladder before quarantine.
"""
from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CHAMPION = Path(
    "/home/username/new_algo/kaggle/fleet/route_carves_hierarchy2/"
    "submission_bio-sprout-03-forager_pow15_conflict_sub_a0.12.csv"
)
DEFAULT_OUT = Path("/home/username/new_algo/kaggle/fleet/route_carves")
COMP = "drw-crypto-market-prediction"


def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return (x - float(x.mean())) / (float(x.std()) + 1e-12)


def rank_z(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy(np.float64)
    return z(r)


def sqrt_z(x: np.ndarray) -> np.ndarray:
    x = z(x)
    return z(np.sign(x) * np.sqrt(np.abs(x)))


def tanh_z(x: np.ndarray) -> np.ndarray:
    return z(np.tanh(z(x)))


def clip_z(x: np.ndarray, q: float = 0.975) -> np.ndarray:
    x = z(x)
    lo, hi = np.quantile(x, [1.0 - q, q])
    return z(np.clip(x, lo, hi))


def pow15_z(x: np.ndarray) -> np.ndarray:
    x = z(x)
    return z(np.sign(x) * np.abs(x) ** 1.5)


def read_submission(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise SystemExit(f"{path}: expected id + prediction columns")
    return df, df.iloc[:, -1].to_numpy(np.float64)


def write_candidate(base: pd.DataFrame, pred: np.ndarray, out: Path) -> None:
    df = base.copy()
    df.iloc[:, -1] = z(pred).astype(np.float32)
    df.to_csv(out, index=False)


def review_score(row: dict) -> float:
    target_novelty = 0.0018
    novelty = max(0.0, float(row["novelty_from_champion"]))
    novelty_fit = -abs(np.log10(novelty + 1e-9) - np.log10(target_novelty))
    return float(
        row["corr_candidate_champion"]
        + 0.08 * row["corr_candidate_route"]
        + 0.03 * row["route_information_gain"]
        + 0.015 * novelty_fit
    )


def iter_carves(champion: np.ndarray, route: np.ndarray, route_name: str,
                out_dir: Path, alphas: list[float]):
    c = z(champion)
    lenses = {
        "raw": z(route),
        "rank": rank_z(route),
        "sqrt": sqrt_z(route),
        "tanh": tanh_z(route),
        "clip": clip_z(route),
        "pow15": pow15_z(route),
    }

    for lens_name, r in lenses.items():
        corr = float(np.corrcoef(c, r)[0, 1])
        resid = z(r - corr * c)
        disagreement = z(np.abs(c - r))
        agreement = z(c * r)
        conflict = disagreement >= np.quantile(disagreement, 0.95)
        agree_tail = agreement >= np.quantile(agreement, 0.95)
        route_tail = np.abs(r) >= np.quantile(np.abs(r), 0.95)
        for alpha in alphas:
            # The hierarchy is explicit in candidate names:
            # reduce -> residual -> carve -> terraform -> contrast.
            specs = {
                f"{route_name}_{lens_name}_reduce_a{alpha:g}":
                    ("reduce", z((1.0 - alpha) * c + alpha * r)),
                f"{route_name}_{lens_name}_resid_add_a{alpha:g}":
                    ("residual_add", z(c + alpha * resid)),
                f"{route_name}_{lens_name}_resid_sub_a{alpha:g}":
                    ("residual_subtract", z(c - alpha * resid)),
                f"{route_name}_{lens_name}_agree_add_a{alpha:g}":
                    ("agreement_carve", z(c + alpha * resid * agree_tail.astype(float))),
                f"{route_name}_{lens_name}_conflict_sub_a{alpha:g}":
                    ("conflict_carve", z(c - alpha * resid * conflict.astype(float))),
            }
            tail_blend = c.copy()
            tail_blend[route_tail] = (1.0 - alpha) * tail_blend[route_tail] + alpha * r[route_tail]
            specs[f"{route_name}_{lens_name}_tail_blend_a{alpha:g}"] = ("tail_terraform", z(tail_blend))
            tail_damp = c.copy()
            tail_damp[conflict] = (1.0 - alpha) * tail_damp[conflict]
            specs[f"{route_name}_{lens_name}_conflict_damp_a{alpha:g}"] = ("conflict_dampen", z(tail_damp))
            for name, (stage, pred) in specs.items():
                corr_candidate_champion = float(np.corrcoef(pred, c)[0, 1])
                if abs(corr_candidate_champion - 1.0) < 1e-10:
                    continue
                corr_candidate_route = float(np.corrcoef(pred, r)[0, 1])
                path = out_dir / f"submission_{name}.csv"
                row = {
                    "candidate": name,
                    "path": str(path),
                    "route": route_name,
                    "rehab_stage": stage,
                    "lens": lens_name,
                    "alpha": alpha,
                    "corr_champion_route": corr,
                    "corr_candidate_champion": corr_candidate_champion,
                    "corr_candidate_route": corr_candidate_route,
                    "novelty_from_champion": 1.0 - corr_candidate_champion,
                    "route_information_gain": corr_candidate_route - corr,
                    "conflict_tail_frac": float(conflict.mean()),
                    "agree_tail_frac": float(agree_tail.mean()),
                    "route_tail_frac": float(route_tail.mean()),
                    "intent": "rehabilitate_route_before_quarantine",
                }
                row["review_score"] = review_score(row)
                yield row, pred.astype(np.float32, copy=False)


def push_candidate(heap: list, max_candidates: int, counter: int, row: dict, pred: np.ndarray) -> None:
    item = (float(row["review_score"]), counter, row, pred.copy())
    if max_candidates <= 0:
        heapq.heappush(heap, item)
        return
    if len(heap) < max_candidates:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="create conservative route-carving submissions")
    ap.add_argument("--champion", default=str(DEFAULT_CHAMPION))
    ap.add_argument("--routes", nargs="+", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--alphas", default="0.03,0.05,0.08,0.12")
    ap.add_argument("--max-candidates", type=int, default=80)
    ap.add_argument("--submit-top", type=int, default=0, help="submit only the first N review-ordered candidates")
    ap.add_argument("--comp", default=COMP)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_df, champion = read_submission(Path(args.champion))
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    heap: list = []
    counter = 0
    for route_path_s in args.routes:
        route_path = Path(route_path_s)
        route_df, route = read_submission(route_path)
        if len(route_df) != len(base_df):
            raise SystemExit(f"{route_path}: row count mismatch vs champion")
        route_name = route_path.parent.parent.name if route_path.name == "submission.csv" else route_path.stem
        route_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in route_name)
        for row, pred in iter_carves(champion, route, route_name, out_dir, alphas):
            counter += 1
            push_candidate(heap, args.max_candidates, counter, row, pred)

    selected = sorted(heap, key=lambda item: (item[0], item[2]["corr_candidate_route"]), reverse=True)
    manifest: list[dict] = []
    for _, _, row, pred in selected:
        write_candidate(base_df, pred, Path(row["path"]))
        manifest.append(row)

    dfm = pd.DataFrame(manifest)
    if not dfm.empty:
        dfm = dfm.sort_values(["review_score", "corr_candidate_route"], ascending=[False, False])
    manifest_path = out_dir / "route_carve_manifest.csv"
    dfm.to_csv(manifest_path, index=False)
    (out_dir / "route_carve_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(manifest)} candidates -> {out_dir}")
    print(f"manifest: {manifest_path}")

    if args.submit_top:
        for row in dfm.head(max(0, args.submit_top)).to_dict("records"):
            subprocess.run([
                "kaggle", "competitions", "submit", "-c", args.comp,
                "-f", row["path"], "-m", f"route carve {row['candidate']}"
            ], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
