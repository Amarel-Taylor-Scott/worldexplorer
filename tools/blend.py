#!/usr/bin/env python3
"""Blend several submission.csv files into one and (optionally) submit it.

The measured DRW lever: a blend of diverse runs beats any single run (a fleet of
light runs blended -> ~0.083, vs ~0.075 for one thin run). Because the metric is
Pearson (order matters, not scale) and the runs live on different scales, the
default blend is a WEIGHTED RANK-AVERAGE -- robust to any single run's amplitude.

Usage:
  python tools/blend.py A.csv B.csv C.csv [--out blend.csv]
  python tools/blend.py /path/to/fleet/*/output      # dirs holding submission.csv
  python tools/blend.py A.csv B.csv --weights 0.5,0.3,0.2
  python tools/blend.py ... --submit [--comp drw-crypto-market-prediction] [-m MSG]
  python tools/blend.py ... --mean                   # plain mean instead of rank-average
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def _find_csvs(paths) -> list:
    out = []
    for p in paths:
        for hit in glob.glob(p):
            if os.path.isdir(hit):
                s = os.path.join(hit, "submission.csv")
                if os.path.exists(s):
                    out.append(s)
            elif hit.endswith(".csv"):
                out.append(hit)
    return out


def main(argv=None) -> int:
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser(description="blend submission.csv files (rank-average)")
    ap.add_argument("inputs", nargs="+", help="submission.csv files or dirs containing them (globs ok)")
    ap.add_argument("--out", default="blend.csv")
    ap.add_argument("--weights", default=None, help="comma-separated, one per input")
    ap.add_argument("--mean", action="store_true", help="plain mean instead of rank-average")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--comp", default="drw-crypto-market-prediction")
    ap.add_argument("-m", "--message", default="worldexplorer fleet blend")
    a = ap.parse_args(argv)

    files = _find_csvs(a.inputs)
    if len(files) < 1:
        sys.exit(f"no submission.csv found in {a.inputs}")
    print(f"blending {len(files)} files:")
    for f in files:
        print("  -", f)

    base = pd.read_csv(files[0])
    id_col, pred_col = base.columns[0], base.columns[-1]
    n = len(base)
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(files)
    if len(w) != len(files):
        sys.exit(f"--weights has {len(w)} values for {len(files)} files")
    wsum = sum(w) or 1.0

    acc = np.zeros(n, np.float64)
    for f, wi in zip(files, w):
        df = pd.read_csv(f)
        if len(df) != n:
            sys.exit(f"{f}: {len(df)} rows != {n} (mismatched submissions)")
        v = df[df.columns[-1]].to_numpy(np.float64)
        if a.mean:
            v = (v - v.mean()) / (v.std() + 1e-12)          # z-score so scales are comparable
        else:
            order = np.argsort(np.argsort(v))                # ranks 0..n-1
            v = order / max(1, n - 1) - 0.5                  # centered rank in [-0.5, 0.5]
        acc += (wi / wsum) * v

    out = base[[id_col]].copy()
    out[pred_col] = acc.astype(np.float32)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({n} rows, '{pred_col}', "
          f"{'mean-z' if a.mean else 'rank'}-blend of {len(files)})")

    if a.submit:
        subprocess.run(["kaggle", "competitions", "submit", "-c", a.comp,
                        "-f", a.out, "-m", a.message], check=False)
        print(f"submitted {a.out} -> {a.comp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
