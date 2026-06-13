#!/usr/bin/env python3
"""Post-run verdict: ingest a Kaggle run's output artifacts and judge every
open hypothesis against the historical ladder in one page.

Usage:
  python tools/ingest_run.py <output_dir> [--lb SCORE_A SCORE_B]

<output_dir> = the downloaded notebook output (or any dir holding
explorer_run_summary.json + friends). --lb takes the two leaderboard numbers
from the Kaggle UI in either order (the tool reports both ratio readings).

Historical anchors (real DRW runs):
  private ladder: v11 0.08969 (BEST) > v9 0.08653 > v8 0.08537 > ... v25
  0.07827 > v19 0.07491 > v24 0.07210 (WORST)
  sealed cliff: v8/v9/v11 sealed 0.10-0.11 = 3 best private; every heavy-
  search run with sealed > ~0.12 lost 0.02-0.03 private.
  healthy ratios: private ~= 0.79-0.85 x sealed ~= 0.55-0.60 x honest.
"""
import json
import sys
from pathlib import Path

import pandas as pd

V11 = {"private": 0.08969, "sealed": 0.11088, "honest": 0.16287}
LADDER = [("v11", 0.08969), ("v9", 0.08653), ("v8", 0.08537), ("v10", 0.08371),
          ("v12", 0.08491), ("v25", 0.07827), ("v19", 0.07491), ("v24", 0.07210)]


def load(d: Path, name: str):
    p = d / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()) if name.endswith(".json") else pd.read_csv(p)
    except Exception as e:
        print(f"  [!] {name}: unreadable ({e})")
        return None


def fam_of(key: str) -> str:
    try:
        vp = key.split("|", 1)[1]
        return "".join(ch for ch in vp.split("_", 1)[0] if not ch.isdigit())
    except Exception:
        return "?"


def main() -> None:
    d = Path(sys.argv[1])
    lb = None
    if "--lb" in sys.argv:
        i = sys.argv.index("--lb")
        lb = sorted([float(sys.argv[i + 1]), float(sys.argv[i + 2])])
    s = load(d, "explorer_run_summary.json")
    if s is None:
        raise SystemExit(f"no explorer_run_summary.json under {d}")
    print("=" * 74)
    print(f"RUN VERDICT -- {d}")
    print("=" * 74)

    honest = (s.get("honest_scores") or {}).get(s.get("ensemble_winner"), None)
    fwd, sealed = s.get("forward_blend_corr"), s.get("sealed_holdout_corr")
    print(f"\n[scores] honest={honest:.5f}  forward={fwd:.5f}  sealed={sealed:.5f}"
          if None not in (honest, fwd, sealed) else f"\n[scores] {honest} {fwd} {sealed}")
    if sealed:
        zone = ("LIGHT-SEARCH ZONE (the 3 best private runs lived here)" if sealed <= 0.115
                else "RATCHET ZONE (every sealed>0.12 run lost private)" )
        print(f"  sealed vs cliff: {sealed:.5f} vs ~0.115 -> {zone}")
        print(f"  naive private projection 0.79-0.85 x sealed = "
              f"{0.79 * sealed:.5f}..{0.85 * sealed:.5f} (v11 = {V11['private']})")

    gov = load(d, "complexity_governor.json")
    if gov:
        b, lam = gov.get("beta_decay_vs_complexity"), gov.get("lambda_penalty")
        wdc = gov.get("width_decay_corr")
        print(f"\n[governor] beta={b}  lambda={lam}  lessons={gov.get('lessons_measured')}")
        print(f"  verdict: {'DRW punishes capacity as predicted (governor bites)' if (b or 0) > 0 else 'beta<=0: capacity not punished THIS run (governor idle - unexpected on DRW)'}")
        print(f"  width_decay_corr={wdc} -> "
              f"{'WIDE PATHS DECAY LESS - the v30 bias is justified' if (wdc or 0) < -0.02 else ('wide-path bias NOT supported this run' if wdc is not None else 'n/a')}")

    tl = load(d, "testlike_report.json")
    if tl:
        auc = tl.get("auc_holdout")
        print(f"\n[testlike] holdout AUC={auc} -> "
              f"{'REAL covariate shift; testlike partitions carry information' if (auc or 0.5) > 0.6 else 'little detectable shift; partitions are benign extra worlds'}")

    w = s.get("shipped_weights") or {}
    fams = sorted({fam_of(k) for k in w})
    print(f"\n[shipped] selector={(s.get('forensics') or {}).get('shipped_selector')}  "
          f"members={len(w)}  viewport-families={fams}")
    for k, v in sorted(w.items(), key=lambda kv: -kv[1]):
        cplx = (gov or {}).get("member_complexity", {}).get(k)
        print(f"    {v:6.3f}  {k}" + (f"  (complexity {cplx})" if cplx is not None else ""))
    if gov and w:
        cm = gov.get("member_complexity", {})
        got = [cm[k] for k in w if k in cm]
        if got:
            print(f"  weight-avg complexity ~= {sum(got)/len(got):.3f} "
                  f"({'SIMPLE-leaning' if sum(got)/len(got) < 0.35 else 'capacity-leaning'})")

    net = load(d, "winner_network.csv")
    if net is not None and not net.empty:
        share = net["community"].value_counts(normalize=True).iloc[0]
        n_comm = net["community"].nunique()
        ship_comm = net[net["key"].isin(w)]["community"].nunique() if w else None
        print(f"\n[network] {len(net)} nodes, {n_comm} communities, largest share={share:.2f}"
              + (f"; shipped members span {ship_comm} communities" if ship_comm else ""))

    sen = load(d, "segment_senate.csv")
    if sen is not None and not sen.empty and w:
        ship = sen[sen["member"].isin(w)]
        if not ship.empty:
            print(f"[senate] shipped members: max veto={int(ship['veto'].max())}, "
                  f"worst segment corr={ship['worst_corr'].min()}")

    rf = load(d, "redundancy_factor_report.csv")
    if rf is not None and not rf.empty and w:
        ship = rf[rf["member"].isin(w)]
        if not ship.empty:
            print(f"[redundancy] shipped min new_info={ship['new_info'].min():.3f}, "
                  f"max crowding_cos={ship['crowding_cos'].max() if 'crowding_cos' in ship else 'n/a'}")

    dist = load(d, "prediction_distribution_shift.csv")
    if dist is not None and len(dist) == 2:
        print(f"[pred-dist] tail3sd work={dist.iloc[0]['tail_mass_3sd']} "
              f"test={dist.iloc[1]['tail_mass_3sd']} -> "
              f"{'test amplitude HOT (consider shrink next run)' if dist.iloc[1]['tail_mass_3sd'] > 3 * max(dist.iloc[0]['tail_mass_3sd'], 1e-6) else 'amplitude sane'}")

    led = load(d, "learning_ledger.json")
    if led:
        g = led.get("governor", {})
        print(f"[ledger] runs={g.get('count')} survivors={len(led.get('survivors') or [])} "
              f"decayers={len(led.get('decayers') or [])} (attach this output to the NEXT run)")

    if lb:
        lo, hi = lb
        print(f"\n[leaderboard] entered: {lo} / {hi}")
        for tag, val in (("larger-as-private", hi), ("smaller-as-private", lo)):
            beat = [n for n, v in LADDER if val > v]
            pos = f"beats {beat}" if beat else "below the whole ladder"
            print(f"  {tag}: {val} -> {pos}; vs v11 {V11['private']}: {val - V11['private']:+.5f}"
                  + (f"; private/sealed={val / sealed:.2f} (healthy 0.79-0.85)" if sealed else ""))
    else:
        print("\n[leaderboard] rerun with --lb <scoreA> <scoreB> once the Kaggle UI shows both numbers")
    print("\nNext-build queue rides on: governor beta, width_decay_corr, testlike AUC,")
    print("shipped community span, senate vetoes -- see IDEAS_ZOO.md sequencing.")


if __name__ == "__main__":
    main()
