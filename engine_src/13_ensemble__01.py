class HealthMonitor:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.alarms: list[dict[str, Any]] = []

    def check(self, name: str, value: float, threshold: float, bad_if: str, note: str) -> None:
        bad = value > threshold if bad_if == "above" else value < threshold
        self.alarms.append({"check": name, "value": round(float(value), 6), "threshold": threshold,
                            "status": "ALARM" if bad else "ok", "note": note})
        if bad:
            log("HEALTH_ALARM", check=name, value=round(float(value), 5), threshold=threshold)

    def run_checks(self, result: dict[str, Any], oofs: dict[str, np.ndarray], y: np.ndarray,
                   lessons: list[Lesson], forward_blend_corr: float | None) -> pd.DataFrame:
        w = result["weights"]
        hhi = float(sum(v * v for v in w.values()))
        self.check("weight_concentration_top", max(w.values()) if w else 1.0,
                   self.cfg.MAX_TOP_WEIGHT, "above", "one member overpowering the blend")
        self.check("effective_members", 1.0 / max(hhi, 1e-9),
                   self.cfg.MIN_EFFECTIVE_MEMBERS, "below", "ensemble going weak / collapsing")
        names = list(w)
        if len(names) >= 2:
            cors = [abs(pearson(oofs[a], oofs[b])) for i, a in enumerate(names) for b in names[i + 1:]]
            self.check("mean_member_correlation", float(np.mean(cors)),
                       self.cfg.MEMBER_CORR_CAP, "above", "members are near-duplicates")
        blend = sum(w[n] * oofs[n].astype(np.float64) for n in w) if w else np.zeros(len(y))
        blend_c = pearson(y, blend)
        best_single_c = max(pearson(y, oofs[n]) for n in oofs)
        self.check("blend_vs_best_single_fullOOF", best_single_c - blend_c, 0.0, "above",
                   "blend weaker than best single on full OOF (diagnostic)")
        worst = max((l.overfit_ratio for l in lessons if l.decision == "promote"), default=1.0)
        self.check("worst_promoted_overfit_ratio", worst, self.cfg.MAX_OVERFIT_RATIO, "above",
                   "a promoted lesson fits train far better than CV")
        if forward_blend_corr is not None:
            self.check("forward_holdout_blend_corr", forward_blend_corr, 0.0, "below",
                       "blend collapses on the untouched most-recent working slice")
        return pd.DataFrame(self.alarms)


def dominance_report(weights: dict[str, float], oofs: dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    if not weights:
        return pd.DataFrame()
    blend = sum(weights[n] * oofs[n].astype(np.float64) for n in weights)
    blend_c = pearson(y, blend)
    rows = []
    for n, wv in weights.items():
        rest = {k: v for k, v in weights.items() if k != n}
        tot = sum(rest.values())
        loo_c = pearson(y, sum(v / tot * oofs[k].astype(np.float64) for k, v in rest.items())) if tot > 0 else 0.0
        rows.append({"member": n, "weight": round(wv, 4),
                     "member_oof_corr": pearson(y, oofs[n]),
                     "blend_corr": blend_c, "loo_blend_corr": loo_c,
                     "marginal_value": blend_c - loo_c,
                     "corr_to_blend": pearson(oofs[n], blend),
                     "dominance_flag": bool(wv > 0.6 and loo_c >= blend_c - 1e-9)})
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def many_worlds_report(members: dict[str, np.ndarray], y: np.ndarray, seg: np.ndarray,
                       terr: np.ndarray | None, wth: np.ndarray | None,
                       cfg: HarnessConfig) -> pd.DataFrame:
    """v19 MANY-WORLDS CV: do not ask a member to win the average -- ask
    whether it survives EVERY environment. Per member, corr WITHIN each time
    segment / terrain valley / weather band; the MIN across all those worlds
    is its survival floor, and frac_positive is how many worlds it helps in.
    A member with a high full corr but a deeply negative world is a private-LB
    landmine; this surfaces it. Report only -- it gates nothing here."""
    states = [("seg", seg)]
    if terr is not None:
        states.append(("terrain", terr))
    if wth is not None:
        states.append(("weather", wth))
    rows = []
    for nm, oof in members.items():
        row = {"member": nm, "full_corr": round(pearson(y, oof), 4)}
        worldcorrs: list[float] = []
        for wname, ids in states:
            wc = [pearson(y[ids == s], oof[ids == s]) for s in np.unique(ids)
                  if (ids == s).sum() >= cfg.MANY_WORLDS_MIN_ROWS]
            if wc:
                row[f"{wname}_min"] = round(float(min(wc)), 4)
                worldcorrs += wc
        row["world_survival_min"] = round(float(min(worldcorrs)), 4) if worldcorrs else float("nan")
        row["world_frac_positive"] = round(float(np.mean([c > 0 for c in worldcorrs])), 3) if worldcorrs else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("world_survival_min", ascending=False)


def map_elites_archive(lessons: list["Lesson"]) -> pd.DataFrame:
    """v19 MAP-ELITES: keep the best lesson in every cell of a BEHAVIORAL grid
    (family x transform x k-bucket x overfit-bucket), not just the global best.
    A biodiversity archive -- it shows whether the search collapsed into one
    superfamily or kept a wide ecology of distinct, healthy niches."""
    cells: dict[tuple, "Lesson"] = {}
    for l in lessons:
        if l.oof_corr <= 0:
            continue
        kb = "k<32" if l.k < 32 else ("k<96" if l.k < 96 else "k>=96")
        ob = "of<2" if l.overfit_ratio < 2 else ("of<4" if l.overfit_ratio < 4 else "of>=4")
        cell = (l.family, l.transform, kb, ob)
        if cell not in cells or l.oof_corr > cells[cell].oof_corr:
            cells[cell] = l
    rows = [{"family": c[0], "transform": c[1], "k_bucket": c[2], "overfit_bucket": c[3],
             "best_key": l.key, "oof_corr": round(l.oof_corr, 4), "width": round(l.width, 4),
             "wf_corr": round(l.wf_corr, 4)} for c, l in cells.items()]
    return pd.DataFrame(rows).sort_values("oof_corr", ascending=False)


def regime_stress(members: dict[str, np.ndarray], y: np.ndarray, seg: np.ndarray,
                  volume: np.ndarray | None, q: float) -> pd.DataFrame:
    rows = []
    segs = np.unique(seg)
    for n, o in members.items():
        per = [pearson(y[seg == s], o[seg == s]) for s in segs]
        row = {"member": n, "full_corr": pearson(y, o),
               "min_segment_corr": float(min(per)), "mean_segment_corr": float(np.mean(per)),
               "frac_segments_negative": float(np.mean([c <= 0 for c in per]))}
        if volume is not None and np.isfinite(volume).any():
            hi = volume >= np.nanquantile(volume, q)
            row["high_volume_corr"] = pearson(y[hi], o[hi]) if hi.sum() >= 50 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("min_segment_corr", ascending=False)


def winsorize_audit(blend_oof: np.ndarray, y: np.ndarray, qs: tuple[float, ...]) -> dict[str, Any]:
    best_q, best_c = 0.0, pearson(y, blend_oof)
    raw_c = best_c
    for q in qs:
        if q <= 0:
            continue
        lo, hi = np.quantile(blend_oof, q), np.quantile(blend_oof, 1 - q)
        c = pearson(y, np.clip(blend_oof, lo, hi))
        if c > best_c + 1e-6:
            best_q, best_c = q, c
    return {"raw_corr": raw_c, "best_q": best_q, "best_corr": best_c, "apply": bool(best_q > 0)}


