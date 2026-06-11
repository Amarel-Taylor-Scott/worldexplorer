# ----------------------------------------------------------------------------
# 10. Ensemble (pooled + era objectives) + health / dominance / regime stress
# ----------------------------------------------------------------------------

def hill_climb_generic(oofs: dict[str, np.ndarray], idx: np.ndarray,
                       objective: Callable[[np.ndarray], float], max_models: int = 6) -> dict[str, float]:
    scores = {n: objective(oofs[n][idx].astype(np.float64)) for n in oofs}
    sel = sorted(scores, key=scores.get, reverse=True)[:max_models]
    if not sel:
        return {}
    w = {sel[0]: 1.0}
    best = oofs[sel[0]][idx].astype(np.float64).copy()
    best_sc = scores[sel[0]]
    improved = True
    while improved:
        improved = False
        for n in sel:
            for g in np.linspace(0.02, 0.5, 25):
                cand = (1 - g) * best + g * oofs[n][idx]
                sc = objective(cand)
                if sc > best_sc + 1e-9:
                    w = {k: v * (1 - g) for k, v in w.items()}
                    w[n] = w.get(n, 0.0) + float(g)
                    best, best_sc = cand, sc
                    improved = True
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else {}


def nnls_stack_weights(oofs: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray,
                       names: list[str]) -> dict[str, float]:
    if not HAVE_NNLS or not names:
        return {}
    A = np.stack([np.asarray(oofs[n][idx], np.float64) for n in names], axis=1)
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    b = np.asarray(y[idx], np.float64)
    b = (b - b.mean()) / (b.std() + 1e-12)
    try:
        wv, _ = nnls(A, b)
    except Exception:
        return {}
    s = float(wv.sum())
    if s <= 0:
        return {}
    return {n: float(wv[i] / s) for i, n in enumerate(names) if wv[i] > 1e-9}


def weather_moe_fit(oofs: dict[str, np.ndarray], y: np.ndarray, wth: np.ndarray,
                    idx: np.ndarray, top: list[str],
                    cfg: HarnessConfig) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """v9: per-weather-state corr weights shrunk toward the global corr
    weights by state population (a = n/(n+SHRINK)). Returns (global, states)."""
    yw, ww = y[idx], wth[idx]
    g = {n: max(pearson(yw, oofs[n][idx]), 0.0) + 1e-6 for n in top}
    gs = sum(g.values())
    g = {n: v / gs for n, v in g.items()}
    states: dict[int, dict[str, float]] = {}
    for s in np.unique(ww):
        m = ww == s
        n_s = int(m.sum())
        if n_s < cfg.WEATHER_MIN_ROWS:
            continue
        cw = {n: max(pearson(yw[m], oofs[n][idx][m]), 0.0) + 1e-6 for n in top}
        cs = sum(cw.values())
        cw = {n: v / cs for n, v in cw.items()}
        a = n_s / (n_s + cfg.WEATHER_MOE_SHRINK)
        states[int(s)] = {n: a * cw[n] + (1.0 - a) * g[n] for n in top}
    return g, states


def weather_moe_apply(oofs: dict[str, np.ndarray], wth: np.ndarray, idx: np.ndarray,
                      g: dict[str, float], states: dict[int, dict[str, float]]) -> np.ndarray:
    ww = wth[idx]
    out = np.zeros(len(ww), np.float64)
    for s in np.unique(ww):
        m = ww == s
        wmap = states.get(int(s), g)
        out[m] = sum(wmap[n] * oofs[n][idx][m].astype(np.float64) for n in wmap)
    return out


SHIP_SHAPES = ("raw", "rank", "sqrt", "pow15", "tanh")


def _shape_pred(p: np.ndarray, shape: str) -> np.ndarray:
    """v19 prediction shape alchemy: Pearson ignores linear scale but not
    nonlinear AMPLITUDE. Audition output shapes (rank/power/tanh) after the
    blend; the winner is chosen on the forward slice, default 'raw' (no-op)."""
    p = np.asarray(p, np.float64)
    if shape == "rank":
        r = np.empty(len(p), np.float64)
        r[np.argsort(p, kind="stable")] = np.arange(len(p), dtype=np.float64)
        return (r - r.mean()) / (r.std() + 1e-9)
    if shape == "sqrt":
        return np.sign(p) * np.abs(p) ** 0.5
    if shape == "pow15":
        return np.sign(p) * np.abs(p) ** 1.5
    if shape == "tanh":
        return np.tanh(p / (np.std(p) + 1e-9))
    return p


def chorus_factor(parts: dict[str, np.ndarray], weights: dict[str, float], beta: float) -> np.ndarray:
    """v18 chorus shrinkage: a per-row multiplier in (0, 1] from member
    AGREEMENT -- low when the committee splits, ~1 when it concurs. The members
    are already z-scored, so cross-member dispersion per row is the disagreement;
    factor = exp(-beta * normalized_dispersion). beta=0 -> all ones (no-op)."""
    nm0 = next(iter(weights))
    if beta <= 0 or len(weights) < 2:
        return np.ones(len(parts[nm0]), np.float64)
    M = np.vstack([np.asarray(parts[nm], np.float64) for nm in weights])
    disp = M.std(axis=0)
    disp = disp / (float(disp.mean()) + 1e-9)
    return np.exp(-beta * disp)


def apply_weights_rows(parts: dict[str, np.ndarray], weights: dict[str, float],
                       is_median: bool, weather_states: dict[int, dict[str, float]] | None,
                       wth_rows: np.ndarray | None) -> np.ndarray:
    """One blending door for forward / sealed / test rows: weather-conditional
    when the weather_moe strategy won, global weights or median otherwise."""
    if weather_states is not None and wth_rows is not None:
        out = np.zeros(len(wth_rows), np.float64)
        for s in np.unique(wth_rows):
            m = wth_rows == s
            wmap = weather_states.get(int(s), weights)
            out[m] = sum(wmap[n] * parts[n][m] for n in wmap)
        return out
    if is_median:
        return np.median(np.vstack([parts[nm] for nm in weights]), axis=0)
    return sum(weights[nm] * parts[nm] for nm in weights)


def parliament_weights(oofs: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray,
                       names: list[str], cfg: HarnessConfig) -> dict[str, float]:
    """v18 PARLIAMENT BLEND -- antitrust for the ensemble. Start from corr
    weights, then choose a shrink-toward-equal level that maximizes
    corr(blend) MINUS penalties on concentration (HHI) and the single largest
    weight. One model/family ships dominant only if its corr edge outweighs the
    democracy tax. Competes in the same honest tournament, so it can only win
    if it actually generalizes -- no-op-safe."""
    if not names:
        return {}
    full = {n: max(pearson(y[idx], oofs[n][idx]), 0.0) + 1e-6 for n in names}
    ssum = sum(full.values())
    base = {n: v / ssum for n, v in full.items()}
    eq = 1.0 / len(names)
    best_w, best_obj = base, -1e9
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        w = {n: (1.0 - lam) * base[n] + lam * eq for n in names}
        blend = sum(w[n] * oofs[n][idx].astype(np.float64) for n in names)
        corr = pearson(y[idx], blend)
        hhi = float(sum(v * v for v in w.values()))
        mx = float(max(w.values()))
        obj = corr - cfg.PARLIAMENT_HHI * hhi - cfg.PARLIAMENT_MAXW * mx
        if obj > best_obj:
            best_obj, best_w = obj, w
    return best_w


def nested_ensemble(oofs: dict[str, np.ndarray], y: np.ndarray, seg: np.ndarray,
                    cfg: HarnessConfig, embargo: int,
                    wth: np.ndarray | None = None) -> dict[str, Any]:
    names = list(oofs)
    strategies = ["best_single", "equal_top", "corr_weighted", "hill_climb", "era_mean_hill",
                  "late_era_hill", "minimax_era", "parliament", "median"]
    if HAVE_NNLS:
        strategies.append("nnls_stack")
    if wth is not None and len(np.unique(wth)) >= 2:
        strategies.append("weather_moe")    # v9: regime-conditional blending
    outer: dict[str, list[float]] = {s: [] for s in strategies}
    for inner, out_idx in purged_segment_splits(seg, cfg.N_SPLITS, embargo):
        inner_sc = {n: pearson(y[inner], oofs[n][inner]) for n in names}
        top = sorted(inner_sc, key=inner_sc.get, reverse=True)[:8]
        top_oofs = {n: oofs[n] for n in top}

        def score(w: dict[str, float]) -> float:
            if not w:
                return -1.0
            blend = sum(w[n] * oofs[n][out_idx].astype(np.float64) for n in w)
            return score_metric(y[out_idx], blend)   # v26: target metric (Pearson default)

        pooled_obj = lambda v: pearson(y[inner], v)                       # noqa: E731
        era_obj = lambda v: era_mean_corr(y[inner], v, seg[inner])        # noqa: E731
        late_obj = lambda v: late_era_mean_corr(y[inner], v, seg[inner])  # noqa: E731  (v13)
        # v18 minimax-era (ANTIFRAGILE): maximize the WORST per-segment corr --
        # breed for the regimes that produce the 0.036 decay, not the average
        mm_obj = lambda v: _worst_seg_corr(y[inner], v, seg[inner])       # noqa: E731

        outer["best_single"].append(score({top[0]: 1.0}))
        outer["equal_top"].append(score({n: 1 / len(top) for n in top}))
        cw = {n: max(inner_sc[n], 0.0) + 1e-6 for n in top}
        s = sum(cw.values())
        outer["corr_weighted"].append(score({n: v / s for n, v in cw.items()}))
        outer["hill_climb"].append(score(hill_climb_generic(top_oofs, inner, pooled_obj)))
        outer["era_mean_hill"].append(score(hill_climb_generic(top_oofs, inner, era_obj)))
        outer["late_era_hill"].append(score(hill_climb_generic(top_oofs, inner, late_obj)))
        outer["minimax_era"].append(score(hill_climb_generic(top_oofs, inner, mm_obj)))
        outer["parliament"].append(score(parliament_weights(oofs, y, inner, top, cfg)))
        med = np.median(np.vstack([oofs[n][out_idx] for n in top]), axis=0)
        outer["median"].append(pearson(y[out_idx], med))
        if HAVE_NNLS:
            outer["nnls_stack"].append(score(nnls_stack_weights(oofs, y, inner, top)))
        if "weather_moe" in outer:
            g_w, st_w = weather_moe_fit(oofs, y, wth, inner, top, cfg)
            outer["weather_moe"].append(
                pearson(y[out_idx], weather_moe_apply(oofs, wth, out_idx, g_w, st_w)))
    honest = {s: float(np.mean(v)) for s, v in outer.items()}
    winner = max(honest, key=honest.get)
    if honest[winner] <= honest["best_single"] + 1e-9:
        winner = "best_single"

    full_sc = {n: pearson(y, oofs[n]) for n in names}
    top_full = sorted(full_sc, key=full_sc.get, reverse=True)[:8]
    top_full_oofs = {n: oofs[n] for n in top_full}
    all_idx = np.arange(len(y))
    weather_states: dict[int, dict[str, float]] | None = None
    if winner == "best_single":
        weights = {top_full[0]: 1.0}
    elif winner == "equal_top":
        weights = {n: 1 / len(top_full) for n in top_full}
    elif winner == "corr_weighted":
        cw = {n: max(full_sc[n], 0.0) + 1e-6 for n in top_full}
        s = sum(cw.values())
        weights = {n: v / s for n, v in cw.items()}
    elif winner == "median":
        weights = {n: 1 / len(top_full) for n in top_full}
    elif winner == "nnls_stack":
        weights = nnls_stack_weights(oofs, y, all_idx, top_full) or {top_full[0]: 1.0}
    elif winner == "era_mean_hill":
        weights = hill_climb_generic(top_full_oofs, all_idx,
                                     lambda v: era_mean_corr(y, v, seg)) or {top_full[0]: 1.0}
    elif winner == "late_era_hill":
        weights = hill_climb_generic(top_full_oofs, all_idx,
                                     lambda v: late_era_mean_corr(y, v, seg)) or {top_full[0]: 1.0}
    elif winner == "minimax_era":
        weights = hill_climb_generic(top_full_oofs, all_idx,
                                     lambda v: _worst_seg_corr(y, v, seg)) or {top_full[0]: 1.0}
    elif winner == "parliament":
        weights = parliament_weights(oofs, y, all_idx, top_full, cfg) or {top_full[0]: 1.0}
    elif winner == "weather_moe":
        # global map ships as result['weights'] (reports / health read it);
        # the per-state maps condition the actual blend wherever weather ids
        # exist (forward, sealed, test) -- one door, apply_weights_rows.
        weights, weather_states = weather_moe_fit(oofs, y, wth, all_idx, top_full, cfg)
    else:
        weights = hill_climb_generic(top_full_oofs, all_idx,
                                     lambda v: pearson(y, v)) or {top_full[0]: 1.0}
    return {"honest": honest, "winner": winner, "weights": weights, "is_median": winner == "median",
            "weather_states": weather_states,
            "outer": {k: list(map(float, v)) for k, v in outer.items()}}


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


