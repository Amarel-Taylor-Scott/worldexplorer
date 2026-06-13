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


def agreement_weighted_weights(oofs: dict[str, np.ndarray], y: np.ndarray, idx: np.ndarray,
                               names: list[str], cfg: HarnessConfig) -> dict[str, float]:
    """v35 'AGREEING MEANING' blend (user-directed): weight each member by its
    signal AND by how much its prediction is CORROBORATED by the rest of the
    committee (mean |corr| to the other members). A member whose call is
    independently echoed is more trustworthy on a feature-disjoint / out-of-
    support test than a lone strong-but-unconfirmed one -- agreement is a proxy
    for the stable shared meaning that survives the shift. Competes in the same
    honest nested tournament as every other strategy, so it ships ONLY if it
    actually generalizes (it cannot collapse the blend to a redundant cluster
    unnoticed -- the outer folds + the viewport-family cap arbitrate)."""
    if len(names) < 2:
        return {names[0]: 1.0} if names else {}
    M = {n: np.asarray(oofs[n][idx], np.float64) for n in names}
    corr = {n: max(pearson(y[idx], M[n]), 0.0) for n in names}
    agree = {}
    for n in names:
        others = [abs(pearson(M[n], M[m])) for m in names if m != n]
        agree[n] = float(np.mean(others)) if others else 0.0
    w = {n: corr[n] * (agree[n] ** float(cfg.AGREEMENT_POWER)) for n in names}
    s = sum(w.values())
    return {n: v / s for n, v in w.items()} if s > 0 else {n: 1.0 / len(names) for n in names}


def nested_ensemble(oofs: dict[str, np.ndarray], y: np.ndarray, seg: np.ndarray,
                    cfg: HarnessConfig, embargo: int,
                    wth: np.ndarray | None = None,
                    prs: np.ndarray | None = None) -> dict[str, Any]:
    names = list(oofs)
    strategies = ["best_single", "equal_top", "corr_weighted", "hill_climb", "era_mean_hill",
                  "late_era_hill", "minimax_era", "parliament", "median"]
    if HAVE_NNLS:
        strategies.append("nnls_stack")
    if wth is not None and len(np.unique(wth)) >= 2:
        strategies.append("weather_moe")    # v9: regime-conditional blending
    if prs is not None and len(np.unique(prs)) >= 2 and getattr(cfg, "PRESSURE_MOE", False):
        strategies.append("pressure_moe")   # v32: microstructure-PRESSURE-conditional blending
    if getattr(cfg, "CONSENSUS_ENSEMBLE", False) and len(names) >= 2:
        strategies.append("agreement_weighted")   # v35: agreeing-meaning blend (committee-corroborated)
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
        if "pressure_moe" in outer:
            g_p, st_p = weather_moe_fit(oofs, y, prs, inner, top, cfg)
            outer["pressure_moe"].append(
                pearson(y[out_idx], weather_moe_apply(oofs, prs, out_idx, g_p, st_p)))
        if "agreement_weighted" in outer:
            outer["agreement_weighted"].append(score(agreement_weighted_weights(oofs, y, inner, top, cfg)))
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
    elif winner == "pressure_moe":
        # v32 (IDEAS_ZOO census->built): same one-door mechanics, keyed by the
        # PRESSURE gauge's microstructure states instead of generic dispersion
        weights, weather_states = weather_moe_fit(oofs, y, prs, all_idx, top_full, cfg)
    elif winner == "agreement_weighted":
        # v35 agreeing-meaning blend (committee-corroborated member weights)
        weights = agreement_weighted_weights(oofs, y, all_idx, top_full, cfg) or {top_full[0]: 1.0}
    else:
        weights = hill_climb_generic(top_full_oofs, all_idx,
                                     lambda v: pearson(y, v)) or {top_full[0]: 1.0}
    return {"honest": honest, "winner": winner, "weights": weights, "is_median": winner == "median",
            "weather_states": weather_states,
            "moe_gauge": ("pressure" if winner == "pressure_moe"
                          else ("weather" if winner == "weather_moe" else None)),
            "outer": {k: list(map(float, v)) for k, v in outer.items()}}



def moe_states(moe_gauge: "str | None", X: np.ndarray) -> "np.ndarray | None":
    """v32: the one place that maps a MoE winner to its row-state assigner --
    pressure_moe rows get PRESSURE states, everything else weather states
    (harmless when no MoE won: apply_weights_rows ignores the vector)."""
    if moe_gauge == "pressure" and PRESSURE is not None:
        return PRESSURE.assign(X)
    return GAUGE.assign(X) if GAUGE is not None else None
