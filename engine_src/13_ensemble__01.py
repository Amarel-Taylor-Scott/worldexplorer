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



def redundancy_factor_report(members: dict[str, np.ndarray],
                             factor_scores: "np.ndarray | None") -> pd.DataFrame:
    """v31 (IDEAS_ZOO B2): per-member ensemble NEW-INFORMATION + latent-factor
    crowding, observation only. new_info = 1 - R^2(member ~ rest of the blend
    pool) -- information the others do not already span (stricter than corr
    caps). Factor exposures = corr(member oof, target-free PCA factor scores);
    the max pairwise exposure-cosine surfaces 'different names, same latent
    bet' -- the crowding channel behind every measured monoculture."""
    nms = list(members)
    if len(nms) < 2:
        return pd.DataFrame()
    M = np.vstack([np.asarray(members[nm], np.float64) for nm in nms])
    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-12)
    expo = Ccos = None
    if factor_scores is not None and factor_scores.shape[0] == M.shape[1]:
        F = np.asarray(factor_scores, np.float64)
        Fz = (F - F.mean(0)) / (F.std(0) + 1e-12)
        expo = (M @ Fz) / M.shape[1]
        ecos = expo / (np.linalg.norm(expo, axis=1, keepdims=True) + 1e-12)
        Ccos = ecos @ ecos.T
        np.fill_diagonal(Ccos, -1.0)
    rows = []
    for i, nm in enumerate(nms):
        others = np.delete(np.arange(len(nms)), i)
        A = M[others].T
        beta, *_ = np.linalg.lstsq(A, M[i], rcond=None)
        r2 = 1.0 - float(np.var(M[i] - A @ beta))
        r2 = min(1.0, max(0.0, r2))
        row = {"member": nm, "new_info": round(1.0 - r2, 4), "r2_vs_rest": round(r2, 4)}
        if expo is not None:
            for f in range(expo.shape[1]):
                row[f"factor_{f}"] = round(float(expo[i, f]), 4)
            j = int(np.argmax(Ccos[i]))
            row["crowding_partner"] = nms[j]
            row["crowding_cos"] = round(float(Ccos[i, j]), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("new_info")

def segment_senate_report(members: dict[str, np.ndarray], y: np.ndarray, seg: np.ndarray,
                          cfg: HarnessConfig) -> pd.DataFrame:
    """v32 SEGMENT SENATE (IDEAS_ZOO C2, observation): every time segment votes
    on every member -- yes (corr > SENATE_YES), abstain, veto (corr <
    SENATE_VETO). A member with a great mean but several vetoes is a few good
    eras hiding many bad ones; era-mean metrics blur exactly this."""
    segs = np.unique(seg)
    rows = []
    for nm, p in members.items():
        per = [pearson(y[seg == s], p[seg == s]) for s in segs
               if int((seg == s).sum()) >= 50]
        yes = sum(1 for c in per if c > cfg.SENATE_YES)
        veto = sum(1 for c in per if c < cfg.SENATE_VETO)
        rows.append({"member": nm, "segments": len(per),
                     "yes": yes, "abstain": len(per) - yes - veto, "veto": veto,
                     "mean_corr": round(float(np.mean(per)), 5) if per else None,
                     "worst_corr": round(float(np.min(per)), 5) if per else None,
                     "veto_frac": round(veto / max(1, len(per)), 3)})
    return pd.DataFrame(rows).sort_values("veto", ascending=False)


def prediction_distribution_report(blend_oof: np.ndarray, test_pred: np.ndarray) -> pd.DataFrame:
    """v32 PREDICTION-DISTRIBUTION SHIFT (IDEAS_ZOO v65 §43, observation):
    moments + tail mass of the working-region blend vs the shipped test
    prediction, both z-scored by the WORKING distribution -- test predictions
    far more extreme than anything validation ever scored = amplitude risk."""
    a = np.asarray(blend_oof, np.float64)
    b = np.asarray(test_pred, np.float64)
    mu, sd = a.mean(), a.std() + 1e-12
    az, bz = (a - mu) / sd, (b - mu) / sd
    def stats(v):
        return {"mean_z": round(float(v.mean()), 4), "std_z": round(float(v.std()), 4),
                "skew": round(float(((v - v.mean()) ** 3).mean() / (v.std() + 1e-12) ** 3), 4),
                "kurt": round(float(((v - v.mean()) ** 4).mean() / (v.std() + 1e-12) ** 4), 4),
                "q01": round(float(np.quantile(v, 0.01)), 4), "q99": round(float(np.quantile(v, 0.99)), 4),
                "tail_mass_3sd": round(float(np.mean(np.abs(v) > 3.0)), 5)}
    rows = [{"distribution": "working_blend_oof", **stats(az)},
            {"distribution": "test_prediction", **stats(bz)}]
    return pd.DataFrame(rows)


def feature_topology_report(graph, shift, cols, X, y) -> pd.DataFrame:
    """v35 observation: the FEATURE TOPOLOGY. Per target-free community -- its
    size, internal coherence (how tightly its features move together), mean/max
    |corr| to y, its DIRECTIONAL signal consensus (does the community agree on a
    direction?), and its mean train->test shift. The map of 'common-topology
    features' that shows WHERE the agreeing, low-shift, stable meaning lives --
    high coherence + high consensus + LOW shift = a community whose shared signal
    should survive the disjoint test. Pure observation; gates nothing."""
    if graph is None or getattr(graph, "community", None) is None:
        return pd.DataFrame()
    comm = graph.community
    step = max(1, len(X) // 20000)
    Xs, ys = X[::step], y[::step]
    rows = []
    for cm in range(int(graph.n_communities)):
        idx = np.where(comm == cm)[0]
        if len(idx) == 0:
            continue
        c = corr_vector(Xs[:, idx], ys)
        ac = np.abs(c)
        mean_signed = float(c.mean())
        coher = (float(graph.coherence[cm]) if graph.coherence is not None
                 and cm < len(graph.coherence) else 0.0)
        consensus = abs(mean_signed) / (float(ac.mean()) + 1e-9)
        sh = float(np.mean([shift[int(j)] for j in idx])) if shift is not None else float("nan")
        rows.append({"community": int(cm), "size": int(len(idx)),
                     "topology_coherence": round(coher, 4),
                     "mean_abs_corr_y": round(float(ac.mean()), 4),
                     "max_abs_corr_y": round(float(ac.max()), 4),
                     "signal_consensus": round(consensus, 4),
                     "mean_train_test_shift": round(sh, 4) if np.isfinite(sh) else None})
    return pd.DataFrame(rows).sort_values("mean_abs_corr_y", ascending=False)


def antifragility_report(members: dict[str, np.ndarray], member_lessons: dict,
                         y: np.ndarray, seg: np.ndarray, terr, wth, cfg) -> pd.DataFrame:
    """v35 observation: the ANTI-FRAGILITY of each candidate member -- not just
    'does it survive' but how it holds under STRESS. Blends the worst-world corr
    FLOOR across time/terrain/weather, the consistency gap (mean - worst world),
    the stability probe (lower RMSE = steadier under input noise), and the
    predator's perturbation width (positive under a shrunken viewport = robust).
    High score = strong stable meaning that does NOT break when the world shifts
    -- the members a high-stability / anti-fragile blend should lean on. Gates
    nothing here; it is the map the user asked to be able to see + steer by."""
    worlds = [("seg", seg)]
    if terr is not None:
        worlds.append(("terrain", terr))
    if wth is not None:
        worlds.append(("weather", wth))
    rows = []
    for nm in members:
        l = member_lessons.get(nm)
        floors = [pearson(y[ids == s], members[nm][ids == s])
                  for _, ids in worlds for s in np.unique(ids)
                  if int((ids == s).sum()) >= cfg.FORENSIC_MIN_WORLD_ROWS]
        wfloor = float(min(floors)) if floors else 0.0
        mean_c = float(np.mean(floors)) if floors else 0.0
        stab = float(l.stability) if (l is not None and np.isfinite(l.stability)) else float("nan")
        perturb = float(l.perturb_width) if (l is not None and np.isfinite(l.perturb_width)) else float("nan")
        af = wfloor - 0.5 * max(0.0, mean_c - wfloor)        # reward a high, CONSISTENT floor
        if np.isfinite(stab):
            af -= 0.1 * stab                                  # penalize fragility under noise
        if np.isfinite(perturb):
            af += 0.5 * max(0.0, perturb)                     # reward holding under perturbation
        rows.append({"member": nm, "world_floor": round(wfloor, 4),
                     "mean_world": round(mean_c, 4),
                     "consistency_gap": round(mean_c - wfloor, 4),
                     "stability_rmse": round(stab, 4) if np.isfinite(stab) else None,
                     "perturb_width": round(perturb, 4) if np.isfinite(perturb) else None,
                     "antifragility": round(af, 4)})
    return pd.DataFrame(rows).sort_values("antifragility", ascending=False)
