# ----------------------------------------------------------------------------
# 9b. Topography reports -- path texture, trail families, terrain trails,
#     symmetry fields. Descriptive sub-models of the lessons' PATHS: they
#     never gate promotion (the honest doors stay fixed); they feed member
#     DIVERSITY (texture-family cap) and the run's map of the world.
# ----------------------------------------------------------------------------

def path_texture_vector(oof: np.ndarray, y: np.ndarray, seg: np.ndarray,
                        terr: np.ndarray | None, wth: np.ndarray | None,
                        min_rows: int) -> dict[str, float]:
    """Describe one trail (an OOF path) the way a hiker would:
      roughness      : how bumpy the per-segment corr profile is
      side textures  : residual scatter above vs below the trail (what the
                       slope looks like on each side of the path)
      side_asym      : log-ratio of the two side textures (signed terrain tilt)
      wake_ac1       : lag-1 autocorrelation of the residual -- structure the
                       path failed to model trailing behind it like a wake
      steep_affinity : does the path bet bigger where the ground is steep
      terrain_min/spread : altitude profile across the atlas's valleys."""
    o = np.asarray(oof, np.float64)
    yv = np.asarray(y, np.float64)
    oz = (o - o.mean()) / (o.std() + 1e-12)
    slope = float(np.mean(oz * (yv - yv.mean())))
    r = yv - slope * oz                              # residual off the trail
    per = [pearson(yv[seg == s], o[seg == s]) for s in np.unique(seg)]
    above, below = r[oz > 0], r[oz < 0]
    sa = float(above.std()) if len(above) > 50 else float("nan")
    sb = float(below.std()) if len(below) > 50 else float("nan")
    tex = {"roughness": float(np.std(per)),
           "side_std_above": sa, "side_std_below": sb,
           "side_asym": float(np.log((sa + 1e-9) / (sb + 1e-9)))
           if np.isfinite(sa) and np.isfinite(sb) else 0.0,
           "wake_ac1": pearson(r[:-1], r[1:]) if len(r) > 1000 else 0.0,
           "steep_affinity": pearson(np.abs(oz), np.abs(yv - yv.mean())),
           # v13 GAIT: the slope of the per-segment corr profile -- an
           # ascending trail strengthens toward the present, a fading one
           # weakens. The measured shadow of regime decay, per trail.
           "gait": (pearson(np.asarray(per, np.float64),
                            np.arange(len(per), dtype=np.float64))
                    if len(per) >= 4 else 0.0)}
    t_min = t_spread = float("nan")
    if terr is not None:
        tc = [pearson(yv[terr == t], o[terr == t])
              for t in np.unique(terr) if (terr == t).sum() >= min_rows]
        if tc:
            t_min, t_spread = float(min(tc)), float(max(tc) - min(tc))
    tex["terrain_min"] = t_min
    tex["terrain_spread"] = t_spread
    w_min = w_spread = float("nan")
    if wth is not None:
        wc = [pearson(yv[wth == s], o[wth == s])
              for s in np.unique(wth) if (wth == s).sum() >= min_rows]
        if wc:
            w_min, w_spread = float(min(wc)), float(max(wc) - min(wc))
    tex["weather_min"] = w_min
    tex["weather_spread"] = w_spread
    return tex


def leader_cluster(vecs: np.ndarray, radius: float) -> list[int]:
    """Single-pass leader clustering (order = strongest trail first): a vector
    joins the first leader within `radius`, else founds a new family."""
    leaders: list[np.ndarray] = []
    fams: list[int] = []
    for v in vecs:
        hit = -1
        for fi, c in enumerate(leaders):
            if float(np.linalg.norm(v - c)) <= radius:
                hit = fi
                break
        if hit < 0:
            leaders.append(v.copy())
            hit = len(leaders) - 1
        fams.append(hit)
    return fams


TEXTURE_FEATURES = ("roughness", "side_asym", "wake_ac1", "steep_affinity",
                    "terrain_spread", "weather_spread", "sense_gap", "gait")


def texture_layer(lessons: list["Lesson"], y: np.ndarray, seg: np.ndarray,
                  terr: np.ndarray | None, wth: np.ndarray | None,
                  cfg: HarnessConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Texture every positive trail, then leader-cluster them into TRAIL
    FAMILIES in z-scored texture space. Two paths can be decorrelated yet
    fail the same way -- same texture family -- so the member selector caps
    how many of one family the blend may carry."""
    cand = sorted((l for l in lessons if l.oof_corr > 0 and float(np.std(l.oof)) > 1e-9),
                  key=lambda l: -l.oof_corr)
    if not cand:
        return pd.DataFrame(), {}
    rows = []
    for l in cand:
        tex = path_texture_vector(l.oof, y, seg, terr, wth, cfg.TERRAIN_MIN_ROWS)
        tex["sense_gap"] = float(l.sense_gap) if np.isfinite(l.sense_gap) else 0.0
        rows.append({"key": l.key, "decision": l.decision, "oof_corr": l.oof_corr,
                     "width": l.width, "wf_corr": l.wf_corr, **tex})
    df = pd.DataFrame(rows)
    M = np.nan_to_num(df[list(TEXTURE_FEATURES)].to_numpy(np.float64), nan=0.0)
    M = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
    df["trail_family"] = leader_cluster(M, cfg.TEXTURE_FAMILY_RADIUS)
    return df, dict(zip(df["key"], df["trail_family"]))


def terrain_trail_report(lessons: list["Lesson"], y: np.ndarray,
                         terr: np.ndarray | None, min_rows: int) -> pd.DataFrame:
    """Per-terrain corr for every promoted (or predator-killed) trail: which
    valleys each path crosses well, and where it dies. Killed lessons stay in
    the report -- the autopsy is part of the map."""
    if terr is None:
        return pd.DataFrame()
    rows = []
    for l in lessons:
        if l.decision not in ("promote", "predator_killed"):
            continue
        for t in np.unique(terr):
            m = terr == t
            if m.sum() >= min_rows:
                rows.append({"key": l.key, "decision": l.decision, "terrain": int(t),
                             "rows": int(m.sum()), "corr": pearson(y[m], l.oof[m])})
    return pd.DataFrame(rows)


def label_archaeology(X: np.ndarray, y: np.ndarray, seg: np.ndarray, cols: list[str],
                      lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """v18 LABEL ARCHAEOLOGY -- the target is anonymized, so dig up its fossil
    record. For each feature sweep corr(feature_{t-h}, y_t) over a grid of lags
    h, and measure y's OWN autocorrelation. If y turns out to be (a transform
    of) some feature's FUTURE value -- common in anonymized comps -- the best
    (feature, lag) pair reveals it and reframes the whole problem. TRAIN-SIDE
    ONLY, zero leakage: it never touches test and gates nothing; it is a map."""
    step = max(1, len(y) // 60_000)
    Xs, ys = X[::step], np.asarray(y[::step], np.float64)
    rows = []
    # y autocorrelation (the label's own memory / horizon)
    yac = []
    for h in lags:
        if len(ys) > h + 100:
            yac.append((h, pearson(ys[h:], ys[:-h])))
    # per-lag best feature: corr(X shifted back by h, y)
    for h in lags:
        if len(ys) <= h + 100:
            continue
        c = corr_vector(Xs[:-h], ys[h:])
        j = int(np.argmax(np.abs(c)))
        rows.append({"lag": h, "best_feature": cols[j] if j < len(cols) else str(j),
                     "best_lagged_corr": float(c[j]),
                     "y_autocorr": float(dict(yac).get(h, float("nan")))})
    return pd.DataFrame(rows)


def symmetry_field_report(X: np.ndarray, y: np.ndarray, cols: list[str],
                          top_n: int = 32) -> pd.DataFrame:
    """Even-vs-odd response field: for each strong feature, compare the signed
    (odd) response corr(y, z) with the folded (even) response corr(y, |z|).
    Even-dominant features are exactly where fold_abs/fold_pairs viewports
    should win -- this report is the measured motivation, not a gate."""
    step = max(1, len(y) // 30_000)
    Xs, ys = X[::step], np.asarray(y[::step], np.float64)
    c = corr_vector(Xs, ys)
    order = np.argsort(-np.abs(c))[:top_n]
    rows = []
    for j in order:
        x = Xs[:, int(j)].astype(np.float64)
        z = (x - x.mean()) / (x.std() + 1e-12)
        odd = pearson(ys, z)
        even = pearson(ys, np.abs(z))
        rows.append({"feature": cols[int(j)], "odd_corr_signed": odd,
                     "even_corr_folded": even, "even_excess": abs(even) - abs(odd),
                     "verdict": "even_dominant_fold_candidate" if abs(even) > abs(odd)
                     else "odd_dominant"})
    return pd.DataFrame(rows)


def dream_replay(oof: np.ndarray, y: np.ndarray, seg: np.ndarray,
                 n_rep: int, rng: np.random.Generator) -> tuple[float, float]:
    """v9: block-bootstrap the trail into the runs the world COULD have shown
    us -- resample whole segments with replacement, recompute pooled corr.
    Returns (dream_p05, dream_p50). Free: stored OOF only, gates nothing."""
    segs = np.unique(seg)
    seg_idx = [np.where(seg == s)[0] for s in segs]
    vals = []
    for _ in range(n_rep):
        pick = rng.integers(0, len(segs), len(segs))
        idx = np.concatenate([seg_idx[int(p)] for p in pick])
        vals.append(pearson(y[idx], oof[idx]))
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.50))


def texture_generalization_report(tex_df: pd.DataFrame) -> pd.DataFrame:
    """v9 meta-report: the harness studying its own trails -- which texture
    dimensions predict walk-forward drift (oof_corr - wf_corr) and width?
    Purely descriptive; n is small, treat as exploratory cartography."""
    if tex_df.empty or "wf_corr" not in tex_df.columns:
        return pd.DataFrame()
    d = tex_df[np.isfinite(tex_df["wf_corr"])].copy()
    if len(d) < 8:
        return pd.DataFrame()
    drift = (d["oof_corr"] - d["wf_corr"]).to_numpy(np.float64)
    width = d["width"].to_numpy(np.float64)
    rows = []
    for f in TEXTURE_FEATURES:
        x = np.nan_to_num(d[f].to_numpy(np.float64), nan=0.0)
        rows.append({"texture": f, "n": len(d),
                     "corr_vs_wf_drift": pearson(x, drift),
                     "corr_vs_width": pearson(x, width)})
    return pd.DataFrame(rows)


def texture_words(row: pd.Series, tex_df: pd.DataFrame) -> str:
    """Translate a trail's texture vector into chronicle words via quantile
    position among all textured trails."""
    words = []
    for f, (lo_w, hi_w) in {"roughness": ("smooth", "rugged"),
                            "side_asym": ("right-tilted", "left-tilted"),
                            "wake_ac1": ("clean-wake", "long-wake"),
                            "steep_affinity": ("flat-walking", "steep-seeking"),
                            "weather_spread": ("all-weather", "fair-weather"),
                            "gait": ("fading", "ascending")}.items():
        v = row.get(f, np.nan)
        col = tex_df[f].to_numpy(np.float64)
        col = col[np.isfinite(col)]
        if not np.isfinite(v) or len(col) < 4:
            continue
        q = float(np.mean(col <= v))
        if q <= 0.25:
            words.append(lo_w)
        elif q >= 0.75:
            words.append(hi_w)
    return ", ".join(words) if words else "unremarkable"


def trail_verb(skill: str, family: str, transform: str) -> str:
    """v13: the chronicle's VERBS -- how a champion moves through the world.
    Pure language; describes, never gates."""
    by_skill = {"swell_rider": "rides the swell", "scout_lattice": "scouts ahead",
                "relay_caravan": "walks the caravan", "steepness_gate": "bets the steep",
                "terrain_router": "rules the valleys", "codebook": "reads the codebook",
                "majority_vote": "calls a vote", "terrace": "terraces the slope",
                "rapids": "runs the rapids", "mlp_assoc": "dreams in layers",
                "nonlinear_assoc": "branches", "gbdt_lib": "branches",
                "theil_sen": "splits the difference", "local_interp": "asks the neighbors",
                "residual_ladder": "climbs rung by rung", "recency_linear": "trusts the dawn",
                "bagged_linear": "walks in committee", "linear_assoc": "draws the line",
                "single_factor": "follows one star", "bin_association": "counts the bins"}
    by_family = {"mycelium": "following pheromone", "weather": "watching the sky",
                 "terrain": "reading the atlas", "watershed": "claiming a valley",
                 "springs": "drinking from springs", "echo": "hearing echoes",
                 "shadow": "searching the dark", "periphery": "glancing sideways",
                 "compass": "trusting true north", "dawn": "facing the sunrise",
                 "both_clocks": "checking both clocks", "medoid": "keeping the medoids"}
    by_tf = {"doppler": "listening for motion", "lateral_line": "feeling the flow",
             "fold_abs": "folding the land", "fold_pairs": "folding the land",
             "prism": "refracting the light", "moire": "watching the interference",
             "tide": "reading the tide", "dual_exposure": "seeing double",
             "rank": "judging by order", "sign_only": "seeing in one bit",
             "quantize2": "squinting hard", "quantize4": "squinting",
             "quantize8": "squinting a little", "foveated": "fixing its fovea"}
    parts = [by_skill.get(skill, "walks")]
    if family in by_family:
        parts.append(by_family[family])
    if transform in by_tf:
        parts.append(by_tf[transform])
    return ", ".join(parts)


def build_trap_map(X: np.ndarray, y: np.ndarray, seg: np.ndarray,
                   cfg: HarnessConfig) -> tuple[set[int], pd.DataFrame]:
    """v10 threat-first scan: a MIRAGE is a feature that looks strong pooled
    but flips its corr sign across folds -- the desert lake explorers walk
    toward and die. Marked before any budget is spent; corr-driven families
    demote traps to the back of their rankings. Target-aware but fold-honest
    in effect: it only REORDERS candidate rankings computed per fold anyway."""
    step = max(1, len(y) // 30_000)
    Xs, ys, ss = X[::step], y[::step], seg[::step]
    c_full = corr_vector(Xs, ys)
    order = np.argsort(-np.abs(c_full))[: cfg.TRAP_SCAN_TOP]
    signs = []
    for tr, va in purged_segment_splits(ss, cfg.N_SPLITS, 1):
        signs.append(np.sign(corr_vector(Xs[va][:, order], ys[va])))
    S = np.vstack(signs)                              # (folds, top)
    maj = np.sign(S.sum(axis=0) + 1e-9)
    flip = (S != maj).mean(axis=0)
    strong = np.abs(c_full[order]) >= np.median(np.abs(c_full[order]))
    traps = {int(order[j]) for j in range(len(order))
             if flip[j] >= cfg.TRAP_FLIP_RATE and strong[j]}
    df = pd.DataFrame([{"col_idx": int(order[j]), "full_abs_corr": float(abs(c_full[order[j]])),
                        "fold_flip_rate": float(flip[j]),
                        "verdict": "MIRAGE" if int(order[j]) in traps else "solid"}
                       for j in range(len(order))]).sort_values("fold_flip_rate", ascending=False)
    return traps, df


def jnd_probe(X: np.ndarray, y: np.ndarray, seg: np.ndarray, cols: list[str],
              cfg: HarnessConfig, embargo: int) -> dict[str, Any]:
    """v10 sensory-threshold calibration: plant synthetic alpha of known
    strength s into a COPY of the labels (within-run permutation of y plus
    s * z(one real feature)) and ask whether the standard draft pipeline
    DETECTS it (draft width clears the absolute bar). The psychometric curve
    is the harness's measured information-processing floor. Gates nothing."""
    rng = np.random.default_rng(stable_seed(cfg.SEED, "jnd"))
    step = max(1, len(y) // 30_000)
    c = corr_vector(X[::step], y[::step])
    j = int(np.argsort(-np.abs(c))[min(50, X.shape[1] - 1)])   # a mid-strength direction
    xz = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-9)
    yz = (y - y.mean()) / (y.std() + 1e-9)
    spec = ViewportSpec(name="top16_identity", family="top", k=16,
                        transform="identity", proj_dim=8)
    curve = []
    for s in cfg.JND_STRENGTHS:
        y_s = (s * xz + math.sqrt(max(0.0, 1 - s * s)) * yz[rng.permutation(len(yz))]).astype(np.float32)
        d = run_draft("linear_assoc", spec, X, y_s, seg, cols, cfg, embargo,
                      stable_seed(cfg.SEED, "jnd", s))
        curve.append({"planted_strength": float(s),
                      "draft_corr": d["draft_corr"], "draft_width": d["draft_width"],
                      "detected": bool(d["draft_width"] >= cfg.DRAFT_ABS_PASS)})
    detected = [r["planted_strength"] for r in curve if r["detected"]]
    return {"curve": curve, "jnd": (min(detected) if detected else None),
            "note": "smallest planted corr the standard draft pipeline detects; calibration only"}


def write_chronicle(parts: dict[str, Any]) -> None:
    """v9: world_chronicle.md -- the run as a story. A report, not a gate;
    the map of the world deserves a legend humans actually read."""
    L: list[str] = []
    A = L.append
    A("# World Chronicle -- " + str(parts.get("title", "explorer run")))
    A("")
    A(f"The world had {parts['features']} visible minerals across "
      f"{parts['train_rows']} recorded minutes ({parts['data_source']}); "
      f"the last {parts['sealed_rows']} minutes were sealed behind glass before "
      f"anyone was allowed to look.")
    A("")
    A("## The land and the sky")
    tp = parts.get("terrain_pop", {})
    if tp:
        A(f"The atlas mapped {len(tp)} terrains (valley populations: "
          + ", ".join(f"T{t}={n}" for t, n in tp.items()) + ").")
    wp = parts.get("weather_pop", {})
    if wp:
        names = {0: "calm", len(wp) - 1: "storm"}
        A(f"The gauge read {len(wp)} weathers ("
          + ", ".join(f"{names.get(s, 'mid')}={n}" for s, n in wp.items()) + " rows).")
    if parts.get("even_dominant", 0):
        A(f"{parts['even_dominant']} strong minerals answered to magnitude rather than "
          f"direction -- folding country.")
    A("")
    A("## The expedition")
    for line in parts.get("embodiment_lines", []):
        A("- " + line)
    A("")
    A("## The explorers")
    for line in parts.get("explorer_lines", []):
        A("- " + line)
    A("")
    A("## Trails and textures")
    A(f"{parts.get('n_lessons', 0)} trails were walked; {parts.get('n_promoted', 0)} "
      f"held under both geometries. The texture atlas grouped them into "
      f"{parts.get('n_families', 0)} trail families.")
    for line in parts.get("champion_lines", []):
        A("- " + line)
    A("")
    A("## The predator's ledger")
    kills = parts.get("kill_lines", [])
    if kills:
        for line in kills:
            A("- " + line)
    else:
        A("Every attacked trail survived the predator (sub-period, terrain, weather, "
          "null tax, perturbation).")
    A("")
    A("## Dreams")
    dl = parts.get("dream_lines", [])
    if dl:
        for line in dl:
            A("- " + line)
    else:
        A("No promoted trail was dreamed (nothing to bootstrap).")
    A("")
    A("## The shipped party and the sealed verdict")
    A(parts.get("shipping_line", ""))
    if parts.get("sealed_line"):
        A(parts["sealed_line"])
    A("")
    A("*Every mechanism in this chronicle entered through the same doors: draft "
      "gate -> dual-geometry width -> predator -> sealed silence. The chronicle "
      "describes; it never gates.*")
    (OUT / "world_chronicle.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("world_chronicle_written", sections=6)


