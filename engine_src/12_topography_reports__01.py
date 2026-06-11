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


