# ----------------------------------------------------------------------------
# 6. Lessons (dual geometry) + shared library + draft gate
# ----------------------------------------------------------------------------

@dataclass
class Lesson:
    explorer: str
    stage: str
    skill: str
    viewport: str
    family: str
    transform: str
    key: str
    seed: int
    oof: np.ndarray
    fold_corrs: list[float]
    oof_corr: float
    width: float
    seed_var: float
    stability: float
    fit_corr: float
    overfit_ratio: float
    uniqueness: float
    cost: int
    decision: str
    reason: str
    wf_fold_corrs: list[float] = field(default_factory=list)
    wf_corr: float = float("nan")
    wf_width: float = float("nan")
    era_corr: float = float("nan")
    draft_width: float = float("nan")
    null_corr: float = float("nan")
    deflated_corr: float = float("nan")
    k: int = 0
    worst3_corr: float = float("nan")
    perturb_width: float = float("nan")
    terrain_min_corr: float = float("nan")
    predator_verdict: str = ""
    weather_min_corr: float = float("nan")
    beacon_min_corr: float = float("nan")   # v15: worst populated beacon-basin corr
    palindrome_corr: float = float("nan")   # v18: corr after a time-REVERSED refit (inversion = momentum mirage)
    recent_corr: float = float("nan")   # v13: mean corr over the FINAL 2 segments (gait's tail)
    dream_p05: float = float("nan")
    dream_p50: float = float("nan")
    used_cols: tuple[int, ...] = ()     # columns the full fit actually used (mycelium substrate)
    sense_gap: float = float("nan")     # |pearson - spearman| of the OOF (tail-dependence sense)
    surprise: float = float("nan")      # |predicted width - measured width| (set by the library)


# v30: the wide-path annealing clock -- lessons measured so far this run
# (reset by the harness at run start; advanced by SharedLibrary.add).
WIDTH_BIAS = {"n": 0}


def width_share() -> float:
    """v30: the search currency's WIDTH share. Starts at WIDTH_BIAS_START
    (initial bias toward WIDE paths -- robust lower-bound strength over raw
    corr) and anneals with a half-life in LESSONS toward the v4 0.5 balance.
    0.5 = exact historical no-op; the bias shapes the EARLY trajectory only."""
    start = float(getattr(CFG, "WIDTH_BIAS_START", 0.5))
    hl = max(1.0, float(getattr(CFG, "WIDTH_BIAS_HALFLIFE", 60)))
    return 0.5 + (start - 0.5) * (0.5 ** (WIDTH_BIAS["n"] / hl))


def width_bias_beta() -> float:
    """The bandit-side blend strength implied by width_share: 0 at the
    historical 0.5 balance (exact no-op), 1 at a pure-width currency."""
    return max(0.0, min(1.0, 2.0 * (width_share() - 0.5)))


def lesson_fitness(l: Lesson) -> float:
    """v4 fitness: honest signal AND the weaker geometry's robustness.
    v30: the width share anneals from WIDTH_BIAS_START -> 0.5 (initial
    wide-path bias; 0.5 reproduces the historical 50/50 exactly)."""
    wf_w = l.wf_width if np.isfinite(l.wf_width) else l.width
    ww = width_share()
    return (1.0 - ww) * l.oof_corr + ww * min(l.width, wf_w)


def run_lesson(explorer: str, stage: str, skill: str, spec: ViewportSpec,
               X: np.ndarray, y: np.ndarray, seg: np.ndarray, cols: list[str],
               cfg: HarnessConfig, embargo: int, seed: int,
               library_oofs: dict[str, np.ndarray],
               draft_width: float = float("nan")) -> Lesson:
    stochastic = is_stochastic(skill, spec)

    def one_cv(rep_seed: int, with_probe: bool) -> tuple[np.ndarray, list[float], float]:
        rng = np.random.default_rng(rep_seed)
        oof = np.zeros(len(y), np.float32)
        fold_corrs, stabs = [], []
        for tr, va in purged_segment_splits(seg, cfg.N_SPLITS, embargo):
            state = fit_skill(skill, spec, X[tr], y[tr], seg[tr], cols, rng, cfg, rep_seed)
            pred = predict_skill(state, X[va])
            oof[va] = pred
            fold_corrs.append(pearson(y[va], pred))
            if with_probe:
                stabs.append(stability_probe(state, X[va], pred, rng, cfg))
        return oof, fold_corrs, float(np.mean(stabs)) if stabs else 0.0

    # geometry A: leave-segments-out (stationarity)
    # On 2-GPU boxes, the stochastic seed-repetition CV runs CONCURRENTLY with
    # the main CV, one per device -- torch/xgboost release the GIL during GPU
    # compute, so this is the one place a single-process kernel gets true 2x.
    gpu_kind = SKILL_REGISTRY[skill]["kind"] == "mlp"
    executor, rep_future = None, None
    if stochastic and cfg.SEED_REPS_STOCHASTIC > 0 and N_GPUS >= 2 and gpu_kind:
        executor = ThreadPoolExecutor(max_workers=1)
        rep_future = executor.submit(one_cv, seed + 101, False)
    oof, fold_corrs, stability = one_cv(seed, with_probe=True)
    oof_corr = score_metric(y, oof)          # v26: target metric (Pearson default = DRW)
    era_c = era_mean_corr(y, oof, seg)

    # v10 cross-sense: the same path felt with the rank sense (Spearman);
    # a large gap means the alpha is tail-driven -- one sense is being fooled
    def _ranks(a: np.ndarray) -> np.ndarray:
        r = np.empty(len(a), np.float64)
        r[np.argsort(a, kind="stable")] = np.arange(len(a), dtype=np.float64)
        return r
    sense_gap = abs(oof_corr - pearson(_ranks(oof), _ranks(y)))

    # geometry B: purged expanding walk-forward (deployability)
    rng_wf = np.random.default_rng(seed + 7)
    oof_wf = np.zeros(len(y), np.float32)
    covered = np.zeros(len(y), bool)
    wf_corrs: list[float] = []
    for tr, va in walk_forward_splits(seg, cfg.WF_FOLDS, embargo):
        state = fit_skill(skill, spec, X[tr], y[tr], seg[tr], cols, rng_wf, cfg, seed + 7)
        pred = predict_skill(state, X[va])
        oof_wf[va] = pred
        covered[va] = True
        wf_corrs.append(pearson(y[va], pred))
    if wf_corrs:
        wf_corr = pearson(y[covered], oof_wf[covered]) if covered.sum() >= 100 else float(np.mean(wf_corrs))
        wf_width = path_width(wf_corrs)
        wf_available = True
    else:
        wf_corr, wf_width, wf_available = oof_corr, path_width(fold_corrs), False

    seed_var = 0.0
    if stochastic and cfg.SEED_REPS_STOCHASTIC > 0:
        reps = [oof_corr]
        if rep_future is not None:
            o2, _, _ = rep_future.result()           # ran on the other GPU
            reps.append(pearson(y, o2))
            for r in range(1, cfg.SEED_REPS_STOCHASTIC):
                o2, _, _ = one_cv(seed + 101 * (r + 1), with_probe=False)
                reps.append(pearson(y, o2))
        else:
            for r in range(cfg.SEED_REPS_STOCHASTIC):
                o2, _, _ = one_cv(seed + 101 * (r + 1), with_probe=False)
                reps.append(pearson(y, o2))
        seed_var = float(np.std(reps))
    if executor is not None:
        executor.shutdown(wait=False)

    full_state = fit_skill(skill, spec, X, y, seg, cols, np.random.default_rng(seed), cfg, seed)
    fit_corr = pearson(y, predict_skill(full_state, X))
    overfit_ratio = float(fit_corr / max(oof_corr, 1e-6)) if oof_corr > 0 else float("inf")
    used_cols = tuple(int(i) for i in full_state.get("idx", ()))   # mycelium substrate
    del full_state
    free_gpu_mem()                   # per-lesson GPU memory hygiene

    uniq = 1.0
    if library_oofs:
        uniq = float(max(0.0, 1.0 - max(abs(pearson(oof, o)) for o in library_oofs.values())))

    stab_pen = cfg.STABILITY_PENALTY * max(0.0, stability - cfg.STABILITY_TOL)
    width = path_width(fold_corrs, seed_var=seed_var, stability_penalty=stab_pen)

    reasons = []
    if oof_corr <= 0:
        reasons.append("no_honest_correlation")
    if width <= 0:
        reasons.append("narrow_or_negative_width")
    if wf_available and wf_width <= 0:
        reasons.append("negative_walk_forward_width")     # dual-geometry gate
    if overfit_ratio > cfg.MAX_OVERFIT_RATIO:
        reasons.append("overpowering_fit_vs_cv")
    decision = "promote" if not reasons else ("expand" if oof_corr > 0 else "reject")

    return Lesson(explorer, stage, skill, spec.name, spec.family, spec.transform,
                  f"{skill}|{spec.name}", seed, oof, fold_corrs,
                  oof_corr, width, seed_var, stability, fit_corr,
                  min(overfit_ratio, 999.0), uniq, SKILL_REGISTRY[skill]["cost"], decision, "|".join(reasons),
                  wf_fold_corrs=wf_corrs, wf_corr=wf_corr, wf_width=wf_width,
                  era_corr=era_c, draft_width=draft_width, k=spec.k, used_cols=used_cols,
                  sense_gap=sense_gap)


def run_draft(skill: str, spec: ViewportSpec, X: np.ndarray, y: np.ndarray, seg: np.ndarray,
              cols: list[str], cfg: HarnessConfig, embargo: int, seed: int) -> dict[str, float]:
    """Successive-halving stage 1: cheap tile + 2 purged folds + reduced iters.
    Frustum culling / speculate-then-verify from the hardware playbook."""
    step = max(1, len(y) // cfg.DRAFT_ROWS)
    Xd, yd, segd = X[::step], y[::step], seg[::step]
    emb = max(1, embargo // step)
    dcfg = replace(cfg, MLP_MAX_ITER=max(4, cfg.MLP_MAX_ITER // 3),
                   GBDT_ESTIMATORS=max(40, cfg.GBDT_ESTIMATORS // 4),
                   HGB_ITERS=max(30, cfg.HGB_ITERS // 3),
                   KNN_BANK=min(cfg.KNN_BANK, 6000),
                   LADDER_MAX_ROUNDS=max(20, cfg.LADDER_MAX_ROUNDS // 2),
                   MLP_MAX_ROWS=min(cfg.MLP_MAX_ROWS, 15_000))
    rng = np.random.default_rng(seed)
    corrs = []
    for tr, va in purged_segment_splits(segd, cfg.DRAFT_FOLDS, emb):
        state = fit_skill(skill, spec, Xd[tr], yd[tr], segd[tr], cols, rng, dcfg, seed)
        corrs.append(pearson(yd[va], predict_skill(state, Xd[va])))
    return {"draft_corr": float(np.mean(corrs)) if corrs else 0.0,
            "draft_width": path_width(corrs)}


def run_car(skill: str, spec: ViewportSpec, X: np.ndarray, y: np.ndarray, seg: np.ndarray,
            cols: list[str], cfg: HarnessConfig, embargo: int, seed: int) -> dict[str, float]:
    """v10 middle rung of the vehicle fleet: a coarse-time pass (CAR_ROWS,
    CAR_FOLDS) between the airplane draft and the full hike. Cars are fast on
    mapped terrain; an expensive candidate that stalls here never pays full
    hike price."""
    return run_draft(skill, spec, X, y, seg, cols,
                     replace(cfg, DRAFT_ROWS=cfg.CAR_ROWS, DRAFT_FOLDS=cfg.CAR_FOLDS),
                     embargo, seed)


class DraftGate:
    """Adaptive cull bar: pass if draft width clears the running quantile of
    prior drafts, the absolute bar, or during warmup."""

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.widths: list[float] = []
        self.drafted = 0
        self.culled = 0

    def passes(self, width: float) -> bool:
        self.drafted += 1
        hist = list(self.widths)
        self.widths.append(width)
        if len(hist) < self.cfg.DRAFT_WARMUP:
            return True
        if width >= self.cfg.DRAFT_ABS_PASS:
            return True
        ok = width >= float(np.quantile(hist, self.cfg.DRAFT_PASS_QUANTILE))
        if not ok:
            self.culled += 1
        return ok

    def report(self) -> dict[str, Any]:
        return {"drafted": self.drafted, "culled": self.culled,
                "cull_rate": self.culled / max(1, self.drafted),
                "median_draft_width": float(np.median(self.widths)) if self.widths else None}


class SharedLibrary:
    def __init__(self) -> None:
        self.lessons: list[Lesson] = []
        self.gain_by_key: dict[str, list[float]] = {}
        self.width_by_key: dict[str, list[float]] = {}   # v30: robust-width per key (wide-path bias)
        self.runs: dict[str, int] = {}
        self.surprise_ema: dict[str, float] = {}    # v10 predictive-coding residuals

    def predict_width(self, skill: str, family: str, transform: str) -> float | None:
        """The library's world-model: expected width of a candidate from its
        genome-space neighbors (lessons sharing skill, family, or transform)."""
        vals = [l.width for l in self.lessons
                if l.skill == skill or l.family == family or l.transform == transform]
        return float(np.mean(vals)) if len(vals) >= 3 else None

    def coord_surprise(self, skill: str, family: str, transform: str) -> float:
        vals = [self.surprise_ema.get(c, 0.0)
                for c in (f"s:{skill}", f"f:{family}", f"t:{transform}")]
        return float(np.mean(vals))

    def add(self, lesson: Lesson) -> None:
        # v10 surprise: score the prediction BEFORE absorbing the lesson
        pred = self.predict_width(lesson.skill, lesson.family, lesson.transform)
        if pred is not None and np.isfinite(lesson.width):
            lesson.surprise = abs(lesson.width - pred)
            for c in (f"s:{lesson.skill}", f"f:{lesson.family}", f"t:{lesson.transform}"):
                prev = self.surprise_ema.get(c, lesson.surprise)
                self.surprise_ema[c] = 0.7 * prev + 0.3 * lesson.surprise
        self.lessons.append(lesson)
        self.gain_by_key.setdefault(lesson.key, []).append(lesson.oof_corr)
        wf_w = lesson.wf_width if np.isfinite(lesson.wf_width) else lesson.width
        self.width_by_key.setdefault(lesson.key, []).append(
            float(min(lesson.width, wf_w)) if np.isfinite(lesson.width) else 0.0)
        WIDTH_BIAS["n"] += 1                        # v30: advance the wide-path annealing clock
        self.runs[lesson.key] = self.runs.get(lesson.key, 0) + 1
        if lesson.decision == "promote" and lesson.oof_corr > 0:
            # stigmergy: deposit pheromone on the columns this trail used
            for i in lesson.used_cols:
                MYCELIUM[i] = MYCELIUM.get(i, 0.0) + float(lesson.oof_corr)
            # v11 quorum: a phase-1 organism casts a vote for this family
            if lesson.stage in ("microbe", "forager", "navigator", "apex"):
                QUORUM.setdefault(lesson.family, set()).add(lesson.explorer)
            # v11 waggle: a rich find posts a dance for newborn recruits
            if lesson.width >= CFG.WAGGLE_MIN:
                DANCES.append((float(lesson.width), lesson.family, lesson.transform, lesson.k))
            # v11 horizontal gene transfer: the proven genome joins the pool
            GENE_POOL.append(Genome(lesson.skill, lesson.family, lesson.transform, lesson.k))

    def note_draft_cull(self, key: str) -> None:
        self.runs[key] = self.runs.get(key, 0) + 1

    def mean_gain(self, key: str) -> float:
        v = self.gain_by_key.get(key, [])
        return float(np.mean(v)) if v else 0.0

    def family_gain(self, family: str) -> float:
        v = [l.oof_corr for l in self.lessons if l.family == family]
        return float(np.mean(v)) if v else 0.0

    def mean_width(self, key: str) -> float:
        v = self.width_by_key.get(key, [])
        return float(np.mean(v)) if v else 0.0

    def family_width(self, family: str) -> float:
        v = [min(l.width, l.wf_width if np.isfinite(l.wf_width) else l.width)
             for l in self.lessons if l.family == family and np.isfinite(l.width)]
        return float(np.mean(v)) if v else 0.0

    def oofs(self) -> dict[str, np.ndarray]:
        return {f"{l.key}:{i}": l.oof for i, l in enumerate(self.lessons)}

    def promoted(self) -> list[Lesson]:
        return [l for l in self.lessons if l.decision == "promote"]

    def can_run(self, skill: str, spec: ViewportSpec) -> bool:
        cap = 2 if is_stochastic(skill, spec) else 1
        return self.runs.get(f"{skill}|{spec.name}", 0) < cap


