# ----------------------------------------------------------------------------
# 2. Core math (Pearson world) + split geometries
# ----------------------------------------------------------------------------

def pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    if len(y) < 3:
        return 0.0
    sy, sp = float(np.std(y)), float(np.std(p))
    if sy <= 1e-12 or sp <= 1e-12 or not (np.isfinite(sy) and np.isfinite(sp)):
        return 0.0
    return float(np.mean((y - y.mean()) * (p - p.mean())) / (sy * sp))


# ----------------------------------------------------------------------------
# v26 GENERALIZATION SEAM -- pluggable TARGET metric. PROFILE defaults to DRW
# (Pearson on time-ordered data). resolve_profile() may switch the metric under
# cfg.METRIC="auto". Every metric returns higher=better, ~0=neutral, in [-1,1]
# -- a drop-in for all the ">0" / corr-weighting logic, so a binary or RMSE
# target works with the SAME selection machinery. score_metric is what the
# harness's DECISIONS use; corr_vector (feature ranking) stays Pearson because
# point-biserial corr still ranks features for classification.
# ----------------------------------------------------------------------------
PROFILE: dict = {"target_kind": "continuous", "temporal": True, "metric": "pearson"}


def _gini(y, p) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        yv = np.asarray(y, np.float64)
        yb = (yv > np.median(yv)).astype(int) if len(np.unique(yv)) > 2 else yv.astype(int)
        if len(np.unique(yb)) < 2:
            return 0.0
        return float(2.0 * roc_auc_score(yb, np.asarray(p, np.float64)) - 1.0)
    except Exception:
        return pearson(y, p)


def _spearman(y, p) -> float:
    def _rk(a):
        a = np.asarray(a, np.float64); r = np.empty(len(a), np.float64)
        r[np.argsort(a, kind="stable")] = np.arange(len(a), dtype=np.float64); return r
    return pearson(_rk(y), _rk(p))


def _neg_rmse(y, p) -> float:
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return -float(np.sqrt(np.mean((y - p) ** 2)) / (np.std(y) + 1e-9))


def score_metric(y, p) -> float:
    """The harness's TARGET metric for selection/scoring. Pearson by default
    (DRW, byte-equivalent); under cfg.METRIC='auto' the data-profile sensor may
    switch it to gini (classification), spearman, or neg-rmse."""
    m = PROFILE.get("metric", "pearson")
    if m == "gini":
        return _gini(y, p)
    if m == "spearman":
        return _spearman(y, p)
    if m == "rmse":
        return _neg_rmse(y, p)
    return pearson(y, p)


def profile_data(X, y, cols, cfg) -> dict:
    """Data-type sensor: target TYPE (binary/multiclass/continuous) + whether
    rows are TIME-ORDERED (mean lag-1 autocorr of high-variance features -- iid
    tabular ~0, a time series >> 0). Reports a recommendation; whether it is
    APPLIED is governed by cfg.METRIC / cfg.GEOMETRY (see resolve_profile)."""
    y = np.asarray(y); n = len(y); yu = np.unique(y)
    if len(yu) <= 2:
        tk = "binary"
    elif len(yu) <= min(20, max(3, n // 50)) and np.allclose(yu, np.round(yu)):
        tk = "multiclass"
    else:
        tk = "continuous"
    step = max(1, n // 20000); Xs = np.asarray(X[::step], np.float64)
    ac1 = 0.0
    if len(Xs) > 200 and Xs.shape[1] > 0:
        top = np.argsort(-Xs.var(0))[:32]
        a, b = Xs[:-1][:, top], Xs[1:][:, top]
        az = (a - a.mean(0)) / (a.std(0) + 1e-9); bz = (b - b.mean(0)) / (b.std(0) + 1e-9)
        ac1 = float(np.mean(np.abs((az * bz).mean(0))))
    return {"target_kind": tk, "n_unique": int(len(yu)), "feature_autocorr": round(ac1, 4),
            "temporal_detected": bool(ac1 >= cfg.PROFILE_MIN_AC1),
            "recommended_metric": ("gini" if tk in ("binary", "multiclass") else "pearson")}


def resolve_profile(prof, cfg) -> dict:
    """Sensor reading + config -> the active PROFILE. cfg defaults
    ('pearson','temporal') reproduce DRW exactly; 'auto' follows the sensor."""
    metric = prof["recommended_metric"] if cfg.METRIC == "auto" else cfg.METRIC
    temporal = (prof["temporal_detected"] if cfg.GEOMETRY == "auto"
                else cfg.GEOMETRY == "temporal")
    return {"target_kind": prof["target_kind"], "temporal": bool(temporal), "metric": metric}


def era_mean_corr(y: np.ndarray, p: np.ndarray, seg: np.ndarray) -> float:
    """Numerai-style era objective: mean of per-segment scores."""
    vals = []
    for s in np.unique(seg):
        m = seg == s
        if m.sum() >= 50:
            vals.append(score_metric(y[m], p[m]))
    return float(np.mean(vals)) if vals else 0.0


def late_era_mean_corr(y: np.ndarray, p: np.ndarray, seg: np.ndarray) -> float:
    """v13: the era objective restricted to the MOST RECENT third of segments
    -- the ground the deployed model will actually stand on. Used by the
    late_era_hill ensemble strategy, which competes in the same nested honest
    assessment as every other strategy."""
    segs = np.unique(seg)
    take = segs[-max(2, len(segs) // 3):]
    vals = [score_metric(y[seg == s], p[seg == s]) for s in take if (seg == s).sum() >= 50]
    return float(np.mean(vals)) if vals else 0.0


def _worst_seg_corr(y: np.ndarray, p: np.ndarray, seg: np.ndarray) -> float:
    """v18: the WORST per-segment score -- the minimax/antifragile objective.
    Optimizing this breeds blends for the regimes that produce the decay,
    not the comfortable average. Competes in the same honest tournament."""
    vals = [score_metric(y[seg == s], p[seg == s]) for s in np.unique(seg) if (seg == s).sum() >= 50]
    return float(min(vals)) if vals else 0.0


def corr_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = (y - y.mean()).astype(np.float64)
    xn = np.sqrt(np.einsum("ij,ij->j", Xc, Xc, dtype=np.float64))
    yn = math.sqrt(float(yc @ yc))
    num = Xc.T.astype(np.float64) @ yc
    den = xn * yn
    out = np.zeros(X.shape[1], np.float64)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def path_width(corrs: list[float], seed_var: float = 0.0, stability_penalty: float = 0.0) -> float:
    vals = [c for c in corrs if np.isfinite(c)]
    if not vals:
        return -seed_var - stability_penalty
    mu, sigma, worst = float(np.mean(vals)), float(np.std(vals)), float(min(vals))
    return mu - sigma + min(0.0, worst) - seed_var - stability_penalty


def fwht(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    size = 1 << max(1, (d - 1)).bit_length()
    z = np.zeros((n, size), dtype=np.float32)
    z[:, :d] = x
    h = 1
    while h < size:
        for i in range(0, size, h * 2):
            a = z[:, i:i + h].copy()
            b = z[:, i + h:i + 2 * h].copy()
            z[:, i:i + h] = a + b
            z[:, i + h:i + 2 * h] = a - b
        h *= 2
    return z / math.sqrt(size)


def purged_segment_splits(seg: np.ndarray, n_splits: int, embargo: int):
    """Leave-segments-out with embargo stripped from training (geometry A)."""
    n = len(seg)
    splits = min(n_splits, max(2, len(np.unique(seg))))
    for tr, va in GroupKFold(n_splits=splits).split(np.zeros(n), groups=seg):
        va_s = np.sort(va)
        runs, start, prev = [], int(va_s[0]), int(va_s[0])
        for v in va_s[1:]:
            v = int(v)
            if v != prev + 1:
                runs.append((start, prev))
                start = v
            prev = v
        runs.append((start, prev))
        bad = np.zeros(n, bool)
        for a, b in runs:
            bad[max(0, a - embargo): min(n, b + embargo + 1)] = True
        yield tr[~bad[tr]], va


def walk_forward_splits(seg: np.ndarray, n_folds: int, embargo: int):
    """Purged expanding walk-forward (geometry B): train on everything before
    a validation segment, minus an embargo gap. Estimates deployability."""
    segs = np.unique(seg)
    if len(segs) < 3:
        return
    val_segs = segs[-min(n_folds, len(segs) - 2):]
    for vs in val_segs:
        va = np.where(seg == vs)[0]
        tr_end = int(va.min()) - embargo
        if tr_end <= 200:
            continue
        yield np.arange(0, tr_end), va


