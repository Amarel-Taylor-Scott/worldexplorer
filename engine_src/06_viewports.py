# ----------------------------------------------------------------------------
# 4. Viewports (bounded windows, fold-honest, rank-cached)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewportSpec:
    name: str
    family: str             # top | anon | market | decor | stable | medoid | lastN
    k: int
    transform: str          # identity | quantize8 | pca | pair_aug | rand_proj | signed_hadamard | pca_aug
    proj_dim: int = 16


FAMILIES = ("top", "anon", "market", "decor", "stable", "medoid", "lastN", "dawn", "both_clocks",
            "terrain", "weather", "mycelium", "shadow", "periphery", "compass",
            "springs", "watershed", "echo", "beacon", "fault", "phyllotaxis", "invariant",
            "head", "mid", "tail",   # v22: positional feature-ORDER blocks (general; feature order is signal)
            "stabsel", "irm")        # v24: bootstrap-L1 stability selection + invariant-risk (slope) selection
ALL_TRANSFORMS = ("identity", "quantize8", "quantize4", "quantize2", "rank", "sign_only",
                  "pca", "pair_aug", "rand_proj", "signed_hadamard", "pca_aug", "foveated",
                  "fold_abs", "fold_pairs", "dual_exposure", "doppler", "lateral_line",
                  "prism", "moire", "tide", "fractal", "reaction_diffusion",
                  "random_fourier", "curvature", "lorentz_boost")

# bits-per-feature proxy for the BIT_BUDGET frontier (input-side accounting;
# pca/rand_proj reduce dimension downstream -- documented proxy, not a claim)
TRANSFORM_BITS: dict[str, int] = {
    "identity": 32, "pca": 32, "pair_aug": 32, "rand_proj": 32,
    "signed_hadamard": 32, "pca_aug": 32, "fold_abs": 32, "fold_pairs": 32,
    "quantize8": 8, "foveated": 8, "rank": 6, "quantize4": 4, "quantize2": 2, "sign_only": 1,
    "dual_exposure": 9,                 # rank(5 bits) + quantize4(4 bits), two eyes
    "doppler": 32,                      # level + causal delta channels (motion sense)
    "lateral_line": 32,                 # neighbor-relative flow channels (fish near-field sense)
    "prism": 96,                        # v13: 3 spectral bands -> frontier pushes prism to tiny k
    "moire": 64,                        # v13: level + level x local-agitation interference
    "tide": 32,                         # v13: the causal long swell subtracted
    "fractal": 96,                      # v16: 3-scale self-similar pyramid (trees/fractals)
    "reaction_diffusion": 64,           # v16: level + activator-inhibitor band (Turing spots/stripes)
    "random_fourier": 32,               # v18: RFF lift -> ridge approximates RBF-kernel ridge (fabric expansion)
    "curvature": 32,                    # v18: 2nd causal difference (acceleration; the dual of doppler velocity)
    "lorentz_boost": 32,                # v19: relativistic level/velocity mix (moving observer; beta = row volatility)
}

# Global target-free atlas of the space (set once per run by the harness;
# None during direct run_lesson calls -- every consumer degrades gracefully).
ATLAS: "TerrainAtlas | None" = None

# Global target-free weather gauge (v9): row-local volatility states.
GAUGE: "WeatherGauge | None" = None

# Global target-free pressure gauge (v20): order-book microstructure states.
PRESSURE: "PressureGauge | None" = None

# Mycelium pheromone map (v9): col_idx -> accumulated promoted-lesson credit.
# Deposited by SharedLibrary.add on promotion; read by the mycelium family.
# Reset at run start. Dict ops are GIL-atomic; a stale read in a lane thread
# only yields a slightly older ranking, which the honest doors absorb.
MYCELIUM: dict[int, float] = {}

# v10 embodiment globals (reset per run; same GIL-atomicity note as MYCELIUM):
TRAPS: set[int] = set()            # mirage feature indices (threat-first scan)
TABOO: dict[str, float] = {}       # venom memory: motif -> kill penalty
SURVEY: dict[str, float] = {}      # satellite signal density per family

# v11 menagerie globals (reset per run):
QUORUM: dict[str, set] = {}        # family -> set of distinct species that promoted there (bee/bacterial consensus)
DANCES: list[tuple] = []           # waggle dances: (quality, family, transform, k) posted by rich finds
GENE_POOL: list = []               # promoted Genomes available for horizontal transfer (plasmid donors)

# v21 forensic globals (reset per run): feature-cluster ids (target-free) and a
# row-habitat sensor (volume bands); populated lazily by the forensic layer.
FCLUST: "np.ndarray | None" = None
HABITAT: "np.ndarray | None" = None
QUARANTINE: set = set()            # working-region rows the row-court flagged (measured; applied only if it wins)

# v14 ecology globals (reset per run):
# RED_MYCELIUM is the repellent channel: predator kills + trap mirages deposit
# here on the columns they used; corr-driven rankings subtract it. Same
# GIL-atomic dict-op note as MYCELIUM (the green attractant channel).
RED_MYCELIUM: dict[int, float] = {}
SEEDBANK: list = []                # measured-loser Genomes germinated from a prior run's cairn (temporal biodiversity)


class WeatherGauge:
    """Target-free, ROW-LOCAL weather: each row's instantaneous dispersion
    (mean |z| over the gauge columns -- market features first, high-variance
    anonymous fill) quantile-binned into WEATHER_STATES bands. Row-local means
    order-free: assign() is exact on any row subset (folds, sealed, test),
    with zero leakage -- y never touches the gauge. State 0 = calm ... last
    state = storm."""

    def __init__(self, n_states: int) -> None:
        self.n_states = int(n_states)
        self.g_idx: list[int] = []
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.edges: np.ndarray | None = None

    def fit(self, X: np.ndarray, cols: list[str]) -> "WeatherGauge":
        mkt = [i for i, c in enumerate(cols) if not c.startswith("X")]
        var = X[:: max(1, len(X) // 20_000)].var(axis=0)
        anon = [int(i) for i in np.argsort(-var) if cols[int(i)].startswith("X")]
        self.g_idx = (mkt[:16] + anon[: max(0, 8 - len(mkt[:16]))]) or list(range(min(8, X.shape[1])))
        sub = X[:: max(1, len(X) // 60_000)][:, self.g_idx]
        self.mu = sub.mean(axis=0, keepdims=True).astype(np.float32)
        self.sd = (sub.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        v = np.abs((sub - self.mu) / self.sd).mean(axis=1)
        qs = np.linspace(0.0, 1.0, self.n_states + 1)[1:-1]
        self.edges = np.quantile(v, qs).astype(np.float32)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        v = np.abs((X[:, self.g_idx] - self.mu) / self.sd).mean(axis=1)
        return np.searchsorted(self.edges, v).astype(np.int32)


class PressureGauge:
    """v20 MICROSTRUCTURE PRESSURE -- the order-book twin of the weather gauge.
    Weather bins generic row dispersion; pressure bins a signed ORDER-BOOK
    STRESS built from the engineered market features (order/flow imbalance,
    Kyle lambda, depth depletion, stress, signed volume). Row-local, target-
    free (y never touches it), so assign() is exact on any slice. State 0 =
    slack book ... last state = maximum pressure. Crypto microstructure regimes
    that generic dispersion misses; weather was the v9 winner, this is its
    flow-aware sibling. Degrades to the weather signal if no market columns."""

    def __init__(self, n_states: int) -> None:
        self.n_states = int(n_states)
        self.p_idx: list[int] = []
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.edges: np.ndarray | None = None

    def fit(self, X: np.ndarray, cols: list[str]) -> "PressureGauge":
        want = ("mkt_order_imbalance", "mkt_flow_imbalance", "mkt_kyle_lambda",
                "mkt_depth_depletion", "mkt_stress", "mkt_signed_volume",
                "mkt_buy_ratio", "mkt_vol_per_trade")
        idx = {c: i for i, c in enumerate(cols)}
        self.p_idx = [idx[c] for c in want if c in idx]
        if not self.p_idx:                       # no market features -> fall back to high-var dispersion
            var = X[:: max(1, len(X) // 20_000)].var(axis=0)
            self.p_idx = [int(i) for i in np.argsort(-var)[:8]]
        sub = X[:: max(1, len(X) // 60_000)][:, self.p_idx]
        self.mu = sub.mean(axis=0, keepdims=True).astype(np.float32)
        self.sd = (sub.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        v = np.abs((sub - self.mu) / self.sd).mean(axis=1)
        qs = np.linspace(0.0, 1.0, self.n_states + 1)[1:-1]
        self.edges = np.quantile(v, qs).astype(np.float32)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        v = np.abs((X[:, self.p_idx] - self.mu) / self.sd).mean(axis=1)
        return np.searchsorted(self.edges, v).astype(np.int32)


class TerrainAtlas:
    """Unsupervised map of the feature space, fit on WORKING-region X only --
    y never touches it, so terrain ids are leak-free everywhere (sealed rows
    and the test set included). Provides:
      assign(X)   : terrain id per row (MiniBatchKMeans regimes over market +
                    high-variance columns) -- "which valley is this minute in"
      altitude(X) : distance to the row's own terrain centroid -- "how far up
                    the mountain from the valley floor"
      f_rank      : features ranked by ANOVA-F BETWEEN terrains (target-free)
                    -- the mountains' defining minerals (the 'terrain' family)."""

    def __init__(self, n_clusters: int, seed: int) -> None:
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.km: MiniBatchKMeans | None = None
        self.d_idx: list[int] = []
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, cols: list[str], max_rows: int) -> "TerrainAtlas":
        mkt = [i for i, c in enumerate(cols) if not c.startswith("X")]
        var = X[:: max(1, len(X) // 20_000)].var(axis=0)
        anon = [int(i) for i in np.argsort(-var) if cols[int(i)].startswith("X")][:16]
        self.d_idx = (mkt[:24] + anon) or list(range(min(16, X.shape[1])))
        D = X[:: max(1, len(X) // max_rows)][:, self.d_idx]
        self.mu = D.mean(axis=0, keepdims=True).astype(np.float32)
        self.sd = (D.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.seed,
                                  n_init=3, batch_size=4096).fit((D - self.mu) / self.sd)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        D = (X[:, self.d_idx] - self.mu) / self.sd
        return self.km.predict(D).astype(np.int32)

    def altitude(self, X: np.ndarray) -> np.ndarray:
        D = (X[:, self.d_idx] - self.mu) / self.sd
        t = self.km.predict(D)
        return np.linalg.norm(D - self.km.cluster_centers_[t], axis=1).astype(np.float32)

    def f_rank(self, X_tr: np.ndarray, pool: list[int]) -> list[int]:
        step = max(1, len(X_tr) // 30_000)
        Xs = X_tr[::step]
        ts = self.assign(Xs)
        gm = Xs[:, pool].mean(axis=0)
        between = np.zeros(len(pool), np.float64)
        within = np.zeros(len(pool), np.float64) + 1e-9
        for c in np.unique(ts):
            m = ts == c
            if m.sum() < 20:
                continue
            mu_c = Xs[m][:, pool].mean(axis=0)
            between += float(m.sum()) * (mu_c - gm) ** 2
            within += ((Xs[m][:, pool] - mu_c) ** 2).sum(axis=0)
        f = between / within
        return [pool[i] for i in np.argsort(-f)]

class BeaconField:
    """v15: ITEMS DROPPED AT UNIQUE TYPOLOGIES that emit a radial field warping
    the feature space around them. The user's idea, made fold-honest: beacons
    are placed at TARGET-FREE coordinates (rare-terrain centroids + high-
    altitude novelty peaks from the atlas -- y never touches them), and each
    emits a Gaussian RBF field

        field_b(row) = exp(-||z(row) - beacon_b||^2 / (2 sigma_b^2))

    over the atlas's standardized discriminative space. These fields become
    NEW FEATURE CHANNELS appended to the matrix -- every explorer can see them,
    rank them, and bend its model around the landmarks (how the feature space
    flows near a dropped item). Leak-free everywhere (folds, sealed, test):
    positions + bandwidths are functions of X and the target-free atlas only.
    'Appropriate training' is then the ordinary doors -- a field channel that
    is noise gets a low |corr| and is never selected."""

    def __init__(self, atlas: "TerrainAtlas", cfg: HarnessConfig) -> None:
        self.atlas = atlas
        self.cfg = cfg
        self.centroids: np.ndarray | None = None    # (B, d) in standardized atlas space
        self.sigmas: np.ndarray | None = None        # (B,)
        self.kinds: list[str] = []                   # 'rare' | 'novelty' per beacon

    def _z(self, X: np.ndarray) -> np.ndarray:
        return ((X[:, self.atlas.d_idx] - self.atlas.mu) / self.atlas.sd).astype(np.float32)

    def fit(self, X: np.ndarray) -> "BeaconField":
        Z = self._z(X)
        n = len(Z)
        ts = self.atlas.assign(X)
        alt = self.atlas.altitude(X)
        beacons, kinds = [], []
        # (1) RARE TERRAINS: drop a beacon at each sparse valley's centroid
        for t in np.unique(ts):
            m = ts == t
            if 0 < m.sum() < self.cfg.BEACON_RARE_FRAC * n:
                beacons.append(Z[m].mean(axis=0))
                kinds.append("rare")
        # (2) NOVELTY PEAKS: cluster the highest-altitude (most-outlying) rows
        hi = np.where(alt >= np.quantile(alt, 0.97))[0]
        if len(hi) >= self.cfg.BEACON_ALTITUDE_PEAKS * 5:
            npk = min(self.cfg.BEACON_ALTITUDE_PEAKS, self.cfg.BEACON_MAX - len(beacons))
            if npk > 0:
                km = MiniBatchKMeans(n_clusters=npk, random_state=self.cfg.SEED,
                                     n_init=3, batch_size=2048).fit(Z[hi])
                for c in km.cluster_centers_:
                    beacons.append(c.astype(np.float32))
                    kinds.append("novelty")
        if not beacons:                              # always drop at least one landmark
            beacons.append(Z[np.argmax(alt)])
            kinds.append("novelty")
        self.centroids = np.vstack(beacons[: self.cfg.BEACON_MAX]).astype(np.float32)
        self.kinds = kinds[: self.cfg.BEACON_MAX]
        # per-beacon bandwidth: the classic median-distance RBF heuristic on a
        # row sample (target-free) -- the field's natural reach in this space
        sub = Z[:: max(1, n // 20_000)]
        sig = []
        for b in self.centroids:
            d = np.linalg.norm(sub - b, axis=1)
            sig.append(max(1e-3, float(np.median(d)) * self.cfg.BEACON_BANDWIDTH))
        self.sigmas = np.asarray(sig, np.float32)
        return self

    def field(self, X: np.ndarray) -> np.ndarray:
        """(n, B) RBF activations -- the dropped items' radial influence."""
        Z = self._z(X)
        out = np.empty((len(Z), len(self.centroids)), np.float32)
        for b in range(len(self.centroids)):
            d2 = np.sum((Z - self.centroids[b]) ** 2, axis=1)
            out[:, b] = np.exp(-d2 / (2.0 * self.sigmas[b] ** 2))
        return out

    def assign(self, X: np.ndarray) -> np.ndarray:
        """Nearest-beacon basin id per row (which landmark dominates here)."""
        return np.argmax(self.field(X), axis=1).astype(np.int32)


# Global beacon field (set once per run; None when BEACON_DROP is off or no atlas).
BEACONS: "BeaconField | None" = None

# (fold_signature, family[, k]) -> ranked feature indices. The signature hashes
# strided y CONTENT, so shuffled-label null runs rebuild their own rankings.
_RANK_CACHE: dict[tuple, list[int]] = {}


def _fold_sig(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple:
    step = max(1, len(X_tr) // 4096)
    yh = hashlib.sha1(np.ascontiguousarray(y_tr[::step]).tobytes()).hexdigest()[:12]
    return (X_tr.shape[0], X_tr.shape[1], round(float(X_tr[::step, 0].sum()), 4), yh)


def _family_pool(family: str, cols: list[str]) -> list[int]:
    if family == "anon":
        idx = [i for i, c in enumerate(cols) if c.startswith("X")]
    elif family == "market":
        idx = [i for i, c in enumerate(cols) if not c.startswith("X")]
    elif family == "beacon":
        # v15: fish in the dropped items' field channels (+ a few strong reals
        # so a beacon viewport is never starved when no field channel helps)
        idx = [i for i, c in enumerate(cols) if c.startswith("beacon_")]
        if not idx:
            idx = list(range(len(cols)))
    elif family == "lastN":
        xs = [i for i, c in enumerate(cols) if c.startswith("X")] or list(range(len(cols)))
        idx = xs[-CFG.LASTN_BLOCK:]
    elif family in ("head", "mid", "tail"):
        # v22: contiguous feature-ORDER blocks (the host appended its strongest
        # features last -> the trailing block often carries the cleanest alpha).
        # General, not DRW-specific: head/mid/tail are all tested; measurement
        # picks. Falls through to corr-ranking WITHIN the block.
        xs = [i for i, c in enumerate(cols) if c.startswith("X")] or list(range(len(cols)))
        b = min(CFG.POSITIONAL_BLOCK, len(xs))
        if family == "head":
            idx = xs[:b]
        elif family == "tail":
            idx = xs[-b:]
        else:
            s = max(0, (len(xs) - b) // 2)
            idx = xs[s:s + b]
    else:
        idx = list(range(len(cols)))
    return idx or list(range(len(cols)))


def _compass_rankvec(ordered: list[int], cand: list[int]) -> np.ndarray:
    """v11 compass helper: map an ordered feature list to a rank-position
    vector aligned to `cand` (position 0 = best). Lower = more-agreed-upon."""
    pos = {c: r for r, c in enumerate(ordered)}
    return np.array([pos.get(c, len(cand)) for c in cand], np.float64)


def _rank_stable(X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray, pool: list[int]) -> list[int]:
    segs = np.unique(seg_tr)
    if len(segs) < 2:
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        return [pool[i] for i in np.argsort(-c)]
    per = []
    for s in segs:
        m = seg_tr == s
        per.append(corr_vector(X_tr[m][:, pool], y_tr[m]) if m.sum() >= 50 else np.zeros(len(pool)))
    P = np.vstack(per)
    score = np.abs(P.mean(axis=0)) - P.std(axis=0)
    return [pool[i] for i in np.argsort(-score)]


def _rank_decor(X_tr: np.ndarray, ranked: list[int], k: int) -> list[int]:
    cand = ranked[: min(len(ranked), max(2 * k, 64), 320)]
    sub = X_tr[:: max(1, len(X_tr) // 20_000)][:, cand]
    sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-6)
    C = np.abs(np.corrcoef(sub, rowvar=False))
    keep: list[int] = []
    kept_pos: list[int] = []
    for i in range(len(cand)):
        if all(C[i, j] < 0.90 for j in kept_pos):
            keep.append(cand[i])
            kept_pos.append(i)
        if len(keep) >= k:
            break
    return keep or cand[:k]


def _rank_medoid(X_tr: np.ndarray, y_tr: np.ndarray, pool: list[int], threshold: float) -> list[int]:
    """DRW 1st-place style: leader-cluster features at |corr|>=threshold over
    the top-|target-corr| candidates, keep each cluster's medoid."""
    c = np.abs(corr_vector(X_tr[:, pool], y_tr))
    order = np.argsort(-c)
    cand = [pool[i] for i in order[: min(CFG.MEDOID_POOL, len(pool))]]
    if len(cand) < 3:
        return cand
    sub = X_tr[:: max(1, len(X_tr) // 12_000)][:, cand]
    sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-6)
    C = np.abs(np.corrcoef(sub, rowvar=False))
    clusters: list[list[int]] = []
    for i in range(len(cand)):
        placed = False
        for members in clusters:
            if C[i, members[0]] >= threshold:
                members.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    medoid_pos: list[int] = []
    for members in clusters:
        if len(members) == 1:
            medoid_pos.append(members[0])
        else:
            sums = C[np.ix_(members, members)].sum(axis=1)
            medoid_pos.append(members[int(np.argmax(sums))])
    medoid_pos.sort()
    return [cand[p] for p in medoid_pos]


def _ranked_for(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
                cols: list[str]) -> list[int]:
    sig = _fold_sig(X_tr, y_tr)
    cache_key = (sig, spec.family) if spec.family != "decor" else (sig, "decor", spec.k)
    hit = _RANK_CACHE.get(cache_key)
    if hit is not None:
        return hit
    pool = _family_pool(spec.family, cols)
    if spec.family == "stable":
        ranked = _rank_stable(X_tr, y_tr, seg_tr, pool)
    elif spec.family == "medoid":
        ranked = _rank_medoid(X_tr, y_tr, pool, CFG.MEDOID_THRESHOLD)
    elif spec.family in ("dawn", "both_clocks"):
        ranked = _rank_two_clocks(X_tr, y_tr, pool, rising_only=(spec.family == "dawn"))
    elif spec.family == "terrain":
        if ATLAS is not None:
            ranked = ATLAS.f_rank(X_tr, pool)     # fully target-free ranking
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "weather":
        # robust intersection across weather bands: rank by the WEAKEST
        # per-state |corr| -- alpha that survives storms (v9)
        if GAUGE is not None:
            wth = GAUGE.assign(X_tr)
            worst = np.full(len(pool), np.inf, np.float64)
            seen = 0
            for s in np.unique(wth):
                msk = wth == s
                if msk.sum() < 300:
                    continue
                worst = np.minimum(worst, np.abs(corr_vector(X_tr[msk][:, pool], y_tr[msk])))
                seen += 1
            if seen >= 2:
                ranked = [pool[i] for i in np.argsort(-worst)]
            else:
                c = np.abs(corr_vector(X_tr[:, pool], y_tr))
                ranked = [pool[i] for i in np.argsort(-c)]
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "pressure":
        # v20 MICROSTRUCTURE: the order-book twin of 'weather' -- rank features
        # by their WEAKEST |corr| across order-book pressure states (alpha that
        # survives both slack and stressed books). Same robust-intersection
        # idea, on flow/depth pressure instead of generic dispersion.
        if PRESSURE is not None:
            prs = PRESSURE.assign(X_tr)
            worst = np.full(len(pool), np.inf, np.float64)
            seen = 0
            for s in np.unique(prs):
                msk = prs == s
                if msk.sum() < 300:
                    continue
                worst = np.minimum(worst, np.abs(corr_vector(X_tr[msk][:, pool], y_tr[msk])))
                seen += 1
            ranked = ([pool[i] for i in np.argsort(-worst)] if seen >= 2
                      else [pool[i] for i in np.argsort(-np.abs(corr_vector(X_tr[:, pool], y_tr)))])
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "mycelium":
        # stigmergy: follow the pheromone other explorers' PROMOTED lessons
        # deposited on these columns; |corr| as scent while the net is young
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        if MYCELIUM:
            scent = np.array([MYCELIUM.get(int(i), 0.0) for i in pool], np.float64)
            if CFG.MYCELIUM_SATURATE:
                # v23: sqrt-saturate the pheromone (sublinear -- a rich column
                # cannot run away) and let |corr| genuinely co-rank, both
                # normalized to [0,1]. Breaks the v12/v19 monoculture feedback
                # loop (scent -> reuse -> more scent) while keeping mycelium a
                # useful family. corr alone still leads when the net is young.
                s_n = np.sqrt(np.maximum(scent, 0.0))
                s_n = s_n / (s_n.max() + 1e-12)
                c_n = c / (c.max() + 1e-12)
                score = 0.65 * s_n + 0.35 * c_n
            else:
                score = scent + 0.01 * c              # legacy: corr only breaks ties
        else:
            score = c
        ranked = [pool[i] for i in np.argsort(-score)]
    elif spec.family == "shadow":
        # negative space: HIGH variance, LOW |corr| -- the big quiet regions
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        v = X_tr[:: max(1, len(X_tr) // 20_000)][:, pool].var(axis=0)
        loud = v >= np.quantile(v, CFG.SHADOW_VAR_Q)
        quiet = np.where(loud, -c, -np.inf)           # among loud, quietest first
        ranked = [pool[i] for i in np.argsort(-quiet)]
    elif spec.family == "periphery":
        # v10: motion in the corner of the eye -- |corr| shift between the
        # early 75% and late 25% of the fold, discounted where the mycelium
        # is already thick (everyone is fixating there anyway)
        cut = max(100, int(0.75 * len(y_tr)))
        c_early = np.abs(corr_vector(X_tr[:cut][:, pool], y_tr[:cut]))
        c_late = (np.abs(corr_vector(X_tr[cut:][:, pool], y_tr[cut:]))
                  if len(y_tr) - cut >= 100 else c_early)
        shift = np.abs(c_late - c_early)
        pher = np.array([MYCELIUM.get(int(i), 0.0) for i in pool], np.float64)
        ranked = [pool[i] for i in np.argsort(-(shift / (1.0 + 10.0 * pher)))]
    elif spec.family == "springs":
        # v13 persistence wells: slow geology, not fast weather -- rank by
        # lag-1 self-autocorrelation x |corr| (a spring that flows today
        # flowed yesterday too); both measured on the training fold only
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        sub = X_tr[:: max(1, len(X_tr) // 20_000)][:, pool]
        if len(sub) >= 200:
            a, b = sub[:-1], sub[1:]
            az = (a - a.mean(0)) / (a.std(0) + 1e-9)
            bz = (b - b.mean(0)) / (b.std(0) + 1e-9)
            ac1 = np.clip((az * bz).mean(0), 0.0, None)
            ranked = [pool[i] for i in np.argsort(-(ac1 * c))]
        else:
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "watershed":
        # v13 valley specialists: how much the BEST single-terrain |corr|
        # exceeds the pooled |corr| -- the expert of ONE valley (the exact
        # complement of 'weather', which demands all-band robustness)
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        if ATLAS is not None:
            t_ids = ATLAS.assign(X_tr)
            best_t = np.zeros(len(pool), np.float64)
            seen_t = 0
            for t in np.unique(t_ids):
                m = t_ids == t
                if m.sum() < 300:
                    continue
                best_t = np.maximum(best_t, np.abs(corr_vector(X_tr[m][:, pool], y_tr[m])))
                seen_t += 1
            ranked = ([pool[i] for i in np.argsort(-(best_t - c))] if seen_t >= 2
                      else [pool[i] for i in np.argsort(-c)])
        else:
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "echo":
        # v13: columns still ringing with YESTERDAY'S outcome -- ranked by
        # |corr(x_t, y_{t-1})| on the training fold. The model still maps
        # x -> y; only the RANKING listens backward (no test-time y needed).
        if len(y_tr) > 300:
            c = np.abs(corr_vector(X_tr[1:][:, pool], y_tr[:-1]))
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "fault":
        # v16 CRACKS: rank features by the largest DISCONTINUITY in their
        # per-segment corr between ADJACENT time segments -- where the
        # feature's relationship to y FRACTURES at a regime boundary. The
        # complement of 'stable'/'springs': fault surfaces the breaks so a
        # regime-aware skill (recency, caravan, weather) can model them.
        segs = np.unique(seg_tr)
        if len(segs) >= 3:
            per = []
            for s in segs:
                m = seg_tr == s
                per.append(corr_vector(X_tr[m][:, pool], y_tr[m]) if m.sum() >= 50
                           else np.zeros(len(pool)))
            P = np.vstack(per)
            frac = np.abs(np.diff(P, axis=0)).max(axis=0)     # biggest adjacent jump
            ranked = [pool[i] for i in np.argsort(-frac)]
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "invariant":
        # v19 INVARIANT FEATURE COURT (causal robustness): not "which features
        # correlate most" but "which keep their relationship to y across MANY
        # ENVIRONMENTS" -- time segments AND target-free terrain/weather states.
        # score = mean|corr| - lambda * std(corr across worlds). Distinct from
        # 'stable' (temporal only): invariance survives environment partitions,
        # the features least likely to break under regime shift.
        worlds = []
        for s in np.unique(seg_tr):
            m = seg_tr == s
            if m.sum() >= 80:
                worlds.append(corr_vector(X_tr[m][:, pool], y_tr[m]))
        for organ in (ATLAS, GAUGE):
            if organ is not None:
                try:
                    ids = organ.assign(X_tr)
                    for s in np.unique(ids):
                        m = ids == s
                        if m.sum() >= 80:
                            worlds.append(corr_vector(X_tr[m][:, pool], y_tr[m]))
                except Exception:
                    pass
        if len(worlds) >= 3:
            W = np.vstack(worlds)
            score = np.abs(W.mean(axis=0)) - np.std(W, axis=0)   # mean signal minus cross-world instability
            ranked = [pool[i] for i in np.argsort(-score)]
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "stabsel":
        # v24 STABILITY SELECTION (Meinshausen-Buhlmann): rank features by how
        # OFTEN they survive an L1 (Lasso) fit across bootstrap subsamples of
        # the training fold -- finite-sample false-discovery control, the
        # principled "which of 800 features are real". Pre-screen to the top
        # STABSEL_POOL by |corr| (fold-honest), stabilize among those, append
        # the rest. Cached per (fold, family) so it is paid once, not per lesson.
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        order0 = np.argsort(-c)
        ntop = min(CFG.STABSEL_POOL, len(pool))
        cand = [pool[i] for i in order0[:ntop]]
        rest = [pool[i] for i in order0[ntop:]]
        try:
            step = max(1, len(X_tr) // 20000)
            Xc = X_tr[::step][:, cand].astype(np.float64)
            yc = y_tr[::step].astype(np.float64)
            Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
            yc = yc - yc.mean()
            freq = np.zeros(len(cand), np.float64)
            rng_s = np.random.default_rng(stable_seed(CFG.SEED, "stabsel", len(cand), len(Xc)))
            nB = max(2, int(CFG.STABSEL_BOOT))
            half = max(2, len(cand) // 2)
            for _ in range(nB):
                bi = rng_s.integers(0, len(Xc), len(Xc))
                jc = rng_s.choice(len(cand), size=half, replace=False)
                try:
                    m = Lasso(alpha=CFG.STABSEL_ALPHA, max_iter=300)
                    m.fit(Xc[bi][:, jc], yc[bi])
                    freq[jc[np.abs(m.coef_) > 1e-8]] += 1.0
                except Exception:
                    pass
            freq /= nB
            ranked = [cand[i] for i in np.argsort(-freq, kind="stable")] + rest
        except Exception:
            ranked = [pool[i] for i in order0]
    elif spec.family == "irm":
        # v24 INVARIANT-RISK (IRM-flavoured) feature selection: keep features
        # whose univariate SLOPE to y is INVARIANT across ENVIRONMENTS (time
        # segments + target-free terrain/weather states). score = |mean_e b_e|
        # - std_e(b_e) - penalty*signflip*|mean_e b_e|. The slope twin of
        # 'invariant' (which uses |corr|), explicitly punishing sign flips --
        # the causally-stable signal least likely to break under regime shift.
        envs = []
        for s in np.unique(seg_tr):
            m = seg_tr == s
            if m.sum() >= 80:
                envs.append(m)
        for organ in (ATLAS, GAUGE):
            if organ is not None:
                try:
                    ids = organ.assign(X_tr)
                    for s in np.unique(ids):
                        m = ids == s
                        if m.sum() >= 80:
                            envs.append(m)
                except Exception:
                    pass
        if len(envs) >= 3:
            slopes = []
            for m in envs:
                Xe = X_tr[m][:, pool].astype(np.float64)
                ye = y_tr[m].astype(np.float64)
                Xz = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-9)
                slopes.append((Xz * (ye - ye.mean())[:, None]).mean(0))   # univariate slope/feature
            S = np.vstack(slopes)
            mean_b = np.abs(S.mean(axis=0))
            std_b = S.std(axis=0)
            flip = np.mean(np.sign(S) != np.sign(S.mean(axis=0))[None, :], axis=0)
            score = mean_b - std_b - CFG.IRM_SIGNFLIP_PENALTY * flip * mean_b
            ranked = [pool[i] for i in np.argsort(-score)]
        else:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            ranked = [pool[i] for i in np.argsort(-c)]
    elif spec.family == "phyllotaxis":
        # v16 SPIRALS: sunflower-seed packing maximizes coverage. Order
        # features by |corr|, then SELECT by a golden-ratio low-discrepancy
        # stride so the chosen viewport spreads across the corr spectrum
        # (strong + medium together) -- deterministic decorrelation by optimal
        # spacing, the phyllotaxis dual of greedy 'decor'.
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        by_corr = [pool[i] for i in np.argsort(-c)]
        nN = len(by_corr)
        phi = 0.6180339887498949
        order, seen = [], set()
        for i in range(nN):
            p = int((i * phi % 1.0) * nN)
            while p in seen:
                p = (p + 1) % nN
            seen.add(p)
            order.append(by_corr[p])
        ranked = order
    elif spec.family == "compass":
        # v11 bird multi-sensor navigation: rank by AGREEMENT across three
        # target-free frames -- magnetic (terrain separation), sun (time
        # stability across segments), star (weather-band robustness). The
        # features all compasses point at are true north. Restricted to the
        # strongest |corr| candidates so the three rank-agreements are cheap.
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        cand = [pool[i] for i in np.argsort(-c)[: min(CFG.COMPASS_POOL, len(pool))]]
        ranks = []
        # sun compass: stability across time segments (mean - std of per-seg corr)
        ranks.append(_compass_rankvec(_rank_stable(X_tr, y_tr, seg_tr, cand), cand))
        # magnetic compass: separation between terrains (atlas f_rank), target-free
        if ATLAS is not None:
            ranks.append(_compass_rankvec(ATLAS.f_rank(X_tr, cand), cand))
        # star compass: weakest-weather-band robustness, target-free bands
        if GAUGE is not None:
            wth = GAUGE.assign(X_tr)
            worst = np.full(len(cand), np.inf, np.float64)
            seen = 0
            for s in np.unique(wth):
                m = wth == s
                if m.sum() < 300:
                    continue
                worst = np.minimum(worst, np.abs(corr_vector(X_tr[m][:, cand], y_tr[m])))
                seen += 1
            if seen >= 2:
                ranks.append(_compass_rankvec([cand[i] for i in np.argsort(-worst)], cand))
        consensus = np.mean(np.vstack(ranks), axis=0) if ranks else c[np.argsort(-c)][: len(cand)]
        ranked = [cand[i] for i in np.argsort(consensus)] + [p for p in pool if p not in set(cand)]
    elif spec.family == "decor":
        base_key = (sig, "top")
        base = _RANK_CACHE.get(base_key)
        if base is None:
            c = np.abs(corr_vector(X_tr[:, pool], y_tr))
            base = [pool[i] for i in np.argsort(-c)]
            _RANK_CACHE[base_key] = base
        ranked = _rank_decor(X_tr, base, spec.k)
    else:
        c = np.abs(corr_vector(X_tr[:, pool], y_tr))
        ranked = [pool[i] for i in np.argsort(-c)]
    if spec.family in ("top", "anon", "stable", "lastN", "dawn", "both_clocks",
                       "springs", "watershed", "echo"):
        # v14 repellent stigmergy: poisoned columns (predator kills + trap
        # mirages deposited red pheromone here) get a SOFT positional penalty
        # -- explorers are warned away from bad ground, not just drawn to good
        if RED_MYCELIUM:
            mx = max(RED_MYCELIUM.values()) + 1e-9
            span = len(ranked)
            pos = {col: r for r, col in enumerate(ranked)}
            ranked = sorted(ranked, key=lambda col: pos[col]
                            + CFG.RED_PHEROMONE_W * span * RED_MYCELIUM.get(int(col), 0.0) / mx)
        # v10 threat-first: mirage features go to the HARD back of corr-driven
        # rankings (stable partition -- order within each group preserved)
        if TRAPS:
            ranked = [i for i in ranked if i not in TRAPS] + [i for i in ranked if i in TRAPS]
    _RANK_CACHE[cache_key] = ranked
    return ranked


def _rank_two_clocks(X_tr: np.ndarray, y_tr: np.ndarray, pool: list[int], rising_only: bool) -> list[int]:
    """Two clocks: full training fold vs its most recent 25% (fold rows are
    time-ordered). 'dawn' (rising_only) ranks features whose recent-clock corr
    EXCEEDS their full-clock corr first -- a measured bet on the new regime.
    'both_clocks' ranks by the weaker of the two clocks (robust intersection)."""
    cut = int(0.75 * len(X_tr))
    c_full = np.abs(corr_vector(X_tr[:, pool], y_tr))
    if len(X_tr) - cut >= 200:
        c_rec = np.abs(corr_vector(X_tr[cut:][:, pool], y_tr[cut:]))
    else:
        c_rec = c_full
    if rising_only:
        rising = c_rec > c_full
        score = np.where(rising, c_rec, c_rec - 1.0)   # rising first, then rest by recent corr
    else:
        score = np.minimum(c_full, c_rec)
    return [pool[i] for i in np.argsort(-score)]


def _pair_op(op: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if op == 0:
        return a * b
    if op == 1:
        return a - b
    if op == 2:
        return a + b
    return a / (np.abs(b) + 1.0)


def build_viewport(spec: ViewportSpec, X_tr: np.ndarray, y_tr: np.ndarray, seg_tr: np.ndarray,
                   cols: list[str], rng: np.random.Generator) -> tuple[list[int], Callable[[np.ndarray], np.ndarray]]:
    ranked = _ranked_for(spec, X_tr, y_tr, seg_tr, cols)
    idx = ranked[: spec.k]

    Z_tr = X_tr[:, idx]
    mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
    signs = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=len(idx)) if spec.transform == "signed_hadamard" else None
    proj = (rng.normal(size=(len(idx), min(spec.proj_dim, len(idx)))).astype(np.float32)
            / math.sqrt(min(spec.proj_dim, len(idx)))) if spec.transform == "rand_proj" else None

    # v18 random_fourier (FABRIC EXPANSION): lift features into a random
    # trigonometric basis so a downstream linear/ridge fit approximates an
    # RBF-KERNEL ridge -- curved feature-space capacity in the winning family.
    # Bandwidth = median pairwise distance heuristic (target-free).
    rff_W = rff_b = rff_mu = rff_sd = None
    if spec.transform == "random_fourier" and len(idx) >= 2:
        rff_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        rff_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        sub = ((Z_tr[:: max(1, len(Z_tr) // 4000)] - rff_mu) / rff_sd).astype(np.float32)
        gamma = 1.0 / max(1e-6, float(np.median(np.var(sub, axis=0))) * len(idx))
        D = min(256, max(32, 4 * len(idx)))
        rff_W = (rng.normal(size=(len(idx), D)).astype(np.float32) * math.sqrt(2.0 * gamma))
        rff_b = (rng.uniform(0, 2 * math.pi, size=D)).astype(np.float32)

    comps = None
    if spec.transform in ("pca", "pca_aug"):
        sub = Z_tr[:: max(1, len(Z_tr) // 20_000)] - mu
        try:
            _, _, vt = np.linalg.svd(sub, full_matrices=False)
            n_comp = min(spec.proj_dim if spec.transform == "pca" else 8, vt.shape[0])
            comps = vt[:n_comp].astype(np.float32)
        except Exception:
            comps = None

    q_lo = q_hi = None
    q_levels = {"quantize8": 255, "quantize4": 15, "quantize2": 3}.get(spec.transform, 15)
    if spec.transform in ("quantize8", "quantize4", "quantize2", "dual_exposure"):
        q_lo = np.percentile(Z_tr, 0.5, axis=0, keepdims=True).astype(np.float32)
        q_hi = np.percentile(Z_tr, 99.5, axis=0, keepdims=True).astype(np.float32)

    dual_grids = None
    if spec.transform == "dual_exposure":
        qs = np.linspace(0.0, 1.0, 32)[1:-1]            # 30 interior quantiles ~ 5 bits
        dual_grids = np.quantile(Z_tr, qs, axis=0).astype(np.float32)   # (30, k)

    # v11 lateral line (fish near-field flow): each feature's deviation from
    # the local consensus of its most-correlated neighbors. Built on the
    # training fold only (feature-feature corr, no y) -> fold-honest.
    lat_mu = lat_sd = lat_nbr = None
    if spec.transform == "lateral_line" and len(idx) >= 3:
        lat_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        lat_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        sub = ((Z_tr[:: max(1, len(Z_tr) // 20_000)] - lat_mu) / lat_sd)
        C = np.abs(np.nan_to_num(np.corrcoef(sub.T), nan=0.0))
        np.fill_diagonal(C, -1.0)
        nn = min(CFG.LATERAL_NEIGHBORS, len(idx) - 1)
        lat_nbr = np.argsort(-C, axis=1)[:, :nn].astype(np.int32)   # (k, nn) neighbor cols

    # v13 prism: train-fold spectral band edges (3-band piecewise light)
    prism_lo = prism_hi = None
    if spec.transform == "prism":
        prism_lo = np.quantile(Z_tr, 0.33, axis=0, keepdims=True).astype(np.float32)
        prism_hi = np.quantile(Z_tr, 0.66, axis=0, keepdims=True).astype(np.float32)

    # v13 moire: viewport self-dispersion interference stats
    moire_mu = moire_sd = None
    if spec.transform == "moire":
        moire_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        moire_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    # v8 folding: global (fold_abs) and regional (fold_pairs) space symmetry
    fold_mu = fold_sd = None
    fold_pair_idx: list[tuple[int, int]] = []
    if spec.transform in ("fold_abs", "fold_pairs"):
        fold_mu = Z_tr.mean(axis=0, keepdims=True).astype(np.float32)
        fold_sd = (Z_tr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        if spec.transform == "fold_pairs" and len(idx) >= 4:
            sub = (Z_tr[:: max(1, len(Z_tr) // 24_000)] - fold_mu) / fold_sd
            C = np.corrcoef(sub.T)
            C = np.nan_to_num(C, nan=0.0)
            order = np.dstack(np.unravel_index(np.argsort(C, axis=None), C.shape))[0]
            used: set[int] = set()
            for i, j in order:                      # most anti-correlated first
                i, j = int(i), int(j)
                if i >= j or i in used or j in used or C[i, j] >= -0.05:
                    continue
                fold_pair_idx.append((i, j))
                used.update((i, j))
                if len(fold_pair_idx) >= CFG.FOLD_PAIRS:
                    break

    rank_grids = None
    if spec.transform == "rank":
        qs = np.linspace(0.0, 1.0, 66)[1:-1]            # 64 interior quantiles ~ 6 bits
        rank_grids = np.quantile(Z_tr, qs, axis=0).astype(np.float32)   # (64, k)

    sign_med = None
    if spec.transform == "sign_only":
        sign_med = np.median(Z_tr, axis=0, keepdims=True).astype(np.float32)

    fov = None
    if spec.transform == "foveated" and len(idx) > 8:
        nf = 8
        npr = min(64, len(idx) - nf)
        peri = slice(nf, nf + npr)
        back = slice(nf + npr, len(idx))
        mu_p = Z_tr[:, peri].mean(axis=0, keepdims=True).astype(np.float32)
        comps_p = None
        try:
            subp = Z_tr[:: max(1, len(Z_tr) // 20_000), peri] - mu_p
            _, _, vtp = np.linalg.svd(subp, full_matrices=False)
            comps_p = vtp[: min(4, vtp.shape[0])].astype(np.float32)
        except Exception:
            comps_p = None
        p_lo = p_hi = None
        if comps_p is not None:
            P_tr = (Z_tr[:, peri] - mu_p) @ comps_p.T
            p_lo = np.percentile(P_tr, 0.5, axis=0, keepdims=True).astype(np.float32)
            p_hi = np.percentile(P_tr, 99.5, axis=0, keepdims=True).astype(np.float32)
        back_w = back_mu = back_sd = None
        if back.stop > back.start:
            bw = corr_vector(Z_tr[:, back], y_tr)
            s_abs = float(np.sum(np.abs(bw)))
            if s_abs > 1e-12:
                back_w = (bw / s_abs).astype(np.float32)
                back_mu = Z_tr[:, back].mean(axis=0).astype(np.float32)
                back_sd = (Z_tr[:, back].std(axis=0) + 1e-6).astype(np.float32)
        fov = {"nf": nf, "peri": peri, "back": back, "mu_p": mu_p, "comps": comps_p,
               "p_lo": p_lo, "p_hi": p_hi, "bw": back_w, "bmu": back_mu, "bsd": back_sd}

    pair_m, pair_mu, pair_sd, pair_keep = 0, None, None, []
    if spec.transform == "pair_aug" and len(idx) >= 2:
        pair_m = min(CFG.PAIR_BASE, len(idx))
        pair_mu = Z_tr[:, :pair_m].mean(0).astype(np.float32)
        pair_sd = (Z_tr[:, :pair_m].std(0) + 1e-6).astype(np.float32)
        step = max(1, len(Z_tr) // 24_000)
        Bz = (Z_tr[::step, :pair_m] - pair_mu) / pair_sd
        ys = y_tr[::step]
        feats, ops = [], []
        for i in range(pair_m):
            for j in range(i + 1, pair_m):
                for op in range(4):
                    feats.append(_pair_op(op, Bz[:, i], Bz[:, j]))
                    ops.append((op, i, j))
        if feats:
            F = np.stack(feats, axis=1).astype(np.float32)
            sc = np.abs(corr_vector(F, ys))
            pair_keep = [ops[t] for t in np.argsort(-sc)[: CFG.PAIR_KEEP]]

    def transform(Z: np.ndarray) -> np.ndarray:
        if spec.transform == "signed_hadamard" and Z.shape[1] >= 2:
            return fwht(Z * signs)
        if spec.transform == "rand_proj" and proj is not None:
            return Z @ proj
        if spec.transform == "pca" and comps is not None:
            return (Z - mu) @ comps.T
        if spec.transform == "pca_aug" and comps is not None:
            return np.concatenate([Z, (Z - mu) @ comps.T], axis=1).astype(np.float32)
        if spec.transform in ("quantize8", "quantize4", "quantize2") and q_lo is not None:
            span = np.maximum(q_hi - q_lo, 1e-6)
            code = np.clip(np.round((Z - q_lo) / span * q_levels), 0, q_levels)
            return (q_lo + code / float(q_levels) * span).astype(np.float32)
        if spec.transform == "doppler":
            # the motion sense: levels + causal first differences. First row's
            # delta is zero; rows after fold gaps carry slightly stale deltas
            # (a handful per fold) -- measured approximation, documented.
            D = np.diff(Z, axis=0, prepend=Z[:1])
            return np.concatenate([Z, D], axis=1).astype(np.float32)
        if spec.transform == "prism" and prism_lo is not None:
            # refraction: the same light split into three spectral bands --
            # piecewise-linear sight for linear skills (train-fold quantiles)
            return np.concatenate([Z, Z * (Z <= prism_lo), Z * (Z >= prism_hi)],
                                  axis=1).astype(np.float32)
        if spec.transform == "moire" and moire_mu is not None:
            # interference: each column times the viewport's OWN local
            # agitation -- regime-conditional slopes with no gate model
            Zz = (Z - moire_mu) / moire_sd
            agitation = np.abs(Zz).mean(axis=1, keepdims=True)
            return np.concatenate([Zz, Zz * agitation], axis=1).astype(np.float32)
        if spec.transform == "tide":
            # the slow swell subtracted (causal EMA); rows after fold gaps
            # carry slightly stale tide -- same documented caveat as doppler
            ema = pd.DataFrame(Z).ewm(span=CFG.TIDE_SPAN, adjust=False).mean().to_numpy(np.float32)
            return (Z - ema).astype(np.float32)
        if spec.transform == "fractal":
            # v16 TREES/FRACTALS: the same signal at three resolutions, a
            # self-similar multiresolution pyramid (level + two coarse-grained
            # causal scales). Mandelbrot's self-similarity made a viewport.
            zdf = pd.DataFrame(Z)
            s1 = zdf.ewm(span=8, adjust=False).mean().to_numpy(np.float32)
            s2 = zdf.ewm(span=32, adjust=False).mean().to_numpy(np.float32)
            return np.concatenate([Z, s1, s2], axis=1).astype(np.float32)
        if spec.transform == "reaction_diffusion":
            # v16 SPOTS/STRIPES (Turing): activator (short-range excitation)
            # MINUS inhibitor (long-range diffusion) -- the band-pass morphogen
            # channel that makes standing-wave patterns. Distinct from tide
            # (high-pass): this keeps the MID band where patterns live.
            zdf = pd.DataFrame(Z)
            act = zdf.ewm(span=4, adjust=False).mean().to_numpy(np.float32)
            inh = zdf.ewm(span=32, adjust=False).mean().to_numpy(np.float32)
            return np.concatenate([Z, (act - inh)], axis=1).astype(np.float32)
        if spec.transform == "random_fourier" and rff_W is not None:
            # the fabric expansion: sqrt(2/D) cos(Wz + b), a Monte-Carlo RBF
            # feature map -- ridge on these = approximate kernel ridge
            Zz = (Z - rff_mu) / rff_sd
            return (math.sqrt(2.0 / rff_W.shape[1])
                    * np.cos(Zz @ rff_W + rff_b)).astype(np.float32)
        if spec.transform == "curvature":
            # acceleration: the SECOND causal difference (the dual of doppler's
            # velocity) -- where the world's motion is itself changing
            d1 = np.diff(Z, axis=0, prepend=Z[:1])
            d2 = np.diff(d1, axis=0, prepend=d1[:1])
            return np.concatenate([Z, d2], axis=1).astype(np.float32)
        if spec.transform == "lorentz_boost":
            # v19 RELATIVITY: a moving-observer mix of level x and velocity v.
            # beta = clipped row volatility (target-free, from the gauge idea);
            # calm rows (beta~0) see mostly level (-> degrades to doppler),
            # storm rows see velocity-boosted level. x'=g(x-bv), v'=g(v-bx).
            v = np.diff(Z, axis=0, prepend=Z[:1])
            beta = np.clip(np.abs(v).mean(axis=1, keepdims=True), 0.0, 0.99) * 0.6
            g = 1.0 / np.sqrt(1.0 - beta ** 2 + 1e-6)
            xp = g * (Z - beta * v)
            vp = g * (v - beta * Z)
            return np.concatenate([xp, vp], axis=1).astype(np.float32)
        if spec.transform == "lateral_line" and lat_nbr is not None:
            # near-field flow: levels + each feature's divergence from the
            # local consensus of its correlated neighbors (the eddy it sits in)
            Zz = (Z - lat_mu) / lat_sd
            consensus = Zz[:, lat_nbr].mean(axis=2)        # (n, k) neighbor mean
            return np.concatenate([Zz, Zz - consensus], axis=1).astype(np.float32)
        if spec.transform == "dual_exposure" and dual_grids is not None and q_lo is not None:
            # two eyes on the same features: rank (order) + quantize4 (magnitude)
            r = np.empty(Z.shape, dtype=np.float32)
            for j in range(Z.shape[1]):
                r[:, j] = np.searchsorted(dual_grids[:, j], Z[:, j])
            r /= np.float32(dual_grids.shape[0] + 1)
            span = np.maximum(q_hi - q_lo, 1e-6)
            code = np.clip(np.round((Z - q_lo) / span * 15), 0, 15)
            q = (q_lo + code / 15.0 * span).astype(np.float32)
            return np.concatenate([r, q], axis=1).astype(np.float32)
        if spec.transform == "fold_abs" and fold_mu is not None:
            # global folding: reflect every feature about its mean plane --
            # the EVEN-response detector (terrain that rises on both sides)
            return np.abs((Z - fold_mu) / fold_sd).astype(np.float32)
        if spec.transform == "fold_pairs" and fold_mu is not None:
            Zz = (Z - fold_mu) / fold_sd
            if fold_pair_idx:
                folded = np.stack([(Zz[:, i] - Zz[:, j]) * 0.5 for i, j in fold_pair_idx], axis=1)
                ridge_ = np.stack([np.abs(Zz[:, i] + Zz[:, j]) * 0.5 for i, j in fold_pair_idx], axis=1)
                return np.concatenate([Zz, folded, ridge_], axis=1).astype(np.float32)
            return Zz.astype(np.float32)
        if spec.transform == "rank" and rank_grids is not None:
            out = np.empty(Z.shape, dtype=np.float32)
            for j in range(Z.shape[1]):
                out[:, j] = np.searchsorted(rank_grids[:, j], Z[:, j])
            return out / np.float32(rank_grids.shape[0] + 1)
        if spec.transform == "sign_only" and sign_med is not None:
            return np.sign(Z - sign_med).astype(np.float32)
        if spec.transform == "foveated" and fov is not None:
            parts = [Z[:, : fov["nf"]]]
            if fov["comps"] is not None:
                P = (Z[:, fov["peri"]] - fov["mu_p"]) @ fov["comps"].T
                if fov["p_lo"] is not None:
                    span = np.maximum(fov["p_hi"] - fov["p_lo"], 1e-6)
                    code = np.clip(np.round((P - fov["p_lo"]) / span * 255), 0, 255)
                    P = fov["p_lo"] + code / 255.0 * span
                parts.append(P.astype(np.float32))
            if fov["bw"] is not None:
                bcol = ((Z[:, fov["back"]] - fov["bmu"]) / fov["bsd"]) @ fov["bw"]
                parts.append(bcol.reshape(-1, 1).astype(np.float32))
            return np.concatenate(parts, axis=1).astype(np.float32)
        if spec.transform == "pair_aug" and pair_keep:
            Zb = (Z[:, :pair_m] - pair_mu) / pair_sd
            extra = np.stack([_pair_op(op, Zb[:, i], Zb[:, j]) for op, i, j in pair_keep], axis=1)
            return np.concatenate([Z, extra], axis=1).astype(np.float32)
        return Z

    return idx, transform


