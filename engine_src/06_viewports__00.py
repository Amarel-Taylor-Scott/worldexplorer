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
            "stabsel", "irm") \
    + (("sign_stability",) if CFG.SIGNSTAB_FAMILY else ())
# v24: stabsel/irm = bootstrap-L1 stability selection + invariant-risk (slope) selection
# v28: sign_stability = the 4th-place sign-flip gate as a ranker family (flag-gated)
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
