# ----------------------------------------------------------------------------
# 8. Phase 2 -- metaheuristic evolution over lesson genomes (mealpy-inspired)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Genome:
    skill: str
    family: str
    transform: str
    k: int

    def repaired(self) -> "Genome":
        skill = self.skill if self.skill in SKILL_REGISTRY else "linear_assoc"
        family = self.family if self.family in FAMILIES else "top"
        transform = self.transform if self.transform in ALL_TRANSFORMS else "identity"
        if SKILL_REGISTRY[skill]["needs_identity"]:
            transform = "identity"
        k = int(np.clip(int(self.k), CFG.K_MIN, CFG.K_MAX))
        # bit-budget frontier: a genome may spend its bits on width OR precision
        k = min(k, max(CFG.K_MIN, CFG.BIT_BUDGET // TRANSFORM_BITS.get(transform, 32)))
        return Genome(skill, family, transform, k)

    def spec(self) -> ViewportSpec:
        return ViewportSpec(name=f"{self.family}{self.k}_{self.transform}", family=self.family,
                            k=self.k, transform=self.transform, proj_dim=32 if self.k >= 64 else 16)

    @property
    def key(self) -> str:
        return f"{self.skill}|{self.family}{self.k}_{self.transform}"


def _rand_cat(rng: np.random.Generator, options) -> str:
    return str(options[int(rng.integers(len(options)))])


def op_de_crossover(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    pick = rng.choice(len(parents), size=3, replace=False)
    a, b, c = (parents[int(i)][0] for i in pick)
    skill = a.skill if rng.random() < 0.6 else (b.skill if rng.random() < 0.5 else c.skill)
    family = a.family if rng.random() < 0.6 else (b.family if rng.random() < 0.5 else c.family)
    transform = a.transform if rng.random() < 0.6 else (b.transform if rng.random() < 0.5 else c.transform)
    k = int(round(a.k + CFG.DE_F * (b.k - c.k)))
    return Genome(skill, family, transform, k).repaired()


def op_gwo_guided(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    leaders = [g for g, _ in sorted(parents, key=lambda t: -t[1])[:3]]
    skill = _rand_cat(rng, [g.skill for g in leaders])
    family = _rand_cat(rng, [g.family for g in leaders])
    transform = _rand_cat(rng, [g.transform for g in leaders])
    k_mean = float(np.mean([g.k for g in leaders]))
    k = int(round(k_mean * (1.0 + rng.normal(0, 0.15))))
    return Genome(skill, family, transform, k).repaired()


def op_woa_spiral(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    best = max(parents, key=lambda t: t[1])[0]
    skill = best.skill if rng.random() >= 0.3 else _rand_cat(rng, list(SKILL_REGISTRY))
    family = best.family if rng.random() >= 0.3 else _rand_cat(rng, FAMILIES)
    transform = best.transform if rng.random() >= 0.3 else _rand_cat(rng, ALL_TRANSFORMS)
    k = int(round(best.k * math.exp(rng.normal(0, 0.25))))
    return Genome(skill, family, transform, k).repaired()


def op_levy_flight(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    base_k = max(parents, key=lambda t: t[1])[0].k if parents else 32
    step = float(np.clip(rng.standard_cauchy() * 0.5, -3.0, 3.0))
    k = int(round(base_k * (2.0 ** step)))
    return Genome(_rand_cat(rng, list(SKILL_REGISTRY)), _rand_cat(rng, FAMILIES),
                  _rand_cat(rng, ALL_TRANSFORMS), k).repaired()


def op_frontier_surf(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    """v9: exploit the measured prior that champions live near the bit-budget
    boundary (v4: bagged_linear|quantize8 k~135 was still climbing). Copy the
    best parent and push k to the frontier for its transform, jittered."""
    best = max(parents, key=lambda t: t[1])[0]
    bits = TRANSFORM_BITS.get(best.transform, 32)
    k_star = max(CFG.K_MIN, CFG.BIT_BUDGET // bits)
    k = int(round(k_star * (1.0 + rng.normal(0, 0.10))))
    return Genome(best.skill, best.family, best.transform, k).repaired()


def op_chemotaxis(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    """v11 bacterial run-and-tumble, v14 with a LOCAL BASELINE: E. coli chase
    concentration above the BACKGROUND, not absolute -- so it cannot be trapped
    in a globally-mediocre basin. Compare a parent pair against the population
    MEDIAN fitness (the local baseline); RUN (extend the step) while the better
    parent BEATS THE ROOM, TUMBLE (randomize) harder when it is merely average."""
    baseline = float(np.median([f for _, f in parents])) if parents else 0.0
    if len(parents) >= 2:
        a, b = (parents[int(i)] for i in rng.choice(len(parents), size=2, replace=False))
        hi, lo = (a, b) if a[1] >= b[1] else (b, a)
        improving = hi[1] > baseline                           # v14: above local background
        run_len = 1.0 + (rng.random() if improving else 0.0)   # extend the run while ahead of the room
        k = int(round(hi[0].k + run_len * (hi[0].k - lo[0].k)))
        g = hi[0]
    else:
        g = parents[0][0]
        improving = parents[0][1] > baseline if parents else False
        k = int(round(g.k * (1.0 + rng.normal(0, 0.2))))
    tumble = rng.random() < (0.2 if improving else 0.5)        # tumble harder when merely average
    transform = _rand_cat(rng, ALL_TRANSFORMS) if tumble else g.transform
    family = _rand_cat(rng, FAMILIES) if tumble else g.family
    return Genome(g.skill, family, transform, k).repaired()


def _rarest_cat(vals: list[str], universe, rng: np.random.Generator) -> str:
    """v14 anti-flock helper: the universe member LEAST represented among the
    parents (ties broken randomly) -- the direction away from the crowd."""
    counts = {u: 0 for u in universe}
    for v in vals:
        if v in counts:
            counts[v] += 1
    lo = min(counts.values())
    rare = [u for u, c in counts.items() if c == lo]
    return str(rare[int(rng.integers(len(rare)))])


def op_antiflock(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    """v14 contrarian scout -- the dual of op_flock. Most explorers chase the
    leaders; a few must walk AGAINST the crowd to catch a regime turn before
    consensus does. Pushes AWAY from the population centroid: rarest
    skill/family/transform among the parents, and k reflected across the
    population mean (mirror image of a random member)."""
    genomes = [g for g, _ in parents]
    if len(genomes) < 2:
        return op_chemotaxis(parents, rng)
    mean_k = float(np.mean([g.k for g in genomes]))
    me = genomes[int(rng.integers(len(genomes)))]
    k = int(round(2.0 * mean_k - me.k))                        # reflect across the mean
    return Genome(_rarest_cat([g.skill for g in genomes], list(SKILL_REGISTRY), rng),
                  _rarest_cat([g.family for g in genomes], FAMILIES, rng),
                  _rarest_cat([g.transform for g in genomes], ALL_TRANSFORMS, rng),
                  k).repaired()


def _genome_dist(a: Genome, b: Genome) -> float:
    return (abs(a.k - b.k) / 64.0 + 2.0 * (a.skill != b.skill)
            + (a.family != b.family) + (a.transform != b.transform))


def op_flock(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    """v11 starling murmuration: align with the ~K TOPOLOGICAL nearest
    neighbors in genome space (not the global leaders) -- emergent order, no
    leader. The child takes the plurality skill/family/transform of the
    recipient's neighborhood and the neighborhood-mean k."""
    me = parents[int(rng.integers(len(parents)))][0]
    nbrs = sorted((g for g, _ in parents), key=lambda g: _genome_dist(me, g))[:7]
    if len(nbrs) < 2:
        return op_chemotaxis(parents, rng)

    def plurality(vals):
        u, c = np.unique(np.array(vals, dtype=object), return_counts=True)
        return u[int(np.argmax(c))]
    skill = plurality([g.skill for g in nbrs])
    family = plurality([g.family for g in nbrs])
    transform = plurality([g.transform for g in nbrs])
    k = int(round(float(np.mean([g.k for g in nbrs]))))
    return Genome(skill, family, transform, k).repaired()


def op_plasmid(parents: list[tuple[Genome, float]], rng: np.random.Generator) -> Genome:
    """v11 horizontal gene transfer: bacteria swap plasmids laterally, not
    only by descent. Graft one proven MOTIF FRAGMENT (skill, transform, or k)
    from any library-promoted genome (the GENE_POOL) onto a current parent."""
    recipient = max(parents, key=lambda t: t[1])[0]
    if not GENE_POOL:
        return op_chemotaxis(parents, rng)
    donor = GENE_POOL[int(rng.integers(len(GENE_POOL)))]
    gene = int(rng.integers(3))
    if gene == 0:
        return Genome(donor.skill, recipient.family, recipient.transform, recipient.k).repaired()
    if gene == 1:
        return Genome(recipient.skill, recipient.family, donor.transform, recipient.k).repaired()
    return Genome(recipient.skill, donor.family, recipient.transform, donor.k).repaired()


EVOLUTION_OPERATORS: dict[str, Callable[[list[tuple[Genome, float]], np.random.Generator], Genome]] = {
    "de_crossover": op_de_crossover,
    "gwo_guided": op_gwo_guided,
    "woa_spiral": op_woa_spiral,
    "levy_flight": op_levy_flight,
    "frontier_surf": op_frontier_surf,
    "chemotaxis": op_chemotaxis,        # v11 bacterial run-and-tumble (v14: local baseline)
    "flock": op_flock,                  # v11 starling murmuration
    "plasmid": op_plasmid,              # v11 horizontal gene transfer
    "antiflock": op_antiflock,          # v14 contrarian scout (the dual of flock)
}

# v14 ISLAND EVOLUTION: each evolutionary epoch is an ISLAND with its own
# operator bias; migration between epochs is the existing re-seed from the
# full library (every island's champions flow in). Allopatric diversity.
ISLAND_PROFILES = (
    {"name": "pioneer",      "boost": ("levy_flight", "woa_spiral", "chemotaxis", "antiflock")},
    {"name": "exploiter",    "boost": ("frontier_surf", "gwo_guided")},
    {"name": "recombinator", "boost": ("plasmid", "de_crossover", "flock")},
)


def parse_genome_key(key: str) -> Genome | None:
    """Parse 'skill|familyK_transform' (e.g. 'bagged_linear|top135_quantize8')
    back into a Genome -- the cross-run warm-start channel."""
    try:
        skill, rest = key.split("|", 1)
        m = re.match(r"^([a-zA-Z_]+?)(\d+)_(.+)$", rest)
        if m is None or skill not in SKILL_REGISTRY:
            return None
        family, k, transform = m.group(1), int(m.group(2)), m.group(3)
        if family not in FAMILIES or transform not in ALL_TRANSFORMS:
            return None
        return Genome(skill, family, transform, k).repaired()
    except Exception:
        return None


class EvolutionEngine:
    def __init__(self, cfg: HarnessConfig, library: SharedLibrary,
                 spec_lookup: dict[str, ViewportSpec], gate: DraftGate) -> None:
        self.cfg = cfg
        self.library = library
        self.spec_lookup = spec_lookup
        self.gate = gate
        self.budget = cfg.EVOLUTION_BUDGET
        self.epoch = 0                  # v12: metabolism refuels evolution in epochs
        self.island_bias: dict[str, float] = {}   # v14: per-epoch operator bias (island)
        self.op_gains: dict[str, list[float]] = {op: [] for op in EVOLUTION_OPERATORS}
        self.op_gains["warm_start"] = []
        self.history: list[dict[str, Any]] = []

    def _seed_population(self) -> list[tuple[Genome, float]]:
        pool: dict[str, tuple[Genome, float]] = {}
        # v10: predator-killed lessons no longer seed the population -- a
        # measured kill is venom, not a parent (their motifs carry taboo too)
        ranked = sorted((l for l in self.library.lessons
                         if l.oof_corr > 0 and l.decision != "predator_killed"),
                        key=lesson_fitness, reverse=True)
        for l in ranked:
            spec = self.spec_lookup.get(l.key)
            if spec is None:
                continue
            g = Genome(l.skill, spec.family, spec.transform, spec.k).repaired()
            if g.key not in pool:
                pool[g.key] = (g, lesson_fitness(l))
            if len(pool) >= self.cfg.EVOLUTION_POP:
                break
        return list(pool.values())

    def _pick_operator(self, rng: np.random.Generator, n_parents: int) -> str:
        ops = list(EVOLUTION_OPERATORS)
        means = np.array([float(np.mean(self.op_gains[o])) if self.op_gains[o] else 0.0 for o in ops])
        w = np.exp(means / self.cfg.OPERATOR_SOFTMAX_TAU)
        w = w / w.sum()
        floor = self.cfg.OPERATOR_FLOOR / len(ops)
        w = np.maximum(w, floor)
        if self.island_bias:               # v14: this epoch's island boosts its operators
            w = w * np.array([self.island_bias.get(o, 1.0) for o in ops])
        w = w / w.sum()
        op = ops[int(rng.choice(len(ops), p=w))]
        if op == "de_crossover" and n_parents < 3:
            op = "levy_flight"
        return op

    def run(self, X: np.ndarray, y: np.ndarray, seg: np.ndarray, cols: list[str],
            embargo: int, journal_rows: list[dict[str, Any]]) -> None:
        cfg = self.cfg
        pop = self._seed_population()
        if not pop:
            log("evolution_skipped", reason="no positive-corr lessons to seed population")
            return
        rng = np.random.default_rng(stable_seed(cfg.SEED, "evolution", self.epoch))
        g_best = max(f for _, f in pop)
        T = cfg.SA_T0
        stale = 0
        # v14 island: this epoch's operator bias (migration = the re-seed above
        # pulled every prior island's champions from the full library)
        if cfg.ISLANDS:
            prof = ISLAND_PROFILES[self.epoch % len(ISLAND_PROFILES)]
            self.island_bias = {o: (3.0 if o in prof["boost"] else 1.0) for o in EVOLUTION_OPERATORS}
            log("evolution_island", epoch=self.epoch, island=prof["name"],
                boosts="|".join(prof["boost"]))
        else:
            self.island_bias = {}
        log("evolution_start", epoch=self.epoch, population=len(pop), budget=self.budget,
            g_best=round(g_best, 5), operators=list(EVOLUTION_OPERATORS))

        for gen in range(cfg.EVOLUTION_MAX_GENERATIONS):
            if self.budget <= 0 or stale >= cfg.EVOLUTION_PATIENCE:
                break
            if META is not None and not META.allow("evolve"):
                log("evolution_out_of_time", epoch=self.epoch, generation=gen,
                    budget_left=self.budget)
                break
            pop_median = float(np.median([f for _, f in pop]))
            produced = 0

            # generate this generation's candidate children up front (cheap);
            # operator credit becomes per-generation adaptive under pairing
            cands: list[tuple[Genome, str]] = []
            tries = 0
            while len(cands) < cfg.EVOLUTION_OFFSPRING and tries < cfg.EVOLUTION_OFFSPRING * 8:
                tries += 1
                # v11: reserve a fraction of offspring for horizontal gene
                # transfer so proven motifs reliably jump (donors = GENE_POOL)
                if GENE_POOL and rng.random() < cfg.HGT_RATE:
                    op_name = "plasmid"
                else:
                    op_name = self._pick_operator(rng, len(pop))
                cand = EVOLUTION_OPERATORS[op_name](pop, rng)
                if SKILL_REGISTRY[cand.skill]["cost"] > self.budget:
                    continue
                if not self.library.can_run(cand.skill, cand.spec()):
                    continue
                if any(c.key == cand.key for c, _ in cands):
                    continue
                venom = (TABOO.get(f"{cand.skill}|{cand.transform}", 0.0)
                         + TABOO.get(f"{cand.skill}|{cand.family}", 0.0))
                if venom >= cfg.TABOO_SKIP:           # v10: don't re-grasp the snake
                    continue
                cands.append((cand, op_name))

            # cross-run warm start: measured champions from prior real runs
            # enter generation 0 first (skipped if already explored/dedup'd)
            if gen == 0:
                warm = []
                # v14 seed bank: a prior run's measured LOSERS germinate first
                # (temporal biodiversity -- the diversity a regime shift rewards)
                warm_keys = list(cfg.WARM_GENOMES)
                if getattr(cfg, "WIDE_SEEDS", False):
                    # v33 wide seeds (user-directed): wide/stable/agreeing
                    # motifs measured through the same gen-0 doors
                    warm_keys += [k for k in cfg.WIDE_WARM_GENOMES if k not in warm_keys]
                germ = [g.key for g in SEEDBANK[: self.cfg.SEED_GERMINATE]] if self.epoch == 0 else []
                if germ:
                    warm_keys = germ + warm_keys
                    log("seedbank_germinate", count=len(germ), keys="|".join(germ))
                for wk in warm_keys:
                    g = parse_genome_key(wk)
                    if (g is not None and self.library.can_run(g.skill, g.spec())
                            and SKILL_REGISTRY[g.skill]["cost"] <= self.budget
                            and all(c.key != g.key for c, _ in cands)):
                        warm.append((g, "warm_start"))
                if warm:
                    cands = warm + cands
                    log("evolution_warm_start", injected=len(warm),
                        keys="|".join(g.key for g, _ in warm))

            def _eval_child(child: Genome, child_op: str, oofs_snap: dict) -> tuple:
                spec_c = child.spec()
                key_c = child.key
                d_width = float("nan")
                if SKILL_REGISTRY[child.skill]["cost"] >= cfg.DRAFT_MIN_COST:
                    d = run_draft(child.skill, spec_c, X, y, seg, cols, cfg, embargo,
                                  stable_seed(cfg.SEED, "draft", key_c))
                    d_width = d["draft_width"]
                    if not self.gate.passes(d_width):
                        return ("culled", child, child_op, d)
                    car = run_car(child.skill, spec_c, X, y, seg, cols, cfg, embargo,
                                  stable_seed(cfg.SEED, "car", key_c))
                    if car["draft_width"] <= 0:       # v10: stalled on the road
                        car["car_stalled"] = 1.0
                        return ("culled", child, child_op, car)
                seed = stable_seed(cfg.SEED, key_c, self.library.runs.get(key_c, 0))
                lesson = run_lesson(f"evo_{child_op}", "evolution", child.skill, spec_c,
                                    X, y, seg, cols, cfg, embargo, seed, oofs_snap,
                                    draft_width=d_width)
                return ("done", child, child_op, lesson)

            # evaluate in lane-paired concurrent batches (gpu lane + cpu lane)
            hetero = CFG.HETERO_PAIRING and N_GPUS > 0
            remaining = list(cands)
            evaluated: list[tuple] = []
            while remaining and self.budget > 0 and (META is None or META.allow("evolve")):
                batch = [remaining.pop(0)]
                if hetero and remaining:
                    l0 = lesson_lane(batch[0][0].skill)
                    for j, c in enumerate(remaining):
                        if lesson_lane(c[0].skill) != l0:
                            batch.append(remaining.pop(j))
                            break
                oofs_snap = self.library.oofs()
                if len(batch) == 2:
                    with ThreadPoolExecutor(max_workers=2) as exe:
                        futs = [exe.submit(_eval_child, c, op, oofs_snap) for c, op in batch]
                        outs = [f.result() for f in futs]
                else:
                    outs = [_eval_child(batch[0][0], batch[0][1], oofs_snap)]
                for status, child, op_name, payload in outs:
                    key = child.key
                    if status == "culled":
                        stalled = bool(payload.get("car_stalled"))
                        self.library.note_draft_cull(key)
                        self.budget -= 2 if stalled else 1
                        self.history.append({"generation": gen, "operator": op_name, "key": key,
                                             "oof_corr": payload["draft_corr"], "width": payload["draft_width"],
                                             "fitness": float("nan"), "gain_vs_pop_median": float("nan"),
                                             "accepted": False, "replaced": "",
                                             "decision": "car_stalled" if stalled else "draft_culled",
                                             "budget_left": self.budget})
                        log("evo_car_stalled" if stalled else "evo_draft_culled",
                            gen=gen, op=op_name, key=key,
                            draft_width=round(payload["draft_width"], 4), budget=self.budget)
                        continue
                    evaluated.append((child, op_name, payload))

            # SA acceptance + credit assignment, sequential and deterministic
            for child, op_name, lesson in evaluated:
                spec = child.spec()
                key = child.key
                self.library.add(lesson)
                self.spec_lookup[lesson.key] = spec
                self.budget -= lesson.cost
                produced += 1
                fit = lesson_fitness(lesson)
                gain = fit - pop_median
                self.op_gains[op_name].append(gain)

                accepted, replaced = False, ""
                if len(pop) < cfg.EVOLUTION_POP:
                    pop.append((child, fit))
                    accepted = True
                else:
                    best_i = int(np.argmax([f for _, f in pop]))
                    j = int(rng.integers(len(pop)))
                    if j == best_i and len(pop) > 1:
                        j = (j + 1) % len(pop)
                    df = fit - pop[j][1]
                    if df > 0 or rng.random() < math.exp(min(0.0, df) / max(T, 1e-9)):
                        replaced = pop[j][0].key
                        pop[j] = (child, fit)
                        accepted = True
                if fit > g_best:
                    g_best = fit
                    stale = -1

                self.history.append({"generation": gen, "operator": op_name, "key": key,
                                     "oof_corr": lesson.oof_corr, "width": lesson.width,
                                     "fitness": fit, "gain_vs_pop_median": gain,
                                     "accepted": accepted, "replaced": replaced,
                                     "decision": lesson.decision, "budget_left": self.budget})
                journal_rows.append({"explorer": f"evo_{op_name}", "lesson_idx": len(self.history),
                                     "stage": "evolution", "key": key, "family": spec.family,
                                     "transform": spec.transform, "k": spec.k,
                                     "oof_corr": lesson.oof_corr, "width": lesson.width,
                                     "wf_corr": lesson.wf_corr, "wf_width": lesson.wf_width,
                                     "stability": lesson.stability, "seed_var": lesson.seed_var,
                                     "overfit_ratio": lesson.overfit_ratio, "decision": lesson.decision,
                                     "reason": lesson.reason, "budget_left": self.budget})
                log("evo_lesson", gen=gen, op=op_name, key=key,
                    oof_corr=round(lesson.oof_corr, 4), wf=round(lesson.wf_corr, 4),
                    fitness=round(fit, 4), accepted=accepted,
                    g_best=round(g_best, 4), budget=self.budget)
            T *= cfg.SA_DECAY
            stale = 0 if stale < 0 else stale + 1
            if produced == 0:
                break
        log("evolution_end", epoch=self.epoch, lessons=len(self.history),
            g_best=round(g_best, 5), budget_left=self.budget)

    def operator_report(self) -> pd.DataFrame:
        rows = []
        for op, gains in self.op_gains.items():
            fits = [h["fitness"] for h in self.history
                    if h["operator"] == op and np.isfinite(h["fitness"])]
            culls = [h for h in self.history if h["operator"] == op and h["decision"] == "draft_culled"]
            rows.append({"operator": op, "full_runs": len(gains), "draft_culled": len(culls),
                         "mean_gain_vs_pop_median": float(np.mean(gains)) if gains else np.nan,
                         "mean_fitness": float(np.mean(fits)) if fits else np.nan,
                         "best_fitness": float(np.max(fits)) if fits else np.nan})
        return pd.DataFrame(rows).sort_values("best_fitness", ascending=False)


