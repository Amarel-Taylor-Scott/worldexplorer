# Idea ledger — winner networks & feature-space artifacts

User direction (2026-06-12): *"shared exploration winners create networks and
sub-models that communicate with each other"* and *"nodes/artifacts placed in
certain areas of the feature space have additional functionality."*

Every idea here is triaged against the **measured law** before it is built:
in-region doors cannot see out-of-period decay; every capacity-adding
mechanism so far inflated sealed and lost private (the complexity ratchet;
sealed cliff at ~0.115). So: **observation layers and selection hardeners can
ship freely; capacity adders need the governor's price tag, the doors, and a
decisive test.** Rejected ideas stay listed with their reason (test-everything
doctrine: omission prevents learning).

## Census — what already exists (don't rebuild)

| User concept | Already live as |
|---|---|
| winners leave signals others follow | MYCELIUM pheromone (promoted lessons deposit on used columns; `mycelium` family reads it; v23 sqrt-saturation breaks the runaway loop) |
| winners recruit others | DANCES (waggle recruitment of newborn explorers), QUORUM (family switched on colony-wide), GENE_POOL (proven genomes enter evolution as plasmids) |
| sub-models building on winners | DIVE phase = 1-hop residual communication (submarines hunt `y − slope·champion_oof`) |
| models seeing each other's outputs | `library.oofs()` snapshot → uniqueness scoring; nested_ensemble stacking; CHORUS shrinkage = 1-round agreement message at predict time |
| cross-run communication | cairn seed bank + v27 learning ledger (survivors/decayers/β) |
| artifacts dropped in feature space | BEACONS (v15): items at rare-terrain + novelty coordinates emit RBF **field channels** every explorer sees (passive functionality) |
| region-local sub-models | terrain_router (per-terrain experts), codebook (per-centroid LUT), weather_moe (regime-conditional blending) |

The genuinely NEW parts of the user's two ideas: (1) an explicit **network
object** over winners with structure-aware consequences, (2) **active**
artifacts — placed nodes that carry behavior, not just a field.

## 1. WINNER NETWORK — graph over promoted trails

### 1a. `winner_network` report — **BUILT (v30.1, observation only)**
Nodes = top promoted lessons (capped); edges = |output-corr| ≥ 0.5 with
input-Jaccard annotation; communities = leader clustering at corr ≥ 0.7.
Writes `winner_network.csv` (node, family, skill, community, degree,
max_corr, mean_abs_corr) + a summary log line (nodes/edges/communities/
largest-community share). Zero behavior change, no rng, exception-wrapped.
*Why first:* every later network mechanism needs this object, and the next
real run starts measuring whether network communities ≠ viewport families
(the v12/v19 monocultures would have appeared here as one giant community).

### 1b. Network-aware member selection — **QUEUED, strong v31 candidate (zero capacity)**
Use 1a's COMMUNITIES as the diversity unit in `_try_admit`, alongside the v23
viewport-family cap: a blend may carry at most N members of one *prediction
community*. Catches what family/texture caps miss (different families that
converged to one prediction = one bet — exactly the v24 failure where the
shipped 3-member blend was 72% mycelium *behavior* across "different"
textures). Selection hardener; no capacity. **Decisive test:** does the
shipped blend's largest community share drop, and does private rise?

### 1c. Council channels (generation-2 stacked lessons) — **QUEUED pending v30 evidence (capacity!)**
The most literal "winners communicate": append the fold-honest OOF
predictions of K diverse community representatives as feature channels; run a
small second generation of lessons on [X | council channels]. Leak-safety:
channels are OOF in-fold (already leak-free by construction); test-time
channels = the members' full-train refit predictions (the final-refit path
already computes these). Risk: this is STACKING = capacity = ratchet food —
governor must price council lessons at high complexity, doors + robust
selector judge them, family cap treats `council` as its own family.
**Decisive test:** council members may ship ONLY if the robust selector picks
them across partitions; watch sealed→private ratio vs non-council runs.

### 1d. Message-passing predict (iterative agreement reweighting) — **DECLINED**
Multi-round consensus at predict time is a pure in-working-region fitting
surface (the kind that picked rank-shape and λ=1.0 shrink in v19 and gamed
sealed). Chorus shrinkage already provides the 1-round, forward-chosen,
default-0 version. More rounds = more ways to fit the tail we select on.

## 2. ACTIVE ARTIFACTS — placed nodes with functionality

Placement must stay **target-free** (the beacon rule: terrain/novelty
coordinates, never y); only the artifact's *function* may be fold-honest.

### 2a. Lighthouse (risk-off artifact) — **PARTIALLY EXISTS / QUEUED as variant**
A node at a fragile region; rows inside its basin get the blend shrunk toward
equal-weight/zero (risk-off). Census: CHORUS shrinkage already shrinks
per-row by member disagreement (forward-chosen β, default 0), and the court
shrinks globally on high criticality. NEW variant worth testing: anchor the
shrink to **terrain basins with measured negative member floors**
(many_worlds already computes per-world floors) instead of instantaneous
disagreement. Forward-chosen strength, default 0 = no-op-safe. Zero capacity.
**Decisive test:** forward + sealed must improve with strength > 0 chosen;
ship only then.

### 2b. Workshop (basin-local correction model) — **QUEUED (small capacity, governor-priced)**
At each beacon, fit a tiny ridge on basin rows (RBF-weighted); its prediction
is an additive correction whose global weight is forward-chosen (default 0).
This is the "artifact with functionality" in its smallest honest form —
k≤8, one per beacon, ≤7 beacons. Capacity exists but is bounded and priced;
the no-op default keeps it door-safe. **Decisive test:** chosen weight > 0
AND forward gain ≥ SHAPE_MARGIN-style threshold; watch decay of corrected vs
uncorrected on the sealed audit.

### 2c. Waypoint relays (artifact chains / multi-hop residual paths) — **QUEUED, design only**
Generalize DIVE to a chain: artifact A's sub-model output hands its residual
to artifact B downstream (a path through feature space, each node a
specialist). Elegant, but it is stacked capacity with extra plumbing; build
only after 1c/2b produce real-run evidence that gated network capacity can
clear the doors. Not before.

### 2d. Artifact memory across runs — **FOLDED INTO LEDGER (cheap, when 2b lands)**
If workshops/lighthouses ever ship, persist their coordinates + measured
deltas into the learning ledger so the next run germinates artifacts where
they historically helped (the seed-bank pattern applied to artifacts).

## Sequencing

1. **v30 run (pending)** ships the observation layer (1a) — first
   winner-network measurements arrive free.
2. **v31** = 1b (network-community cap in member selection) — zero capacity,
   directly attacks the residual monoculture channel.
3. **2a lighthouse variant** next among artifacts (zero capacity, no-op-safe).
4. **1c council / 2b workshop** only after the v30/v31 evidence says gated
   capacity can survive the doors (governor β on real DRW is the gate).
5. 1d stays declined; 2c waits on 1c/2b.
