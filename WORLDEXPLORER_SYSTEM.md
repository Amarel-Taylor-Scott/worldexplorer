# WorldExplorer System

WorldExplorer is a scientific feature-space exploration system. It is not just a
model trainer or a leaderboard blender. It builds many candidate views of a data
world, attacks them with hostile validation, records what survived, and carries
mathematical memory forward to the next runtime.

The core idea is:

```text
Do not only remember conclusions.
Remember the mathematical state that made those conclusions true.
```

WorldExplorer treats each run as a branch through a high-dimensional terrain:

```text
WorldState_0
  -> build coordinate systems
  -> map row and feature topology
  -> generate paths and viewports
  -> train candidate models
  -> attack candidates with predators and forensic courts
  -> ship only robust survivors
  -> write durable memory for the next run
```

The newer computational-atlas layer extends this into a versioned state graph:

```text
WorldState_v = {
  geometry,
  topology,
  transformations,
  model parameters,
  information matrices,
  residual/evaluation surfaces,
  route values,
  hypotheses,
  contradictions,
  grokking candidates
}
```

Each transformation is an operator:

```text
WorldState_next = operator(WorldState_current)
```

Examples of operators:

```text
normalize
rank-transform
quantize
PCA / PLS projection
feature mask
feature topology residual
row-space carve
route carve
residual add/subtract
tree partition
model refit
ensemble reweight
quarantine
revival
grokking incubation
```

## Current Deployment Reality

This local checkout currently has no GitHub remote configured:

```text
git remote -v
# no remotes
```

The current Kaggle kernels are full embedded script kernels, not slim GitHub
downloaders. The grokking kernel metadata currently has:

```json
{
  "enable_internet": false,
  "code_file": "kernel.py",
  "kernel_type": "script"
}
```

That means the kernel must contain the actual runnable code. A slim GitHub
downloader kernel is possible, but it requires all of these:

```text
1. A real GitHub remote with the code pushed.
2. Kaggle kernel metadata with enable_internet=true.
3. A bootstrap script that downloads a pinned commit or release artifact.
4. A fallback or checksum strategy so the run is reproducible.
```

For competition work, the embedded offline kernel is more reproducible and less
fragile. The downloader approach is lighter and easier to update, but depends on
network availability and a stable public or token-accessible source.

There is already a thin bootstrap template at:

```text
kaggle/bootstrap_kernel.py
```

That template is designed to either install from GitHub when Kaggle internet is
enabled or load an attached Kaggle Dataset copy of the repo when internet is
disabled. However, this checkout currently has no configured git remote, and the
template's public GitHub URL was not reachable through an unauthenticated
`git ls-remote` check:

```text
https://github.com/Amarel-Taylor-Scott/worldexplorer.git
-> pushed from this checkout on master
```

So the accurate current status is:

```text
bootstrap template exists
fleet bootstrap generator exists
current active kernels do not use it
public GitHub source is now configured as origin
embedded offline kernels remain the safest competition path
```

The new slim-kernel path is generated with:

```bash
python tools/fleet.py bootstrap \
  --name wx-github-v020 \
  --internet \
  --repo git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git \
  --repo-ref v0.2.0 \
  --time-budget 120
```

This writes:

```text
/home/username/new_algo/kaggle/fleet/wx-github-v020/kernel.py
/home/username/new_algo/kaggle/fleet/wx-github-v020/kernel-metadata.json
/home/username/new_algo/kaggle/fleet/wx-github-v020_manifest.json
```

Then the standard fleet push/status/collect lifecycle works:

```bash
python tools/fleet.py push --manifest /home/username/new_algo/kaggle/fleet/wx-github-v020_manifest.json
python tools/fleet.py status --manifest /home/username/new_algo/kaggle/fleet/wx-github-v020_manifest.json
python tools/fleet.py collect --manifest /home/username/new_algo/kaggle/fleet/wx-github-v020_manifest.json
```

This should be pinned to a tag or commit SHA for reproducible Kaggle runs.

## High-Level Runtime Loop

At a high level, an `ExplorerHarness` run does this:

```text
1. Load config, data, prior ledgers, and runtime budget.
2. Build row-space sensors: terrain, weather, pressure, beacons, test-likeness.
3. Build feature-space sensors: feature graph, shift vector, topology reports.
4. Generate many candidate paths from skills, families, transforms, and k-widths.
5. Train and promote candidates through validation gates.
6. Attack promoted paths with predator falsification tests.
7. Run robust OOS / forensic selection.
8. Decide whether the forensic layer should override the incumbent blend.
9. Write submission and many reports.
10. Write durable memory: ledgers, graphs, matrices, and policy for the next run.
```

The important point is that path generation and model selection are not one
linear pipeline. WorldExplorer is closer to an exploration ecology:

```text
many candidate organisms
many viewports
many attacks
many local terrain checks
many failed paths remembered as anti-priors
```

## Data And World Coordinates

The generalized dataset can be thought of as:

```text
D = { (x_i, y_i, t_i, e_i, m_i) }
```

Where:

```text
x_i = feature vector
y_i = target
t_i = time/order coordinate
e_i = environment/regime/domain
m_i = metadata: quality, missingness, source, permissions
```

Matrix form:

```text
X = feature matrix
Y = label/target matrix
E = environment membership matrix
M = metadata matrix
```

WorldExplorer builds multiple coordinate systems over this data:

```text
raw features
ranked features
quantized features
PCA/PLS projections
feature-community views
topology residual views
time/motion views
terrain/weather/pressure views
test-like views
```

Each view is a possible hypothesis about where the signal lives.

## Row-Space Topology

The row-space topology maps examples into regions, plateaus, peaks, and weird
pockets.

Key objects:

```text
TerrainAtlas      -> row partitions, altitude/novelty, terrain assignment
WeatherGauge      -> volatility/state partition
PressureGauge     -> microstructure/pressure partition
BeaconField       -> radial landmarks around important regions
TESTLIKE          -> target-free probability that a row looks like test data
```

Mathematically, a row topology can be represented as:

```text
Z = T(X)
W_row[i,j] = similarity(z_i, z_j)
L_row = D_row - W_row
P[i,r] = membership of row i in region r
```

Where:

```text
Z      = transformed row coordinates
W_row  = row neighbor graph
L_row  = row graph Laplacian
P      = region or plateau membership matrix
```

This is the formal version of the "surface" WorldExplorer walks over.

## Feature-Space Topology

The feature-space topology maps how features agree, disagree, duplicate,
confound, shift, or form communities.

Current code concepts:

```text
FeatureGraph        -> feature-feature correlation topology
FEATURE_SHIFT       -> per-feature train-to-test distribution shift
feature communities -> target-free feature clusters
feature reports     -> signal, shift, coherence, and risk summaries
```

Mathematically:

```text
W_feat[j,k] =
  alpha_1 * |corr(x_j, x_k)|
+ alpha_2 * mutual_information(x_j, x_k)
+ alpha_3 * conditional_dependency(j,k)
- alpha_4 * shift_instability(j,k)
```

Future durable tensors should include:

```text
A_feat[e,j,k] = feature-feature agreement in environment e
D_feat[e,j,k] = feature-feature disagreement in environment e
U_feat[e,j,k] = uncertainty about that relation
A_y[e,j,m]    = feature-target agreement by environment and method
```

This lets the system distinguish:

```text
features agree globally but disagree locally
features disagree globally because mixed regions are merged
features agree only due to leakage or shift
features become useful only after a transformation
```

## Material Field Formalism

WorldExplorer can describe the feature space as loose heterogeneous material,
but the metaphor has a strict mathematical meaning. A material is not an
accepted story. It is a scoped, derived signal component with support,
composition, evidence, risk, and provenance.

```text
MaterialParticle p =
{
  support_mask      m_p over rows or regions,
  activation_vector v_p,
  feature_mix       a_p over original/derived columns,
  region_mix        r_p over terrain partitions,
  label_relation    beta_p or attribution summary,
  stability         survival under changed worlds,
  impurity          leakage/shift/noise/confounding risk,
  reactivity        useful interactions with other particles/operators,
  provenance        source run, operator, parent checkpoint
}
```

The material inventory is a derived representation:

```text
raw evidence stays immutable
derived material worlds may branch
operators reshape derived material only
every reshape must leave an impact record
```

This keeps the idea general. A material can be a feature, feature community,
region-local feature, residual pocket, projection direction, model-family
disagreement, or route-carve effect. It should not become a hard-coded theory
about one competition or one model family.

Material operations are typed hypotheses:

```text
sift        select/filter/downweight
wash        denoise/de-shift/impute
grind       quantize/rank/bin/compress
smelt       extract stable target-aligned signal
alloy       combine materials into interactions/projections
carve       isolate a region or pocket
route       send a pocket to a specialist or blend
polish      calibrate/residual-shape/output transform
quarantine  suppress in scope while preserving provenance
revive      retest old material under a new topology
anneal      long-horizon branch/grokking incubation
```

None of these operations become foundation by default.

## Evidence Ladder And Anti-Drift Rules

WorldExplorer must avoid turning every interesting grain into a new branch.
Branches are useful only when they are evidence-bearing and bounded.

The evidence ladder is:

```text
D rejected/hazard memory
  unsupported, high drift, high false-agreement risk, or high stress

C route-limited or quarantined
  local signal exists but global support is weak or scope is narrow

B branch with independent retest
  promising local/global signal, but not enough independent support

A supported candidate
  local signal, global effect, worst-world proxy, low drift, and support agree
```

Promotion is allowed only after a candidate satisfies the relevant gates:

```text
independent seed/split/world confirmation
parent and sibling ablation
worst-world floor checked
false agreement checked
overfit and complexity controlled
foundation stress not increased without branch reason
private/public or forward gap not a trap
```

The default for a weird but interesting branch is:

```text
branch-only
```

not:

```text
main-world edit
```

The system should also track validation-world reuse. If the same validation
world, route-carve review surface, or external score context is used many
times, its selection value is discounted until a new independent world or
mutated adversarial split confirms the move.

## Viewports, Families, And Transforms

WorldExplorer does not train one model on all features. It creates viewports:

```text
skill | family | transform | k
```

Examples:

```text
linear_assoc | top      | quantize4 | 80
swell_rider  | mycelium | quantize4 | 30
bagged_linear| medoid   | pca       | 120
single_factor| weather  | identity  | 1
```

Families choose which features or regions to use. Transforms change the
coordinate system.

Important families and concepts:

```text
top              = high direct target association
stable           = stable across splits
mycelium         = follows promoted feature pheromones
red pheromone    = repellent memory from killed/trap features
weather          = environment-specific signal
terrain          = row-region-specific signal
testlike_stable  = stable under train/test-like pressure
medoid           = feature-community representative
periphery        = less-used exploratory feature regions
```

Important transforms:

```text
identity
rank
quantize2/4/8
sign_only
pca
pls
fold_abs
fold_pairs
dual_exposure
doppler
lateral_line
moire
prism
tide
curvature
```

Examples of mathematical meaning:

```text
lateral_line = feature minus correlated-neighbor flow
beacon       = radial basis coordinate around a terrain landmark
PCA/PLS      = learned projection matrix
fold_abs     = symmetry/absolute-value fold
doppler      = motion/temporal derivative-like transform
```

## Candidate Paths And Lessons

A path is a candidate model/view combination plus its evidence.

Each lesson records things like:

```text
path key
skill
family
transform
k
selected columns
OOF score
forward/working-fold score
width
decay
world-floor behavior
promotion decision
predator verdict
shipping decision
```

Promoted lessons can deposit positive feature pheromone. Predator-killed paths
can deposit repellent memory. This is how later candidates reuse or avoid
parts of the feature world.

## Predator Layer

The predator is a falsification engine. It attacks promising paths to find
which ones are fragile.

Typical attacks include:

```text
null tax
perturbation tests
sub-period attacks
time-reversal / palindrome checks
terrain death
weather death
beacon death
decay attacks
```

A promoted path is not trusted until it survives hostile worlds. A failed path
is not discarded silently; it becomes negative memory.

## Forensic Regime Science

The forensic layer measures whether the selected blend survives many worlds:

```text
time blocks
terrain regions
weather states
pressure/habitat states
test-like partitions
feature clusters
bad-row influence
row quarantine candidates
regime split candidates
```

It writes reports such as:

```text
robust_oos_selection.csv
forensic_selection_decision.json
shipping_court_report.csv
feature_topology_report.csv
row_influence_court.csv
regime_change_passports.csv
```

The forensic layer can override the normal incumbent only when the measured
robust court says the override improves the CV-to-forward behavior.

## Robust OOS Selection

WorldExplorer does not select only by average validation. It considers:

```text
global score
worst-world behavior
terrain minimum
weather minimum
test-like partitions
decay
width
overfit ratio
complexity
model diversity
feature-family diversity
```

This is why a locally good path may remain branch-only: it may fix one region
while damaging another.

## Memory Across Runs

A run writes artifacts that the next run can use:

```text
world_cairn.json
learning_ledger.json
explorer_findings_graph.json
complexity_governor.json
predator_report.csv
feature_topology_report.csv
robust_oos_selection.csv
forensic_selection_decision.json
submission.csv
```

The older cross-run memory carries:

```text
survivors
anti-priors
governor warm-start information
feature pheromones
route priors
advisor-readable findings
```

The newer atlas compiler turns these reports into typed matrices.

## Computational Atlas Layer

The computational atlas is generated under:

```text
/home/username/new_algo/kaggle/fleet/memory_matrices/
```

Key files:

```text
run_memory_matrix.csv
path_memory_matrix.csv
typology_memory_matrix.csv
typology_coverage_matrix.csv
typology_vector_field.csv
feature_space_memory_matrix.csv
surface_surgery_matrix.csv
impact_field_matrix.csv
foundation_stress_matrix.csv
route_strength_matrix.csv
validation_budget_ledger.csv
evidence_gate_matrix.csv
proof_carrying_paths.jsonl
contradiction_graph.csv
grokking_incubation_matrix.csv
projection_memory_matrix.csv
collinearity_memory_matrix.csv
operation_memory_matrix.csv
relation_edges.csv
numeric_memory_bundle.npz
numeric_memory_schema.json
tensor_artifact_manifest.json
checkpoint_graph.json
operator_graph_edges.csv
open_hypotheses.json
next_runtime_policy.json
computational_atlas_manifest.json
attention_inputs.json
```

These files convert run reports into a generalized memory fabric:

```text
actual numeric vectors
actual information matrices
operator edges
checkpoint graph nodes
surface edits
impact fields
foundation stress
route/action strengths
validation-world reuse pressure
evidence gates
proof-carrying candidates
contradictions
grokking incubators
next-runtime policy
```

## Surface Surgery

Surface surgery stores local edits and risks:

```text
surface_id
source kind
coordinate
local signal
global stability
agreement score
disagreement score
uncertainty
false agreement risk
false disagreement risk
overfit risk
rearrangement gain
grokking priority
suggested surgery
reversal plan
```

This represents actions such as:

```text
branch before carve
ablate route/lens/alpha
compress a feature community
restore a quarantined feature
split a region
train a specialist
compare parent and child surfaces
```

The rule is:

```text
Local fixes must pay rent globally.
```

## Impact Fields

Every candidate surface move gets a ripple estimate:

```text
impact_id
local_effect
global_effect
stability_effect
agreement_delta
disagreement_delta
uncertainty_delta
false_agreement_risk
false_disagreement_risk
overfit_risk
ripple_radius
side_effect_load
foundation_stress_delta
move_quality
branch_priority
move_status
```

The mathematical ideal is:

```text
DeltaS_action = Surface(World_after) - Surface(World_before)
```

So a future runtime can ask:

```text
what changed locally?
what changed globally?
which disagreements appeared?
which regions got worse?
should this remain branch-only?
```

## Foundation Stress

Foundation stress estimates whether the current coordinate system may be wrong.

Stress rises when:

```text
local repairs create many distant disagreements
false agreement risk rises
overfit risk rises
model-family rankings flip
topology becomes unstable
contradictions accumulate
complexity keeps growing to patch small regions
```

When stress is high, the policy should branch from an earlier world and try a
different foundation:

```text
alternate normalization
alternate partition
alternate label treatment
alternate projection
alternate causal/environment split
```

## Validation Budget And Evidence Gates

The atlas now separates exploration from acceptance with explicit guard
artifacts:

```text
validation_budget_ledger.csv
evidence_gate_matrix.csv
proof_carrying_paths.jsonl
```

`validation_budget_ledger.csv` records visible validation surfaces, how many
candidates have been searched against them, and how much to discount them as
independent evidence.

`evidence_gate_matrix.csv` turns each impact field into a branch/promotion
decision:

```text
local effect
global effect
worst-world proxy
independent support
false-agreement risk
false-disagreement risk
overfit risk
foundation stress
branch priority
evidence grade
decision
```

`proof_carrying_paths.jsonl` stores evidence/risk certificates for candidate
paths and operations. A proof object is not a proof of truth; it is a compact,
machine-checkable record of what supports the candidate, what risks remain,
and what must be true before promotion.

## Contradiction Graph

The contradiction graph stores claims and relationships:

```text
claim A supports claim B
claim A weakens claim B
claim A contradicts claim B
claim A revives old claim B
claim A supersedes claim B
```

This prevents the system from turning every report into unquestioned truth.

Example:

```text
external score supports operation
but route information gain is negative
=> contradiction requiring route ablation and private-public gap replay
```

## Route Strength

Route strength learns how useful action types are under terrain signatures.

It stores:

```text
terrain_signature
action_type
operator_type
expected_gain
uncertainty
success_probability
overfit_risk
transferability
fragility
complementarity
branch_value
status
```

This is a value function over exploration actions:

```text
Q(world_signature, action) -> expected value and risk
```

## Grokking Incubation

Grokking incubation is a quarantined long-horizon research lane.

It exists because some branches may show delayed generalization. Flat early
validation is usually a reason to stop, but sometimes internal structure is
still forming.

Grokking branches may continue when they have evidence such as:

```text
structured residuals
low label-noise estimate
consistent gradient direction
improving representation separation
systematic model disagreement
reasonable overfit risk
unexplored terrain value
```

But grokking branches cannot ship directly.

They must pass:

```text
beats parent on robust score
worst-world nonnegative
false-agreement risk checked
overfit ratio controlled
private/public or forward gap not a trap
```

The current grokking lane is represented by:

```text
grokking_incubation_matrix.csv
next_runtime_policy.json
grokking_incubation_report.json  # emitted by completed incubator kernels
```

## Next Runtime Policy

The atlas compiler now emits:

```text
next_runtime_policy.json
```

This is a machine-consumable policy, not a prose summary. It contains:

```text
budget_allocation
hard_guards
protected_priors
hazard_exclusions
branch_queue
grokking_queue
retest_queue
reduction_queue
contradiction_queue
collect_actions
promotion_gates
```

Current policy shape:

```text
normal_search
safe_repair_and_consolidation
grokking_incubation
random_sprouts
```

Hard guards:

```text
grokking branches cannot ship directly
local surface edits need global ripple checks
contradictions require ablation before promotion
hazards are masks or penalties, not deletions
```

## Kaggle Fleet

The fleet tool builds and runs Kaggle kernels:

```bash
python3 tools/fleet.py build
python3 tools/fleet.py sprout --count 6
python3 tools/fleet.py grok --count 3 --prefix grok-atlas2
python3 tools/fleet.py push --manifest /home/username/new_algo/kaggle/fleet/grok-atlas2_manifest.json
python3 tools/fleet.py status --manifest /home/username/new_algo/kaggle/fleet/grok-atlas2_manifest.json
python3 tools/fleet.py harvest-grok --manifest /home/username/new_algo/kaggle/fleet/grok-atlas2_manifest.json
```

`harvest-grok` is deliberately safe:

```text
checks only grokking incubators
collects only completed outputs
never submits
prints pending/failed/collected counts
```

## Embedded Kernel Versus GitHub Downloader

Current mode:

```text
embedded offline kernel.py
enable_internet=false
full code injected into Kaggle script
```

Pros:

```text
reproducible
works without internet
does not depend on GitHub availability
does not require secrets
competition-safe
```

Cons:

```text
large kernel.py
slower to update
harder to inspect diffs in Kaggle UI
```

Possible slim downloader mode:

```text
small kernel.py
enable_internet=true
download repo/archive from a verified GitHub remote or attached dataset
run pinned entrypoint
```

Generated slim downloader mode:

```bash
python tools/fleet.py bootstrap \
  --name wx-github-v020 \
  --internet \
  --repo git+https://github.com/Amarel-Taylor-Scott/worldexplorer.git \
  --repo-ref v0.2.0 \
  --override SEED=7
```

Offline dataset mode:

```bash
python tools/fleet.py bootstrap \
  --name wx-dataset-v020 \
  --engine-dataset /kaggle/input/worldexplorer-engine \
  --dataset taylorsamarel/worldexplorer-engine \
  --repo-ref v0.2.0
```

Pros:

```text
small Kaggle kernel
fast iteration when code lives in GitHub
clear source provenance if pinned to a commit
```

Cons:

```text
requires GitHub remote
requires Kaggle internet
less robust if network fails
must pin commits/checksums for reproducibility
private repos require token handling
```

Recommended approach:

```text
Use embedded offline kernels for serious competition runs.
Use slim GitHub downloader kernels only for development or public reproducible demos,
and pin them to immutable commit SHAs.
```

## Where The System Should Go Next

The current atlas stores many derived matrices and graph views. The next major
upgrade is to persist more actual tensors and model states:

```text
feature graph adjacency matrices
row graph sparse matrices
region membership matrices
feature-target agreement tensors
projection matrices
feature masks
region masks
model weights and biases
optimizer state
Fisher/Hessian approximations
tree split and leaf statistics
attention/router weights
replay buffers
prediction and residual matrices
checkpoint diffs
```

The end state is:

```text
WorldExplorer learns a graph of worlds.

Each world stores geometry, topology, operators, parameters, matrices,
surfaces, beliefs, and route values.

Each action stores what changed, where it helped, where it hurt, what it
contradicted, how risky it is, and when future runtimes should challenge it.
```

That is the durable mathematical memory needed for warm multi-runtime
scientific exploration.
