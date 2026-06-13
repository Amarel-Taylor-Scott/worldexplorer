# ----------------------------------------------------------------------------
# 7. Phase 1 -- developmental explorers (curriculum + UCB initialization)
# ----------------------------------------------------------------------------

# v11 TROPHIC ONTOGENY: the developmental ladder is no longer human
# (infant/child/adolescent/adult) but a SENSORY-RANGE ascent -- which is what
# the ladder always mechanically was (k = perceptual reach). microbe senses
# locally; the apex predator scans the whole range. Same k-growth + graduation
# machinery underneath; only the framing left the species behind.
STAGES = [
    ("microbe",   0, {"max_k": 4,   "max_lessons": 3,  "graduate_gain": 0.01}),
    ("forager",   1, {"max_k": 16,  "max_lessons": 4,  "graduate_gain": 0.02}),
    ("navigator", 2, {"max_k": 64,  "max_lessons": 6,  "graduate_gain": 0.04}),
    ("apex",      3, {"max_k": 160, "max_lessons": 99, "graduate_gain": float("inf")}),  # v8 real run: best phase-1 lesson sat AT the 128 menu edge
]

EXPLORER_TRAITS = [
    {"name": "cautious_cartographer", "metaheuristic": "local_search_hill_climber",
     "curiosity": 0.2, "caution": 0.9, "sociality": 0.5,
     "skill_prior": {"single_factor": 0.9, "bin_association": 0.3, "majority_vote": 0.9, "theil_sen": 0.85,
                     "recency_linear": 0.6, "local_interp": 0.2,
                     "linear_assoc": 0.9, "bagged_linear": 0.95, "residual_ladder": 0.7,
                     "nonlinear_assoc": 0.3, "mlp_assoc": 0.3, "gbdt_lib": 0.3,
                     "codebook": 0.5, "terrain_router": 0.5, "steepness_gate": 0.4, "scout_lattice": 0.8,
                     "relay_caravan": 0.7, "swell_rider": 0.6},
     "transform_prior": {"identity": 0.9, "rank": 0.85, "sign_only": 0.7,
                         "quantize8": 0.85, "quantize4": 0.6, "quantize2": 0.5, "pca": 0.8, "pair_aug": 0.4,
                         "rand_proj": 0.5, "signed_hadamard": 0.2, "pca_aug": 0.4, "foveated": 0.3,
                         "fold_abs": 0.4, "fold_pairs": 0.3, "dual_exposure": 0.6, "doppler": 0.5},
     "family_prior": {"top": 0.9, "anon": 0.8, "market": 0.3, "decor": 0.7, "stable": 0.7,
                      "medoid": 0.8, "lastN": 0.6, "dawn": 0.3, "both_clocks": 0.8, "terrain": 0.6,
                      "weather": 0.6, "mycelium": 0.7, "shadow": 0.2, "periphery": 0.3}},
    {"name": "curious_scout", "metaheuristic": "cuckoo_levy_flight",
     "curiosity": 0.95, "caution": 0.2, "sociality": 0.3,
     "skill_prior": {"single_factor": 0.4, "bin_association": 0.3, "majority_vote": 0.5, "theil_sen": 0.4,
                     "recency_linear": 0.5, "local_interp": 0.2,
                     "linear_assoc": 0.6, "bagged_linear": 0.8, "residual_ladder": 0.7,
                     "nonlinear_assoc": 0.9, "mlp_assoc": 0.6, "gbdt_lib": 0.9,
                     "codebook": 0.7, "terrain_router": 0.7, "steepness_gate": 0.7, "scout_lattice": 0.95,
                     "relay_caravan": 0.6, "swell_rider": 0.7},
     "transform_prior": {"identity": 0.4, "rank": 0.7, "sign_only": 0.6,
                         "quantize8": 0.85, "quantize4": 0.7, "quantize2": 0.8, "pca": 0.8, "pair_aug": 0.8,
                         "rand_proj": 0.85, "signed_hadamard": 0.7, "pca_aug": 0.85, "foveated": 0.9,
                         "fold_abs": 0.8, "fold_pairs": 0.7, "dual_exposure": 0.8, "doppler": 0.9,
                         "prism": 0.9, "moire": 0.85, "tide": 0.7},
     "family_prior": {"top": 0.6, "anon": 0.7, "market": 0.4, "decor": 0.7, "stable": 0.6,
                      "medoid": 0.7, "lastN": 0.6, "dawn": 0.8, "both_clocks": 0.5, "terrain": 0.7,
                      "weather": 0.7, "mycelium": 0.4, "shadow": 0.9, "periphery": 0.9,
                      "echo": 0.85, "springs": 0.6}},
    {"name": "regime_watcher", "metaheuristic": "pso_social_follower",
     "curiosity": 0.5, "caution": 0.6, "sociality": 0.95,
     "skill_prior": {"single_factor": 0.5, "bin_association": 0.3, "majority_vote": 0.7, "theil_sen": 0.5,
                     "recency_linear": 0.95, "local_interp": 0.2,
                     "linear_assoc": 0.7, "bagged_linear": 0.95, "residual_ladder": 0.8,
                     "nonlinear_assoc": 0.5, "mlp_assoc": 0.5, "gbdt_lib": 0.5,
                     "codebook": 0.5, "terrain_router": 0.95, "steepness_gate": 0.8, "scout_lattice": 0.85,
                     "relay_caravan": 0.95, "swell_rider": 0.85},
     "transform_prior": {"identity": 0.8, "rank": 0.8, "sign_only": 0.5,
                         "quantize8": 0.8, "quantize4": 0.6, "quantize2": 0.5, "pca": 0.85, "pair_aug": 0.5,
                         "rand_proj": 0.7, "signed_hadamard": 0.3, "pca_aug": 0.6, "foveated": 0.5,
                         "fold_abs": 0.6, "fold_pairs": 0.5, "dual_exposure": 0.6, "doppler": 0.85},
     "family_prior": {"top": 0.7, "anon": 0.7, "market": 0.3, "decor": 0.7, "stable": 0.95,
                      "medoid": 0.7, "lastN": 0.7, "dawn": 0.9, "both_clocks": 0.95, "terrain": 0.95,
                      "weather": 0.95, "mycelium": 0.6, "shadow": 0.3, "periphery": 0.7}},
    {"name": "feature_alchemist", "metaheuristic": "de_recombinator",
     "curiosity": 0.7, "caution": 0.5, "sociality": 0.6,
     "skill_prior": {"single_factor": 0.4, "bin_association": 0.3, "majority_vote": 0.6, "theil_sen": 0.5,
                     "recency_linear": 0.5, "local_interp": 0.2,
                     "linear_assoc": 0.85, "bagged_linear": 0.85, "residual_ladder": 0.7,
                     "nonlinear_assoc": 0.6, "mlp_assoc": 0.5, "gbdt_lib": 0.7,
                     "codebook": 0.6, "terrain_router": 0.6, "steepness_gate": 0.6, "scout_lattice": 0.85,
                     "relay_caravan": 0.6, "swell_rider": 0.6},
     "transform_prior": {"identity": 0.6, "rank": 0.6, "sign_only": 0.4,
                         "quantize8": 0.8, "quantize4": 0.6, "quantize2": 0.6, "pca": 0.8, "pair_aug": 0.95,
                         "rand_proj": 0.6, "signed_hadamard": 0.3, "pca_aug": 0.9, "foveated": 0.85,
                         "fold_abs": 0.8, "fold_pairs": 0.9, "dual_exposure": 0.9, "doppler": 0.7},
     "family_prior": {"top": 0.7, "anon": 0.7, "market": 0.3, "decor": 0.7, "stable": 0.7,
                      "medoid": 0.95, "lastN": 0.9, "dawn": 0.6, "both_clocks": 0.6, "terrain": 0.6,
                      "weather": 0.6, "mycelium": 0.7, "shadow": 0.5, "periphery": 0.6}},
    # v8: the topographer -- lives on the terrain atlas, folding, and the
    # quantized space sub-models; stigmergy = follows the texture trails
    # other explorers leave in the shared library.
    {"name": "terrain_surveyor", "metaheuristic": "stigmergy_ant_colony",
     "curiosity": 0.6, "caution": 0.5, "sociality": 0.7,
     "skill_prior": {"single_factor": 0.4, "bin_association": 0.4, "majority_vote": 0.6, "theil_sen": 0.4,
                     "recency_linear": 0.6, "local_interp": 0.2,
                     "linear_assoc": 0.7, "bagged_linear": 0.8, "residual_ladder": 0.6,
                     "nonlinear_assoc": 0.5, "mlp_assoc": 0.5, "gbdt_lib": 0.5,
                     "codebook": 0.6, "terrain_router": 0.9, "steepness_gate": 0.85, "scout_lattice": 0.9,
                     "relay_caravan": 0.8, "swell_rider": 0.7, "terrace": 0.8, "rapids": 0.7},
     "transform_prior": {"identity": 0.5, "rank": 0.7, "sign_only": 0.6,
                         "quantize8": 0.8, "quantize4": 0.7, "quantize2": 0.85, "pca": 0.6, "pair_aug": 0.5,
                         "rand_proj": 0.5, "signed_hadamard": 0.4, "pca_aug": 0.5, "foveated": 0.6,
                         "fold_abs": 0.6, "fold_pairs": 0.6, "dual_exposure": 0.7, "doppler": 0.6},
     "family_prior": {"top": 0.7, "anon": 0.6, "market": 0.4, "decor": 0.6, "stable": 0.7,
                      "medoid": 0.6, "lastN": 0.6, "dawn": 0.5, "both_clocks": 0.6, "terrain": 0.95,
                      "weather": 0.9, "mycelium": 0.95, "shadow": 0.6, "periphery": 0.7,
                      "watershed": 0.95, "springs": 0.7}},
    # ---- v11 MENAGERIE: organisms whose namesake strategy is a real mechanism.
    # birth_stage/max_stage give each species its OWN trophic ontogeny.
    {"name": "bacterium", "metaheuristic": "chemotaxis_run_and_tumble", "species": "bacterium",
     "birth_stage": 0, "max_stage": 1, "behavior": "chemotaxis",  # no childhood, never grows a big eye
     "curiosity": 0.85, "caution": 0.15, "sociality": 0.8,
     "skill_prior": {"single_factor": 0.7, "majority_vote": 0.8, "bin_association": 0.6,
                     "linear_assoc": 0.7, "bagged_linear": 0.75, "codebook": 0.7, "scout_lattice": 0.6},
     "transform_prior": {"identity": 0.6, "sign_only": 0.8, "rank": 0.7, "quantize2": 0.9,
                         "quantize4": 0.7, "lateral_line": 0.6, "doppler": 0.6},
     "family_prior": {"top": 0.7, "anon": 0.7, "mycelium": 0.95, "shadow": 0.8, "compass": 0.5,
                      "terrain": 0.6, "weather": 0.6}},
    {"name": "starling", "metaheuristic": "topological_flocking", "species": "starling",
     "birth_stage": 1, "max_stage": 3, "behavior": "flock", "recruit": True,
     "curiosity": 0.6, "caution": 0.3, "sociality": 0.98,
     "skill_prior": {"majority_vote": 0.8, "linear_assoc": 0.8, "bagged_linear": 0.9,
                     "scout_lattice": 0.85, "terrain_router": 0.7, "relay_caravan": 0.7},
     "transform_prior": {"identity": 0.7, "quantize8": 0.85, "quantize2": 0.7, "pca": 0.7,
                         "rank": 0.7, "lateral_line": 0.7, "doppler": 0.7},
     "family_prior": {"top": 0.8, "stable": 0.85, "compass": 0.95, "both_clocks": 0.8,
                      "terrain": 0.8, "weather": 0.8, "mycelium": 0.7}},
    {"name": "electric_fish", "metaheuristic": "active_electrolocation", "species": "fish",
     "birth_stage": 1, "max_stage": 3, "behavior": "electrosense",
     "curiosity": 0.8, "caution": 0.4, "sociality": 0.5,
     "skill_prior": {"single_factor": 0.6, "linear_assoc": 0.8, "bagged_linear": 0.85,
                     "steepness_gate": 0.8, "scout_lattice": 0.8, "swell_rider": 0.7,
                     "rapids": 0.85},
     "transform_prior": {"identity": 0.6, "lateral_line": 0.95, "doppler": 0.85, "rank": 0.7,
                         "quantize8": 0.7, "pca": 0.6, "tide": 0.9, "moire": 0.8},
     "family_prior": {"top": 0.7, "anon": 0.7, "shadow": 0.7, "periphery": 0.8, "compass": 0.7,
                      "weather": 0.85, "terrain": 0.7, "echo": 0.8}},
    {"name": "honeybee", "metaheuristic": "waggle_recruitment", "species": "bee",
     "birth_stage": 1, "max_stage": 3, "behavior": "waggle", "recruit": True,
     "curiosity": 0.7, "caution": 0.4, "sociality": 0.95,
     "skill_prior": {"majority_vote": 0.7, "linear_assoc": 0.8, "bagged_linear": 0.9,
                     "scout_lattice": 0.85, "codebook": 0.7, "terrain_router": 0.75},
     "transform_prior": {"identity": 0.7, "quantize8": 0.85, "quantize2": 0.75, "rank": 0.7,
                         "pca": 0.7, "lateral_line": 0.6},
     "family_prior": {"top": 0.85, "stable": 0.8, "mycelium": 0.9, "compass": 0.8, "terrain": 0.8,
                      "both_clocks": 0.75, "weather": 0.7}},
    {"name": "nutcracker", "metaheuristic": "scatter_hoard_caching", "species": "bird",
     "birth_stage": 2, "max_stage": 3, "behavior": "cache",   # born ranging; remembers caches
     "curiosity": 0.55, "caution": 0.6, "sociality": 0.6,
     "skill_prior": {"linear_assoc": 0.8, "bagged_linear": 0.9, "recency_linear": 0.85,
                     "relay_caravan": 0.85, "scout_lattice": 0.8, "terrain_router": 0.8,
                     "terrace": 0.85},
     "transform_prior": {"identity": 0.7, "quantize8": 0.85, "quantize2": 0.7, "pca": 0.75,
                         "rank": 0.7, "doppler": 0.6, "tide": 0.7},
     "family_prior": {"top": 0.8, "stable": 0.9, "mycelium": 0.95, "terrain": 0.85, "compass": 0.8,
                      "both_clocks": 0.8, "weather": 0.8, "springs": 0.9, "watershed": 0.7}},
]

# v11: fill ontogeny/species defaults for the proven v1-v10 personas (they
# keep their measured priors; they are simply re-seated in the phylogeny as
# full-range generalist organisms born at the smallest trophic level).
for _t in EXPLORER_TRAITS:
    _t.setdefault("species", _t["name"])
    _t.setdefault("birth_stage", 0)
    _t.setdefault("max_stage", 3)
    _t.setdefault("behavior", "generalist")
    _t.setdefault("recruit", False)


def stage_viewport_menu(stage_idx: int, max_k: int, n_cols: int) -> list[ViewportSpec]:
    transforms = ["identity", "sign_only", "rank"]      # primitives are microbe-native
    if stage_idx >= 2:
        transforms += ["quantize8", "quantize4", "quantize2", "dual_exposure", "doppler",
                       "lateral_line", "prism", "moire", "tide", "fractal",
                       "reaction_diffusion", "curvature", "lorentz_boost", "pca", "pair_aug"]
    if stage_idx >= 3:
        transforms += ["rand_proj", "signed_hadamard", "pca_aug", "foveated",
                       "fold_abs", "fold_pairs", "random_fourier"]
    specs = []
    for fam in FAMILIES:
        for t in transforms:
            k = min(max_k, max(2, n_cols))
            specs.append(ViewportSpec(name=f"{fam}{k}_{t}", family=fam, k=k, transform=t,
                                      proj_dim=32 if stage_idx == 3 else 16))
    return specs


if CFG.WIDE_PERSONA:
    # v33 ALBATROSS (user-directed wide-glider): rides wide, stable, agreeing
    # signal channels -- consensus skills (vote/bayes/ols/pls), conservative
    # encodings, stability-bred families. Inserted at roster slot 7 so the
    # proven first seven personas are untouched; _setup raises N_EXPLORERS to 8.
    EXPLORER_TRAITS.insert(min(7, len(EXPLORER_TRAITS)), {
        "name": "albatross", "metaheuristic": "dynamic_soaring_glider", "species": "albatross",
        "behavior": "wide_path_glider", "width_pref": 0.9,   # v34: leans WIDE (robust width)
        "curiosity": 0.35, "caution": 0.8, "sociality": 0.7,
        "skill_prior": {"single_factor": 0.5, "bin_association": 0.3, "majority_vote": 0.95,
                        "theil_sen": 0.7, "recency_linear": 0.5, "local_interp": 0.1,
                        "linear_assoc": 0.85, "bagged_linear": 0.9, "residual_ladder": 0.5,
                        "nonlinear_assoc": 0.2, "mlp_assoc": 0.1, "gbdt_lib": 0.2,
                        "codebook": 0.4, "terrain_router": 0.5, "steepness_gate": 0.4,
                        "scout_lattice": 0.6, "relay_caravan": 0.5, "swell_rider": 0.85,
                        "linear_ols": 0.9, "huber_linear": 0.85, "elastic_net": 0.7,
                        "pls": 0.9, "bayes_ridge": 0.95, "ard_linear": 0.7, "greedy_ols": 0.8},
        "transform_prior": {"identity": 0.95, "rank": 0.9, "sign_only": 0.85,
                            "quantize8": 0.7, "quantize4": 0.8, "quantize2": 0.6,
                            "pca": 0.6, "pair_aug": 0.2, "rand_proj": 0.2,
                            "signed_hadamard": 0.1, "pca_aug": 0.3, "foveated": 0.3,
                            "fold_abs": 0.4, "fold_pairs": 0.2, "dual_exposure": 0.6,
                            "doppler": 0.4, "tide": 0.5, "moire": 0.3},
        "family_prior": {"top": 0.7, "anon": 0.6, "market": 0.4, "decor": 0.8,
                         "stable": 0.95, "medoid": 0.8, "lastN": 0.5, "dawn": 0.4,
                         "both_clocks": 0.9, "terrain": 0.6, "weather": 0.8,
                         "mycelium": 0.5, "shadow": 0.1, "periphery": 0.2,
                         "invariant": 0.95, "sign_stability": 0.95, "irm": 0.8,
                         "stabsel": 0.85, "pls_weight": 0.85, "tail": 0.8}})

if CFG.SENSORY_ROSTER:
    # v34 SENSORY MENAGERIE (user-directed: "more explorer primitives with
    # different functionalities and ways of looking at their surroundings").
    # Each persona is a DIFFERENT point on the narrow<->wide axis (width_pref,
    # so one run mixes both) AND a DIFFERENT sense (sparse family/transform
    # priors; .get default 0.5 leaves the rest neutral). Appended after the
    # proven roster + albatross; _setup bumps N_EXPLORERS so they run when the
    # metabolism has airtime. SENSORY_ROSTER off => this whole block is skipped.
    # Spliced in right AFTER the albatross (not appended) so the wide<->narrow
    # spread gets airtime ahead of the older v11 menagerie tail at light budgets.
    _SENSORY_PERSONAS = [
        # KESTREL -- the NARROW counterpart to the wide albatross: a sharp stoop
        # on ONE tiny high-corr target (narrow lucky ridgelines, not robust width)
        {"name": "kestrel", "metaheuristic": "stooping_hunter", "species": "kestrel",
         "behavior": "narrow_sharp_stoop", "width_pref": 0.1,
         "curiosity": 0.4, "caution": 0.3, "sociality": 0.4,
         "skill_prior": {"single_factor": 0.95, "theil_sen": 0.9, "greedy_ols": 0.9,
                         "linear_ols": 0.85, "majority_vote": 0.6, "bin_association": 0.5,
                         "mlp_assoc": 0.1, "gbdt_lib": 0.2, "nonlinear_assoc": 0.2},
         "transform_prior": {"identity": 0.8, "sign_only": 0.9, "quantize2": 0.85,
                             "rank": 0.7, "quantize4": 0.6},
         "family_prior": {"top": 0.95, "tail": 0.85, "medoid": 0.8, "decor": 0.7,
                          "sign_stability": 0.8, "shadow": 0.1}},
        # MANTIS_SHRIMP -- twelve-channel spectral vision: reads many SPECTRAL
        # BANDS of the same feature at once (prism/moire/dual/lateral/fractal/RD)
        {"name": "mantis_shrimp", "metaheuristic": "spectral_multichannel", "species": "mantis_shrimp",
         "behavior": "many_band_vision", "width_pref": 0.5,
         "curiosity": 0.85, "caution": 0.3, "sociality": 0.4,
         "skill_prior": {"linear_assoc": 0.8, "bagged_linear": 0.8, "scout_lattice": 0.8,
                         "rapids": 0.75, "terrace": 0.7},
         "transform_prior": {"prism": 0.95, "moire": 0.9, "dual_exposure": 0.9,
                             "lateral_line": 0.85, "fractal": 0.8, "reaction_diffusion": 0.8,
                             "curvature": 0.7, "identity": 0.4},
         "family_prior": {"top": 0.6, "anon": 0.6, "decor": 0.7, "medoid": 0.7,
                          "weather": 0.7, "terrain": 0.6}},
        # OWL -- nocturnal: hunts the QUIET regions everyone else ignores
        # (shadow/periphery/fault) -- low-signal, high-variance negative space
        {"name": "owl", "metaheuristic": "silent_night_hunter", "species": "owl",
         "behavior": "low_signal_forager", "width_pref": 0.6,
         "curiosity": 0.9, "caution": 0.5, "sociality": 0.2,
         "skill_prior": {"single_factor": 0.5, "majority_vote": 0.7, "linear_assoc": 0.7,
                         "bagged_linear": 0.7, "codebook": 0.6},
         "transform_prior": {"identity": 0.6, "sign_only": 0.7, "rank": 0.6, "quantize4": 0.6},
         "family_prior": {"shadow": 0.95, "periphery": 0.9, "fault": 0.8, "echo": 0.75,
                          "watershed": 0.7, "top": 0.3, "medoid": 0.3}},
        # BLOODHOUND -- tracks a FAINT PERSISTENT scent: pheromone trails +
        # persistence wells (mycelium/echo/springs/watershed), slow geology
        {"name": "bloodhound", "metaheuristic": "scent_tracking", "species": "bloodhound",
         "behavior": "faint_trail_follower", "width_pref": 0.7,
         "curiosity": 0.6, "caution": 0.5, "sociality": 0.85,
         "skill_prior": {"linear_assoc": 0.8, "bagged_linear": 0.85, "relay_caravan": 0.85,
                         "recency_linear": 0.8, "swell_rider": 0.8, "terrain_router": 0.75},
         "transform_prior": {"identity": 0.7, "tide": 0.8, "doppler": 0.7, "quantize8": 0.7,
                             "dual_exposure": 0.6},
         "family_prior": {"mycelium": 0.95, "echo": 0.9, "springs": 0.9, "watershed": 0.85,
                          "stable": 0.8, "both_clocks": 0.75}},
        # SPIDER -- feels which FEATURES vibrate the web: lives on the selection-
        # hardener families (stabsel/irm/sign_stability/pls_weight/invariant)
        {"name": "spider", "metaheuristic": "web_vibration_sense", "species": "spider",
         "behavior": "feature_selection_web", "width_pref": 0.6,
         "curiosity": 0.5, "caution": 0.7, "sociality": 0.6,
         "skill_prior": {"linear_ols": 0.85, "pls": 0.9, "bayes_ridge": 0.85, "ard_linear": 0.8,
                         "elastic_net": 0.85, "huber_linear": 0.75, "bagged_linear": 0.7},
         "transform_prior": {"identity": 0.9, "rank": 0.7, "quantize4": 0.6},
         "family_prior": {"stabsel": 0.95, "irm": 0.9, "sign_stability": 0.95,
                          "pls_weight": 0.9, "invariant": 0.9, "medoid": 0.7}},
        # OCTOPUS -- eight independent arms, each a DIFFERENT transform of the
        # same view (pairwise/projected/curved channels)
        {"name": "octopus", "metaheuristic": "distributed_arms", "species": "octopus",
         "behavior": "multi_transform_arms", "width_pref": 0.4,
         "curiosity": 0.8, "caution": 0.4, "sociality": 0.5,
         "skill_prior": {"linear_assoc": 0.8, "bagged_linear": 0.8, "scout_lattice": 0.8,
                         "steepness_gate": 0.7, "rapids": 0.7},
         "transform_prior": {"fold_pairs": 0.9, "pca_aug": 0.85, "pair_aug": 0.9,
                             "curvature": 0.8, "dual_exposure": 0.7, "foveated": 0.7,
                             "identity": 0.4},
         "family_prior": {"decor": 0.8, "medoid": 0.8, "top": 0.6, "anon": 0.6,
                          "compass": 0.7, "phyllotaxis": 0.7}},
    ]
    _ins = next((i for i, t in enumerate(EXPLORER_TRAITS) if t.get("name") == "albatross"), 6) + 1
    EXPLORER_TRAITS[_ins:_ins] = _SENSORY_PERSONAS


# v36 ADVISOR LOOP globals: set by the harness from an optional advisor_
# instructions.json (an external model / LLM's guidance); read by ucb_pick as
# ADDITIVE exploration priors. Empty => no-op. The advisor never sees labels or
# the leaderboard and never ships a model -- it only nudges WHERE/HOW to look,
# and every nudge is still measured through the honest doors.
ADVISOR: dict = {}              # raw loaded advisor instructions (for the chronicle + report)
ADVISOR_PRIORS: dict = {}       # {"family": {...}, "transform": {...}, "skill": {...}} lift weights


class Explorer:
    def __init__(self, traits: dict[str, Any], cfg: HarnessConfig) -> None:
        self.traits = traits
        self.name = str(traits["name"])
        self.species = str(traits.get("species", traits["name"]))
        self.behavior = str(traits.get("behavior", "generalist"))
        self.cfg = cfg
        self.budget = cfg.LESSON_BUDGET
        self.season = 1                  # v12: which metabolic season bore this organism
        # v11 trophic ontogeny: each species is born at its own level and
        # never grows past its own ceiling (a bacterium stays small)
        self.birth_stage = int(traits.get("birth_stage", 0))
        self.max_stage = int(traits.get("max_stage", len(STAGES) - 1))
        self.stage_idx = self.birth_stage
        # v11 waggle: newborn recruited toward the best advertised dance
        self.recruit_family = None
        self.recruit_transform = None
        if traits.get("recruit") and DANCES:
            q, fam, tf, _k = max(DANCES, key=lambda d: d[0])
            self.recruit_family, self.recruit_transform = fam, tf
            log("waggle_recruit", explorer=self.name, to_family=fam, to_transform=tf,
                quality=round(q, 4))

    def candidates(self, n_cols: int) -> list[tuple[str, ViewportSpec]]:
        _, stage_lvl, params = STAGES[self.stage_idx]
        menu = stage_viewport_menu(self.stage_idx, params["max_k"], n_cols)
        out = []
        for skill, meta in SKILL_REGISTRY.items():
            if meta["stage"] > stage_lvl:
                continue
            for spec in menu:
                if meta["needs_identity"] and spec.transform != "identity":
                    continue
                out.append((skill, spec))
        return out

    def ucb_pick(self, library: SharedLibrary, n_cols: int,
                 lane: str | None = None, exclude_key: str | None = None) -> tuple[str, ViewportSpec] | None:
        best, best_score = None, -1e9
        total = sum(library.runs.values()) + 1
        for skill, spec in self.candidates(n_cols):
            key = f"{skill}|{spec.name}"
            if lane is not None and lesson_lane(skill) != lane:
                continue
            if exclude_key is not None and key == exclude_key:
                continue
            if not library.can_run(skill, spec):
                continue
            if SKILL_REGISTRY[skill]["cost"] > self.budget:
                continue
            prior = 0.02 * self.traits["skill_prior"].get(skill, 0.5)
            prior += 0.01 * self.traits["transform_prior"].get(spec.transform, 0.5)
            prior += 0.01 * self.traits["family_prior"].get(spec.family, 0.5)
            if SURVEY:
                # v10 satellite map: families where orbit saw signal get lift
                prior += 0.01 * SURVEY.get(spec.family, 0.0) / (max(SURVEY.values()) + 1e-9)
            social_gain = max(library.mean_gain(key), library.family_gain(spec.family))
            wb = width_bias_beta()
            if self.cfg.SENSORY_ROSTER:
                # v34: per-explorer WIDTH_PREF scales the wide-path blend, so the
                # population MIXES wide (albatross, pref 0.9) and narrow (kestrel,
                # pref 0.1) trails in the SAME run. 0.5 = neutral = x1.0 = v33.
                wb = max(0.0, min(1.0, wb * 2.0 * float(self.traits.get("width_pref", 0.5))))
            if wb > 0.0:
                # v30 initial wide-path bias: the early social signal listens to
                # WIDTH (robust lower-bound strength); anneals to pure corr-gain
                social_gain = (1.0 - wb) * social_gain + wb * max(
                    library.mean_width(key), library.family_width(spec.family))
            social = self.traits["sociality"] * social_gain
            explore = (0.5 + self.traits["curiosity"]) * self.cfg.UCB_C * math.sqrt(
                math.log(total + 1) / (library.runs.get(key, 0) + 1))
            # v10 surprise curiosity: attention flows where the map is wrong
            explore += self.traits["curiosity"] * self.cfg.SURPRISE_W * \
                library.coord_surprise(skill, spec.family, spec.transform)
            caution_pen = self.traits["caution"] * 0.004 * SKILL_REGISTRY[skill]["cost"]
            # v10 venom memory: one-trial fear on predator-killed motifs
            venom = self.cfg.TABOO_W * (TABOO.get(f"{skill}|{spec.transform}", 0.0)
                                        + TABOO.get(f"{skill}|{spec.family}", 0.0))
            # v11 quorum sensing: a family >= QUORUM_MIN distinct species have
            # promoted is switched ON for the whole colony (a vote, not a gradient)
            quorum_lift = self.cfg.QUORUM_BOOST if len(QUORUM.get(spec.family, ())) >= self.cfg.QUORUM_MIN else 0.0
            # v11 waggle: this newborn was recruited toward a rich patch
            recruit = 0.0
            if self.recruit_family is not None:
                recruit = self.cfg.WAGGLE_W * (
                    (spec.family == self.recruit_family) + (spec.transform == self.recruit_transform))
            # v27 self-improvement: cross-run GENERALIZATION prior -- lift skills/
            # families whose accumulated track record is high-oof AND low-decay,
            # penalize chronic decayers. Evidence-shrunk; no-op with no prior ledger.
            gen_prior = 0.0
            if LEDGER_PRIOR:
                for st in (LEDGER_PRIOR.get("family_decay", {}).get(spec.family),
                           LEDGER_PRIOR.get("skill_decay", {}).get(skill)):
                    if st and int(st.get("n", 0)) >= 2:
                        gen_prior += self.cfg.LEDGER_PRIOR_W * (float(st.get("mean_oof", 0.0))
                                                               - 2.0 * max(0.0, float(st.get("mean_decay", 0.0))))
            # v36 advisor loop: an external model's instructions nudge WHERE/HOW
            # to look (additive prior on family/transform/skill). Still measured
            # through the honest doors; empty/off => 0 => no-op.
            advisor = 0.0
            if self.cfg.ADVISOR_INGEST and ADVISOR_PRIORS:
                advisor = self.cfg.ADVISOR_PRIOR_W * (
                    float(ADVISOR_PRIORS.get("family", {}).get(spec.family, 0.0))
                    + float(ADVISOR_PRIORS.get("transform", {}).get(spec.transform, 0.0))
                    + float(ADVISOR_PRIORS.get("skill", {}).get(skill, 0.0)))
            score = (prior + social + explore - caution_pen - venom + quorum_lift
                     + recruit + gen_prior + advisor)
            if score > best_score:
                best, best_score = (skill, spec), score
        return best

    def maybe_graduate(self, stage_gains: list[float]) -> bool:
        if self.stage_idx >= self.max_stage:
            return False
        _, _, params = STAGES[self.stage_idx]
        mastered = bool(stage_gains) and max(stage_gains) >= params["graduate_gain"]
        exhausted = len(stage_gains) >= params["max_lessons"]
        if mastered or exhausted:
            self.stage_idx += 1
            log("explorer_graduates", explorer=self.name, to=STAGES[self.stage_idx][0],
                via="mastery" if mastered else "exhaustion", best=round(max(stage_gains or [0.0]), 4))
            return True
        return False


