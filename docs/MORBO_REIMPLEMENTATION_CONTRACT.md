# MORBO-like Backend Reimplementation Contract

## 1. Purpose

This document defines the contract for reimplementing the useful parts of the legacy `parameter_scan` MORBO/BO code into the modern `campaign-optimization` repository.

The goal is not to port the old code verbatim. The goal is to preserve the scientifically useful optimizer invariants while adapting them to the current file-based, auditable, HPC-safe campaign workflow.

The legacy source of interest is:

```text
/home/jrope/workspaces/cursor_workspace/Python/parameter_scan
```

especially:

```text
commands/morbo_backend.py
commands/morbo_observations.py
tests/commands/test_morbo_backend.py
tests/commands/test_morbo_observations.py
```

The old Lynx inventory identifies `parameter_scan` as the proto-framework that evolved toward the current campaign workflow, containing `commands/`, `slurm/`, `tests/`, `wake_analysis/`, `database_manager.py`, `main_morbo.py`, `config.ini`, SQLite DBs and legacy notebooks.

The old MORBO layer already contains meaningful abstractions: `TrainingData`, `ReferencePointPolicy`, `CandidateProposal`, `RegionalPolicy`, `OptimizerBackend`, `SearchSpaceCodec`, `MultiObjectiveBayesianBackend`, `ObjectiveDefinition`, `ObjectiveSpec`, `MorboObservationStore`, `FrontierRecord`, `RegionRecord`, and `RehydratedObservation`.

## 2. Terminology: is this MORBO?

### 2.1 What MORBO means externally

In the literature, MORBO refers to **Multi-Objective Bayesian Optimization over high-dimensional search spaces**. The PMLR paper describes MORBO as a scalable method for multi-objective BO in high-dimensional search spaces, using multiple local regions of the design space in parallel with a coordinated strategy.

The official archived repository `facebookresearch/morbo` states that it is the code associated with the paper *Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces*.

### 2.2 What the legacy code currently is

The legacy implementation is **multi-objective** in the practical sense:

* It supports multiple objectives through `ObjectiveSpec`.
* Each objective has a metric, sense, transform and scale.
* `min` objectives are canonicalized by sign flip so that the internal optimizer maximizes all objectives.
* It computes and persists a Pareto frontier.
* It uses a hypervolume-style acquisition through BoTorch’s `qLogNoisyExpectedHypervolumeImprovement`.
* It manages multiple local regions with expansion, contraction and restart logic.

BoTorch’s own documentation describes multi-objective BO as learning the Pareto front, i.e. the set of optimal trade-offs between competing objectives, and lists `qLogNoisyExpectedHypervolumeImprovement` as a supported multi-objective acquisition function.

Therefore:

```text
The legacy backend is genuinely multi-objective.
```

However, it should **not** be called a faithful or validated implementation of MORBO yet.

### 2.3 Correct name for v1

The v1 reimplementation shall be documented as:

```text
MORBO-like regional multi-objective Bayesian optimization backend
```

or shorter:

```text
morbo_like
```

The name `morbo_like` is intentional. It means:

```text
- multi-objective: yes
- Pareto/frontier-aware: yes
- regional/trust-region-inspired: yes
- file-contract compatible: yes
- exact MORBO paper implementation: no, not yet
```

## 3. Non-goals

The new backend must not:

1. Launch WarpX.
2. Launch `sbatch`, `srun`, `mpirun`, or `mpiexec`.
3. Read raw HDF5/openPMD.
4. Touch `input.py`, `lwfa_3d.py`, or physics inputs.
5. Delete raw data.
6. Depend on campaign-local SQLite as a mutable runtime database.
7. Generate ad hoc SLURM scripts.
8. Own cleanup policy.
9. Edit `campaign-workflow` state, validation evidence, manifests, or post directories.
10. Depend on hardcoded names such as `combination_0001` or `lwfa_3d.py`.

The backend is an optimizer only. It consumes reduced observations and emits candidate recommendations.

## 4. Inputs and outputs

### 4.1 Required inputs

The backend consumes file-based products already produced by the current pipeline:

```text
observations.csv
objective_table.csv
optimizer_config.json
candidate_history.tsv or candidate_history.json
```

The exact upstream generation remains owned by `campaign-optimization` existing CLIs:

```text
build_observations
build_objectives
propose_recommendations / run_iteration
```

### 4.2 Optional inputs

```text
optimizer_state.json
frontier.json
regions.json
pending_candidates.tsv
previous recommended_candidates.tsv
```

These are used to rehydrate the optimizer state across iterations.

### 4.3 Required outputs

The backend must emit:

```text
recommended_candidates.tsv
optimizer_state.json
optimizer_report.json
```

### 4.4 Optional outputs

```text
frontier.json
regions.json
training_data_summary.json
candidate_diagnostics.tsv
```

## 5. Proposed module layout

The reimplementation should be modular and pure where possible:

```text
campaign_optimizer/
  morbo/
    __init__.py
    search_space.py
    objectives.py
    observations.py
    frontier.py
    regions.py
    state.py
    backend.py
```

No module in `campaign_optimizer/morbo/` may import `campaign_workflow`.

No module in `campaign_optimizer/morbo/` may invoke SLURM, WarpX, MPI, HDF5 readers, or cleanup tools.

## 6. Components to reimplement

### 6.1 `search_space.py`

Responsibility:

```text
Encode, decode, project, sample and canonicalize candidate parameter dictionaries.
```

Required classes:

```text
FloatRange
IntRange
Choice
SearchSpaceCodec
```

Required behavior:

* Stable parameter order.
* Projection of out-of-range continuous values into bounds.
* Integer rounding and clipping.
* Log-space encode/decode for positive log parameters.
* Categorical encode/decode.
* Canonical candidate signature.
* Deterministic behavior under fixed seed.

Legacy basis:

The old `SearchSpaceCodec` provides `signature`, `canonical_signature_payload`, `project_params`, `sample`, `sample_within_bounds`, `encoded_dim`, `encode`, and `decode`.

Important correction:

Categorical one-hot encoding may be kept in v1, but the contract must explicitly document that trust-region bounds over one-hot categorical axes are approximate. A future version should support a cleaner categorical policy.

### 6.2 `objectives.py`

Responsibility:

```text
Define objective mappings from reduced metrics to optimizer objectives.
```

Required classes:

```text
ObjectiveDefinition
ObjectiveSpec
ObjectiveSpecError
ObjectiveResolutionError
```

Required transforms:

```text
identity
abs
log10
log10p
sqrt
square
```

Required behavior:

* Validate objective name.
* Validate source metric name.
* Validate objective sense: `max` or `min`.
* Validate transform.
* Validate finite numeric values.
* Apply transform.
* Apply scale.
* Canonicalize all objectives into maximization space.

Legacy basis:

The old `ObjectiveDefinition` supports `name`, `metric`, `sense`, `transform`, `scale`, validates transform/sense, applies transforms and canonicalizes `min` objectives by negation.

### 6.3 `observations.py`

Responsibility:

```text
Build optimizer-ready observations from reduced file products.
```

Required data models:

```text
ObservedTrial
PendingTrial
SkippedTrial
TrainingData
RehydratedObservation
```

Required behavior:

* Load reduced observations from CSV/TSV.
* Resolve metrics into objectives using `ObjectiveSpec`.
* Reject observations with missing or invalid objectives.
* Preserve skip reasons.
* Build `TrainingData` with:

  * candidate IDs
  * parameter dictionaries
  * encoded X
  * canonical Y
  * raw objective values
  * objective names
  * senses

Legacy basis:

The old `MorboObservationStore` can rehydrate observed trials, fetch raw metrics, fetch statuses, persist objectives and distinguish observed/completed/failed/skipped trials. The new implementation keeps this concept but replaces SQLite runtime storage with file-based JSON/CSV state.

### 6.4 `frontier.py`

Responsibility:

```text
Compute Pareto frontier and related frontier-aware utilities.
```

Required functions:

```text
dominates(lhs, rhs, objective_names)
compute_pareto_frontier(observations, objective_names)
crowding_distance(observation, population, objective_names)
```

Important correction:

The old `_dominates()` uses `dict.values()`. The new implementation must compare objectives using explicit `objective_names`.

Correct contract:

```text
lhs dominates rhs iff:
  for every objective name: lhs[name] >= rhs[name]
  and for at least one objective name: lhs[name] > rhs[name]
```

No frontier computation may rely on dictionary insertion order.

### 6.5 `regions.py`

Responsibility:

```text
Manage local trust-region-inspired search regions.
```

Required data models:

```text
RegionalPolicy
RegionRecord
RegionAssignment
```

Required behavior:

* Initialize up to `max_regions` regions from diverse frontier observations.
* Assign observations to nearest active region in encoded space.
* Track region radius.
* Track observation count and frontier count.
* Track success and failure streaks.
* Expand radius after attributed success.
* Contract radius after attributed failure.
* Restart region after too many failures or minimum radius.
* Reseed restarted regions from frontier-aware candidates.
* Persist and rehydrate regions from `optimizer_state.json`.

Legacy basis:

The old `RegionalPolicy` contains `max_regions`, `initial_radius`, `min_radius`, `max_radius`, `expansion_factor`, `contraction_factor`, `restart_failure_threshold`, and `local_min_observations`.

The old backend initializes regions, assigns observations to nearest region, updates success/failure, expands/contracts radii and restarts regions.

Important correction:

The old logic partly uses monotonic `simulation_id` through `last_processed_simulation_id`. The new implementation must not rely on simulation IDs being monotonic. It should use explicit fields:

```text
iteration
batch_id
candidate_id
observed_at
proposal_origin_region_id
```

### 6.6 `state.py`

Responsibility:

```text
Persist and rehydrate optimizer state without SQLite runtime.
```

Required files:

```text
optimizer_state.json
frontier.json
regions.json
candidate_registry.json
```

Minimum `optimizer_state.json` content:

```json
{
  "schema_version": "morbo_like_state_v1",
  "backend": "morbo_like",
  "seed": 42,
  "objective_spec": {},
  "search_space_signature": "...",
  "seen_candidate_signatures": [],
  "pending_candidate_signatures": [],
  "ref_point_policy": {},
  "ref_point_raw": {},
  "ref_point_model_units": {},
  "y_transform": {},
  "regions": [],
  "last_strategy": "...",
  "created_at": "...",
  "updated_at": "..."
}
```

The file must be written atomically.

### 6.7 `backend.py`

Responsibility:

```text
Coordinate observations, frontier, regions and candidate generation.
```

Required backend modes:

```text
global_random
regional_random
regional_model
```

Required behavior:

1. Rehydrate state.
2. Build observations.
3. Compute canonical training data.
4. Compute frontier.
5. Compute/update regions.
6. Generate candidates.
7. Deduplicate against seen and pending signatures.
8. Emit candidates with diagnostics.

Minimum candidate output columns:

```text
candidate_id
candidate_signature
backend
strategy
region_id
iteration
batch_id
parameter columns...
```

## 7. BoTorch and model-based suggestions

The v1 implementation may be phased.

### 7.1 Phase A: no BoTorch required

Implement:

```text
global_random
regional_random
dedup
frontier
regions
state rehydration
```

This phase must pass all pure `unittest` tests and should not require `torch`, `botorch`, or `gpytorch`.

### 7.2 Phase B: optional BoTorch backend

Implement model-based regional suggestions only after Phase A.

The old backend uses:

```text
ModelListGP
SingleTaskGP
qLogNoisyExpectedHypervolumeImprovement
SobolQMCNormalSampler
optimize_acqf
```

BoTorch supports multi-objective BO and `qLogNoisyExpectedHypervolumeImprovement` as a multi-objective acquisition function.

Important correction:

The legacy code standardizes `Y_model` before model fitting, but computes/persists the reference point in raw canonical objective units. That is dimensionally inconsistent. The new implementation must ensure the reference point is in the same units as the model output.

Required state fields for Phase B:

```text
ref_point_raw
ref_point_model_units
y_transform
```

Required behavior:

```text
- compute canonical_Y_raw
- compute y_transform per objective
- train model in transformed units
- transform ref_point_raw into model units
- use ref_point_model_units for acquisition
- report raw-unit frontier and model-unit acquisition metadata separately
```

## 8. Explicitly documented limitations of v1

The v1 backend is not a faithful MORBO implementation.

It is:

```text
MORBO-like
regional
multi-objective
file-contract compatible
```

It is not yet:

```text
a validated reproduction of the Daulton/Eriksson/Balandat/Bakshy MORBO algorithm
```

Reasons:

1. The region scheduler is manually reimplemented and not yet checked against the official algorithm.
2. Trust-region success/failure is frontier/attribution-based, not yet validated against the paper’s exact update rules.
3. Categorical dimensions are handled approximately.
4. Batch allocation across regions is round-robin in the legacy code, not necessarily the coordinated strategy described in MORBO.
5. BoTorch model scaling/reference point handling requires correction.
6. No benchmark comparison against official MORBO code has been performed.
7. No synthetic high-dimensional benchmark suite has been run.
8. No high-dimensional multiobjective campaign has yet validated sample efficiency.

## 9. What would be required for “real MORBO”

To call a future backend `morbo` rather than `morbo_like`, the project must complete a separate validation phase.

Required work:

1. Read and summarize the MORBO paper and official implementation.
2. Identify exact algorithmic contracts:

   * region initialization
   * region success/failure update
   * restart rule
   * local surrogate fitting
   * acquisition function
   * coordinated batch allocation
   * reference point handling
   * pending/fantasized points
3. Compare our implementation against the official archived repository.
4. Add benchmark tests on synthetic multiobjective high-dimensional problems, preferably including DTLZ-style tests similar to the MORBO paper/repository.
5. Validate behavior with 2, 3 and 4 objectives.
6. Validate behavior in dimensions representative of our intended WarpX scans.
7. Add regression tests proving:

   * Pareto frontier matches expected non-dominated set
   * hypervolume does not degrade under simple known cases
   * region restarts happen deterministically
   * no duplicate candidates are emitted
   * pending candidates are respected
   * reference point is correctly transformed
8. Document deviations from official MORBO if any remain.

Only after that should the backend name become:

```text
morbo
```

Until then:

```text
morbo_like
```

## 10. Tests to migrate from legacy

The following legacy tests should be adapted into `campaign-optimization` using `unittest`.

### 10.1 Objective tests

```text
test_adapt_metrics_to_objectives_applies_transform_and_sense
test_evaluate_trial_eligibility_requires_terminal_success_and_all_objectives
test_rehydrate_observed_trials_from_file_state
```

### 10.2 Codec tests

```text
test_categorical_codec_projection_and_roundtrip_are_deterministic
test_duplicate_signature_canonicalization_collapses_equivalent_floats
test_int_and_log_roundtrip_stability
```

### 10.3 Training data tests

```text
test_training_data_uses_canonical_objectives_for_mixed_senses
test_backend_falls_back_to_random_when_observations_are_insufficient
test_pending_trials_are_not_trivially_resuggested
```

### 10.4 Frontier tests

```text
test_pareto_frontier_is_persisted_from_observed_trials
test_region_success_uses_true_multiobjective_improvement_not_scalar_sum
test_dominated_point_with_better_scalar_sum_does_not_count_as_success
test_dominance_uses_explicit_objective_order_not_dict_values
```

### 10.5 Region tests

```text
test_region_state_persistence_and_restart_rehydration
test_observation_to_region_assignment_is_deterministic
test_local_trust_region_candidate_generation_respects_bounds
test_region_update_policy_after_success_and_failure
test_region_restart_reseeds_when_failures_accumulate
test_duplicate_pending_filtering_still_works_under_regional_suggestion
test_backend_falls_back_gracefully_when_too_few_observations_for_regions
test_region_success_uses_attributed_proposals_not_geometric_reassignment
test_center_update_is_deterministic_under_revised_rule
test_reseed_choice_is_frontier_aware_and_deterministic
```

The old MORBO tests are already `unittest`, which makes them much more reusable than the older pytest-based CRUD tests in `parameter_scan`.

## 11. Acceptance criteria for v1

The v1 `morbo_like` backend is accepted when:

1. `python -m unittest discover -s tests -p "test_*.py"` passes locally.
2. The backend can run without BoTorch installed in random/regional-random mode.
3. The backend never launches simulations.
4. The backend never reads raw HDF5/openPMD.
5. The backend writes only optimizer-owned outputs.
6. Candidate signatures are stable across repeated runs.
7. Pending candidates are not re-suggested.
8. Seen candidates are not re-suggested.
9. Mixed max/min objectives are canonicalized correctly.
10. Pareto frontier computation is deterministic.
11. Region state can be rehydrated from files.
12. Region restart behavior is deterministic under fixed seed.
13. The backend emits `recommended_candidates.tsv` compatible with the existing candidate batch builder.
14. The documentation clearly states that this is `morbo_like`, not validated full MORBO.

## 12. Recommended implementation phases

### Phase 0 — Documentation only

Add:

```text
docs/MORBO_REIMPLEMENTATION_CONTRACT.md
```

No code changes.

### Phase 1 — Pure core

Add:

```text
campaign_optimizer/morbo/search_space.py
campaign_optimizer/morbo/objectives.py
campaign_optimizer/morbo/frontier.py
```

Add `unittest` tests for codec, objectives and Pareto frontier.

No BoTorch dependency.

### Phase 2 — File-based state and regional random

Add:

```text
campaign_optimizer/morbo/observations.py
campaign_optimizer/morbo/state.py
campaign_optimizer/morbo/regions.py
campaign_optimizer/morbo/backend.py
```

Implement:

```text
global_random
regional_random
dedup
pending handling
state rehydration
frontier persistence
regions persistence
```

No BoTorch dependency required.

### Phase 3 — CLI integration

Add a backend option to the existing recommendation CLI/config:

```text
backend = morbo_like
```

The backend must produce the same downstream candidate files expected by the current `campaign-optimization` flow.

### Phase 4 — Optional BoTorch model mode

Add:

```text
regional_model
```

Only after:

```text
- reference point scaling is corrected
- y_transform metadata is persisted
- fallback to regional_random is robust
- tests cover model/ref-point unit consistency
```

### Phase 5 — Real MORBO validation

Separate future phase. Not part of v1.

Deliverables:

```text
docs/MORBO_REAL_VALIDATION_PLAN.md
benchmark scripts
comparison against official MORBO implementation
synthetic high-dimensional multiobjective benchmarks
```

## 13. Final decision

The legacy code proves that the old optimizer was more advanced than a simple scalar BO loop. It already had the right high-level concepts:

```text
- objective canonicalization
- Pareto frontier
- seen/pending deduplication
- local regions
- expansion/contraction/restart
- persistent optimizer state
- multiobjective acquisition intent
```

But it should be modernized into a file-based `morbo_like` backend before being trusted in SUNRISE or future Lynx campaigns.

The correct near-term goal is:

```text
Implement a robust MORBO-like regional multi-objective backend in campaign-optimization.
```

The correct long-term goal is:

```text
Validate and, if justified, promote morbo_like -> morbo after comparison with the MORBO paper and official implementation.
```
