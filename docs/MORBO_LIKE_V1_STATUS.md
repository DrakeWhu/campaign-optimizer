# MORBO-like v1 status

## Current status

The `morbo_like` backend is implemented as a regional multi-objective optimizer backend inside the existing recommendation flow.

It is selected through configuration:

```json
{
  "recommendation": {
    "backend": "morbo_like"
  }
}
```

There is no separate CLI command. The existing commands remain the entry points:

```bash
python -m campaign_optimizer.cli.propose_recommendations --config optimizer.json --iteration N
python -m campaign_optimizer.cli.run_iteration --config optimizer.json --iteration N
```

## Implemented
- `backend = "morbo_like"` selectable from the existing recommendation flow.
- No duplicate recommendation CLI.
- File-based optimizer state.
- Candidate seen/pending registry.
- Search-space encoding/decoding/signatures.
- Objective canonicalization to maximization space.
- Observation rehydration from reduced metrics/objective tables.
- Pareto frontier computation.
- Regional state with expansion/contraction/restart bookkeeping.
- `global_random` fallback.
- `regional_random` proposals.
- Optional `regional_model` proposals through BoTorch qLogNEHVI/qNEHVI.
- Model diagnostics and configurable random fallback.
- `recommended_candidates.tsv` compatible with the capillary candidate-batch builder.
- `candidate_batch.tsv` generation through the existing downstream path.
- Sidecar files:
    - `morbo_optimizer_state.json`
    - `morbo_frontier.json`
    - `morbo_regions.json`
    - `surrogate_summary.json`

## Explicit non-goals still enforced

The MORBO-like backend does not:

- launch WarpX;
- launch SLURM jobs;
- read raw HDF5/openPMD diagnostics;
- edit WarpX/PICMI input files;
- delete raw data;
- own cleanup policy;
- modify `campaign-workflow`.

It consumes reduced observations and emits candidate recommendations.

## Historical SUNRISE smoke report

The following is a previously recorded result, not a run repeated during the
2026-10-20 Cloud baseline. See `BASELINE_2026-10-20.md` for newly executed tests.

A file-based smoke test was run on SUNRISE against reduced data from:

```text
/gpfs/home/jrodriguez/warpx_runs/capillaries_bo_transverse_campaign
```

Smoke output root:

```text
/gpfs/home/jrodriguez/warpx_runs/morbo_like_smoke_20260703_114039
```

Result:

```text
OK    inputs/observations.csv
OK    inputs/objective_table.csv
OK    outputs/recommended_candidates.tsv
OK    outputs/candidate_batch.tsv
OK    outputs/morbo_optimizer_state.json
OK    outputs/morbo_frontier.json
OK    outputs/morbo_regions.json
OK    outputs/surrogate_summary.json

observations rows:      351
fit eligible rows:      241
recommendations rows:   6
candidate_batch rows:   6
backend:                morbo_like
surrogate_backend:      morbo_like_no_botorch
last_strategy:          regional_random
```

This validates the file-based workflow on real reduced SUNRISE data:

```text
reduced campaign data
→ observations.csv
→ objective_table.csv
→ MORBO-like sync/frontier/regions/state
→ recommended_candidates.tsv
→ candidate_batch.tsv
```

No WarpX simulations, SLURM jobs, raw HDF5/openPMD reads, or cleanup actions were performed.

## Name and limitations

This backend is intentionally named:

```text
morbo_like
```

It is not yet a validated faithful implementation of the MORBO paper.

Current status:

```text
multi-objective: yes
Pareto/frontier-aware: yes
regional/trust-region-inspired: yes
file-contract compatible: yes
validated full MORBO implementation: no
```

## Optional model implementation and validation limits

`campaign_optimizer/morbo/botorch_model.py` implements the optional
`regional_model` path with lazy Torch/BoTorch/GPyTorch imports. It fits models
and ranks candidates from sampled regional pools. The backend preserves
`global_random` and `regional_random` paths and records model diagnostics.
Random fallback depends on `recommendation.botorch.fallback_to_random`;
do not describe fallback as a successful model fit.

The code uses raw canonical objective units for acquisition reference points
and records standardization metadata. Implementation presence is not proof of
numerical correctness or equivalence to published MORBO.

The Cloud baseline ran without Torch, BoTorch, GPyTorch, Optimas, or Ax.
`tests/test_morbo_botorch_optional.py` checks configuration, disabled behavior,
and fallback; it does not establish a successful real BoTorch fit.
Actual model/acquisition validation remains a separate task.

## Why this is still called `morbo_like`

The backend is intentionally called `morbo_like`, not `morbo`.

The current implementation is MORBO-inspired and already includes several core ingredients:

```text
- multi-objective canonical objective handling;
- Pareto frontier tracking;
- regional/trust-region-like state;
- region expansion/contraction/restart bookkeeping;
- random regional fallback;
- optional BoTorch qLogNEHVI/qNEHVI regional model proposals;
- persistent file-based optimizer state;
- integration with the existing recommendation and candidate-batch workflow.
```

However, it is not yet claimed to be a faithful implementation of the published MORBO algorithm.

The published MORBO method performs multi-objective Bayesian optimization over high-dimensional spaces using multiple local trust regions in parallel with a coordinated strategy. Our current implementation uses regional state and region-constrained candidate pools, but it still needs additional validation before it can be described as MORBO without qualification.

Current gaps:

```text
1. Local models per trust region are not yet implemented as in the published method.
   The current BoTorch mode fits a model on the selected training set and ranks
   candidates sampled from regional pools.

2. Region coordination, batch allocation, success/failure rules, restart behavior,
   and trust-region radius updates have not yet been audited against the MORBO
   paper or the official implementation.

3. Acquisition optimization is currently implemented by ranking sampled regional
   candidate pools. This is practical for the campaign workflow, but it is not yet
   validated as equivalent to the acquisition optimization used by MORBO.

4. Mixed/categorical parameter handling is approximate. In particular, categorical
   one-hot geometry inside trust-region bounds can bias or freeze category choices.

5. The implementation has not yet been benchmarked against the official MORBO code
   on standard high-dimensional multi-objective test problems.

6. The implementation has been workflow-validated on real reduced SUNRISE WarpX
   campaign data, but not algorithmically validated as a faithful MORBO reproduction.
```

Therefore the correct description is:

```text
MORBO-like regional multi-objective Bayesian recommendation backend.
```

A more explicit description is:

```text
A file-based, campaign-workflow-compatible, MORBO-inspired regional
multi-objective backend with Pareto frontier tracking, trust-region-like
state, BoTorch qLogNEHVI/qNEHVI proposal mode, and random fallbacks.
```

Acceptable wording only when a report has evidence of actual BoTorch execution:

```text
We implemented a MORBO-inspired regional multi-objective Bayesian optimization
backend using BoTorch qLogNEHVI, integrated with the WarpX campaign optimization
workflow.
```

For dependency-free runs, report the actual random strategy and any fallback
reason instead of implying that Bayesian model fitting occurred.

Avoid for now:

```text
We used MORBO.
```

That wording should only be used after validating the implementation against the
published MORBO algorithm and/or the official reference implementation.
