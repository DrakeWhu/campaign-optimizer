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

## SUNRISE smoke test

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

## Next planned stage

The next stage is optional BoTorch-backed regional model proposals:

```text
The next stage is optional BoTorch-backed regional model proposals:
```

The existing random modes must remain available:

```text
The existing random modes must remain available:
```

The BoTorch mode must be optional and must fall back to `regional_random` if fitting or acquisition evaluation fails.

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

Acceptable wording for reports or internal documentation:

```text
We implemented a MORBO-inspired regional multi-objective Bayesian optimization
backend using BoTorch qLogNEHVI, integrated with the WarpX campaign optimization
workflow.
```

Avoid for now:

```text
We used MORBO.
```

That wording should only be used after validating the implementation against the
published MORBO algorithm and/or the official reference implementation.