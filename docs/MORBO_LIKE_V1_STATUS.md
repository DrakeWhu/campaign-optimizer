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