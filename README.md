# campaign-optimizer

File-based optimization for simulation campaigns. The package reads reduced
observations and validation evidence and produces recommendations,
`candidate_batch.tsv`, and `batch_campaign_plan.json` for campaign-workflow.
It does not run WarpX, submit SLURM jobs, or delete diagnostics.

## Layout and entry points

- `campaign_optimizer/capillary/`: capillary observations, scores, and batches.
- `campaign_optimizer/multichannel/`: configurable observations and parameter
  spaces, including Sobol-to-MORBO-like proposals.
- `campaign_optimizer/morbo/`: objective canonicalization, Pareto frontier,
  regions, state, and optional model proposals.
- `campaign_optimizer/cli/`: finite commands including `build_observations`,
  `build_objectives`, `propose_recommendations`, and `run_iteration`.
- `examples/`: optimizer configuration examples.

The nozzle campaign uses the configurable multichannel adapter; it is not a
separate optimizer problem kind. See `docs/OBSERVATION_CONTRACT_REVIEW.md`.

## Development environment

The core workflows use NumPy and pandas; Sobol generation uses SciPy and
reporting uses matplotlib. BoTorch/Torch/GPyTorch and Optimas/Ax are separate
optional backend dependencies. This repository currently has no packaging
manifest; do not assume `pip install .` is supported.

From the repository root, with the intended scientific interpreter:

```bash
PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_*.py' -v
```

The suite writes temporary fixtures. Use an isolated Git export when the
checkout must remain untouched. The dated interpreter, package inventory,
remote revision check, and results are in `docs/BASELINE_2026-10-20.md`.

## Algorithm and contract status

Read `docs/MORBO_LIKE_V1_STATUS.md` for the implemented paths and limitations.
`docs/MORBO_REIMPLEMENTATION_CONTRACT.md` preserves historical design rationale.
The backend is MORBO-inspired and is not claimed to reproduce published MORBO.
Passing the baseline without optional ML packages does not validate those
backends. Review species, timing, optional evidence, and reduced-product
completeness before changing cross-repository observation semantics.
