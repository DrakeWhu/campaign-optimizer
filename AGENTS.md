# Repository guidance: campaign-optimizer

## Scope and contracts

- This repository owns reduced-data ingestion, objectives, recommendation,
  candidate batches, reports, and optimizer state. The external workflow owns
  simulations, scheduler submission, case state, and raw-data cleanup.
- Read `docs/MORBO_REIMPLEMENTATION_CONTRACT.md`,
  `docs/MORBO_LIKE_V1_STATUS.md`, and `docs/OBSERVATION_CONTRACT_REVIEW.md`.
  Historical design milestones are not a current feature inventory.
- Consult `docs/DEVELOPMENT_BRANCH_REVIEW_2026-10-20.md` for existing N2
  contracts and state-migration constraints before implementing overlapping
  work. Those branch implementations are not incorporated by this guidance.
- Preserve `candidate_batch.tsv` and `batch_campaign_plan.json` compatibility
  with campaign-workflow, including custom schemas and materialization units.
- Do not read raw HDF5 to build optimization observations or directly import
  analysis/workflow packages into the optimizer.

## Scientific invariants

- Preserve objective directions, transforms, bounds, units, eligibility,
  seeds, deduplication, and persisted state. Document and test any intentional
  changes; do not silently substitute objectives or algorithms.
- Keep the `morbo_like` name. The optional BoTorch `regional_model` path exists,
  but equivalence to published MORBO is not established.
- Distinguish requested strategy, actual model execution, and random fallback.
  Preserve the configured fallback policy and its diagnostics.
- BoTorch/Torch/GPyTorch and Optimas/Ax are separate optional dependency paths.
  Passing tests with these packages absent does not validate their execution.
- Preserve current first-row, temporal-selection, and validation-evidence
  behavior until the pending observation contracts are reviewed. Do not
  silently reinterpret nozzle compatibility fields as electron-beam metrics.

## Cross-repository runtime and provenance

- Before inspecting or building observations for an HPC campaign, identify the
  actual optimizer checkout, full Git revision, selected Python environment,
  and imported `campaign_optimizer.__file__` **after** the final environment
  script is sourced. `PYTHONPATH` alone does not prove import provenance.
- Treat recorded workflow, optimizer, and guiding revisions as an explicit
  compatibility boundary. A current default branch must not silently replace
  a frozen production worktree. Do not assume a newer observation contract
  was available in a historical campaign.
- Keep historical campaign roots independent when importing reduced evidence;
  validate source identity, units, species, temporal selection, eligibility,
  and state compatibility before any multi-source optimization or state reuse.
- For SUNRISE environment and campaign operations, use the corresponding
  site-specific procedures in campaign-workflow when available. Do not
  duplicate workflow-owned Slurm, quota, Gate B, or cleanup logic here.
- Default to non-mutating inspection; report which commands would create
  observations, reports, candidate batches, or optimizer state before running.

## Verification

- Use `PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_*.py' -v`
  from this repository or an isolated export; use the intended scientific
  interpreter. Tests write temporary optimizer products.
- Record errors, skips, dependencies, and actual backends. See
  `docs/BASELINE_2026-10-20.md` for the dated baseline, not a guarantee for
  later revisions or SUNRISE/LYNX execution.
- Distinguish synthetic correctness from real-campaign performance and
  scientific improvement. Do not submit jobs or modify campaign data without
  task authorization.
- Write code, comments, tests, documentation, commits, and PR text in English.
