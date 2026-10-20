# Observation contract review

Status: current behavior and proposed questions, 2026-10-20.
No ingestion, objective, or eligibility behavior is changed by this document.
Behavior below refers to the audited default revision. The separate
`DEVELOPMENT_BRANCH_REVIEW_2026-10-20.md` records existing N2 capillary contracts,
source/state constraints, and their integration consequences at pinned tips.
They do not resolve generic multichannel first-row selection or the inspected
optional-output evidence mismatch. The unchanged baseline remains historical
Cloud evidence, not validation of those branches or SUNRISE/LYNX.

## Reduced products and evidence

The optimizer consumes reduced tables and workflow evidence, not raw HDF5.
Capillary defaults include `guiding_metrics.csv`,
`guiding_singlecase_score.csv`, `particle_analysis/particle_summary.csv`, and
`particle_analysis/particle_acceptance_curves.csv`. A campaign's required
reduced-output declaration is not automatically the same set.

Before raw deletion, the campaign owner must decide which products are needed
for optimization, baseline comparisons, and later scientific questions.
The workflow's `docs/INTEGRATION_REVIEW.md` records the cross-repository review
agenda. Do not infer scientific completeness from a successful CSV structure
check or from the presence of a guiding CSV alone.

## Species and first-row selection

`campaign_optimizer/multichannel/observations.py` uses `read_first_row_csv`.
It does not select a configured species from a multi-row summary. LFMetrics
can emit multiple species in request order; its SUNRISE wrapper currently
requests electrons alone.

Review before implementation:

- Which species does each objective measure?
- Should missing or duplicate requested species reject an observation?
- Should a summary have exactly one optimizer row or a species-keyed schema?
- How should existing multi-row summaries be migrated without changing the
  population represented by historical objective values?

Summing or selecting populations changes the scientific observable. Do not
silently aggregate species or change a producer's row order.

Nozzle uses `problem.kind="multichannel"` with
`particle_summary="post/mna_case_summary.csv"`. Its selected-particle count
and charge are compatibility fields for one carbon probe. Its actual
objectives are carbon terminal longitudinal energy gain and an exit-field
percentile peak; electron-beam thresholds must not be applied by analogy.

## Temporal selection

The capillary reader uses `target_guiding_iteration` from particle metrics
when it is numeric and the guiding table has an `iteration` column. No
matching guiding row in that path yields `missing_metric`. Without a usable
target or without that column, it accepts a single row or uses
`last_row_fallback`, recording the row policy.

This fallback is not an exact-exit measurement. Decide which campaigns may
use it, which timing metadata must survive reduction, and whether legacy
rows remain fit-eligible. LFMetrics' final dump and nozzle's terminal probe
sample/windowed field peak require distinct policies.

## Optional outputs

The workflow records `required` and `ok` per reduced output and can accept a
case with failed optional outputs. Current capillary and multichannel
ingestion checks success across the examined validation entries, rather than
filtering to the workflow's required set. Thus case validity and fit eligibility
can disagree.

Decide whether fit eligibility follows workflow-required products, a separate
objective-specific product set, or a stricter policy. Ignoring optional
failures without that decision could admit missing scientific inputs; keeping
all failures blocking can exclude usable cases because of visualization
failures.

## Future integration acceptance cases

After scientific approval, cover incomplete particle reduction before cleanup,
reversed/missing/duplicate species rows, exact and missing target iterations,
legacy temporal fallback, and optional-versus-required evidence failures.
Also cover nozzle independently of electron-beam observations.
These are proposed tests, not implemented or passed checks.
