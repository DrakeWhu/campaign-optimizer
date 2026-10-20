# Development branch review — 2026-10-20

Repository: `campaign-optimizer`. Audited base: `0d2b039f820a0ecbf82f978bc23a5f65e69942b7`.
Remote references observed at `2026-10-20T11:41:17.168761+00:00`.

## Evidence scope

Remote heads were fetched into a temporary bare shared clone. Source repositories
were not switched, merged, rebased, or updated with development code. Inspection
used `git show`, `git diff`, `git log`, `git rev-list --left-right --count`,
and `git merge-base --is-ancestor`. Reference acquisition used `git ls-remote
--heads origin` and `git fetch --no-tags origin
'+refs/heads/*:refs/remotes/origin/*'` in the temporary clone. No branch tests,
simulations, scheduler jobs, analysis pipelines, or cleanup commands were run.
The inventory is a dated remote snapshot; findings below cover selected relevant
implementations, not certification of every changed file.

`BASELINE_2026-10-20.md` is preserved byte-for-byte as historical evidence for
the audited base and its recorded Cloud environment. It does not validate these
development tips, future versions, SUNRISE, LYNX, or WarpX execution. Its SHA-256:
`bbf6bf81033c5fd12e20274aeb5610cc1f85700e6c2fdff3314ba9896e6f3231`.

## Remote branch inventory

Counts are commits unique to the audited base / unique to the remote branch.
A zero branch-only count denotes an ancestor or the identical base, not pending
implementation work. Hashes, rather than moving branch names, identify evidence.

| Remote branch | Commit | Base-only / branch-only |
| --- | --- | --- |
| `feat/clpu-baseline-soft50-single-objective` | `1018bd59c6ee656158ac8aba0acb1d4c49f871ec` | 0 / 19 |
| `feat/clpu-soft50-sobol` | `9fc7e612f00a5ca4ee1b85de62167c6690546058` | 0 / 4 |
| `feat/multichannel-soft100-v2` | `e54692a3a64deceb44854f927ae745fad1e312a2` | 0 / 3 |
| `feature/clpu-n2-candidate-identity` | `4fa2edac8d3cc05f9fa45737a012e1b4a0618064` | 0 / 23 |
| `feature/clpu-n2-gate-a` | `73dab76305f547581c57b70a900706374c929141` | 0 / 24 |
| `feature/clpu-n2-observation-contract` | `c565f9071df8a75e48f38f48a4f4cf670029f4b1` | 0 / 20 |
| `feature/clpu-n2-source-manifest` | `5073547eb19c9a4d30662d0938efc4abbf27018a` | 0 / 21 |
| `feature/clpu-n2-state-fingerprint` | `23a442757d6634cb55f588617d8d1de7c11da26e` | 0 / 22 |
| `main` | `0d2b039f820a0ecbf82f978bc23a5f65e69942b7` | 0 / 0 |

## Verified implementation findings

- `feature/clpu-n2-observation-contract` adds opt-in
  `particle_observation_contract="clpu_n2_plateau_all_electrons_v1"` in
  `campaign_optimizer/capillary/observations.py`. It requires exactly ordered
  scopes `all_electrons`, `preionized_background_electrons`, and
  `nitrogen_ionized_electrons`; exit selection with `exact_resolved_v1`; and
  the same particle iteration across scopes. The aggregate summary remains
  authoritative. A unique aggregate Soft50 curve at the configured lower
  threshold must agree on iteration, schema, thresholds, and charges.
  Contract failures make the observation ineligible. Legacy ingestion remains
  available when the explicit contract is absent.
- This contract validates scope order before returning the aggregate row.
  It is not a general species-key lookup and is not a change to multichannel
  ingestion. `multichannel/observations.py` still calls `read_first_row_csv`
  at both `feat/multichannel-soft100-v2` and `feature/clpu-n2-gate-a`.
- The N2 chain is cumulative: observation contract -> source manifest -> state
  fingerprint -> candidate identity -> Gate A. `strict_static_sources_v1`
  checks the configured source count and unique aliases/roots.
  `clpu_n2_fresh_state_v1` fingerprints the search space, objective definition,
  and nitrogen semantics; it restricts bootstrap and later state reuse.
- Candidate-identity handling preserves signatures with the complete active
  keyset and migrates the otherwise complete legacy keyset by assigning
  `nitrogen_fraction=0`. Other nonempty incompatible keysets fail closed.
  The migration has scientific and deduplication consequences; it must not be
  adopted as a documentation correction.
- `capillary/gate_a.py` defines a specific warm-start gate: 40 static sources,
  344 initial observations, six physical parameters encoded into eight
  dimensions, eight candidates, and a direct Soft50 charge objective. It
  requires BoTorch regional modeling without random fallback. These are
  campaign-specific conditions, not generic optimizer defaults or verified
  results from this review.
- Soft100 branches add metrics and configurable objective eligibility checks.
  They do not select the intended species from a multi-row summary. The
  capillary `_read_guiding_metrics_row` and `_section_status`, and multichannel
  `_validation_section_ok`, have identical ASTs to the audited base at the
  Soft100 and Gate A tips. Thus those inspected paths retain the temporal
  fallback and optional-evidence behavior documented in
  `OBSERVATION_CONTRACT_REVIEW.md`.

Evidence (paths relative to `campaign_optimizer/`): `capillary/observations.py`, `config.py`,
`capillary/state_contract.py`, `capillary/candidate_identity.py`,
`capillary/gate_a.py`, `capillary/objectives.py`, and
`multichannel/observations.py` at the listed tips.
The N2 observation tests were read as implementation evidence, not executed.

## Possible conflicts and decisions required

No reviewed development branch changes a documentation path modified in this
stage. This path-level result does not establish semantic compatibility.
The N2 contracts concern capillary populations and exact exits; applying them
unchanged to LFMetrics or nozzle would reinterpret different physical
observables. Nozzle's adapter count/charge represent one carbon probe.

Before integration, choose whether scope order is itself a contract or whether
selection should be species-keyed; define missing/duplicate behavior and
migration of existing summaries. Define which legacy temporal fallbacks remain
fit-eligible and how optional workflow evidence affects objective-specific
eligibility. Review source freezing, historical nitrogen assumptions, candidate
identity, and optimizer-state reuse together. A default-branch unit baseline
with optional ML packages absent cannot certify the Gate A model execution.

## Integration boundary

The current stage is documentation only. No branch implementation, species
selection, objective, temporal policy, scientific-validation rule, or raw
cleanup behavior is adopted. Scientific decisions above remain pending.
Any later integration requires its own version-specific tests and site evidence.
