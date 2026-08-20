from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_json, write_tsv
from campaign_optimizer.morbo import (
    BotorchRegionalConfig,
    CategoricalRegionalPolicy,
    MorboLikeBackend,
    ObjectiveSpec,
    OptimizerState,
    RegionalPolicy,
    TrialInput,
    load_optimizer_state,
    save_frontier_records,
    save_optimizer_state,
    save_region_records,
)

from .parameters import (
    build_codec,
    build_search_space,
    parameter_names,
    reference_parameters,
    sobol_parameters,
)


def _load_previous_state(
    config: OptimizerConfig, iteration: int, outputs_dir: Path
) -> OptimizerState | None:
    current = outputs_dir / "morbo_optimizer_state.json"
    if current.is_file():
        return load_optimizer_state(current)
    for previous_iteration in range(iteration - 1, -1, -1):
        path = (
            config.iteration_dir(previous_iteration)
            / "outputs"
            / "morbo_optimizer_state.json"
        )
        if path.is_file():
            return load_optimizer_state(path)
    return None


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _proposal_region_map_from_state(
    state: OptimizerState | None,
    *,
    backend: MorboLikeBackend,
    trials: list[TrialInput],
) -> dict[str, str]:
    """Map observed trial IDs back to the region that proposed them.

    Materialization does not guarantee that recommendation candidate IDs survive
    as observation IDs, so attribution is recovered through the stable candidate
    signature persisted in pending_proposals.  When observation provenance does
    not contain that signature (historical campaigns), the legacy reconstruction
    from trial.params remains the compatibility fallback.
    """

    if state is None:
        return {}

    pending = (
        state.extra.get("pending_proposals", [])
        if isinstance(state.extra, dict)
        else []
    )
    if not isinstance(pending, list):
        return {}

    region_by_signature: dict[str, str] = {}
    for item in pending:
        if not isinstance(item, dict):
            continue
        signature = str(item.get("candidate_signature", "")).strip()
        region_id = str(item.get("region_id", "")).strip()
        if signature and region_id:
            region_by_signature[signature] = region_id

    out: dict[str, str] = {}
    for trial in trials:
        signature = _optional_text(trial.metadata.get("candidate_signature", ""))
        if not signature:
            try:
                signature = backend.space_codec.signature(trial.params)
            except Exception:
                continue
        region_id = region_by_signature.get(signature)
        if region_id:
            out[trial.candidate_id] = region_id

    return out


def _fit_rows(
    observations: pd.DataFrame, objectives: pd.DataFrame
) -> pd.DataFrame:
    if observations.empty or objectives.empty:
        return observations.head(0).copy()
    eligible_ids = objectives.loc[
        objectives["fit_eligible"].astype(str).str.lower() == "true",
        "observation_id",
    ]
    return observations[
        observations["observation_id"].isin(eligible_ids)
    ].copy()


def _used_sobol_indices(observations: pd.DataFrame) -> list[int]:
    if "sobol_index" not in observations.columns:
        return []
    values = pd.to_numeric(observations["sobol_index"], errors="coerce")
    return sorted({int(value) for value in values.dropna().tolist() if value >= 0})


def _base_recommendation_row(
    *,
    iteration: int,
    rank: int,
    candidate_id: str,
    source: str,
    signature: str,
    params: dict[str, Any],
    sobol_index: int | str = "",
    region_id: str | None = None,
    strategy: str | None = None,
    acquisition_value: float | None = None,
) -> dict[str, Any]:
    return {
        "recommendation_id": f"iter_{iteration:03d}_rank_{rank:03d}",
        "optimizer_iteration": int(iteration),
        "candidate_id": candidate_id,
        "rank": int(rank),
        "recommendation_status": "proposed",
        "candidate_source": source,
        "recommendation_backend": "sobol_then_morbo",
        "surrogate_backend": (
            "none_sobol_design" if source in {"reference", "sobol"} else "morbo_like"
        ),
        "ranking_source": source,
        "acquisition_value": "" if acquisition_value is None else acquisition_value,
        "candidate_signature": signature,
        "region_id": "" if region_id is None else region_id,
        "morbo_strategy": "" if strategy is None else strategy,
        "sobol_index": sobol_index,
        **params,
    }


def _write_sobol_recommendations(
    *,
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
    observations: pd.DataFrame,
    eligible_count: int,
    minimum_observations: int,
) -> Path:
    parameter_space = config.parameter_space()
    rec_cfg = config.recommendation_config()
    design = dict(rec_cfg.get("initial_design", {}) or {})
    seed = int(design.get("seed", rec_cfg.get("seed", 12345)))
    target = int(design.get("sobol_points", 16))
    initial_batch_size = int(design.get("initial_batch_size", 8))
    continuation_size = int(
        design.get("continuation_batch_size", rec_cfg.get("n_candidates", 4))
    )
    include_reference = bool(design.get("include_reference", True))

    used = _used_sobol_indices(observations)
    next_index = max(used) + 1 if used else 0
    remaining_balanced_design = max(target - len(used), 0)
    if remaining_balanced_design > 0:
        requested = initial_batch_size
        count = min(requested, remaining_balanced_design)
        phase = "balanced_sobol"
    else:
        count = continuation_size
        phase = "sobol_until_model_ready"

    codec = build_codec(parameter_space)
    known_signatures: set[str] = set()
    for _, row in observations.iterrows():
        try:
            known_signatures.add(
                codec.signature({name: row[name] for name in parameter_names(parameter_space)})
            )
        except Exception:
            continue

    rows: list[dict[str, Any]] = []
    rank = 1
    if iteration == 0 and include_reference:
        params = reference_parameters(parameter_space)
        signature = codec.signature(params)
        rows.append(
            _base_recommendation_row(
                iteration=iteration,
                rank=rank,
                candidate_id=f"mc_{iteration:03d}_reference",
                source="reference",
                signature=signature,
                params=params,
            )
        )
        known_signatures.add(signature)
        rank += 1

    accepted = 0
    candidate_index = next_index
    while accepted < count:
        params = sobol_parameters(
            parameter_space,
            start_index=candidate_index,
            count=1,
            seed=seed,
        )[0]
        signature = codec.signature(params)
        if signature not in known_signatures:
            rows.append(
                _base_recommendation_row(
                    iteration=iteration,
                    rank=rank,
                    candidate_id=f"mc_{iteration:03d}_sobol_{candidate_index:06d}",
                    source="sobol",
                    signature=signature,
                    params=params,
                    sobol_index=candidate_index,
                )
            )
            known_signatures.add(signature)
            accepted += 1
            rank += 1
        candidate_index += 1

    output = write_tsv(
        outputs_dir / "recommended_candidates.tsv", pd.DataFrame(rows)
    )
    write_json(
        outputs_dir / "surrogate_summary.json",
        {
            "schema_version": 1,
            "problem_kind": "multichannel",
            "backend": "sobol_then_morbo",
            "status": "initial_design",
            "phase": phase,
            "seed": seed,
            "sobol_target": target,
            "sobol_indices_already_used": used,
            "sobol_indices_proposed": [
                int(row["sobol_index"])
                for row in rows
                if row["candidate_source"] == "sobol"
            ],
            "eligible_observations": int(eligible_count),
            "minimum_observations": int(minimum_observations),
            "candidate_rows": len(rows),
        },
    )
    return output


def _morbo_trials(
    history: pd.DataFrame,
    *,
    parameter_space: dict[str, Any],
    objective_spec: ObjectiveSpec,
) -> list[TrialInput]:
    trials: list[TrialInput] = []
    names = parameter_names(parameter_space)
    for _, row in history.sort_values("observation_id", kind="stable").iterrows():
        params = {name: float(row[name]) for name in names}
        metrics = {
            objective.metric: float(row[objective.metric])
            for objective in objective_spec.objectives
        }
        trials.append(
            TrialInput(
                candidate_id=str(row["observation_id"]),
                params=params,
                raw_metrics=metrics,
                simulation_status="finished",
                metadata={
                    "source_case_id": str(row.get("source_case_id", "")),
                    "source_case_name": str(row.get("source_case_name", "")),
                    "candidate_signature": _optional_text(
                        row.get("candidate_signature", "")
                    ),
                },
            )
        )
    return trials


def _write_morbo_recommendations(
    *,
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
    history: pd.DataFrame,
) -> Path:
    parameter_space = config.parameter_space()
    rec_cfg = config.recommendation_config()
    objective_spec = ObjectiveSpec.from_dict(config.objective_config())
    previous_state = _load_previous_state(config, iteration, outputs_dir)

    regional_policy = RegionalPolicy.from_dict(rec_cfg.get("regional_policy", {}) or {})
    botorch_payload = dict(rec_cfg.get("botorch", {}) or {})
    categorical_payload = dict(rec_cfg.get("categorical_policy", {}) or {})
    backend = MorboLikeBackend(
        build_search_space(parameter_space),
        objective_spec,
        seed=int(rec_cfg.get("seed", 12345)),
        min_observations=rec_cfg.get("min_observations"),
        regional_policy=regional_policy,
        state=previous_state,
        suggestion_mode=str(rec_cfg.get("suggestion_mode", "regional_model")),
        botorch_config=BotorchRegionalConfig.from_dict(botorch_payload),
        categorical_policy=CategoricalRegionalPolicy.from_dict(categorical_payload),
    )
    trials = _morbo_trials(
        history,
        parameter_space=parameter_space,
        objective_spec=objective_spec,
    )
    sync_result = backend.sync(
        trials,
        proposal_region_map=_proposal_region_map_from_state(
            previous_state,
            backend=backend,
            trials=trials,
        ),
    )

    count = int(rec_cfg.get("n_candidates", 4))
    proposals = backend.suggest_proposals(count)
    backend.register_pending(proposals)

    rows = []
    for rank, proposal in enumerate(proposals, start=1):
        rows.append(
            _base_recommendation_row(
                iteration=iteration,
                rank=rank,
                candidate_id=f"mc_{iteration:03d}_morbo_{rank - 1:03d}",
                source="morbo",
                signature=proposal.candidate_signature,
                params=dict(proposal.params),
                region_id=proposal.region_id,
                strategy=proposal.strategy,
                acquisition_value=proposal.acquisition_value,
            )
        )

    backend.state = backend.state.with_updates(
        last_strategy=backend.last_strategy,
        extra={
            **dict(backend.state.extra),
            "problem_kind": "multichannel",
            "pending_proposals": [dict(row) for row in rows],
        },
    )
    save_optimizer_state(outputs_dir / "morbo_optimizer_state.json", backend.state)
    save_frontier_records(
        outputs_dir / "morbo_frontier.json", list(sync_result.frontier)
    )
    save_region_records(outputs_dir / "morbo_regions.json", list(sync_result.regions))

    output = write_tsv(
        outputs_dir / "recommended_candidates.tsv", pd.DataFrame(rows)
    )
    write_json(
        outputs_dir / "surrogate_summary.json",
        {
            "schema_version": 1,
            "problem_kind": "multichannel",
            "backend": "sobol_then_morbo",
            "surrogate_backend": (
                "morbo_like_botorch_qnehvi"
                if backend.last_strategy == "regional_model"
                else "morbo_like_random_fallback"
            ),
            "status": "ok",
            "phase": "morbo",
            "fit_rows": len(history),
            "candidate_rows": len(rows),
            "encoded_dimensions": backend.space_codec.encoded_dim(),
            "minimum_observations": backend.min_observations,
            "frontier_count": len(sync_result.frontier),
            "region_count": len(sync_result.regions),
            "last_strategy": backend.last_strategy,
            "botorch_config": backend.botorch_config.as_dict(),
            "model_diagnostics": backend.last_model_diagnostics,
        },
    )
    return output


def propose_recommendations(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    outputs_dir = iter_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    observations = read_table(iter_dir / "inputs" / "observations.csv")
    objectives = read_table(iter_dir / "inputs" / "objective_table.csv")
    history = _fit_rows(observations, objectives)

    parameter_space = config.parameter_space()
    objective_spec = ObjectiveSpec.from_dict(config.objective_config())
    codec = build_codec(parameter_space)
    rec_cfg = config.recommendation_config()
    minimum_observations = int(
        rec_cfg.get("min_observations")
        or max(4, len(objective_spec.objectives) + 1, codec.encoded_dim() + 1)
    )
    design = dict(rec_cfg.get("initial_design", {}) or {})
    sobol_target = int(design.get("sobol_points", 16))
    used_sobol = _used_sobol_indices(observations)

    if len(used_sobol) < sobol_target or len(history) < minimum_observations:
        return _write_sobol_recommendations(
            config=config,
            iteration=iteration,
            outputs_dir=outputs_dir,
            observations=observations,
            eligible_count=len(history),
            minimum_observations=minimum_observations,
        )

    return _write_morbo_recommendations(
        config=config,
        iteration=iteration,
        outputs_dir=outputs_dir,
        history=history,
    )