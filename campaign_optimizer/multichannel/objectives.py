from __future__ import annotations

import math
from typing import Any

import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_csv
from campaign_optimizer.morbo import ObjectiveSpec, evaluate_trial_eligibility

from .parameters import parameter_names


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def build_objectives(config: OptimizerConfig, iteration: int):
    iter_dir = config.iteration_dir(iteration)
    inputs_dir = iter_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    observations = read_table(inputs_dir / "observations.csv")
    objective_config = config.objective_config()
    objective_spec = ObjectiveSpec.from_dict(objective_config)
    eligibility = dict(objective_config.get("eligibility", {}) or {})
    minimum_metric = str(
        eligibility.get("minimum_metric", "n_macroparticles_selected")
    ).strip()
    minimum_default = eligibility.get("minimum_selected_macroparticles", 200)
    minimum_value = _finite_float(
        eligibility.get("minimum_value", minimum_default)
    )
    if not minimum_metric or minimum_value is None:
        raise ValueError("eligibility minimum_metric/minimum_value are invalid")

    require_positive_charge = bool(
        eligibility.get("require_positive_charge", True)
    )
    positive_metric = str(
        eligibility.get("positive_metric", "charge_selected_pC")
    ).strip()
    if require_positive_charge and not positive_metric:
        raise ValueError("eligibility positive_metric must not be empty")

    required_status_metric = str(
        eligibility.get("required_status_metric", "")
    ).strip()
    required_status_value = str(
        eligibility.get("required_status_value", "ok")
    )
    require_finite_parameters = bool(
        eligibility.get("require_finite_parameters", True)
    )

    rows: list[dict[str, Any]] = []
    for _, observation in observations.iterrows():
        raw_metrics: dict[str, float] = {}
        for objective in objective_spec.objectives:
            value = _finite_float(observation.get(objective.metric))
            if value is not None:
                raw_metrics[objective.metric] = value

        reasons: list[str] = []
        if str(observation.get("simulation_status", "")) != "finished":
            reasons.append("simulation_not_reduced_valid")

        minimum_observed = _finite_float(observation.get(minimum_metric))
        if minimum_observed is None or minimum_observed < minimum_value:
            if minimum_metric == "n_macroparticles_selected":
                reasons.append("insufficient_selected_macroparticles")
            else:
                reasons.append(f"insufficient_metric:{minimum_metric}")

        positive_observed = _finite_float(observation.get(positive_metric))
        if require_positive_charge and (
            positive_observed is None or positive_observed <= 0.0
        ):
            if positive_metric == "charge_selected_pC":
                reasons.append("nonpositive_or_missing_charge")
            else:
                reasons.append(f"nonpositive_or_missing:{positive_metric}")

        if required_status_metric:
            observed_status = str(observation.get(required_status_metric, ""))
            if observed_status != required_status_value:
                reasons.append(f"status_mismatch:{required_status_metric}")

        if require_finite_parameters:
            for name in parameter_names(config.parameter_space()):
                if _finite_float(observation.get(name)) is None:
                    reasons.append(f"missing_parameter:{name}")

        resolved = evaluate_trial_eligibility(
            simulation_status=("finished" if not reasons else "incomplete"),
            raw_metrics=raw_metrics,
            spec=objective_spec,
            candidate_id=observation.get("observation_id"),
        )
        if not resolved.eligible:
            reasons.append(resolved.reason)

        row: dict[str, Any] = {
            "observation_id": observation.get("observation_id", ""),
            "fit_eligible": not reasons and resolved.eligible,
            "fit_exclusion_reason": ";".join(dict.fromkeys(reasons)),
            "eligibility_minimum_metric": minimum_metric,
            "eligibility_minimum_value": minimum_value,
            minimum_metric: minimum_observed,
            positive_metric: positive_observed,
        }
        for objective in objective_spec.objectives:
            raw_value = raw_metrics.get(objective.metric)
            row[objective.metric] = raw_value
            row[objective.name] = (
                resolved.objective_values.get(objective.name)
                if resolved.eligible
                else None
            )
            row[f"canonical_{objective.name}"] = (
                resolved.canonical_objective_values.get(objective.name)
                if resolved.eligible
                else None
            )
        rows.append(row)

    if rows:
        objective_table = pd.DataFrame(rows)
    else:
        columns = [
            "observation_id",
            "fit_eligible",
            "fit_exclusion_reason",
            "eligibility_minimum_metric",
            "eligibility_minimum_value",
            minimum_metric,
            positive_metric,
        ]
        for objective in objective_spec.objectives:
            columns.extend(
                [objective.metric, objective.name, f"canonical_{objective.name}"]
            )
        objective_table = pd.DataFrame(columns=list(dict.fromkeys(columns)))

    return write_csv(inputs_dir / "objective_table.csv", objective_table)
