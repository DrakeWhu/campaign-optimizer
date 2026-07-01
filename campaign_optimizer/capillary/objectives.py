from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_csv
from campaign_optimizer.state import stable_hash


SCORE_SPECS = {
    "score_guiding_v1": [
        "metric_guiding_singlecase_score_v1",
        "metric_guiding_final_score",
        "metric_guiding_score",
    ],
    "score_beamlike_v1": [
        "metric_comparison_beam_beamlike_gain_score",
        "metric_particle_beamlike_gain_score",
        "metric_particle_beamlike_score",
    ],
    "score_transverse_v1": [
        "metric_particle_beam_transverse_quality_score_channel",
        "metric_particle_beam_transverse_quality_score",
    ],
    "score_acceptance_v1": [
        "metric_acceptance_Q_Ege100MeV_theta10mrad_pC",
    ],
}


def _score_specs_from_config(objective_config: dict[str, Any]) -> dict[str, list[str]]:
    specs = {score: list(columns) for score, columns in SCORE_SPECS.items()}

    override = objective_config.get("score_specs", {}) or {}
    if not isinstance(override, dict):
        raise ValueError("objective.score_specs must be an object")

    for score, columns in override.items():
        if not isinstance(columns, list) or not all(
            isinstance(col, str) and col for col in columns
        ):
            raise ValueError(
                f"objective.score_specs[{score!r}] must be a non-empty list of strings"
            )
        specs[str(score)] = list(columns)

    return specs


def _first_available_numeric(
    row: pd.Series, candidates: list[str]
) -> tuple[float, str, str]:
    first_invalid: tuple[str, str] | None = None

    for col in candidates:
        if col not in row.index:
            continue

        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]

        if pd.isna(value):
            if first_invalid is None:
                first_invalid = ("invalid_nan", col)
            continue

        if not np.isfinite(float(value)):
            if first_invalid is None:
                first_invalid = ("invalid_infinite", col)
            continue

        return float(value), "ok", col

    if first_invalid is not None:
        status, col = first_invalid
        return float("nan"), status, col

    return float("nan"), "missing_metric", ""


def build_objectives(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    inputs_dir = iter_dir / "inputs"
    observations_path = inputs_dir / "observations.csv"

    obs = read_table(observations_path)

    objective_config = config.objective_config()
    objective_hash = stable_hash(objective_config)
    required_for_fit = set(objective_config.get("required_scores_for_fit", []))
    score_specs = _score_specs_from_config(objective_config)

    rows: list[dict[str, Any]] = []

    for _, row in obs.iterrows():
        out: dict[str, Any] = {
            "observation_id": row["observation_id"],
            "objective_schema_version": objective_config.get("schema_version", 1),
            "objective_config_id": objective_config.get(
                "config_id", "capillary_objectives_v1"
            ),
            "objective_config_hash": objective_hash,
            "objective_status": "ok",
            "objective_failure_reason": "",
        }

        plasma_kind = str(row.get("plasma_kind", "")).strip().lower()

        if plasma_kind != "chan":
            for score in score_specs:
                out[score] = float("nan")
                out[f"{score}_status"] = "not_applicable_baseline_or_vacuum"
                out[f"{score}_direction"] = "maximize"
                out[f"{score}_source_metric"] = ""

            out["objective_status"] = "not_optimized_baseline_or_vacuum"
            out["objective_failure_reason"] = (
                f"plasma_kind={plasma_kind} is not an optimizable channel case"
            )
            out["fit_eligible"] = "false"
            rows.append(out)
            continue

        failures = []

        for score, source_cols in score_specs.items():
            value, status, source_col = _first_available_numeric(row, source_cols)

            out[score] = value
            out[f"{score}_status"] = status
            out[f"{score}_direction"] = "maximize"
            out[f"{score}_source_metric"] = source_col

            if score in required_for_fit and status != "ok":
                failures.append(f"{score}:{status}")

        reduced_ok = str(row.get("reduced_validation_status", "")) in {
            "ok",
            "validation_not_required",
        }

        if failures:
            out["objective_status"] = "missing_metric"
            out["objective_failure_reason"] = "; ".join(failures)
        elif not reduced_ok:
            out["objective_status"] = "reduced_validation_failed"
            out["objective_failure_reason"] = str(row.get("failure_reason", ""))

        out["fit_eligible"] = "true" if out["objective_status"] == "ok" else "false"
        rows.append(out)

    objectives = pd.DataFrame(rows)
    return write_csv(inputs_dir / "objective_table.csv", objectives)
