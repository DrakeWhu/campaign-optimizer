from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import math

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
DERIVED_SCORE_NAMES = frozenset(
    {
        "score_beam_longitudinal_v2",
        "score_beam_transverse_v2",
    }
)

DEFAULT_BEAM_GATED_V2_CONFIG = {
    "min_beamlike_score": 1.0e-12,
    "min_beam_yield_score": 1.0e-12,
    "min_charge_hot_pC": 100.0,
    "min_n_macroparticles_hot": 200.0,
    "min_n_macroparticles_transverse": 200.0,
    "min_E95_hot_MeV": 50.0,
    "charge_ref_pC": 1200.0,
    "n_hot_ref": 1000.0,
    "energy_ref_MeV": 220.0,
    "mono_min": 0.30,
    "mono_ref": 0.65,
    "transverse_ref": 0.02,
    "score_scale": 100.0,
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
        if str(score) in DERIVED_SCORE_NAMES:
            raise ValueError(
                f"objective.score_specs[{score!r}] cannot override a derived score"
            )
        specs[str(score)] = list(columns)

    return specs


def _derived_scores_from_config(objective_config: dict[str, Any]) -> list[str]:
    configured = objective_config.get("derived_scores", []) or []
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, list) or not all(
        isinstance(item, str) for item in configured
    ):
        raise ValueError("objective.derived_scores must be a list of strings")

    required = objective_config.get("required_scores_for_fit", []) or []
    if isinstance(required, str):
        required = [required]

    names: list[str] = []
    for name in [*configured, *required]:
        score = str(name)
        if score not in DERIVED_SCORE_NAMES:
            continue
        if score not in names:
            names.append(score)
    return names


def _first_finite(row: pd.Series, candidates: Iterable[str]) -> tuple[float, str]:
    value, status, source = _first_available_numeric(row, list(candidates))
    if status == "ok":
        return value, source
    return float("nan"), ""


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _linear_component(value: float, minimum: float, reference: float) -> float:
    if not math.isfinite(value) or value <= minimum:
        return 0.0
    if reference <= minimum:
        return 1.0
    return _clamp01((value - minimum) / (reference - minimum))


def _log_component(value: float, minimum: float, reference: float) -> float:
    if not math.isfinite(value) or value <= minimum:
        return 0.0
    if reference <= minimum:
        return 1.0
    return _clamp01(math.log10(value / minimum) / math.log10(reference / minimum))


def _beam_v2_config(objective_config: dict[str, Any]) -> dict[str, float]:
    cfg = dict(DEFAULT_BEAM_GATED_V2_CONFIG)
    shared = objective_config.get("beam_gated_v2", {}) or {}
    if not isinstance(shared, dict):
        raise ValueError("objective.beam_gated_v2 must be an object")
    cfg.update(shared)
    return {key: float(value) for key, value in cfg.items()}


def _beam_gate_reason(row: pd.Series, cfg: dict[str, float]) -> str | None:
    n_hot, _ = _first_finite(
        row,
        [
            "metric_particle_n_macroparticles_hot",
            "metric_comparison_beam_n_macroparticles_hot_channel",
        ],
    )
    charge, _ = _first_finite(
        row,
        [
            "metric_particle_charge_hot_pC",
            "metric_comparison_beam_charge_hot_pC_channel",
        ],
    )
    beamlike, beamlike_source = _first_finite(
        row,
        [
            "metric_particle_beamlike_score",
            "metric_comparison_beam_beamlike_score_channel",
        ],
    )
    beam_yield, beam_yield_source = _first_finite(
        row,
        [
            "metric_particle_beam_yield_score",
            "metric_comparison_beam_beam_yield_score_channel",
        ],
    )

    if not math.isfinite(n_hot) or n_hot < cfg["min_n_macroparticles_hot"]:
        return "low_n_macroparticles_hot"
    if not math.isfinite(charge) or charge < cfg["min_charge_hot_pC"]:
        return "low_charge_hot_pC"
    if beamlike_source and beamlike <= cfg["min_beamlike_score"]:
        return "zero_beamlike_score"
    if beam_yield_source and beam_yield <= cfg["min_beam_yield_score"]:
        return "zero_beam_yield_score"
    return None


def _beam_longitudinal_v2(
    row: pd.Series, cfg: dict[str, float]
) -> tuple[float, str, str]:
    gate = _beam_gate_reason(row, cfg)
    if gate is not None:
        return 0.0, "ok", f"derived:hard_gate:{gate}"

    e95, e95_source = _first_finite(
        row,
        [
            "metric_particle_E95_hot_MeV",
            "metric_comparison_beam_E95_hot_MeV_channel",
        ],
    )
    emax, emax_source = _first_finite(
        row,
        [
            "metric_particle_Emax_hot_MeV",
            "metric_particle_Emax_MeV",
            "metric_comparison_beam_Emax_hot_MeV_channel",
        ],
    )

    if not math.isfinite(e95):
        return float("nan"), "missing_metric", "metric_particle_E95_hot_MeV"

    if not math.isfinite(emax):
        emax = e95
        emax_source = e95_source

    charge, charge_source = _first_finite(
        row,
        [
            "metric_particle_charge_hot_pC",
            "metric_comparison_beam_charge_hot_pC_channel",
        ],
    )
    n_hot, n_source = _first_finite(
        row,
        [
            "metric_particle_n_macroparticles_hot",
            "metric_comparison_beam_n_macroparticles_hot_channel",
        ],
    )

    mono = e95 / emax if emax > 0.0 else float("nan")

    charge_component = _log_component(
        charge,
        cfg["min_charge_hot_pC"],
        cfg["charge_ref_pC"],
    )
    statistics_component = _log_component(
        n_hot,
        cfg["min_n_macroparticles_hot"],
        cfg["n_hot_ref"],
    )
    energy_component = _linear_component(
        e95,
        cfg["min_E95_hot_MeV"],
        cfg["energy_ref_MeV"],
    )
    mono_component = _linear_component(
        mono,
        cfg["mono_min"],
        cfg["mono_ref"],
    )

    score = (
        cfg["score_scale"]
        * charge_component
        * statistics_component
        * energy_component
        * mono_component
    )

    source = ";".join(
        item
        for item in [
            "derived:beam_longitudinal_v2",
            e95_source,
            emax_source,
            charge_source,
            n_source,
        ]
        if item
    )
    return float(score), "ok", source


def _beam_transverse_v2(
    row: pd.Series, cfg: dict[str, float]
) -> tuple[float, str, str]:
    gate = _beam_gate_reason(row, cfg)
    if gate is not None:
        return 0.0, "ok", f"derived:hard_gate:{gate}"

    n_transverse, n_source = _first_finite(
        row,
        [
            "metric_particle_n_macroparticles_transverse",
            "metric_particle_n_macroparticles_hot",
            "metric_comparison_beam_n_macroparticles_hot_channel",
        ],
    )
    if (
        not math.isfinite(n_transverse)
        or n_transverse < cfg["min_n_macroparticles_transverse"]
    ):
        return 0.0, "ok", "derived:hard_gate:low_n_macroparticles_transverse"

    transverse, transverse_source = _first_finite(
        row,
        [
            "metric_particle_beam_transverse_quality_score_channel",
            "metric_particle_beam_transverse_quality_score",
        ],
    )
    if not transverse_source:
        return (
            float("nan"),
            "missing_metric",
            "metric_particle_beam_transverse_quality_score",
        )

    transverse_component = _linear_component(transverse, 0.0, cfg["transverse_ref"])
    statistics_component = _log_component(
        n_transverse,
        cfg["min_n_macroparticles_transverse"],
        cfg["n_hot_ref"],
    )

    score = cfg["score_scale"] * transverse_component * statistics_component
    return (
        float(score),
        "ok",
        f"derived:beam_transverse_v2;{transverse_source};{n_source}",
    )


def _compute_derived_score(
    row: pd.Series,
    score: str,
    objective_config: dict[str, Any],
) -> tuple[float, str, str]:
    cfg = _beam_v2_config(objective_config)
    if score == "score_beam_longitudinal_v2":
        return _beam_longitudinal_v2(row, cfg)
    if score == "score_beam_transverse_v2":
        return _beam_transverse_v2(row, cfg)
    raise ValueError(f"Unknown derived score: {score}")


def _validate_required_scores(
    *,
    required_for_fit: set[str],
    score_specs: dict[str, list[str]],
    derived_scores: list[str],
) -> None:
    available = set(score_specs) | set(derived_scores)
    missing = sorted(required_for_fit - available)
    if missing:
        raise ValueError(
            "objective.required_scores_for_fit contains unknown score(s): "
            + ", ".join(missing)
        )


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

    derived_scores = _derived_scores_from_config(objective_config)
    _validate_required_scores(
        required_for_fit=required_for_fit,
        score_specs=score_specs,
        derived_scores=derived_scores,
    )
    all_scores = [*score_specs.keys(), *derived_scores]

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
            for score in all_scores:
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

        # 1. Scores directos: columnas existentes en observations.csv.
        for score, source_cols in score_specs.items():
            value, status, source_col = _first_available_numeric(row, source_cols)

            out[score] = value
            out[f"{score}_status"] = status
            out[f"{score}_direction"] = "maximize"
            out[f"{score}_source_metric"] = source_col

            if score in required_for_fit and status != "ok":
                failures.append(f"{score}:{status}")

        # 2. Scores derivados: funciones físicas compuestas/gated.
        #
        # Aquí entran:
        #   - score_beam_longitudinal_v2
        #   - score_beam_transverse_v2
        #
        # Estos NO salen de una sola columna. Se calculan combinando carga,
        # número de partículas, beamlike/yield, energía y calidad transversal.
        for score in derived_scores:
            value, status, source_col = _compute_derived_score(
                row,
                score,
                objective_config,
            )

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
