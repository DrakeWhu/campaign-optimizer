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
        "score_charge_soft50_v1",
        "score_energy_soft50_v1",
        "score_energy_soft50_v2",
        "score_transverse_soft50_v1",
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

DEFAULT_SOFT50_V1_CONFIG = {
    "energy_low_MeV": 10.0,
    "energy_target_MeV": 50.0,
    "energy_ref_MeV": 200.0,
    "energy_relative_spread_ref": 0.25,
    "energy_spread_weight": 0.20,
    "charge_scale_pC": 10.0,
    "charge_ref_pC": 1000.0,
    "reliability_floor": 0.05,
    "effective_count_reference": 200.0,
    "transverse_charge_activation_pC": 25.0,
    "theta_p95_ref_mrad": 15.0,
    "emitn_xy_ref_um_rad": 2.0,
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


def _soft50_config(objective_config: dict[str, Any]) -> dict[str, float]:
    cfg = dict(DEFAULT_SOFT50_V1_CONFIG)
    override = objective_config.get("soft50_v1", {}) or {}
    if not isinstance(override, dict):
        raise ValueError("objective.soft50_v1 must be an object")
    cfg.update(override)
    out = {key: float(value) for key, value in cfg.items()}
    if out["energy_low_MeV"] < 0.0:
        raise ValueError("soft50 energy_low_MeV must be non-negative")
    if out["energy_target_MeV"] <= out["energy_low_MeV"]:
        raise ValueError("soft50 energy_target_MeV must exceed energy_low_MeV")
    if out["energy_ref_MeV"] <= out["energy_target_MeV"]:
        raise ValueError("soft50 energy_ref_MeV must exceed energy_target_MeV")
    if out["energy_relative_spread_ref"] <= 0.0:
        raise ValueError("soft50 energy_relative_spread_ref must be positive")
    if not 0.0 <= out["energy_spread_weight"] <= 1.0:
        raise ValueError("soft50 energy_spread_weight must be within [0, 1]")
    if out["charge_scale_pC"] <= 0.0 or out["charge_ref_pC"] <= 0.0:
        raise ValueError("soft50 charge scales must be positive")
    if not 0.0 <= out["reliability_floor"] <= 1.0:
        raise ValueError("soft50 reliability_floor must be within [0, 1]")
    if out["effective_count_reference"] <= 0.0:
        raise ValueError("soft50 effective_count_reference must be positive")
    for name in [
        "transverse_charge_activation_pC",
        "theta_p95_ref_mrad",
        "emitn_xy_ref_um_rad",
        "score_scale",
    ]:
        if out[name] <= 0.0:
            raise ValueError(f"soft50 {name} must be positive")
    return out


def _smoothstep01(value: float) -> float:
    u = _clamp01(value)
    return u * u * (3.0 - 2.0 * u)


def _soft50_inputs(
    row: pd.Series,
    cfg: dict[str, float],
) -> tuple[float, float, float, str] | tuple[None, None, None, str]:
    charge, charge_source = _first_finite(
        row, ["metric_particle_charge_soft50_pC"]
    )
    n_effective, n_source = _first_finite(
        row, ["metric_particle_n_effective_soft50"]
    )
    if not math.isfinite(charge):
        return None, None, None, "metric_particle_charge_soft50_pC"
    if not math.isfinite(n_effective):
        return None, None, None, "metric_particle_n_effective_soft50"
    reliability = cfg["reliability_floor"] + (
        1.0 - cfg["reliability_floor"]
    ) * (1.0 - math.exp(-max(n_effective, 0.0) / cfg["effective_count_reference"]))
    return max(charge, 0.0), max(n_effective, 0.0), reliability, ";".join(
        [charge_source, n_source]
    )


def _charge_soft50_v1(
    row: pd.Series, cfg: dict[str, float]
) -> tuple[float, str, str]:
    charge, _, reliability, source = _soft50_inputs(row, cfg)
    if charge is None or reliability is None:
        return float("nan"), "missing_metric", source
    denominator = math.log1p(cfg["charge_ref_pC"] / cfg["charge_scale_pC"])
    component = _clamp01(
        math.log1p(charge / cfg["charge_scale_pC"]) / denominator
    )
    return (
        cfg["score_scale"] * component * reliability,
        "ok",
        f"derived:charge_soft50_v1;{source}",
    )


def _soft_energy_component(energy: float, cfg: dict[str, float]) -> float:
    low = cfg["energy_low_MeV"]
    target = cfg["energy_target_MeV"]
    reference = cfg["energy_ref_MeV"]
    if not math.isfinite(energy) or energy <= low:
        return 0.0
    if energy < target:
        return 0.5 * _smoothstep01((energy - low) / (target - low))
    return 0.5 + 0.5 * _clamp01(
        math.log(energy / target) / math.log(reference / target)
    )


def _energy_soft50_v1(
    row: pd.Series, cfg: dict[str, float]
) -> tuple[float, str, str]:
    charge, _, reliability, base_source = _soft50_inputs(row, cfg)
    if charge is None or reliability is None:
        return float("nan"), "missing_metric", base_source
    energy, energy_source = _first_finite(
        row, ["metric_particle_energy_p90_soft50_MeV"]
    )
    if not math.isfinite(energy):
        if charge <= 0.0:
            return 0.0, "ok", f"derived:energy_soft50_v1:no_accepted_particles;{base_source}"
        return (
            float("nan"),
            "missing_metric",
            "metric_particle_energy_p90_soft50_MeV",
        )
    return (
        cfg["score_scale"] * _soft_energy_component(energy, cfg) * reliability,
        "ok",
        f"derived:energy_soft50_v1;{base_source};{energy_source}",
    )


def _energy_soft50_v2(
    row: pd.Series, cfg: dict[str, float]
) -> tuple[float, str, str]:
    """Prioritize energy while softly preferring a narrower energy spectrum."""

    charge, _, reliability, base_source = _soft50_inputs(row, cfg)
    if charge is None or reliability is None:
        return float("nan"), "missing_metric", base_source
    energy, energy_source = _first_finite(
        row, ["metric_particle_energy_p90_soft50_MeV"]
    )
    if not math.isfinite(energy):
        if charge <= 0.0:
            return 0.0, "ok", f"derived:energy_soft50_v2:no_accepted_particles;{base_source}"
        return (
            float("nan"),
            "missing_metric",
            "metric_particle_energy_p90_soft50_MeV",
        )
    if charge <= 0.0:
        return 0.0, "ok", f"derived:energy_soft50_v2:no_accepted_particles;{base_source}"

    spread, spread_source = _first_finite(
        row, ["metric_particle_energy_relative_spread_rms_soft50"]
    )
    if not math.isfinite(spread) or spread < 0.0:
        return (
            float("nan"),
            "missing_metric",
            "metric_particle_energy_relative_spread_rms_soft50",
        )

    spread_quality = _inverse_quality_component(
        spread, cfg["energy_relative_spread_ref"]
    )
    spread_factor = (
        1.0 - cfg["energy_spread_weight"]
        + cfg["energy_spread_weight"] * spread_quality
    )
    return (
        cfg["score_scale"]
        * _soft_energy_component(energy, cfg)
        * reliability
        * spread_factor,
        "ok",
        f"derived:energy_soft50_v2;{base_source};{energy_source};{spread_source}",
    )


def _inverse_quality_component(value: float, reference: float) -> float:
    if not math.isfinite(value) or value < 0.0 or reference <= 0.0:
        return 0.0
    return 1.0 / (1.0 + (value / reference) ** 2)


def _transverse_soft50_v1(
    row: pd.Series, cfg: dict[str, float]
) -> tuple[float, str, str]:
    charge, _, reliability, base_source = _soft50_inputs(row, cfg)
    if charge is None or reliability is None:
        return float("nan"), "missing_metric", base_source
    if charge <= 0.0:
        return 0.0, "ok", f"derived:transverse_soft50_v1:no_accepted_particles;{base_source}"
    theta, theta_source = _first_finite(
        row, ["metric_particle_theta_r_p95_soft50_mrad"]
    )
    emitn, emitn_source = _first_finite(
        row, ["metric_particle_emitn_xy_soft50_um_rad"]
    )
    if not math.isfinite(theta) or not math.isfinite(emitn):
        return (
            float("nan"),
            "missing_metric",
            "metric_particle_theta_r_p95_soft50_mrad/emitn_xy_soft50_um_rad",
        )
    charge_activation = 1.0 - math.exp(
        -charge / cfg["transverse_charge_activation_pC"]
    )
    quality = math.sqrt(
        _inverse_quality_component(theta, cfg["theta_p95_ref_mrad"])
        * _inverse_quality_component(emitn, cfg["emitn_xy_ref_um_rad"])
    )
    return (
        cfg["score_scale"] * reliability * charge_activation * quality,
        "ok",
        f"derived:transverse_soft50_v1;{base_source};{theta_source};{emitn_source}",
    )


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
    if score == "score_beam_longitudinal_v2":
        return _beam_longitudinal_v2(row, _beam_v2_config(objective_config))
    if score == "score_beam_transverse_v2":
        return _beam_transverse_v2(row, _beam_v2_config(objective_config))
    cfg = _soft50_config(objective_config)
    if score == "score_charge_soft50_v1":
        return _charge_soft50_v1(row, cfg)
    if score == "score_energy_soft50_v1":
        return _energy_soft50_v1(row, cfg)
    if score == "score_energy_soft50_v2":
        return _energy_soft50_v2(row, cfg)
    if score == "score_transverse_soft50_v1":
        return _transverse_soft50_v1(row, cfg)
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

        # 2. Derived scores combine several persisted physical metrics. This
        # includes both the legacy hard-gated v2 scores and the continuous
        # soft50 charge/energy/transverse scores.
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
    if objectives.empty:
        columns = [
            "observation_id",
            "objective_schema_version",
            "objective_config_id",
            "objective_config_hash",
            "objective_status",
            "objective_failure_reason",
        ]
        for score in all_scores:
            columns.extend(
                [
                    score,
                    f"{score}_status",
                    f"{score}_direction",
                    f"{score}_source_metric",
                ]
            )
        columns.append("fit_eligible")
        objectives = pd.DataFrame(columns=columns)
    return write_csv(inputs_dir / "objective_table.csv", objectives)
