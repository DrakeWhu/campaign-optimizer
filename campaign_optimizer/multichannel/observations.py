from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from campaign_optimizer.config import OptimizerConfig, resolve_path
from campaign_optimizer.io import (
    read_first_row_csv,
    read_json_optional,
    read_table,
    write_csv,
    write_json,
)

from .parameters import parameter_names, parameters_from_case_row


BEAM_METRIC_COLUMNS = (
    "n_macroparticles_selected",
    "charge_selected_pC",
    "energy_p95_MeV",
    "theta_r_p95_mrad",
    "emitn_xy_um_rad",
    "forward_fraction",
    "pz_mean_MeV_c",
    "soft100_status",
    "n_macroparticles_soft100",
    "charge_soft100_pC",
    "n_effective_soft100",
    "reliability_soft100",
    "energy_p95_soft100_MeV",
    "energy_relative_spread_rms_soft100",
    "theta_r_p95_soft100_mrad",
    "emitn_xy_soft100_um_rad",
    "charge_Ege100MeV_pC",
    "halo_fraction_soft100",
)


def _safe_case_dir(campaign_root: Path, case_name: str) -> Path:
    path = campaign_root / case_name
    if not path.resolve().is_relative_to(campaign_root.resolve()):
        raise ValueError(f"unsafe case path outside campaign root: {case_name}")
    return path


def _validation_section_ok(
    validation: dict[str, Any] | None, section: str, *, required: bool
) -> bool:
    if not required:
        return True
    if not isinstance(validation, dict):
        return False
    entries = validation.get(section)
    if not isinstance(entries, dict) or not entries:
        return False
    values = [item for item in entries.values() if isinstance(item, dict)]
    return bool(values) and all(item.get("ok") is True for item in values)


def _empty_observations(parameter_space: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "observation_id",
        "source_campaign_name",
        "source_campaign_root",
        "source_case_id",
        "source_case_name",
        "case_state",
        "simulation_status",
        "raw_validation_status",
        "reduced_validation_status",
        "particle_summary_status",
        "failure_reason",
        "sample_source",
        "sobol_index",
        *parameter_names(parameter_space),
        *BEAM_METRIC_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)


def build_observations(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    inputs_dir = iter_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    parameter_space = config.parameter_space()
    require_validation = bool(
        config.validation_config().get("require_case_validation", True)
    )
    sources = config.source_campaigns_for_iteration(iteration)

    write_json(
        inputs_dir / "source_campaigns_effective.json",
        {
            "schema_version": 1,
            "problem_kind": "multichannel",
            "iteration": int(iteration),
            "source_campaigns": sources,
        },
    )

    rows: list[dict[str, Any]] = []
    for source in sources:
        campaign_root = resolve_path(config.base_dir, source["campaign_root"])
        cases_path = resolve_path(campaign_root, source.get("cases_tsv", "cases.tsv"))
        campaign_name = str(source.get("campaign_name", campaign_root.name))
        reduced_outputs = dict(source.get("reduced_outputs", {}) or {})
        particle_summary_rel = str(
            reduced_outputs.get("particle_summary", "post/particle_summary.csv")
        )

        for _, case in read_table(cases_path, sep="\t").iterrows():
            case_row = dict(case)
            case_id = str(case_row.get("CASE_ID", ""))
            case_name = str(case_row.get("CASE_NAME", ""))
            case_dir = _safe_case_dir(campaign_root, case_name)

            state = read_json_optional(case_dir / "state.json") or {}
            validation = read_json_optional(case_dir / "validation.json")
            raw_ok = _validation_section_ok(
                validation, "raw", required=require_validation
            )
            reduced_ok = _validation_section_ok(
                validation, "reduced", required=require_validation
            )

            metrics, metrics_status, metrics_reason = read_first_row_csv(
                case_dir / particle_summary_rel
            )
            case_state = str(state.get("state", "missing_state"))
            finished_state = case_state in {
                "Reduced_validated",
                "Raw_delete_eligible",
                "Raw_deleted",
            }
            simulation_status = (
                "finished"
                if finished_state and raw_ok and reduced_ok and metrics_status == "ok"
                else "incomplete"
            )

            sobol_raw = case_row.get("OPT_SOBOL_INDEX", "")
            try:
                sobol_index: int | str = int(float(sobol_raw))
            except (TypeError, ValueError):
                sobol_index = ""

            row: dict[str, Any] = {
                "observation_id": f"{campaign_name}:{case_name}",
                "source_campaign_name": campaign_name,
                "source_campaign_root": str(campaign_root),
                "source_case_id": case_id,
                "source_case_name": case_name,
                "case_state": case_state,
                "simulation_status": simulation_status,
                "raw_validation_status": "ok" if raw_ok else "not_validated",
                "reduced_validation_status": (
                    "ok" if reduced_ok else "not_validated"
                ),
                "particle_summary_status": metrics_status,
                "failure_reason": "" if metrics_status == "ok" else metrics_reason,
                "sample_source": str(case_row.get("OPT_SAMPLE_SOURCE", "")),
                "sobol_index": sobol_index,
                **parameters_from_case_row(case_row, parameter_space),
            }
            row.update(metrics)
            rows.append(row)

    observations = (
        pd.DataFrame(rows) if rows else _empty_observations(parameter_space)
    )
    first = [
        "observation_id",
        "source_campaign_name",
        "source_campaign_root",
        "source_case_id",
        "source_case_name",
        "case_state",
        "simulation_status",
        "raw_validation_status",
        "reduced_validation_status",
        "particle_summary_status",
        "failure_reason",
        "sample_source",
        "sobol_index",
        *parameter_names(parameter_space),
    ]
    ordered = [column for column in first if column in observations.columns]
    ordered.extend(column for column in observations.columns if column not in ordered)
    return write_csv(inputs_dir / "observations.csv", observations[ordered])
