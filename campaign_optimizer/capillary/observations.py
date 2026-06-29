from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig, resolve_path
from campaign_optimizer.io import (
    read_first_row_csv,
    read_json_optional,
    read_table,
    write_csv,
)

from .parameters import PARAMETER_COLUMNS, canonical_parameters_from_case_row


def _case_dir(campaign_root: Path, case_name: str) -> Path:
    path = campaign_root / case_name

    if not path.resolve().is_relative_to(campaign_root.resolve()):
        raise ValueError(f"unsafe case path outside campaign root: {case_name}")

    return path


def _section_status(
    validation: dict[str, Any] | None,
    section: str,
    *,
    require: bool,
) -> tuple[str, str]:
    if validation is None:
        if require:
            return "missing_validation_evidence", "validation.json not found"
        return "validation_not_required", ""

    value = validation.get(section)

    if not isinstance(value, dict):
        return (
            f"{section}_validation_failed",
            f"validation.json {section} is not an object",
        )

    if not value:
        if require:
            return (
                f"missing_{section}_validation",
                f"validation.json {section} is empty",
            )
        return "validation_not_required", ""

    oks = [entry.get("ok") for entry in value.values() if isinstance(entry, dict)]

    if oks and all(v is True for v in oks):
        return "ok", ""

    return (
        f"{section}_validation_failed",
        f"not all {section} validation entries are ok",
    )


def _load_global_tables(
    campaign_root: Path,
    source: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cap = source.get("capillary", {}) or {}
    tables = cap.get("global_tables", {}) or {}

    joint = None
    acceptance = None

    if tables.get("joint_table"):
        joint = read_table(resolve_path(campaign_root, tables["joint_table"]))

    if tables.get("acceptance_rankings"):
        acceptance = read_table(
            resolve_path(campaign_root, tables["acceptance_rankings"])
        )

    return joint, acceptance


def _metric_from_first_row(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"metric_{prefix}_{key}": value for key, value in row.items()}


def _add_global_metrics(
    base: pd.DataFrame,
    joint: pd.DataFrame | None,
    acceptance: pd.DataFrame | None,
) -> pd.DataFrame:
    out = base.copy()

    if joint is not None and not joint.empty:
        keep = [
            c
            for c in joint.columns
            if c
            in {
                "channel_case_id",
                "uniform_case_id",
                "vacuum_case_id",
                "guiding_final_score",
                "beam_beamlike_gain_score",
                "beam_beam_transverse_quality_score_channel",
                "beam_transverse_quality_score_channel",
                "joint_status",
                "bucket",
            }
        ]
        j = joint[keep].copy()
        j = j.rename(
            columns={
                "guiding_final_score": "metric_guiding_final_score",
                "beam_beamlike_gain_score": "metric_comparison_beam_beamlike_gain_score",
                "beam_beam_transverse_quality_score_channel": "metric_particle_beam_transverse_quality_score_channel",
                "beam_transverse_quality_score_channel": "metric_particle_beam_transverse_quality_score_channel",
            }
        )
        out = out.merge(j, on="channel_case_id", how="left")

    if acceptance is not None and not acceptance.empty:
        keep = [
            c
            for c in acceptance.columns
            if c
            in {
                "channel_case_id",
                "Q_Ege100MeV_theta10mrad_pC_channel",
                "Q_Ege100MeV_theta5mrad_pC_channel",
                "E95_hot_MeV_channel",
                "charge_hot_pC_channel",
                "theta_rms_mrad_channel",
                "theta_r_p95_mrad_channel",
                "beam_transverse_quality_score_channel",
            }
        ]
        a = acceptance[keep].copy()
        a = a.rename(
            columns={
                "Q_Ege100MeV_theta10mrad_pC_channel": "metric_acceptance_Q_Ege100MeV_theta10mrad_pC",
                "Q_Ege100MeV_theta5mrad_pC_channel": "metric_acceptance_Q_Ege100MeV_theta5mrad_pC",
                "E95_hot_MeV_channel": "metric_particle_E95_hot_MeV",
                "charge_hot_pC_channel": "metric_particle_charge_hot_pC",
                "theta_rms_mrad_channel": "metric_particle_theta_rms_mrad",
                "theta_r_p95_mrad_channel": "metric_particle_theta_r_p95_mrad",
                "beam_transverse_quality_score_channel": "metric_particle_beam_transverse_quality_score_channel",
            }
        )

        out = out.merge(a, on="channel_case_id", how="left", suffixes=("", "_acc"))

        duplicate = "metric_particle_beam_transverse_quality_score_channel_acc"
        base_col = "metric_particle_beam_transverse_quality_score_channel"

        if duplicate in out.columns:
            out[base_col] = out.get(base_col, np.nan).combine_first(out[duplicate])
            out = out.drop(columns=[duplicate])

    return out


def build_observations(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    inputs_dir = iter_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    parameter_space = config.parameter_space()
    allowed_lasers = list(parameter_space["laser_cases"])
    require_validation = bool(
        config.validation_config().get("require_case_validation", False)
    )
    allow_fallback = bool(config.data.get("allow_case_name_parameter_fallback", True))

    all_rows: list[dict[str, Any]] = []

    for source in config.source_campaigns():
        campaign_root = resolve_path(config.base_dir, source["campaign_root"])
        cases_path = resolve_path(campaign_root, source.get("cases_tsv", "cases.tsv"))
        campaign_name = source.get("campaign_name", campaign_root.name)

        cases = read_table(cases_path)
        joint, acceptance = _load_global_tables(campaign_root, source)

        rows = []

        for _, case in cases.iterrows():
            case_dict = dict(case)
            case_id = str(case_dict.get("CASE_ID", case_dict.get("case_id", "")))
            case_name = str(case_dict.get("CASE_NAME", case_dict.get("case_name", "")))

            cdir = _case_dir(campaign_root, case_name)

            state = read_json_optional(cdir / "state.json") or {}
            validation = read_json_optional(cdir / "validation.json")

            raw_status, raw_reason = _section_status(
                validation, "raw", require=require_validation
            )
            reduced_status, reduced_reason = _section_status(
                validation, "reduced", require=require_validation
            )

            params = canonical_parameters_from_case_row(
                case_dict,
                allowed_laser_cases=allowed_lasers,
                allow_case_name_fallback=allow_fallback,
            )

            row: dict[str, Any] = {
                "observation_id": f"{campaign_name}:{case_name}",
                "source_campaign_name": campaign_name,
                "source_campaign_root": str(campaign_root),
                "source_case_id": case_id,
                "source_case_name": case_name,
                "case_state": state.get("state", "missing_state"),
                "simulation_status": (
                    "ok"
                    if state.get("state")
                    in {"Reduced_validated", "Raw_delete_eligible", "Raw_deleted"}
                    else "unknown_or_incomplete"
                ),
                "raw_validation_status": raw_status,
                "analysis_status": "ok" if reduced_status == "ok" else reduced_status,
                "reduced_validation_status": reduced_status,
                "failure_kind": ""
                if reduced_status in {"ok", "validation_not_required"}
                else "reduced_validation",
                "failure_reason": reduced_reason or raw_reason,
                "channel_case_id": case_name
                if params.get("plasma_kind") == "chan"
                else "",
                "channel_case_name": case_name
                if params.get("plasma_kind") == "chan"
                else "",
                "uniform_case_id": "",
                "uniform_case_name": "",
                "vacuum_case_id": "",
                "vacuum_case_name": "",
                "baseline_match_status": "not_applicable",
                "baseline_match_reason": "",
                "guiding_metrics_status": "not_checked",
                "particle_summary_status": "not_applicable"
                if params.get("plasma_kind") == "vac"
                else "not_checked",
                "acceptance_curves_status": "not_applicable"
                if params.get("plasma_kind") == "vac"
                else "not_checked",
                "comparison_metrics_status": "not_checked",
                **params,
            }

            reduced_outputs = source.get("reduced_outputs", {}) or {}

            if reduced_outputs.get("guiding_metrics"):
                metrics, status, reason = read_first_row_csv(
                    cdir / reduced_outputs["guiding_metrics"]
                )
                row["guiding_metrics_status"] = status
                if reason and not row["failure_reason"]:
                    row["failure_reason"] = reason
                row.update(_metric_from_first_row("guiding", metrics))

            if (
                reduced_outputs.get("particle_summary")
                and params.get("plasma_kind") != "vac"
            ):
                metrics, status, reason = read_first_row_csv(
                    cdir / reduced_outputs["particle_summary"]
                )
                row["particle_summary_status"] = status
                if reason and not row["failure_reason"]:
                    row["failure_reason"] = reason
                row.update(_metric_from_first_row("particle", metrics))

            if (
                reduced_outputs.get("acceptance_curves")
                and params.get("plasma_kind") != "vac"
            ):
                path = cdir / reduced_outputs["acceptance_curves"]
                try:
                    df_acc = read_table(path)
                    row["acceptance_curves_status"] = (
                        "ok" if not df_acc.empty else "missing_metric"
                    )
                except FileNotFoundError:
                    row["acceptance_curves_status"] = "missing_reduced_output"

            rows.append(row)

        base = pd.DataFrame(rows)
        base = _add_global_metrics(base, joint, acceptance)

        if "uniform_case_id" in base.columns:
            base["baseline_match_status"] = np.where(
                base["uniform_case_id"].astype(str).str.len() > 0,
                "ok",
                base["baseline_match_status"],
            )

        all_rows.extend(base.to_dict(orient="records"))

    observations = pd.DataFrame(all_rows)

    first_cols = [
        "observation_id",
        "source_campaign_name",
        "source_campaign_root",
        "source_case_id",
        "source_case_name",
        "case_state",
        "simulation_status",
        "raw_validation_status",
        "analysis_status",
        "reduced_validation_status",
        "failure_kind",
        "failure_reason",
        *PARAMETER_COLUMNS,
    ]

    ordered = [c for c in first_cols if c in observations.columns]
    ordered += [c for c in observations.columns if c not in ordered]

    observations = observations[ordered]
    return write_csv(inputs_dir / "observations.csv", observations)
