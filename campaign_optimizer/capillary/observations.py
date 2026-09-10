from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_optimizer.config import (
    DEFAULT_REDUCED_OUTPUTS,
    OptimizerConfig,
    resolve_path,
)
from campaign_optimizer.io import (
    read_first_row_csv,
    read_json_optional,
    read_table,
    write_csv,
    write_json,
)

from .parameters import PARAMETER_COLUMNS, canonical_parameters_from_case_row


NITROGEN_PARTICLE_OBSERVATION_CONTRACT = "clpu_n2_plateau_all_electrons_v1"
NITROGEN_SPECIES_SCOPES = [
    "all_electrons",
    "preionized_background_electrons",
    "nitrogen_ionized_electrons",
]


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


def _metric_from_reduced_output_row(
    prefix: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Return metric columns from a reduced-output row.

    Standard analysis sidecars may already export optimizer-facing columns with
    a metric_* prefix. In that case, keep the names as-is instead of adding a
    second prefix. Legacy/plain rows still receive metric_<prefix>_.
    """

    out: dict[str, Any] = {}
    for key, value in row.items():
        name = str(key)
        if name.startswith("metric_"):
            out[name] = value
        else:
            out[f"metric_{prefix}_{name}"] = value
    return out


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _read_first_row_csv(path: Path) -> tuple[dict[str, Any], str, str]:
    metrics, status, reason = read_first_row_csv(path)
    return metrics, status, reason


def _read_guiding_metrics_row(
    path: Path,
    *,
    particle_metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str]:
    if not path.is_file():
        return {}, "missing_reduced_output", f"missing file: {path}"

    try:
        df = read_table(path)
    except Exception as exc:
        return {}, "analysis_failed", f"csv read failed: {exc}"

    if df.empty:
        return {}, "missing_metric", f"empty csv: {path}"

    target_iteration = None
    if particle_metrics:
        target_iteration = particle_metrics.get("target_guiding_iteration")

    if target_iteration is not None and "iteration" in df.columns:
        target = pd.to_numeric(pd.Series([target_iteration]), errors="coerce").iloc[0]
        iterations = pd.to_numeric(df["iteration"], errors="coerce")

        if not pd.isna(target):
            matches = df.loc[iterations == target]
            if not matches.empty:
                out = dict(matches.iloc[0])
                out["row_policy"] = "target_guiding_iteration"
                out["row_policy_target_iteration"] = target_iteration
                return out, "ok", ""

            return (
                {},
                "missing_metric",
                f"guiding_metrics.csv has no row for target_guiding_iteration={target_iteration}",
            )

    if len(df) == 1:
        out = dict(df.iloc[0])
        out["row_policy"] = "single_row"
        return out, "ok", ""

    out = dict(df.iloc[-1])
    out["row_policy"] = "last_row_fallback"
    return out, "ok", ""


def _read_soft50_curve_row(
    path: Path,
    *,
    energy_low_mev: float,
) -> tuple[dict[str, Any], str, str]:
    """Select the persisted all-electron row for the configured soft threshold."""

    if not path.is_file():
        return {}, "missing_reduced_output", f"missing file: {path}"
    try:
        frame = read_table(path)
    except Exception as exc:
        return {}, "analysis_failed", f"csv read failed: {exc}"
    required = {"soft50_energy_low_MeV", "species_scope"}
    if frame.empty or not required.issubset(frame.columns):
        return {}, "missing_metric", f"invalid soft50 curve: {path}"

    energy = pd.to_numeric(frame["soft50_energy_low_MeV"], errors="coerce")
    scopes = frame["species_scope"].astype(str).str.lower()
    selected = frame.loc[
        np.isclose(energy, float(energy_low_mev), rtol=0.0, atol=1.0e-9)
        & scopes.eq("all_electrons")
    ]
    if selected.empty:
        return (
            {},
            "missing_metric",
            f"no all_electrons soft50 row for E_low={energy_low_mev:g} MeV",
        )
    return dict(selected.iloc[-1]), "ok", ""


def _read_nitrogen_particle_summary(
    path: Path,
) -> tuple[dict[str, Any], str, str]:
    """Read the canonical plateau/all-electron summary for the N2 profile."""

    if not path.is_file():
        return {}, "missing_reduced_output", f"missing file: {path}"
    try:
        frame = read_table(path)
    except Exception as exc:
        return {}, "analysis_failed", f"csv read failed: {exc}"
    if frame.empty or "species_scope" not in frame.columns:
        return {}, "particle_observation_contract_failed", f"invalid N2 summary: {path}"

    scopes = frame["species_scope"].astype(str).str.strip().str.lower().tolist()
    if scopes != NITROGEN_SPECIES_SCOPES:
        return (
            {},
            "particle_observation_contract_failed",
            "N2 particle summary scopes must appear exactly once and in order "
            f"{NITROGEN_SPECIES_SCOPES}; observed={scopes}: {path}",
        )

    required_columns = {
        "selection_mode",
        "particle_exit_selection_policy",
        "selected_particle_iteration",
        "soft50_schema_version",
        "soft50_energy_low_MeV",
        "soft50_energy_target_MeV",
        "charge_soft50_pC",
        "charge_Ege50MeV_pC",
        "forward_only",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        return (
            {},
            "particle_observation_contract_failed",
            f"N2 particle summary lacks required columns {missing}: {path}",
        )

    selection_modes = frame["selection_mode"].astype(str).str.strip().str.lower()
    if not selection_modes.eq("exit").all():
        return {}, "particle_observation_contract_failed", f"N2 summary selection_mode must be exit: {path}"
    policies = frame["particle_exit_selection_policy"].astype(str).str.strip()
    if not policies.eq("exact_resolved_v1").all():
        return (
            {},
            "particle_observation_contract_failed",
            f"N2 summary must use exact_resolved_v1 particle exit selection: {path}",
        )

    iterations = pd.to_numeric(frame["selected_particle_iteration"], errors="coerce")
    if iterations.isna().any() or iterations.nunique(dropna=False) != 1:
        return (
            {},
            "particle_observation_contract_failed",
            f"N2 summary scopes do not share one selected particle iteration: {path}",
        )

    return dict(frame.iloc[0]), "ok", ""


def _read_nitrogen_soft50_curve_row(
    path: Path,
    *,
    energy_low_mev: float,
) -> tuple[dict[str, Any], str, str]:
    """Read the unique all-electron N2 curve row used only for concordance."""

    if not path.is_file():
        return {}, "missing_reduced_output", f"missing file: {path}"
    try:
        frame = read_table(path)
    except Exception as exc:
        return {}, "analysis_failed", f"csv read failed: {exc}"

    required = {
        "species_scope",
        "selection_mode",
        "selected_particle_iteration",
        "soft50_schema_version",
        "soft50_energy_low_MeV",
        "soft50_energy_target_MeV",
        "charge_soft50_pC",
        "charge_Ege50MeV_pC",
        "forward_only",
    }
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        return (
            {},
            "particle_observation_contract_failed",
            f"invalid N2 Soft50 curve; missing columns={missing}: {path}",
        )

    scope_order: list[str] = []
    for value in frame["species_scope"].astype(str).str.strip().str.lower():
        if value not in scope_order:
            scope_order.append(value)
    if scope_order != NITROGEN_SPECIES_SCOPES:
        return (
            {},
            "particle_observation_contract_failed",
            "N2 Soft50 curve scopes must preserve aggregate/background/ionized order; "
            f"observed={scope_order}: {path}",
        )

    energy = pd.to_numeric(frame["soft50_energy_low_MeV"], errors="coerce")
    scopes = frame["species_scope"].astype(str).str.strip().str.lower()
    selected = frame.loc[
        np.isclose(energy, float(energy_low_mev), rtol=0.0, atol=1.0e-9)
        & scopes.eq("all_electrons")
    ]
    if len(selected) != 1:
        return (
            {},
            "particle_observation_contract_failed",
            "N2 Soft50 curve must contain exactly one all_electrons row for "
            f"E_low={energy_low_mev:g} MeV; found={len(selected)}: {path}",
        )
    return dict(selected.iloc[0]), "ok", ""


def _numeric_concordance(
    left: Any,
    right: Any,
    *,
    name: str,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
) -> str:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return f"{name} is not numeric: summary={left!r}, curve={right!r}"
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return f"{name} must be finite: summary={left_value}, curve={right_value}"
    if not np.isclose(left_value, right_value, rtol=rtol, atol=atol):
        return f"{name} differs: summary={left_value:.17g}, curve={right_value:.17g}"
    return ""


def _validate_nitrogen_soft50_concordance(
    summary: dict[str, Any],
    curve: dict[str, Any],
    *,
    energy_low_mev: float,
    energy_target_mev: float,
) -> tuple[str, str]:
    """Require the curve to agree with, not overwrite, the canonical summary."""

    text_fields = {
        "species_scope": "all_electrons",
        "selection_mode": "exit",
        "soft50_schema_version": "soft50_v2",
    }
    for name, expected in text_fields.items():
        summary_value = str(summary.get(name, "")).strip().lower()
        curve_value = str(curve.get(name, "")).strip().lower()
        if summary_value != expected.lower() or curve_value != expected.lower():
            return (
                "particle_observation_contract_failed",
                f"N2 Soft50 {name} mismatch: expected={expected!r}, "
                f"summary={summary.get(name)!r}, curve={curve.get(name)!r}",
            )

    expected_numeric = {
        "soft50_energy_low_MeV": float(energy_low_mev),
        "soft50_energy_target_MeV": float(energy_target_mev),
    }
    for name, expected in expected_numeric.items():
        for origin, payload in (("summary", summary), ("curve", curve)):
            reason = _numeric_concordance(payload.get(name), expected, name=f"{origin}.{name}")
            if reason:
                return "particle_observation_contract_failed", f"N2 Soft50 {reason}"

    for name in (
        "selected_particle_iteration",
        "charge_soft50_pC",
        "charge_Ege50MeV_pC",
    ):
        reason = _numeric_concordance(summary.get(name), curve.get(name), name=name)
        if reason:
            return "particle_observation_contract_failed", f"N2 Soft50 {reason}"

    for origin, payload in (("summary", summary), ("curve", curve)):
        forward = str(payload.get("forward_only", "")).strip().lower()
        if forward not in {"true", "1"}:
            return (
                "particle_observation_contract_failed",
                f"N2 Soft50 forward_only must be true in {origin}; got={payload.get('forward_only')!r}",
            )

    return "ok", ""


def _mark_particle_observation_contract_failure(
    row: dict[str, Any],
    *,
    reason: str,
) -> None:
    row["analysis_status"] = "particle_observation_contract_failed"
    row["reduced_validation_status"] = "particle_observation_contract_failed"
    row["failure_kind"] = "particle_observation_contract"
    row["failure_reason"] = reason


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

    effective_sources = config.source_campaigns_for_iteration(iteration)
    write_json(
        inputs_dir / "source_campaigns_effective.json",
        {
            "schema_version": 1,
            "iteration": iteration,
            "source_campaigns": effective_sources,
        },
    )

    for source in effective_sources:
        campaign_root = resolve_path(config.base_dir, source["campaign_root"])
        cases_path = resolve_path(campaign_root, source.get("cases_tsv", "cases.tsv"))
        campaign_name = source.get("campaign_name", campaign_root.name)
        particle_observation_contract = str(
            source.get("particle_observation_contract", "") or ""
        ).strip()
        if particle_observation_contract and (
            particle_observation_contract != NITROGEN_PARTICLE_OBSERVATION_CONTRACT
        ):
            raise ValueError(
                "unsupported particle_observation_contract="
                f"{particle_observation_contract!r} for source {campaign_name!r}"
            )
        nitrogen_contract = (
            particle_observation_contract == NITROGEN_PARTICLE_OBSERVATION_CONTRACT
        )

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
                "candidate_signature": _optional_text(
                    case_dict.get("OPT_CANDIDATE_SIGNATURE", "")
                ),
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
                "particle_observation_contract": (
                    particle_observation_contract or "legacy"
                ),
                "particle_metric_origin": "not_checked",
                "soft50_curve_concordance_status": "not_checked",
                "soft50_curve_concordance_reason": "",
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
                "guiding_singlecase_score_status": "not_checked",
                "particle_summary_status": "not_applicable"
                if params.get("plasma_kind") == "vac"
                else "not_checked",
                "acceptance_curves_status": "not_applicable"
                if params.get("plasma_kind") == "vac"
                else "not_checked",
                "soft50_curves_status": "not_applicable"
                if params.get("plasma_kind") == "vac"
                else "not_checked",
                "comparison_metrics_status": "not_checked",
                **params,
            }

            reduced_outputs = dict(DEFAULT_REDUCED_OUTPUTS)
            reduced_outputs.update(source.get("reduced_outputs", {}) or {})

            particle_metrics: dict[str, Any] = {}

            if (
                reduced_outputs.get("particle_summary")
                and params.get("plasma_kind") != "vac"
            ):
                summary_path = cdir / reduced_outputs["particle_summary"]
                if nitrogen_contract:
                    particle_metrics, status, reason = _read_nitrogen_particle_summary(
                        summary_path
                    )
                    row["particle_metric_origin"] = (
                        "particle_summary:first_row_all_electrons"
                    )
                else:
                    particle_metrics, status, reason = _read_first_row_csv(summary_path)
                    row["particle_metric_origin"] = "particle_summary:first_row_legacy"
                row["particle_summary_status"] = status
                if reason and not row["failure_reason"]:
                    row["failure_reason"] = reason
                if status == "ok":
                    row.update(_metric_from_first_row("particle", particle_metrics))
                elif nitrogen_contract:
                    _mark_particle_observation_contract_failure(row, reason=reason)

            if (
                reduced_outputs.get("soft50_curves")
                and params.get("plasma_kind") != "vac"
            ):
                soft50_cfg = config.objective_config().get("soft50_v1", {}) or {}
                energy_low = float(soft50_cfg.get("energy_low_MeV", 10.0))
                energy_target = float(soft50_cfg.get("energy_target_MeV", 50.0))
                curve_path = cdir / reduced_outputs["soft50_curves"]
                if nitrogen_contract:
                    curve_metrics, status, reason = _read_nitrogen_soft50_curve_row(
                        curve_path,
                        energy_low_mev=energy_low,
                    )
                    row["soft50_curves_status"] = status
                    if status == "ok" and particle_metrics:
                        concordance_status, concordance_reason = (
                            _validate_nitrogen_soft50_concordance(
                                particle_metrics,
                                curve_metrics,
                                energy_low_mev=energy_low,
                                energy_target_mev=energy_target,
                            )
                        )
                        row["soft50_curve_concordance_status"] = concordance_status
                        row["soft50_curve_concordance_reason"] = concordance_reason
                        if concordance_status != "ok":
                            _mark_particle_observation_contract_failure(
                                row,
                                reason=concordance_reason,
                            )
                    elif status != "ok":
                        row["soft50_curve_concordance_status"] = status
                        row["soft50_curve_concordance_reason"] = reason
                        _mark_particle_observation_contract_failure(row, reason=reason)
                    else:
                        reason = "N2 Soft50 concordance cannot run without a valid canonical summary"
                        row["soft50_curve_concordance_status"] = (
                            "particle_observation_contract_failed"
                        )
                        row["soft50_curve_concordance_reason"] = reason
                        _mark_particle_observation_contract_failure(row, reason=reason)
                    # Deliberately do not merge curve metrics into metric_particle_*.
                    # The canonical plateau/all-electron summary remains authoritative.
                else:
                    curve_metrics, status, _reason = _read_soft50_curve_row(
                        curve_path,
                        energy_low_mev=energy_low,
                    )
                    row["soft50_curves_status"] = status
                    row["soft50_curve_concordance_status"] = "legacy_not_required"
                    if status == "ok":
                        row.update(
                            _metric_from_reduced_output_row("particle", curve_metrics)
                        )

            if reduced_outputs.get("guiding_metrics"):
                guiding_metrics, status, reason = _read_guiding_metrics_row(
                    cdir / reduced_outputs["guiding_metrics"],
                    particle_metrics=particle_metrics,
                )
                row["guiding_metrics_status"] = status
                if reason and not row["failure_reason"]:
                    row["failure_reason"] = reason
                row.update(_metric_from_first_row("guiding", guiding_metrics))

            if reduced_outputs.get("guiding_singlecase_score"):
                singlecase_metrics, status, reason = _read_first_row_csv(
                    cdir / reduced_outputs["guiding_singlecase_score"]
                )
                row["guiding_singlecase_score_status"] = status
                if (
                    reason
                    and status != "missing_reduced_output"
                    and not row["failure_reason"]
                ):
                    row["failure_reason"] = reason
                row.update(
                    _metric_from_reduced_output_row(
                        "guiding_singlecase",
                        singlecase_metrics,
                    )
                )

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

    first_cols = [
        "observation_id",
        "source_campaign_name",
        "source_campaign_root",
        "source_case_id",
        "source_case_name",
        "candidate_signature",
        "case_state",
        "simulation_status",
        "raw_validation_status",
        "analysis_status",
        "reduced_validation_status",
        "failure_kind",
        "failure_reason",
        *PARAMETER_COLUMNS,
    ]

    observations = pd.DataFrame(all_rows)
    if observations.empty:
        observations = pd.DataFrame(columns=first_cols)

    ordered = [c for c in first_cols if c in observations.columns]
    ordered += [c for c in observations.columns if c not in ordered]

    observations = observations[ordered]
    return write_csv(inputs_dir / "observations.csv", observations)