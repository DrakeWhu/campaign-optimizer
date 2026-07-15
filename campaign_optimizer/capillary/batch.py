from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import math
import re

import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_json, write_tsv
from campaign_optimizer.state import now_utc, write_optimizer_state

from .parsing import normalize_laser_case, parse_float
from .parameters import resonant_duration_fwhm_fs


RECOMMENDED_REQUIRED_COLUMNS = [
    "recommendation_id",
    "optimizer_iteration",
    "candidate_id",
    "rank",
    "recommendation_status",
    "candidate_source",
    "ranking_source",
    "acquisition_value",
    "laser_case",
    "n0_1e18cm3",
    "plateau_mm_num",
    "diameter_um_num",
    "focus_mm_num",
]

CANDIDATE_BATCH_REQUIRED_COLUMNS = [
    "CASE_ID",
    "CASE_NAME",
    "LASER_CASE",
    "PLASMA_KIND",
    "N0_CM3",
    "PLATEAU_LENGTH_MM",
    "DIAMETER_UM",
    "RADIUS_UM",
    "FOCUS_OFFSET_FROM_PLATEAU_START_MM",
    "NITROGEN_DOPANT_FRACTION",
    "PULSE_RESONANCE_FACTOR",
    "LASER_DURATION_FWHM_FS",
    "CAP_RMAX_UM",
    "CAP_NR",
]

CANDIDATE_BATCH_PROVENANCE_COLUMNS = [
    "OPT_ITERATION",
    "OPT_CANDIDATE_ID",
    "OPT_RECOMMENDATION_ID",
    "OPT_OBJECTIVE_CONFIG_ID",
    "OPT_SOURCE_OBSERVATION_ID",
    "OPT_RANKING_SOURCE",
    "OPT_ACQUISITION_VALUE",
    "OPT_SAMPLE_SOURCE",
    "OPT_SOBOL_INDEX",
    "OPT_REFERENCE_ID",
    "OPT_CANDIDATE_SIGNATURE",
]

EXPECTED_WORKFLOW = [
    "create campaign root",
    "copy or derive cases.tsv from candidate_batch.tsv",
    "copy campaign.json",
    "copy input_template.py",
    "materialize cases",
    "initialize case states",
    "submit SLURM array",
    "validate raw diagnostics",
    "run external analysis",
    "validate reduced outputs",
    "cleanup raw diagnostics through manifests",
]

NON_GOALS = [
    "does not submit jobs",
    "does not call sbatch/srun/mpiexec/mpirun",
    "does not launch WarpX",
    "does not edit physics input",
    "does not edit input_template.py",
    "does not read raw HDF5/openPMD",
    "does not delete data",
    "does not create campaign roots",
    "does not mutate previous campaigns",
]

_SAFE_CASE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def require_columns(df: pd.DataFrame, required: list[str], *, table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required column(s): {', '.join(missing)}"
        )


def optional_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def optional_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    return value


def format_float_token(value: float, *, ndigits: int = 6) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"cannot format non-finite value in CASE_NAME: {value!r}")

    rounded = round(float(value), ndigits)
    if rounded == 0:
        return "0"

    text = f"{rounded:.{ndigits}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def is_safe_relative_case_name(case_name: str) -> bool:
    text = str(case_name).strip()
    if not text:
        return False

    p = PurePosixPath(text.replace("\\", "/"))
    if p.is_absolute():
        return False

    if any(part in {"", ".", ".."} for part in p.parts):
        return False

    return True


def safe_case_name(
    *,
    case_id: int,
    laser_case: str,
    plasma_kind: str,
    n0_1e18cm3: float,
    plateau_mm: float,
    diameter_um: float,
    focus_mm: float,
    nitrogen_fraction: float = 0.0,
    laser_duration_fwhm_fs: float = 30.0,
) -> str:
    raw = (
        f"{case_id:03d}_{laser_case}_{plasma_kind}_"
        f"n{format_float_token(n0_1e18cm3)}e18cm3_"
        f"L{format_float_token(plateau_mm)}mm_"
        f"d{format_float_token(diameter_um)}um_"
        f"foc{format_float_token(focus_mm)}mm_"
        f"N2pct{format_float_token(100.0 * nitrogen_fraction)}_"
        f"tau{format_float_token(laser_duration_fwhm_fs)}fs_rz"
    )
    name = _SAFE_CASE_NAME_RE.sub("_", raw)

    if not is_safe_relative_case_name(name):
        raise ValueError(f"generated unsafe CASE_NAME: {name!r}")

    return name


def compute_cap_rmax_um(
    diameter_um: float,
    radius_um: float,
    policy: dict[str, Any],
) -> float:
    mode = str(policy.get("policy", "fixed")).strip().lower()

    if mode == "fixed":
        value = parse_float(policy.get("value"))
    elif mode == "diameter_factor":
        value = parse_float(policy.get("factor")) * float(diameter_um)
    elif mode == "radius_factor":
        value = parse_float(policy.get("factor")) * float(radius_um)
    elif mode == "radius_plus_margin":
        value = float(radius_um) + parse_float(policy.get("margin_um"))
    else:
        raise ValueError(
            "candidate_batch.cap_rmax_um.policy must be one of: "
            "fixed, diameter_factor, radius_factor, radius_plus_margin"
        )

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"invalid CAP_RMAX_UM computed from policy {policy!r}: {value!r}"
        )

    return float(value)


def compute_cap_nr(cap_rmax_um: float, batch_config: dict[str, Any]) -> int:
    """Preserve radial resolution while the physical capillary radius changes."""

    grid = batch_config.get("cap_radial_grid")
    if not grid:
        value = int(batch_config.get("cap_nr", 192))
        if value <= 0:
            raise ValueError("candidate_batch.cap_nr must be positive")
        return value
    if not isinstance(grid, dict):
        raise ValueError("candidate_batch.cap_radial_grid must be an object")

    dr_um = parse_float(grid.get("dr_um"))
    blocking_factor = int(grid.get("blocking_factor", 1))
    minimum = int(grid.get("min_nr", blocking_factor))
    if not math.isfinite(dr_um) or dr_um <= 0.0:
        raise ValueError("candidate_batch.cap_radial_grid.dr_um must be positive")
    if blocking_factor <= 0 or minimum <= 0:
        raise ValueError(
            "candidate_batch.cap_radial_grid blocking_factor/min_nr must be positive"
        )
    cells = max(minimum, int(math.ceil(float(cap_rmax_um) / dr_um)))
    return int(math.ceil(cells / blocking_factor) * blocking_factor)


def _source_campaigns_for_plan(
    config: OptimizerConfig,
    iteration: int,
) -> list[dict[str, Any]]:
    out = []
    for source in config.source_campaigns_for_iteration(iteration):
        item = {
            "campaign_name": str(source.get("campaign_name", "")),
            "campaign_root": str(source.get("campaign_root", "")),
            "cases_tsv": str(source.get("cases_tsv", "cases.tsv")),
            "campaign_json": str(source.get("campaign_json", "campaign.json")),
        }

        # Preserve audit metadata for dynamically discovered optimization history.
        for key in ["source_kind", "history_iteration"]:
            if key in source:
                item[key] = source[key]

        out.append(item)

    return out


def recommended_candidates_to_candidate_batch(
    recommended: pd.DataFrame,
    *,
    iteration: int,
    objective_config_id: str,
    parameter_space: dict[str, Any],
    batch_config: dict[str, Any],
) -> pd.DataFrame:
    require_columns(
        recommended,
        RECOMMENDED_REQUIRED_COLUMNS,
        table_name="recommended_candidates.tsv",
    )

    allowed_laser_cases = list(
        parameter_space.get("laser_cases", ["f20", "f32", "f40"])
    )

    plasma_kind = str(batch_config.get("plasma_kind", "chan")).strip().lower()
    if plasma_kind != "chan":
        raise ValueError(
            "Phase 2 candidate batch generation only supports PLASMA_KIND=chan"
        )

    raw_cap_rmax_policy = batch_config.get("cap_rmax_um")
    if not isinstance(raw_cap_rmax_policy, dict) or not raw_cap_rmax_policy:
        raise ValueError(
            "optimizer.json must define candidate_batch.cap_rmax_um policy"
        )
    cap_rmax_policy = dict(raw_cap_rmax_policy)

    case_id_start = int(batch_config.get("case_id_start", 0))

    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[int] = set()

    ordered = recommended.sort_values("rank", kind="stable").reset_index(drop=True)

    for i, rec in ordered.iterrows():
        case_id = case_id_start + i
        if case_id in seen_ids:
            raise ValueError(f"duplicate CASE_ID generated: {case_id}")
        seen_ids.add(case_id)

        laser_case = normalize_laser_case(
            rec["laser_case"],
            allowed=allowed_laser_cases,
        )
        n0_1e18 = parse_float(rec["n0_1e18cm3"])
        plateau = parse_float(rec["plateau_mm_num"])
        diameter = parse_float(rec["diameter_um_num"])
        focus = parse_float(rec["focus_mm_num"])
        nitrogen_fraction = parse_float(rec.get("nitrogen_fraction", 0.0))
        resonant_duration_fs = resonant_duration_fwhm_fs(n0_1e18)
        if "pulse_duration_factor" in parameter_space.get("ranges", {}):
            pulse_duration_factor = parse_float(
                rec.get("pulse_duration_factor")
            )
            laser_duration_fwhm_fs = (
                resonant_duration_fs * pulse_duration_factor
            )
        else:
            laser_duration_fwhm_fs = parse_float(
                batch_config.get("laser_duration_fwhm_fs", 30.0)
            )
            pulse_duration_factor = (
                laser_duration_fwhm_fs / resonant_duration_fs
            )

        for name, value in [
            ("n0_1e18cm3", n0_1e18),
            ("plateau_mm_num", plateau),
            ("diameter_um_num", diameter),
            ("focus_mm_num", focus),
            ("nitrogen_fraction", nitrogen_fraction),
            ("pulse_duration_factor", pulse_duration_factor),
            ("laser_duration_fwhm_fs", laser_duration_fwhm_fs),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"recommended_candidates.tsv has non-finite {name}")
        if not 0.0 <= nitrogen_fraction <= 1.0:
            raise ValueError("nitrogen_fraction must be within [0, 1]")

        radius = 0.5 * diameter
        cap_rmax = compute_cap_rmax_um(diameter, radius, cap_rmax_policy)
        if cap_rmax <= radius:
            raise ValueError(
                "CAP_RMAX_UM must exceed the capillary radius; "
                f"got rmax={cap_rmax}, radius={radius}"
            )
        cap_nr = compute_cap_nr(cap_rmax, batch_config)
        case_name = safe_case_name(
            case_id=case_id,
            laser_case=laser_case,
            plasma_kind=plasma_kind,
            n0_1e18cm3=n0_1e18,
            plateau_mm=plateau,
            diameter_um=diameter,
            focus_mm=focus,
            nitrogen_fraction=nitrogen_fraction,
            laser_duration_fwhm_fs=laser_duration_fwhm_fs,
        )

        if case_name in seen_names:
            raise ValueError(f"duplicate CASE_NAME generated: {case_name}")
        seen_names.add(case_name)

        rows.append(
            {
                "CASE_ID": case_id,
                "CASE_NAME": case_name,
                "LASER_CASE": laser_case,
                "PLASMA_KIND": plasma_kind,
                "N0_CM3": n0_1e18 * 1.0e18,
                "PLATEAU_LENGTH_MM": plateau,
                "DIAMETER_UM": diameter,
                "RADIUS_UM": radius,
                "FOCUS_OFFSET_FROM_PLATEAU_START_MM": focus,
                "NITROGEN_DOPANT_FRACTION": nitrogen_fraction,
                "PULSE_RESONANCE_FACTOR": pulse_duration_factor,
                "LASER_DURATION_FWHM_FS": laser_duration_fwhm_fs,
                "CAP_RMAX_UM": cap_rmax,
                "CAP_NR": cap_nr,
                "OPT_ITERATION": int(rec.get("optimizer_iteration", iteration)),
                "OPT_CANDIDATE_ID": str(rec.get("candidate_id", "")),
                "OPT_RECOMMENDATION_ID": str(rec.get("recommendation_id", "")),
                "OPT_OBJECTIVE_CONFIG_ID": objective_config_id,
                "OPT_SOURCE_OBSERVATION_ID": optional_text(
                    rec.get("source_observation_id", "")
                ),
                "OPT_RANKING_SOURCE": optional_text(rec.get("ranking_source", "")),
                "OPT_ACQUISITION_VALUE": optional_scalar(
                    rec.get("acquisition_value", "")
                ),
                "OPT_SAMPLE_SOURCE": optional_text(rec.get("candidate_source", "")),
                "OPT_SOBOL_INDEX": optional_scalar(rec.get("sobol_index", "")),
                "OPT_REFERENCE_ID": optional_text(rec.get("reference_id", "")),
                "OPT_CANDIDATE_SIGNATURE": optional_text(
                    rec.get("candidate_signature", "")
                ),
            }
        )

    out = pd.DataFrame(rows)
    ordered_columns = (
        CANDIDATE_BATCH_REQUIRED_COLUMNS + CANDIDATE_BATCH_PROVENANCE_COLUMNS
    )
    return out[ordered_columns]


def write_batch_campaign_plan(
    *,
    config: OptimizerConfig,
    iteration: int,
    candidate_batch_path: Path,
    recommended_candidates_path: Path,
) -> Path:
    iter_dir = config.iteration_dir(iteration)
    objective_config = config.objective_config()
    batch_config = config.candidate_batch_config()

    def rel(path: Path) -> str:
        return str(path.relative_to(iter_dir)).replace("\\", "/")

    plan = {
        "schema_version": 1,
        "plan_type": "optimizer_candidate_batch",
        "created_at": now_utc(),
        "optimizer_iteration": int(iteration),
        "objective_config_id": objective_config.get(
            "config_id", "capillary_objectives_v1"
        ),
        "candidate_batch": rel(candidate_batch_path),
        "recommended_candidates": rel(recommended_candidates_path),
        "source_campaigns": _source_campaigns_for_plan(config, iteration),
        "campaign_template": dict(batch_config.get("campaign_template", {}) or {}),
        "expected_workflow": list(EXPECTED_WORKFLOW),
        "non_goals": list(NON_GOALS),
    }
    return write_json(iter_dir / "outputs" / "batch_campaign_plan.json", plan)


def build_candidate_batch(config: OptimizerConfig, iteration: int) -> tuple[Path, Path]:
    iter_dir = config.iteration_dir(iteration)
    outputs_dir = iter_dir / "outputs"
    recommended_path = outputs_dir / "recommended_candidates.tsv"

    recommended = read_table(recommended_path, sep="\t")
    objective_config = config.objective_config()

    candidate_batch = recommended_candidates_to_candidate_batch(
        recommended,
        iteration=iteration,
        objective_config_id=str(
            objective_config.get("config_id", "capillary_objectives_v1")
        ),
        parameter_space=config.parameter_space(),
        batch_config=config.candidate_batch_config(),
    )

    candidate_batch_path = write_tsv(
        outputs_dir / "candidate_batch.tsv", candidate_batch
    )
    plan_path = write_batch_campaign_plan(
        config=config,
        iteration=iteration,
        candidate_batch_path=candidate_batch_path,
        recommended_candidates_path=recommended_path,
    )

    write_optimizer_state(
        iter_dir=iter_dir,
        iteration=iteration,
        config=config.data,
        objective_config=objective_config,
        parameter_space=config.parameter_space(),
        observations_path=iter_dir / "inputs" / "observations.csv",
        objectives_path=iter_dir / "inputs" / "objective_table.csv",
        recommendations_path=recommended_path,
        surrogate_summary_path=outputs_dir / "surrogate_summary.json",
        candidate_batch_path=candidate_batch_path,
        batch_campaign_plan_path=plan_path,
    )

    return candidate_batch_path, plan_path
