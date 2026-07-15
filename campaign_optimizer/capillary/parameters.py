from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .parsing import (
    f_number_from_laser_case,
    normalize_laser_case,
    parse_case_name_parameters,
    parse_float,
)


PARAMETER_COLUMNS = [
    "laser_case",
    "f_number",
    "n0_cm3",
    "n0_1e18cm3",
    "plateau_length_mm",
    "plateau_mm_num",
    "diameter_um",
    "diameter_um_num",
    "radius_um",
    "focus_offset_from_plateau_start_mm",
    "focus_mm_num",
    "nitrogen_fraction",
    "nitrogen_percent",
    "pulse_duration_factor",
    "laser_duration_fwhm_fs",
    "resonant_duration_fwhm_fs",
    "cap_rmax_um",
    "cap_nr",
    "sample_source",
    "sobol_index",
    "reference_id",
]

DISTANCE_COLUMNS = [
    "f_number",
    "n0_1e18cm3",
    "plateau_mm_num",
    "diameter_um_num",
    "focus_mm_num",
]

OPTIMIZER_PARAMETER_COLUMNS = [
    "laser_case",
    "n0_1e18cm3",
    "plateau_mm_num",
    "diameter_um_num",
    "focus_mm_num",
    "nitrogen_fraction",
    "pulse_duration_factor",
]


EPSILON_0_SI = 8.8541878128e-12
ELECTRON_MASS_KG = 9.1093837139e-31
ELEMENTARY_CHARGE_C = 1.602176634e-19


def resonant_duration_fwhm_fs(n0_1e18cm3: float) -> float:
    """Return tau=lambda_p/(2c)=pi/omega_p, expressed in femtoseconds."""

    density_m3 = float(n0_1e18cm3) * 1.0e24
    if not math.isfinite(density_m3) or density_m3 <= 0.0:
        raise ValueError("n0_1e18cm3 must be finite and positive")
    omega_p = math.sqrt(
        density_m3 * ELEMENTARY_CHARGE_C**2
        / (ELECTRON_MASS_KG * EPSILON_0_SI)
    )
    return math.pi / omega_p * 1.0e15


def active_optimizer_parameter_columns(
    parameter_space: dict[str, Any],
) -> list[str]:
    """Keep legacy five-dimensional campaigns backward compatible."""

    ranges = parameter_space["ranges"]
    return [
        name
        for name in OPTIMIZER_PARAMETER_COLUMNS
        if name == "laser_case" or name in ranges
    ]


def canonical_parameters_from_case_row(
    row: dict[str, Any],
    *,
    allowed_laser_cases: list[str],
    allow_case_name_fallback: bool,
) -> dict[str, Any]:
    parse_status = "explicit_columns"
    plasma_kind = (
        str(row.get("PLASMA_KIND", row.get("plasma_kind", ""))).strip().lower()
    )

    if "LASER_CASE" in row and str(row.get("LASER_CASE", "")).strip():
        laser_case = normalize_laser_case(
            row["LASER_CASE"], allowed=allowed_laser_cases
        )
        f_number = f_number_from_laser_case(laser_case)
    else:
        laser_case = ""
        f_number = float("nan")

    n0_cm3 = parse_float(row.get("N0_CM3")) if "N0_CM3" in row else float("nan")
    plateau = (
        parse_float(row.get("PLATEAU_LENGTH_MM"))
        if "PLATEAU_LENGTH_MM" in row
        else float("nan")
    )
    diameter = (
        parse_float(row.get("DIAMETER_UM")) if "DIAMETER_UM" in row else float("nan")
    )
    radius = parse_float(row.get("RADIUS_UM")) if "RADIUS_UM" in row else float("nan")
    focus = (
        parse_float(row.get("FOCUS_OFFSET_FROM_PLATEAU_START_MM"))
        if "FOCUS_OFFSET_FROM_PLATEAU_START_MM" in row
        else float("nan")
    )
    cap_rmax = (
        parse_float(row.get("CAP_RMAX_UM")) if "CAP_RMAX_UM" in row else float("nan")
    )
    cap_nr = parse_float(row.get("CAP_NR")) if "CAP_NR" in row else float("nan")
    nitrogen_fraction = (
        parse_float(row.get("NITROGEN_DOPANT_FRACTION"))
        if "NITROGEN_DOPANT_FRACTION" in row
        and str(row.get("NITROGEN_DOPANT_FRACTION", "")).strip()
        else 0.0
    )
    pulse_duration_factor = (
        parse_float(row.get("PULSE_RESONANCE_FACTOR"))
        if "PULSE_RESONANCE_FACTOR" in row
        and str(row.get("PULSE_RESONANCE_FACTOR", "")).strip()
        else float("nan")
    )
    laser_duration_fwhm_fs = (
        parse_float(row.get("LASER_DURATION_FWHM_FS"))
        if "LASER_DURATION_FWHM_FS" in row
        and str(row.get("LASER_DURATION_FWHM_FS", "")).strip()
        else float("nan")
    )
    sample_source = str(row.get("OPT_SAMPLE_SOURCE", "")).strip()
    sobol_raw = row.get("OPT_SOBOL_INDEX", "")
    try:
        sobol_index: int | str = int(float(sobol_raw)) if str(sobol_raw).strip() else ""
    except (TypeError, ValueError):
        sobol_index = ""
    reference_id = str(row.get("OPT_REFERENCE_ID", "")).strip()

    if not np.isfinite(diameter) and np.isfinite(radius):
        diameter = 2.0 * radius

    if not np.isfinite(radius) and np.isfinite(diameter):
        radius = 0.5 * diameter

    resonant_duration = (
        resonant_duration_fwhm_fs(n0_cm3 / 1.0e18)
        if np.isfinite(n0_cm3) and n0_cm3 > 0.0
        else float("nan")
    )
    if not np.isfinite(laser_duration_fwhm_fs) and np.isfinite(
        pulse_duration_factor
    ) and np.isfinite(resonant_duration):
        laser_duration_fwhm_fs = pulse_duration_factor * resonant_duration
    if not np.isfinite(pulse_duration_factor) and np.isfinite(
        laser_duration_fwhm_fs
    ) and np.isfinite(resonant_duration):
        pulse_duration_factor = laser_duration_fwhm_fs / resonant_duration

    missing_core = any(
        not np.isfinite(v) for v in [f_number, n0_cm3, plateau, diameter, focus]
    )

    if missing_core and plasma_kind != "chan":
        parse_status = "baseline_or_vacuum_not_optimized"

    elif missing_core and allow_case_name_fallback:
        parsed = parse_case_name_parameters(
            str(row.get("CASE_NAME", row.get("case_name", "")))
        )

        if parsed.get("fallback_parse_status") == "ok":
            parse_status = "case_name_fallback"
            laser_case = str(parsed["laser_case"])
            f_number = float(parsed["f_number"])
            n0_cm3 = float(parsed["n0_1e18cm3"]) * 1.0e18
            plateau = float(parsed["plateau_mm_num"])
            diameter = float(parsed["diameter_um_num"])
            radius = 0.5 * diameter
            focus = float(parsed["focus_mm_num"])
        else:
            parse_status = "missing_explicit_columns_and_fallback_failed"

    return {
        "laser_case": laser_case,
        "f_number": f_number,
        "plasma_kind": str(plasma_kind).strip().lower(),
        "n0_cm3": n0_cm3,
        "n0_1e18cm3": n0_cm3 / 1.0e18 if np.isfinite(n0_cm3) else float("nan"),
        "plateau_length_mm": plateau,
        "plateau_mm_num": plateau,
        "diameter_um": diameter,
        "diameter_um_num": diameter,
        "radius_um": radius,
        "focus_offset_from_plateau_start_mm": focus,
        "focus_mm_num": focus,
        "nitrogen_fraction": nitrogen_fraction,
        "nitrogen_percent": 100.0 * nitrogen_fraction,
        "pulse_duration_factor": pulse_duration_factor,
        "laser_duration_fwhm_fs": laser_duration_fwhm_fs,
        "resonant_duration_fwhm_fs": resonant_duration,
        "cap_rmax_um": cap_rmax,
        "cap_nr": int(cap_nr) if np.isfinite(cap_nr) else float("nan"),
        "sample_source": sample_source,
        "sobol_index": sobol_index,
        "reference_id": reference_id,
        "parameter_source_status": parse_status,
    }


def scaled_parameter_array(
    df: pd.DataFrame, parameter_space: dict[str, Any]
) -> np.ndarray:
    ranges = parameter_space["ranges"]
    cols = []

    numeric_columns = DISTANCE_COLUMNS[1:]
    if "nitrogen_fraction" in parameter_space["ranges"]:
        numeric_columns = [*numeric_columns, "nitrogen_fraction"]
    if "pulse_duration_factor" in parameter_space["ranges"]:
        numeric_columns = [*numeric_columns, "pulse_duration_factor"]

    for col in ["f_number", *numeric_columns]:
        if col == "f_number":
            lo, hi = 20.0, 40.0
        else:
            lo, hi = [float(v) for v in ranges[col]]

        denom = hi - lo
        if denom <= 0:
            raise ValueError(f"invalid range for {col}: {lo}, {hi}")

        cols.append(
            (pd.to_numeric(df[col], errors="coerce").to_numpy(float) - lo) / denom
        )

    return np.column_stack(cols)


def capillary_search_space(parameter_space: dict[str, Any]) -> dict[str, Any]:
    """Return the native mixed capillary search space used by MORBO."""

    from campaign_optimizer.morbo import Choice, FloatRange

    ranges = parameter_space["ranges"]
    names = active_optimizer_parameter_columns(parameter_space)
    return {
        "laser_case": Choice(
            list(parameter_space.get("laser_cases", ["f20", "f32", "f40"]))
        ),
        **{
            name: FloatRange(*[float(value) for value in ranges[name]])
            for name in names
            if name != "laser_case"
        },
    }


def capillary_sobol_parameters(
    parameter_space: dict[str, Any],
    *,
    start_index: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Map a deterministic scrambled Sobol sequence onto the mixed space."""

    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if count < 0:
        raise ValueError("count must be >= 0")
    if count == 0:
        return []

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise RuntimeError(
            "Sobol initialization requires scipy in the optimizer environment"
        ) from exc

    required = start_index + count
    power = max(0, int(np.ceil(np.log2(max(required, 1)))))
    unit = qmc.Sobol(
        d=len(active_optimizer_parameter_columns(parameter_space)),
        scramble=True,
        seed=int(seed),
    ).random_base2(m=power)[start_index:required]

    lasers = list(parameter_space.get("laser_cases", ["f20", "f32", "f40"]))
    if not lasers:
        raise ValueError("parameter_space.laser_cases must not be empty")
    ranges = parameter_space["ranges"]
    rows: list[dict[str, Any]] = []
    for point in unit:
        laser_index = min(int(float(point[0]) * len(lasers)), len(lasers) - 1)
        row: dict[str, Any] = {"laser_case": lasers[laser_index]}
        for coordinate, name in zip(
            point[1:], active_optimizer_parameter_columns(parameter_space)[1:]
        ):
            low, high = [float(value) for value in ranges[name]]
            row[name] = low + float(coordinate) * (high - low)
        rows.append(row)
    return rows


def capillary_reference_parameters(
    parameter_space: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Validate explicit control points used before the Sobol design."""

    raw = parameter_space.get("references", []) or []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("parameter_space.references must be a list")

    from campaign_optimizer.morbo import SearchSpaceCodec

    codec = SearchSpaceCodec(capillary_search_space(parameter_space))
    out: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("parameter_space.references entries must be objects")
        reference_id = str(item.get("reference_id", f"reference_{index:03d}")).strip()
        names = active_optimizer_parameter_columns(parameter_space)
        missing = [name for name in names if name not in item]
        if missing:
            raise ValueError(f"parameter_space reference {reference_id!r} is missing {missing}")
        params = codec.project_params(
            {name: item[name] for name in names}
        )
        out.append((reference_id, params))
    return out
