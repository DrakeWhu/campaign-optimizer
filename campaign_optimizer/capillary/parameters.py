from __future__ import annotations

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
    "cap_rmax_um",
    "cap_nr",
]

DISTANCE_COLUMNS = [
    "f_number",
    "n0_1e18cm3",
    "plateau_mm_num",
    "diameter_um_num",
    "focus_mm_num",
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

    if not np.isfinite(diameter) and np.isfinite(radius):
        diameter = 2.0 * radius

    if not np.isfinite(radius) and np.isfinite(diameter):
        radius = 0.5 * diameter

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
        "cap_rmax_um": cap_rmax,
        "cap_nr": int(cap_nr) if np.isfinite(cap_nr) else float("nan"),
        "parameter_source_status": parse_status,
    }


def scaled_parameter_array(
    df: pd.DataFrame, parameter_space: dict[str, Any]
) -> np.ndarray:
    ranges = parameter_space["ranges"]
    cols = []

    for col in DISTANCE_COLUMNS:
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
