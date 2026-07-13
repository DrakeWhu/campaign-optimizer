from __future__ import annotations

import math
from typing import Any, Mapping

from campaign_optimizer.morbo import FloatRange, PeriodicRange, SearchSpaceCodec


SUPPORTED_PARAMETER_TYPES = frozenset({"float", "log_float", "periodic"})


def parameter_definitions(parameter_space: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_parameters = parameter_space.get("parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("parameter_space.parameters must be a non-empty list")

    definitions: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_columns: set[str] = set()

    for raw in raw_parameters:
        if not isinstance(raw, dict):
            raise ValueError("parameter_space.parameters entries must be objects")

        name = str(raw.get("name", "")).strip()
        column = str(raw.get("column", name.upper())).strip()
        kind = str(raw.get("type", "float")).strip().lower()
        bounds = raw.get("bounds")

        if not name:
            raise ValueError("parameter definition has an empty name")
        if name in seen_names:
            raise ValueError(f"duplicate parameter name: {name}")
        if not column:
            raise ValueError(f"parameter {name!r} has an empty column")
        if column in seen_columns:
            raise ValueError(f"duplicate parameter column: {column}")
        if kind not in SUPPORTED_PARAMETER_TYPES:
            raise ValueError(f"unsupported parameter type for {name!r}: {kind!r}")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"parameter {name!r} bounds must contain two values")

        low, high = (float(bounds[0]), float(bounds[1]))
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError(f"parameter {name!r} has invalid bounds {bounds!r}")
        if kind == "log_float" and low <= 0.0:
            raise ValueError(f"log parameter {name!r} lower bound must be positive")

        definitions.append(
            {
                **dict(raw),
                "name": name,
                "column": column,
                "type": kind,
                "bounds": [low, high],
            }
        )
        seen_names.add(name)
        seen_columns.add(column)

    return definitions


def parameter_names(parameter_space: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(item["name"] for item in parameter_definitions(parameter_space))


def parameter_columns(parameter_space: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(item["column"] for item in parameter_definitions(parameter_space))


def column_by_name(parameter_space: Mapping[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["column"]
        for item in parameter_definitions(parameter_space)
    }


def build_search_space(parameter_space: Mapping[str, Any]) -> dict[str, Any]:
    search_space: dict[str, Any] = {}
    for item in parameter_definitions(parameter_space):
        low, high = item["bounds"]
        if item["type"] == "periodic":
            search_space[item["name"]] = PeriodicRange(low, high)
        else:
            search_space[item["name"]] = FloatRange(
                low,
                high,
                log=item["type"] == "log_float",
            )
    return search_space


def build_codec(parameter_space: Mapping[str, Any]) -> SearchSpaceCodec:
    return SearchSpaceCodec(build_search_space(parameter_space))


def reference_parameters(parameter_space: Mapping[str, Any]) -> dict[str, float]:
    raw_reference = parameter_space.get("reference")
    if not isinstance(raw_reference, dict):
        raise ValueError("parameter_space.reference must be an object")

    names = parameter_names(parameter_space)
    missing = [name for name in names if name not in raw_reference]
    if missing:
        raise ValueError(f"parameter_space.reference is missing {missing}")

    codec = build_codec(parameter_space)
    return {
        key: float(value)
        for key, value in codec.project_params(raw_reference).items()
    }


def parameters_from_case_row(
    row: Mapping[str, Any], parameter_space: Mapping[str, Any]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in parameter_definitions(parameter_space):
        column = item["column"]
        if column not in row or str(row[column]).strip() == "":
            values[item["name"]] = float("nan")
            continue
        try:
            values[item["name"]] = float(row[column])
        except (TypeError, ValueError):
            values[item["name"]] = float("nan")
    return values


def sobol_parameters(
    parameter_space: Mapping[str, Any],
    *,
    start_index: int,
    count: int,
    seed: int,
) -> list[dict[str, float]]:
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
            "Sobol initialization requires scipy; install scipy in the optimizer environment"
        ) from exc

    definitions = parameter_definitions(parameter_space)
    required = start_index + count
    power = max(0, int(math.ceil(math.log2(max(required, 1)))))
    unit_points = qmc.Sobol(
        d=len(definitions),
        scramble=True,
        seed=int(seed),
    ).random_base2(m=power)

    rows: list[dict[str, float]] = []
    for unit_row in unit_points[start_index:required]:
        params: dict[str, float] = {}
        for coordinate, item in zip(unit_row, definitions):
            low, high = item["bounds"]
            if item["type"] == "log_float":
                value = 10.0 ** (
                    math.log10(low)
                    + float(coordinate) * (math.log10(high) - math.log10(low))
                )
            else:
                value = low + float(coordinate) * (high - low)
            params[item["name"]] = float(value)
        rows.append(params)

    return rows
