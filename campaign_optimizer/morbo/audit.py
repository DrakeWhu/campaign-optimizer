from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .search_space import SearchSpaceCodec


DEFAULT_CATEGORICAL_COLUMNS = ("laser_case",)
DEFAULT_NUMERIC_COLUMNS = ("acquisition_value",)


def audit_morbo_outputs(
    outputs_dir: str | Path,
    *,
    codec: SearchSpaceCodec | None = None,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
) -> dict[str, Any]:
    """Audit MORBO-like recommendation sidecars from one optimizer iteration.

    This function is read-only. It reads TSV/JSON outputs that were already
    produced by ``propose_recommendations`` or ``run_iteration`` and returns a
    JSON-serializable summary.
    """

    out = Path(outputs_dir)
    files = {
        "recommended_candidates": out / "recommended_candidates.tsv",
        "candidate_batch": out / "candidate_batch.tsv",
        "morbo_optimizer_state": out / "morbo_optimizer_state.json",
        "morbo_frontier": out / "morbo_frontier.json",
        "morbo_regions": out / "morbo_regions.json",
        "surrogate_summary": out / "surrogate_summary.json",
    }

    file_status = {
        key: {"path": str(path), "exists": path.is_file()}
        for key, path in files.items()
    }

    rec = _read_tsv_if_exists(files["recommended_candidates"])
    regions = _read_json_if_exists(files["morbo_regions"], default=[])
    frontier = _read_json_if_exists(files["morbo_frontier"], default=[])
    state = _read_json_if_exists(files["morbo_optimizer_state"], default={})
    surrogate = _read_json_if_exists(files["surrogate_summary"], default={})

    known_categories = _known_categories_by_column(codec, categorical_columns)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "outputs_dir": str(out),
        "files": file_status,
        "surrogate": _surrogate_subset(surrogate),
        "state": _state_subset(state),
        "recommendations": _audit_recommendations(
            rec,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        ),
        "regions": _audit_regions(
            regions,
            recommendations=rec,
            codec=codec,
            categorical_columns=categorical_columns,
        ),
        "frontier": _audit_frontier(
            frontier,
            known_categories=known_categories,
        ),
    }
    summary["warnings"] = _audit_warnings(summary, known_categories)
    return summary


def write_morbo_audit_json(
    outputs_dir: str | Path,
    path: str | Path | None = None,
    *,
    codec: SearchSpaceCodec | None = None,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
) -> Path:
    """Write ``morbo_audit.json`` for one optimizer iteration."""

    out = Path(outputs_dir)
    audit_path = Path(path) if path is not None else out / "morbo_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary = audit_morbo_outputs(
        out,
        codec=codec,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    audit_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return audit_path


def render_morbo_audit_text(summary: Mapping[str, Any]) -> str:
    """Render a compact human-readable MORBO audit report."""

    rec = dict(summary.get("recommendations", {}))
    regions = dict(summary.get("regions", {}))
    frontier = dict(summary.get("frontier", {}))
    surrogate = dict(summary.get("surrogate", {}))
    warnings = list(summary.get("warnings", []))

    lines = [
        "MORBO-like audit",
        "================",
        f"outputs_dir: {summary.get('outputs_dir', '')}",
        "",
        "Surrogate",
        "---------",
        f"backend: {surrogate.get('backend', '')}",
        f"surrogate_backend: {surrogate.get('surrogate_backend', '')}",
        f"last_strategy: {surrogate.get('last_strategy', '')}",
        f"suggestion_mode: {surrogate.get('suggestion_mode', '')}",
        f"model_status: {surrogate.get('model_status', '')}",
        "",
        "Recommendations",
        "---------------",
        f"rows: {rec.get('row_count', 0)}",
    ]

    for key in ("candidate_source_counts", "morbo_strategy_counts", "region_counts"):
        counts = rec.get(key, [])
        if counts:
            lines.append(f"{key}: {_format_counts(counts)}")

    categorical = dict(rec.get("categorical_counts", {}))
    for column, counts in categorical.items():
        lines.append(f"{column}: {_format_counts(counts)}")

    numeric = dict(rec.get("numeric", {}))
    for column, stats in numeric.items():
        lines.append(
            f"{column}: count={stats.get('count', 0)} "
            f"min={stats.get('min')} max={stats.get('max')} "
            f"mean={stats.get('mean')}"
        )

    lines.extend(
        [
            "",
            "Regions",
            "-------",
            f"rows: {regions.get('row_count', 0)}",
            f"active: {regions.get('active_count', 0)}",
        ]
    )
    for item in regions.get("records", []):
        decoded = item.get("decoded_center", {})
        lines.append(
            f"- {item.get('region_id')}: status={item.get('status')} "
            f"radius={item.get('radius')} obs={item.get('observation_count')} "
            f"frontier={item.get('frontier_count')} recs={item.get('recommendation_count')} "
            f"decoded_center={decoded}"
        )

    lines.extend(
        [
            "",
            "Frontier",
            "--------",
            f"rows: {frontier.get('row_count', 0)}",
        ]
    )
    for column, counts in dict(frontier.get("candidate_id_token_counts", {})).items():
        lines.append(f"{column}: {_format_counts(counts)}")

    if warnings:
        lines.extend(["", "Warnings", "--------"])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def _read_tsv_if_exists(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _read_json_if_exists(path: Path, *, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_recommendations(
    rec: pd.DataFrame,
    *,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "row_count": int(len(rec)),
        "columns": list(rec.columns),
        "candidate_source_counts": _value_counts(rec, "candidate_source"),
        "morbo_strategy_counts": _value_counts(rec, "morbo_strategy"),
        "region_counts": _value_counts(rec, "region_id"),
        "categorical_counts": {},
        "numeric": {},
    }

    for column in categorical_columns:
        out["categorical_counts"][str(column)] = _value_counts(rec, str(column))

    for column in numeric_columns:
        out["numeric"][str(column)] = _numeric_summary(rec, str(column))

    return out


def _audit_regions(
    regions: Any,
    *,
    recommendations: pd.DataFrame,
    codec: SearchSpaceCodec | None,
    categorical_columns: Sequence[str],
) -> dict[str, Any]:
    records = regions if isinstance(regions, list) else []
    rec_region_counts = _counts_as_dict(_value_counts(recommendations, "region_id"))

    out_records: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []

    for item in records:
        if not isinstance(item, Mapping):
            continue

        center = list(item.get("center", []))
        decoded = _decode_center(codec, center)
        if decoded:
            decoded_rows.append(decoded)

        region_id = _string_value(item.get("region_id", ""))
        metadata = dict(item.get("metadata", {}) or {})
        out_records.append(
            {
                "region_id": region_id,
                "status": _string_value(item.get("status", "")),
                "radius": _safe_float_or_none(item.get("radius")),
                "observation_count": int(item.get("observation_count", 0) or 0),
                "frontier_count": int(item.get("frontier_count", 0) or 0),
                "success_streak": int(item.get("success_streak", 0) or 0),
                "failure_streak": int(item.get("failure_streak", 0) or 0),
                "restart_count": int(item.get("restart_count", 0) or 0),
                "generation": int(item.get("generation", 0) or 0),
                "seed_candidate_id": metadata.get("seed_candidate_id"),
                "center_candidate_id": metadata.get("center_candidate_id"),
                "recommendation_count": int(rec_region_counts.get(region_id, 0)),
                "center": center,
                "decoded_center": decoded,
            }
        )

    return {
        "row_count": len(out_records),
        "active_count": sum(1 for item in out_records if item["status"] == "active"),
        "records": sorted(out_records, key=lambda item: item["region_id"]),
        "decoded_center_categorical_counts": _categorical_counts_from_dicts(
            decoded_rows,
            categorical_columns,
        ),
    }


def _audit_frontier(
    frontier: Any,
    *,
    known_categories: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    records = frontier if isinstance(frontier, list) else []
    candidate_ids = [
        str(item.get("candidate_id", ""))
        for item in records
        if isinstance(item, Mapping)
    ]

    return {
        "row_count": len(candidate_ids),
        "candidate_id_token_counts": {
            column: _token_counts(candidate_ids, options)
            for column, options in known_categories.items()
        },
        "objective_values": _objective_ranges(records, key="objective_values"),
        "canonical_objective_values": _objective_ranges(
            records,
            key="canonical_objective_values",
        ),
    }


def _surrogate_subset(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    diagnostics = payload.get("model_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}

    return {
        "backend": payload.get("backend"),
        "surrogate_backend": payload.get("surrogate_backend"),
        "last_strategy": payload.get("last_strategy"),
        "suggestion_mode": payload.get("suggestion_mode"),
        "model_status": diagnostics.get("status"),
        "model_reason": diagnostics.get("reason"),
        "acquisition_function": diagnostics.get("acquisition_function"),
        "train_rows": diagnostics.get("train_rows"),
        "candidate_pool_rows": diagnostics.get("candidate_pool_rows"),
        "selected_rows": diagnostics.get("selected_rows"),
        "objective_names": diagnostics.get(
            "objective_names", payload.get("objective_names")
        ),
    }


def _state_subset(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    registry = payload.get("candidate_registry", {})
    if not isinstance(registry, Mapping):
        registry = {}
    return {
        "schema_version": payload.get("schema_version"),
        "backend": payload.get("backend"),
        "last_strategy": payload.get("last_strategy"),
        "region_count": len(payload.get("regions", []) or []),
        "ref_point_raw": payload.get("ref_point_raw", {}),
        "ref_point_model_units": payload.get("ref_point_model_units", {}),
        "y_transform_type": dict(payload.get("y_transform", {}) or {}).get("type"),
        "seen_count": len(registry.get("seen_candidate_signatures", []) or []),
        "pending_count": len(registry.get("pending_candidate_signatures", []) or []),
    }


def _audit_warnings(
    summary: Mapping[str, Any],
    known_categories: Mapping[str, Sequence[Any]],
) -> list[str]:
    warnings: list[str] = []
    files = dict(summary.get("files", {}))
    for name, status in files.items():
        if not dict(status).get("exists", False) and name != "candidate_batch":
            warnings.append(f"Missing MORBO output file: {name}")

    surrogate = dict(summary.get("surrogate", {}))
    model_status = surrogate.get("model_status")
    suggestion_mode = surrogate.get("suggestion_mode")
    last_strategy = surrogate.get("last_strategy")
    if suggestion_mode == "regional_model" and model_status not in {"ok", None}:
        warnings.append(f"regional_model did not report ok status: {model_status}")
    if suggestion_mode == "regional_model" and last_strategy != "regional_model":
        warnings.append(
            f"regional_model was requested but last_strategy is {last_strategy}"
        )

    rec = dict(summary.get("recommendations", {}))
    categorical = dict(rec.get("categorical_counts", {}))
    for column, counts in categorical.items():
        non_empty = [
            item for item in counts if item.get("value") not in {"", "nan", "None"}
        ]
        known = [str(item) for item in known_categories.get(column, ())]
        if len(non_empty) == 1 and len(known) > 1:
            warnings.append(
                f"All recommendations share {column}={non_empty[0]['value']} "
                f"while known options are {known}"
            )

    regions = dict(summary.get("regions", {}))
    if regions.get("active_count", 0) > 1:
        region_counts = _counts_as_dict(rec.get("region_counts", []))
        used_regions = [key for key, value in region_counts.items() if value > 0]
        if len(used_regions) == 1:
            warnings.append(
                f"All recommendations came from one region despite "
                f"{regions.get('active_count')} active regions"
            )

    return warnings


def _value_counts(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if df.empty or column not in df.columns:
        return []

    values = [_string_value(value) for value in df[column].tolist()]
    total = len(values)
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return [
        {
            "value": value,
            "count": int(count),
            "fraction": float(count / total) if total else 0.0,
        }
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _numeric_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {"count": 0, "min": None, "max": None, "mean": None}

    numeric = pd.to_numeric(df[column], errors="coerce")
    values = [
        float(value)
        for value in numeric.dropna().tolist()
        if math.isfinite(float(value))
    ]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}

    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _categorical_counts_from_dicts(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        counts: dict[str, int] = {}
        total = 0
        for row in rows:
            if column not in row:
                continue
            value = _string_value(row[column])
            counts[value] = counts.get(value, 0) + 1
            total += 1
        out[str(column)] = [
            {
                "value": value,
                "count": count,
                "fraction": float(count / total) if total else 0.0,
            }
            for value, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
    return out


def _counts_as_dict(counts: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(counts, list):
        return out
    for item in counts:
        if not isinstance(item, Mapping):
            continue
        out[_string_value(item.get("value", ""))] = int(item.get("count", 0) or 0)
    return out


def _known_categories_by_column(
    codec: SearchSpaceCodec | None,
    categorical_columns: Sequence[str],
) -> dict[str, list[Any]]:
    if codec is None:
        return {str(column): [] for column in categorical_columns}

    out: dict[str, list[Any]] = {}
    for column in categorical_columns:
        spec = codec.space.get(str(column))
        options = getattr(spec, "options", None)
        out[str(column)] = list(options) if options is not None else []
    return out


def _decode_center(
    codec: SearchSpaceCodec | None,
    center: Sequence[Any],
) -> dict[str, Any]:
    if codec is None:
        return {}
    try:
        return codec.decode([float(value) for value in center])
    except Exception as exc:
        return {"decode_error": str(exc)}


def _token_counts(
    candidate_ids: Sequence[str],
    options: Sequence[Any],
) -> list[dict[str, Any]]:
    if not options:
        return []

    total = len(candidate_ids)
    rows: list[dict[str, Any]] = []
    for option in options:
        token = str(option)
        count = sum(1 for candidate_id in candidate_ids if token in candidate_id)
        rows.append(
            {
                "value": token,
                "count": int(count),
                "fraction": float(count / total) if total else 0.0,
            }
        )
    return sorted(rows, key=lambda item: (-item["count"], item["value"]))


def _objective_ranges(records: Sequence[Any], *, key: str) -> dict[str, dict[str, Any]]:
    values_by_name: dict[str, list[float]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            continue
        values = item.get(key, {})
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            numeric = _safe_float_or_none(value)
            if numeric is None:
                continue
            values_by_name.setdefault(str(name), []).append(numeric)

    out: dict[str, dict[str, Any]] = {}
    for name, values in values_by_name.items():
        out[name] = {
            "count": len(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
        }
    return out


def _safe_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _format_counts(counts: Any) -> str:
    if not isinstance(counts, list) or not counts:
        return "none"
    return ", ".join(
        f"{item.get('value')}={item.get('count')}"
        for item in counts
        if isinstance(item, Mapping)
    )
