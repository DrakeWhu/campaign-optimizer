from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .frontier import FrontierRecord
from .regions import RegionRecord


SCHEMA_VERSION = "morbo_like_state_v1"
BACKEND_NAME = "morbo_like"


@dataclass(frozen=True)
class CandidateRegistry:
    """Seen/pending candidate signatures used for duplicate prevention."""

    seen_candidate_signatures: tuple[str, ...] = ()
    pending_candidate_signatures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen = _dedupe_sorted_strings(
            self.seen_candidate_signatures, "seen_candidate_signatures"
        )
        pending = _dedupe_sorted_strings(
            self.pending_candidate_signatures,
            "pending_candidate_signatures",
        )
        object.__setattr__(self, "seen_candidate_signatures", seen)
        object.__setattr__(self, "pending_candidate_signatures", pending)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateRegistry":
        return cls(
            seen_candidate_signatures=tuple(
                payload.get("seen_candidate_signatures", ())
            ),
            pending_candidate_signatures=tuple(
                payload.get("pending_candidate_signatures", ())
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen_candidate_signatures": list(self.seen_candidate_signatures),
            "pending_candidate_signatures": list(self.pending_candidate_signatures),
        }

    def mark_seen(self, signatures: Sequence[str]) -> "CandidateRegistry":
        return CandidateRegistry(
            seen_candidate_signatures=(
                *self.seen_candidate_signatures,
                *tuple(str(item) for item in signatures),
            ),
            pending_candidate_signatures=self.pending_candidate_signatures,
        )

    def mark_pending(self, signatures: Sequence[str]) -> "CandidateRegistry":
        new_pending = tuple(str(item) for item in signatures)
        return CandidateRegistry(
            seen_candidate_signatures=(
                *self.seen_candidate_signatures,
                *new_pending,
            ),
            pending_candidate_signatures=(
                *self.pending_candidate_signatures,
                *new_pending,
            ),
        )

    def clear_pending(self, signatures: Sequence[str]) -> "CandidateRegistry":
        to_clear = {str(item) for item in signatures}
        return CandidateRegistry(
            seen_candidate_signatures=self.seen_candidate_signatures,
            pending_candidate_signatures=tuple(
                item
                for item in self.pending_candidate_signatures
                if item not in to_clear
            ),
        )


@dataclass(frozen=True)
class OptimizerState:
    """File-based optimizer state for the MORBO-like backend."""

    schema_version: str = SCHEMA_VERSION
    backend: str = BACKEND_NAME
    seed: int = 42
    objective_spec: Mapping[str, Any] = field(default_factory=dict)
    search_space_signature: str | None = None
    candidate_registry: CandidateRegistry = field(default_factory=CandidateRegistry)
    ref_point_policy: Mapping[str, Any] = field(default_factory=dict)
    ref_point_raw: Mapping[str, float] = field(default_factory=dict)
    ref_point_model_units: Mapping[str, float] = field(default_factory=dict)
    y_transform: Mapping[str, Any] = field(default_factory=dict)
    regions: tuple[RegionRecord, ...] = ()
    last_strategy: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        schema_version = str(self.schema_version)
        backend = str(self.backend)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported optimizer state schema_version '{schema_version}'"
            )
        if backend != BACKEND_NAME:
            raise ValueError(f"unsupported optimizer backend '{backend}'")

        seed = int(self.seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")

        search_space_signature = (
            None
            if self.search_space_signature is None
            else str(self.search_space_signature)
        )
        if search_space_signature is not None and not search_space_signature.strip():
            raise ValueError("search_space_signature must be non-empty when provided")

        created_at = self.created_at or utc_now_iso()
        updated_at = self.updated_at or created_at

        regions = tuple(
            region
            if isinstance(region, RegionRecord)
            else RegionRecord.from_dict(region)
            for region in self.regions
        )
        region_ids = [region.region_id for region in regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "objective_spec", dict(self.objective_spec))
        object.__setattr__(self, "search_space_signature", search_space_signature)
        object.__setattr__(self, "candidate_registry", self.candidate_registry)
        object.__setattr__(self, "ref_point_policy", dict(self.ref_point_policy))
        object.__setattr__(
            self,
            "ref_point_raw",
            _finite_float_dict(self.ref_point_raw, "ref_point_raw"),
        )
        object.__setattr__(
            self,
            "ref_point_model_units",
            _finite_float_dict(self.ref_point_model_units, "ref_point_model_units"),
        )
        object.__setattr__(self, "y_transform", dict(self.y_transform))
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self,
            "last_strategy",
            None if self.last_strategy is None else str(self.last_strategy),
        )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizerState":
        registry_payload = {
            "seen_candidate_signatures": payload.get("seen_candidate_signatures", ()),
            "pending_candidate_signatures": payload.get(
                "pending_candidate_signatures", ()
            ),
        }
        if isinstance(payload.get("candidate_registry"), Mapping):
            registry_payload = payload["candidate_registry"]

        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            backend=str(payload.get("backend", BACKEND_NAME)),
            seed=int(payload.get("seed", 42)),
            objective_spec=dict(payload.get("objective_spec", {})),
            search_space_signature=payload.get("search_space_signature"),
            candidate_registry=CandidateRegistry.from_dict(registry_payload),
            ref_point_policy=dict(payload.get("ref_point_policy", {})),
            ref_point_raw=dict(payload.get("ref_point_raw", {})),
            ref_point_model_units=dict(payload.get("ref_point_model_units", {})),
            y_transform=dict(payload.get("y_transform", {})),
            regions=tuple(
                RegionRecord.from_dict(item) for item in payload.get("regions", ())
            ),
            last_strategy=payload.get("last_strategy"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            extra=dict(payload.get("extra", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        registry = self.candidate_registry.as_dict()
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "seed": self.seed,
            "objective_spec": dict(self.objective_spec),
            "search_space_signature": self.search_space_signature,
            "seen_candidate_signatures": registry["seen_candidate_signatures"],
            "pending_candidate_signatures": registry["pending_candidate_signatures"],
            "candidate_registry": registry,
            "ref_point_policy": dict(self.ref_point_policy),
            "ref_point_raw": dict(self.ref_point_raw),
            "ref_point_model_units": dict(self.ref_point_model_units),
            "y_transform": dict(self.y_transform),
            "regions": [region.as_dict() for region in self.regions],
            "last_strategy": self.last_strategy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "extra": dict(self.extra),
        }

    def with_updates(self, **updates: Any) -> "OptimizerState":
        """Return an updated state with a refreshed ``updated_at`` timestamp."""

        return replace(self, updated_at=utc_now_iso(), **updates)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json_atomic(
    path: str | Path, payload: Mapping[str, Any] | Sequence[Any]
) -> None:
    """Write JSON atomically using a same-directory temporary file.

    This is safe on Windows because the temporary file is closed before
    ``os.replace`` is called.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_optimizer_state(path: str | Path, state: OptimizerState) -> None:
    write_json_atomic(path, state.as_dict())


def load_optimizer_state(path: str | Path) -> OptimizerState:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("optimizer state JSON must be an object")
    return OptimizerState.from_dict(payload)


def save_candidate_registry(path: str | Path, registry: CandidateRegistry) -> None:
    write_json_atomic(path, registry.as_dict())


def load_candidate_registry(path: str | Path) -> CandidateRegistry:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("candidate registry JSON must be an object")
    return CandidateRegistry.from_dict(payload)


def save_frontier_records(path: str | Path, records: Sequence[FrontierRecord]) -> None:
    write_json_atomic(path, [frontier_record_to_dict(record) for record in records])


def load_frontier_records(path: str | Path) -> list[FrontierRecord]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("frontier JSON must be a list")
    return [frontier_record_from_dict(item) for item in payload]


def save_region_records(path: str | Path, records: Sequence[RegionRecord]) -> None:
    write_json_atomic(path, [record.as_dict() for record in records])


def load_region_records(path: str | Path) -> list[RegionRecord]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("regions JSON must be a list")
    return [RegionRecord.from_dict(item) for item in payload]


def frontier_record_to_dict(record: FrontierRecord) -> dict[str, Any]:
    return {
        "candidate_id": record.candidate_id,
        "objective_values": dict(record.objective_values),
        "canonical_objective_values": dict(record.canonical_objective_values),
    }


def frontier_record_from_dict(payload: Mapping[str, Any]) -> FrontierRecord:
    return FrontierRecord(
        candidate_id=str(payload["candidate_id"]),
        objective_values=_finite_float_dict(
            payload.get("objective_values", {}),
            "objective_values",
        ),
        canonical_objective_values=_finite_float_dict(
            payload.get("canonical_objective_values", {}),
            "canonical_objective_values",
        ),
    )


def _dedupe_sorted_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = str(value)
        if not item:
            raise ValueError(f"{label} cannot contain empty strings")
        if item in seen:
            continue
        seen.add(item)
        out.append(item)

    return tuple(sorted(out))


def _finite_float_dict(values: Mapping[str, Any], label: str) -> dict[str, float]:
    out: dict[str, float] = {}

    for key, value in values.items():
        name = str(key)
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} value for '{name}' must be finite")
        out[name] = numeric

    return out
