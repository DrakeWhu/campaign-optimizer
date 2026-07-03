from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


TERMINAL_SUCCESS_STATUSES = frozenset({"finished", "analyzed"})
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})


class ObjectiveSpecError(ValueError):
    """Raised when an objective specification is malformed."""


class ObjectiveResolutionError(ValueError):
    """Raised when reduced metrics cannot be converted into objectives."""


def _identity(value: float) -> float:
    return value


TRANSFORMS: dict[str, Callable[[float], float]] = {
    "identity": _identity,
    "abs": abs,
    "log10": math.log10,
    "log10p": lambda value: math.log10(1.0 + value),
    "sqrt": math.sqrt,
    "square": lambda value: value * value,
}


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Mapping from one reduced metric to one optimizer objective.

    The backend internally treats every objective as a maximization objective.
    Objectives with ``sense="min"`` are canonicalized by sign flip.
    """

    name: str
    metric: str
    sense: str
    transform: str = "identity"
    scale: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        metric = str(self.metric).strip()
        sense = str(self.sense).strip().lower()
        transform = str(self.transform).strip().lower()
        scale = float(self.scale)

        if not name:
            raise ObjectiveSpecError("Objective entry requires a non-empty 'name'")
        if not metric:
            raise ObjectiveSpecError(
                f"Objective '{name}' requires a non-empty 'metric'"
            )
        if sense not in {"max", "min"}:
            raise ObjectiveSpecError(f"Objective '{name}' has invalid sense '{sense}'")
        if transform not in TRANSFORMS:
            raise ObjectiveSpecError(
                f"Objective '{name}' has unsupported transform '{transform}'"
            )
        if not math.isfinite(scale):
            raise ObjectiveSpecError(f"Objective '{name}' scale must be finite")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "sense", sense)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveDefinition":
        return cls(
            name=str(payload.get("name", "")).strip(),
            metric=str(payload.get("metric", "")).strip(),
            sense=str(payload.get("sense", "")).strip().lower(),
            transform=str(payload.get("transform", "identity")).strip().lower(),
            scale=float(payload.get("scale", 1.0)),
        )

    def apply(self, raw_value: Any) -> float:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ObjectiveResolutionError(
                f"Metric '{self.metric}' for objective '{self.name}' is not numeric"
            )

        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise ObjectiveResolutionError(
                f"Metric '{self.metric}' for objective '{self.name}' is not finite"
            )

        try:
            transformed = TRANSFORMS[self.transform](numeric_value)
        except ValueError as exc:
            raise ObjectiveResolutionError(
                f"Transform '{self.transform}' failed for objective '{self.name}': {exc}"
            ) from exc

        value = float(transformed) * self.scale
        if not math.isfinite(value):
            raise ObjectiveResolutionError(
                f"Objective '{self.name}' resolved to a non-finite value"
            )

        return value

    def canonicalize(self, objective_value: float) -> float:
        value = float(objective_value)
        if not math.isfinite(value):
            raise ObjectiveResolutionError(
                f"Objective '{self.name}' cannot canonicalize a non-finite value"
            )
        return value if self.sense == "max" else -value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "sense": self.sense,
            "transform": self.transform,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class ObjectiveSpec:
    """Ordered collection of objective definitions."""

    objectives: tuple[ObjectiveDefinition, ...]

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ObjectiveSpecError(
                "Objective spec must define at least one objective"
            )

        names = [item.name for item in self.objectives]
        if len(names) != len(set(names)):
            raise ObjectiveSpecError("Objective names must be unique")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        objectives_raw = payload.get("objectives")
        if not isinstance(objectives_raw, list) or not objectives_raw:
            raise ObjectiveSpecError(
                "Objective spec must define a non-empty 'objectives' list"
            )

        return cls(
            tuple(ObjectiveDefinition.from_dict(item) for item in objectives_raw)
        )

    @classmethod
    def from_list(cls, payload: list[Any]) -> "ObjectiveSpec":
        if not payload:
            raise ObjectiveSpecError("Objective spec list must be non-empty")
        return cls(tuple(ObjectiveDefinition.from_dict(item) for item in payload))

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.objectives)

    def as_dict(self) -> dict[str, Any]:
        return {"objectives": [item.to_dict() for item in self.objectives]}

    def get(self, objective_name: str) -> ObjectiveDefinition:
        for item in self.objectives:
            if item.name == objective_name:
                return item
        raise KeyError(objective_name)


@dataclass(frozen=True)
class EligibilityResult:
    candidate_id: str | None
    simulation_status: str | None
    eligible: bool
    reason: str
    objective_values: dict[str, float]
    canonical_objective_values: dict[str, float]
    missing_objectives: tuple[str, ...] = ()


def load_objective_spec(
    spec_path: str | Path | None = None,
    spec_data: Mapping[str, Any] | list[Any] | None = None,
) -> ObjectiveSpec:
    """Load an objective spec from JSON data or a JSON file.

    Accepted shapes:

    ``{"objectives": [...]}``

    or directly:

    ``[...]``
    """

    if spec_path is None and spec_data is None:
        raise ObjectiveSpecError("Objective spec requires spec_path or spec_data")

    payload: Any
    if spec_data is not None:
        payload = spec_data
    else:
        if spec_path is None:
            raise ObjectiveSpecError(
                "Objective spec path is required when spec_data is absent"
            )
        with Path(spec_path).expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if isinstance(payload, list):
        return ObjectiveSpec.from_list(payload)

    if isinstance(payload, Mapping):
        return ObjectiveSpec.from_dict(payload)

    raise ObjectiveSpecError("Objective spec must be a mapping or objective list")


def adapt_metrics_to_objectives(
    raw_metrics: Mapping[str, Any],
    spec: ObjectiveSpec,
) -> dict[str, dict[str, Any]]:
    """Convert reduced raw metrics into objective values.

    Returns a mapping keyed by objective name. Each entry contains the physical
    objective value, sense, source metric and canonical maximization value.
    """

    derived: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for objective in spec.objectives:
        if objective.metric not in raw_metrics:
            missing.append(objective.name)
            continue

        value = objective.apply(raw_metrics[objective.metric])
        derived[objective.name] = {
            "value": value,
            "sense": objective.sense,
            "metric": objective.metric,
            "canonical_value": objective.canonicalize(value),
        }

    if missing:
        raise ObjectiveResolutionError(
            f"Missing metrics for objectives: {', '.join(missing)}"
        )

    return derived


def evaluate_trial_eligibility(
    simulation_status: str | None,
    raw_metrics: Mapping[str, Any],
    spec: ObjectiveSpec,
    candidate_id: str | int | None = None,
    success_statuses: Iterable[str] = TERMINAL_SUCCESS_STATUSES,
) -> EligibilityResult:
    """Decide whether a completed trial is usable by the optimizer."""

    candidate_id_value = None if candidate_id is None else str(candidate_id)
    success_status_set = set(success_statuses)

    if simulation_status not in success_status_set:
        return EligibilityResult(
            candidate_id=candidate_id_value,
            simulation_status=simulation_status,
            eligible=False,
            reason=f"simulation status '{simulation_status}' is not terminal-successful",
            objective_values={},
            canonical_objective_values={},
        )

    objective_values: dict[str, float] = {}
    canonical_values: dict[str, float] = {}
    missing: list[str] = []

    for objective in spec.objectives:
        if objective.metric not in raw_metrics:
            missing.append(objective.name)
            continue

        try:
            resolved = objective.apply(raw_metrics[objective.metric])
        except ObjectiveResolutionError:
            missing.append(objective.name)
            continue

        objective_values[objective.name] = resolved
        canonical_values[objective.name] = objective.canonicalize(resolved)

    if missing:
        return EligibilityResult(
            candidate_id=candidate_id_value,
            simulation_status=simulation_status,
            eligible=False,
            reason=f"missing or invalid objectives: {', '.join(missing)}",
            objective_values=objective_values,
            canonical_objective_values=canonical_values,
            missing_objectives=tuple(missing),
        )

    return EligibilityResult(
        candidate_id=candidate_id_value,
        simulation_status=simulation_status,
        eligible=True,
        reason="eligible",
        objective_values=objective_values,
        canonical_objective_values=canonical_values,
    )
