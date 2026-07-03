from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .frontier import FrontierRecord
from .objectives import ObjectiveSpec, evaluate_trial_eligibility
from .search_space import SearchSpaceCodec


@dataclass(frozen=True)
class TrialInput:
    """Reduced trial/candidate data before objective resolution.

    This is the file-based replacement for the legacy scan DB rows. It contains
    only parameter values, reduced metrics and a terminal/reduced status.
    """

    candidate_id: str
    params: Mapping[str, Any]
    raw_metrics: Mapping[str, Any]
    simulation_status: str | None = "finished"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must be non-empty")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "raw_metrics", dict(self.raw_metrics))
        object.__setattr__(
            self,
            "simulation_status",
            None
            if self.simulation_status is None
            else str(self.simulation_status).strip(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObservedTrial:
    """Optimizer-ready observed trial.

    ``canonical_objective_values`` must be in maximization space.
    """

    candidate_id: str
    params: dict[str, Any]
    objective_values: dict[str, float]
    canonical_objective_values: dict[str, float]
    senses: dict[str, str]
    raw_metrics: dict[str, Any]
    simulation_status: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.objective_values:
            raise ValueError("objective_values must be non-empty")
        if not self.canonical_objective_values:
            raise ValueError("canonical_objective_values must be non-empty")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "objective_values", _float_dict(self.objective_values))
        object.__setattr__(
            self,
            "canonical_objective_values",
            _float_dict(self.canonical_objective_values),
        )
        object.__setattr__(
            self, "senses", {str(k): str(v) for k, v in self.senses.items()}
        )
        object.__setattr__(self, "raw_metrics", dict(self.raw_metrics))
        object.__setattr__(
            self,
            "simulation_status",
            None if self.simulation_status is None else str(self.simulation_status),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SkippedTrial:
    candidate_id: str
    reason: str
    simulation_status: str | None
    missing_objectives: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        reason = str(self.reason).strip()
        if not candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not reason:
            raise ValueError("skip reason must be non-empty")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "simulation_status",
            None if self.simulation_status is None else str(self.simulation_status),
        )
        object.__setattr__(
            self,
            "missing_objectives",
            tuple(str(item) for item in self.missing_objectives),
        )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "raw_metrics", dict(self.raw_metrics))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObservationBuildResult:
    observations: tuple[ObservedTrial, ...]
    skipped: tuple[SkippedTrial, ...]

    @property
    def observed_count(self) -> int:
        return len(self.observations)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


@dataclass(frozen=True)
class TrainingData:
    candidate_ids: tuple[str, ...]
    params: tuple[dict[str, Any], ...]
    encoded_X: tuple[tuple[float, ...], ...]
    canonical_Y: tuple[tuple[float, ...], ...]
    objective_names: tuple[str, ...]
    objective_values: tuple[dict[str, float], ...]

    def __post_init__(self) -> None:
        n = len(self.candidate_ids)
        for label, values in {
            "params": self.params,
            "encoded_X": self.encoded_X,
            "canonical_Y": self.canonical_Y,
            "objective_values": self.objective_values,
        }.items():
            if len(values) != n:
                raise ValueError(f"{label} length does not match candidate_ids length")

        object.__setattr__(
            self, "candidate_ids", tuple(str(item) for item in self.candidate_ids)
        )
        object.__setattr__(self, "params", tuple(dict(item) for item in self.params))
        object.__setattr__(
            self,
            "encoded_X",
            tuple(tuple(float(value) for value in row) for row in self.encoded_X),
        )
        object.__setattr__(
            self,
            "canonical_Y",
            tuple(tuple(float(value) for value in row) for row in self.canonical_Y),
        )
        object.__setattr__(
            self, "objective_names", tuple(str(item) for item in self.objective_names)
        )
        object.__setattr__(
            self,
            "objective_values",
            tuple(_float_dict(item) for item in self.objective_values),
        )

    def __len__(self) -> int:
        return len(self.candidate_ids)


def build_observation(
    trial: TrialInput,
    spec: ObjectiveSpec,
    success_statuses: Iterable[str] | None = None,
) -> ObservedTrial | SkippedTrial:
    """Resolve one trial into either an observed or skipped optimizer record."""

    kwargs: dict[str, Any] = {}
    if success_statuses is not None:
        kwargs["success_statuses"] = success_statuses

    eligibility = evaluate_trial_eligibility(
        simulation_status=trial.simulation_status,
        raw_metrics=trial.raw_metrics,
        spec=spec,
        candidate_id=trial.candidate_id,
        **kwargs,
    )

    if not eligibility.eligible:
        return SkippedTrial(
            candidate_id=trial.candidate_id,
            reason=eligibility.reason,
            simulation_status=trial.simulation_status,
            missing_objectives=eligibility.missing_objectives,
            params=dict(trial.params),
            raw_metrics=dict(trial.raw_metrics),
            metadata=dict(trial.metadata),
        )

    senses = {objective.name: objective.sense for objective in spec.objectives}
    return ObservedTrial(
        candidate_id=trial.candidate_id,
        params=dict(trial.params),
        objective_values=eligibility.objective_values,
        canonical_objective_values=eligibility.canonical_objective_values,
        senses=senses,
        raw_metrics=dict(trial.raw_metrics),
        simulation_status=trial.simulation_status,
        metadata=dict(trial.metadata),
    )


def build_observations(
    trials: Sequence[TrialInput],
    spec: ObjectiveSpec,
    success_statuses: Iterable[str] | None = None,
    require_unique_candidate_ids: bool = True,
) -> ObservationBuildResult:
    """Resolve a sequence of reduced trials into optimizer observations."""

    if require_unique_candidate_ids:
        candidate_ids = [trial.candidate_id for trial in trials]
        duplicated = sorted(
            {
                candidate_id
                for candidate_id in candidate_ids
                if candidate_ids.count(candidate_id) > 1
            }
        )
        if duplicated:
            raise ValueError(f"duplicate candidate_id values: {', '.join(duplicated)}")

    observations: list[ObservedTrial] = []
    skipped: list[SkippedTrial] = []

    for trial in trials:
        resolved = build_observation(trial, spec, success_statuses=success_statuses)
        if isinstance(resolved, ObservedTrial):
            observations.append(resolved)
        else:
            skipped.append(resolved)

    return ObservationBuildResult(
        observations=tuple(sorted(observations, key=lambda item: item.candidate_id)),
        skipped=tuple(sorted(skipped, key=lambda item: item.candidate_id)),
    )


def build_training_data(
    observations: Sequence[ObservedTrial],
    codec: SearchSpaceCodec,
    objective_names: Sequence[str],
) -> TrainingData:
    """Build encoded X and canonical Y tensors represented as tuples."""

    names = tuple(str(item) for item in objective_names)
    if not names:
        raise ValueError("objective_names must be non-empty")

    ordered = sorted(observations, key=lambda item: item.candidate_id)

    candidate_ids: list[str] = []
    params_rows: list[dict[str, Any]] = []
    encoded_rows: list[tuple[float, ...]] = []
    canonical_rows: list[tuple[float, ...]] = []
    objective_value_rows: list[dict[str, float]] = []

    for observation in ordered:
        projected_params = codec.project_params(observation.params)
        candidate_ids.append(observation.candidate_id)
        params_rows.append(projected_params)
        encoded_rows.append(tuple(codec.encode(projected_params)))
        canonical_rows.append(
            tuple(
                _required_float(
                    observation.canonical_objective_values,
                    objective_name,
                    observation.candidate_id,
                )
                for objective_name in names
            )
        )
        objective_value_rows.append(
            {
                objective_name: _required_float(
                    observation.objective_values,
                    objective_name,
                    observation.candidate_id,
                )
                for objective_name in names
            }
        )

    return TrainingData(
        candidate_ids=tuple(candidate_ids),
        params=tuple(params_rows),
        encoded_X=tuple(encoded_rows),
        canonical_Y=tuple(canonical_rows),
        objective_names=names,
        objective_values=tuple(objective_value_rows),
    )


def frontier_records_from_observations(
    observations: Sequence[ObservedTrial],
) -> list[FrontierRecord]:
    """Convert observations to frontier records."""

    return [
        FrontierRecord(
            candidate_id=observation.candidate_id,
            objective_values=dict(observation.objective_values),
            canonical_objective_values=dict(observation.canonical_objective_values),
        )
        for observation in sorted(observations, key=lambda item: item.candidate_id)
    ]


def read_trial_inputs_from_table(
    path: str | Path,
    parameter_columns: Sequence[str],
    metric_columns: Sequence[str],
    candidate_id_column: str = "candidate_id",
    status_column: str | None = "simulation_status",
    default_status: str | None = "finished",
) -> list[TrialInput]:
    """Read reduced trial inputs from a CSV/TSV table.

    This helper is intentionally simple. It does not know about campaign
    directories, WarpX, HDF5 or SLURM. It only maps table columns into
    ``TrialInput`` records.
    """

    table_path = Path(path)
    delimiter = "\t" if table_path.suffix.lower() == ".tsv" else ","

    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"table '{table_path}' has no header")

        _require_columns(reader.fieldnames, [candidate_id_column])
        _require_columns(reader.fieldnames, parameter_columns)
        _require_columns(reader.fieldnames, metric_columns)
        if status_column is not None:
            _require_columns(reader.fieldnames, [status_column])

        trials: list[TrialInput] = []
        reserved = {candidate_id_column, *(parameter_columns), *(metric_columns)}
        if status_column is not None:
            reserved.add(status_column)

        for row_index, row in enumerate(reader, start=2):
            candidate_id = str(row.get(candidate_id_column, "")).strip()
            if not candidate_id:
                raise ValueError(f"row {row_index} has an empty candidate_id")

            params = {
                column: _parse_table_value(row[column]) for column in parameter_columns
            }
            raw_metrics = {
                column: _parse_metric_value(row[column])
                for column in metric_columns
                if str(row[column]).strip() != ""
            }
            status = (
                default_status
                if status_column is None
                else row.get(status_column, default_status)
            )
            metadata = {
                key: _parse_table_value(value)
                for key, value in row.items()
                if key not in reserved
            }

            trials.append(
                TrialInput(
                    candidate_id=candidate_id,
                    params=params,
                    raw_metrics=raw_metrics,
                    simulation_status=status,
                    metadata=metadata,
                )
            )

    return trials


def _require_columns(fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _parse_metric_value(value: Any) -> Any:
    parsed = _parse_table_value(value)
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return parsed


def _parse_table_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "":
        return ""

    try:
        return json.loads(text)
    except Exception:
        return text


def _required_float(
    values: Mapping[str, Any], objective_name: str, candidate_id: str
) -> float:
    if objective_name not in values:
        raise KeyError(
            f"candidate '{candidate_id}' is missing objective '{objective_name}'"
        )
    return float(values[objective_name])


def _float_dict(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}
