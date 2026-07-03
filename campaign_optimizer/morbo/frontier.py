from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FrontierRecord:
    """Objective record used for Pareto-frontier computations.

    ``objective_values`` keeps the user/reporting objective values.

    ``canonical_objective_values`` must already be transformed so that larger
    is better for every objective. For objectives with sense="min", this means
    the value should already have been sign-flipped by the objective layer.
    """

    candidate_id: str
    objective_values: dict[str, float]
    canonical_objective_values: dict[str, float]


def dominates(
    lhs: Mapping[str, float],
    rhs: Mapping[str, float],
    objective_names: Sequence[str],
) -> bool:
    """Return true when ``lhs`` Pareto-dominates ``rhs``.

    All comparisons are made in canonical maximization space using the explicit
    ``objective_names`` order. This intentionally avoids relying on
    ``dict.values()`` ordering.
    """

    names = tuple(objective_names)
    if not names:
        raise ValueError("objective_names must be non-empty")

    lhs_values = _objective_vector(lhs, names, label="lhs")
    rhs_values = _objective_vector(rhs, names, label="rhs")

    return all(lv >= rv for lv, rv in zip(lhs_values, rhs_values)) and any(
        lv > rv for lv, rv in zip(lhs_values, rhs_values)
    )


def compute_pareto_frontier(
    records: Sequence[FrontierRecord],
    objective_names: Sequence[str],
) -> list[FrontierRecord]:
    """Compute the non-dominated frontier in canonical objective space.

    Returned records are sorted deterministically by ``candidate_id``.
    Duplicate candidate IDs are rejected because they make frontier persistence
    ambiguous.
    """

    names = tuple(objective_names)
    if not names:
        raise ValueError("objective_names must be non-empty")

    seen_ids: set[str] = set()
    for record in records:
        if record.candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id '{record.candidate_id}'")
        seen_ids.add(record.candidate_id)
        _objective_vector(
            record.canonical_objective_values, names, label=record.candidate_id
        )

    frontier: list[FrontierRecord] = []

    for record in records:
        target = record.canonical_objective_values
        is_dominated = False

        for other in records:
            if other.candidate_id == record.candidate_id:
                continue
            if dominates(other.canonical_objective_values, target, names):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append(record)

    return sorted(frontier, key=lambda item: item.candidate_id)


def crowding_distance(
    record: FrontierRecord,
    population: Sequence[FrontierRecord],
    objective_names: Sequence[str],
) -> float:
    """Compute a deterministic crowding-distance score.

    This is used later for frontier-aware region seeding. Boundary points get
    infinite distance, matching the conventional NSGA-II style intuition and
    the behavior of the legacy MORBO-like code.
    """

    names = tuple(objective_names)
    if not names:
        raise ValueError("objective_names must be non-empty")

    if len(population) <= 2:
        return float("inf")

    candidate_ids = [item.candidate_id for item in population]
    if candidate_ids.count(record.candidate_id) != 1:
        raise ValueError(
            f"record candidate_id '{record.candidate_id}' must appear exactly once in population"
        )

    for item in population:
        _objective_vector(
            item.canonical_objective_values, names, label=item.candidate_id
        )

    distance = 0.0

    for objective_name in names:
        ordered = sorted(
            population,
            key=lambda item: (
                float(item.canonical_objective_values[objective_name]),
                item.candidate_id,
            ),
        )
        ordered_ids = [item.candidate_id for item in ordered]
        idx = ordered_ids.index(record.candidate_id)

        if idx == 0 or idx == len(ordered) - 1:
            return float("inf")

        low = float(ordered[0].canonical_objective_values[objective_name])
        high = float(ordered[-1].canonical_objective_values[objective_name])
        span = high - low

        if abs(span) <= 1.0e-15:
            continue

        prev_value = float(ordered[idx - 1].canonical_objective_values[objective_name])
        next_value = float(ordered[idx + 1].canonical_objective_values[objective_name])
        distance += (next_value - prev_value) / span

    return float(distance)


def _objective_vector(
    values: Mapping[str, float],
    objective_names: Sequence[str],
    label: str,
) -> tuple[float, ...]:
    vector: list[float] = []

    for objective_name in objective_names:
        if objective_name not in values:
            raise KeyError(f"{label} is missing objective '{objective_name}'")

        value = float(values[objective_name])
        if not math.isfinite(value):
            raise ValueError(f"{label} objective '{objective_name}' is not finite")

        vector.append(value)

    return tuple(vector)
