from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .frontier import FrontierRecord, crowding_distance
from .search_space import SearchSpaceCodec


ACTIVE = "active"
RESTARTING = "restarting"
INACTIVE = "inactive"
VALID_REGION_STATUSES = frozenset({ACTIVE, RESTARTING, INACTIVE})


@dataclass(frozen=True)
class RegionalPolicy:
    max_regions: int = 2
    initial_radius: float = 0.25
    min_radius: float = 0.05
    max_radius: float = 0.5
    expansion_factor: float = 1.2
    contraction_factor: float = 0.7
    restart_failure_threshold: int = 3
    local_min_observations: int = 3

    def __post_init__(self) -> None:
        if int(self.max_regions) < 1:
            raise ValueError("max_regions must be >= 1")
        if int(self.restart_failure_threshold) < 1:
            raise ValueError("restart_failure_threshold must be >= 1")
        if int(self.local_min_observations) < 1:
            raise ValueError("local_min_observations must be >= 1")

        min_radius = float(self.min_radius)
        initial_radius = float(self.initial_radius)
        max_radius = float(self.max_radius)
        expansion_factor = float(self.expansion_factor)
        contraction_factor = float(self.contraction_factor)

        for name, value in {
            "min_radius": min_radius,
            "initial_radius": initial_radius,
            "max_radius": max_radius,
            "expansion_factor": expansion_factor,
            "contraction_factor": contraction_factor,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if min_radius <= 0.0:
            raise ValueError("min_radius must be > 0")
        if max_radius > 1.0:
            raise ValueError("max_radius must be <= 1")
        if not (min_radius <= initial_radius <= max_radius):
            raise ValueError(
                "radii must satisfy min_radius <= initial_radius <= max_radius"
            )
        if expansion_factor <= 1.0:
            raise ValueError("expansion_factor must be > 1")
        if not (0.0 < contraction_factor < 1.0):
            raise ValueError("contraction_factor must be in (0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_regions": self.max_regions,
            "initial_radius": self.initial_radius,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "expansion_factor": self.expansion_factor,
            "contraction_factor": self.contraction_factor,
            "restart_failure_threshold": self.restart_failure_threshold,
            "local_min_observations": self.local_min_observations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegionalPolicy":
        return cls(
            max_regions=int(payload.get("max_regions", cls.max_regions)),
            initial_radius=float(payload.get("initial_radius", cls.initial_radius)),
            min_radius=float(payload.get("min_radius", cls.min_radius)),
            max_radius=float(payload.get("max_radius", cls.max_radius)),
            expansion_factor=float(
                payload.get("expansion_factor", cls.expansion_factor)
            ),
            contraction_factor=float(
                payload.get("contraction_factor", cls.contraction_factor)
            ),
            restart_failure_threshold=int(
                payload.get("restart_failure_threshold", cls.restart_failure_threshold)
            ),
            local_min_observations=int(
                payload.get("local_min_observations", cls.local_min_observations)
            ),
        )


@dataclass(frozen=True)
class RegionRecord:
    region_id: str
    status: str
    center: Sequence[float]
    radius: float
    observation_count: int = 0
    frontier_count: int = 0
    success_streak: int = 0
    failure_streak: int = 0
    restart_count: int = 0
    generation: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        region_id = str(self.region_id).strip()
        status = str(self.status).strip().lower()
        center = tuple(float(value) for value in self.center)
        radius = float(self.radius)

        if not region_id:
            raise ValueError("region_id must be non-empty")
        if status not in VALID_REGION_STATUSES:
            raise ValueError(f"unsupported region status '{status}'")
        if not center:
            raise ValueError("region center must be non-empty")
        if any(
            (not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in center
        ):
            raise ValueError("region center values must be finite and inside [0, 1]")
        if not math.isfinite(radius) or radius <= 0.0 or radius > 1.0:
            raise ValueError("region radius must be finite and inside (0, 1]")

        for name in (
            "observation_count",
            "frontier_count",
            "success_streak",
            "failure_streak",
            "restart_count",
            "generation",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")

        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "observation_count", int(self.observation_count))
        object.__setattr__(self, "frontier_count", int(self.frontier_count))
        object.__setattr__(self, "success_streak", int(self.success_streak))
        object.__setattr__(self, "failure_streak", int(self.failure_streak))
        object.__setattr__(self, "restart_count", int(self.restart_count))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "status": self.status,
            "center": list(self.center),
            "radius": self.radius,
            "observation_count": self.observation_count,
            "frontier_count": self.frontier_count,
            "success_streak": self.success_streak,
            "failure_streak": self.failure_streak,
            "restart_count": self.restart_count,
            "generation": self.generation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegionRecord":
        return cls(
            region_id=str(payload["region_id"]),
            status=str(payload["status"]),
            center=payload["center"],
            radius=float(payload["radius"]),
            observation_count=int(payload.get("observation_count", 0)),
            frontier_count=int(payload.get("frontier_count", 0)),
            success_streak=int(payload.get("success_streak", 0)),
            failure_streak=int(payload.get("failure_streak", 0)),
            restart_count=int(payload.get("restart_count", 0)),
            generation=int(payload.get("generation", 0)),
            metadata=dict(payload.get("metadata", {})),
        )


def region_bounds(region: RegionRecord) -> list[tuple[float, float]]:
    return [
        (
            max(0.0, float(center_value) - region.radius),
            min(1.0, float(center_value) + region.radius),
        )
        for center_value in region.center
    ]


def distance_sq(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(
            f"encoded vectors have different dimensions: {len(lhs)} != {len(rhs)}"
        )
    return sum((float(a) - float(b)) ** 2 for a, b in zip(lhs, rhs))


def initialize_regions(
    observations: Sequence[FrontierRecord],
    frontier: Sequence[FrontierRecord],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
    policy: RegionalPolicy,
) -> list[RegionRecord]:
    seeds = seed_observations(observations, frontier, params_by_candidate_id, codec)
    regions: list[RegionRecord] = []

    for idx, seed in enumerate(seeds[: policy.max_regions]):
        regions.append(
            RegionRecord(
                region_id=f"region_{idx:03d}",
                status=ACTIVE,
                center=codec.encode(
                    _params_for(seed.candidate_id, params_by_candidate_id)
                ),
                radius=policy.initial_radius,
                observation_count=0,
                frontier_count=0,
                success_streak=0,
                failure_streak=0,
                restart_count=0,
                generation=0,
                metadata={
                    "seed_candidate_id": seed.candidate_id,
                    "center_candidate_id": seed.candidate_id,
                    "processed_candidate_ids": [],
                },
            )
        )

    return regions


def seed_observations(
    observations: Sequence[FrontierRecord],
    frontier: Sequence[FrontierRecord],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
) -> list[FrontierRecord]:
    frontier_ids = {record.candidate_id for record in frontier}
    frontier_records = [
        record for record in observations if record.candidate_id in frontier_ids
    ]
    remaining_records = [
        record for record in observations if record.candidate_id not in frontier_ids
    ]

    ranked_frontier = sorted(
        frontier_records,
        key=lambda record: (
            -crowding_distance(
                record,
                frontier_records,
                tuple(record.canonical_objective_values.keys()),
            )
            if len(frontier_records) >= 2
            else 0.0,
            record.candidate_id,
        ),
    )
    ranked_remaining = sorted(remaining_records, key=lambda record: record.candidate_id)

    seen_signatures: set[str] = set()
    seeds: list[FrontierRecord] = []
    for record in [*ranked_frontier, *ranked_remaining]:
        params = _params_for(record.candidate_id, params_by_candidate_id)
        signature = codec.signature(params)
        if signature in seen_signatures:
            continue
        seeds.append(record)
        seen_signatures.add(signature)

    return seeds


def assign_observations_to_regions(
    observations: Sequence[FrontierRecord],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
    regions: Sequence[RegionRecord],
) -> dict[str, str]:
    active_regions = sorted(
        (region for region in regions if region.status == ACTIVE),
        key=lambda region: region.region_id,
    )
    if not active_regions:
        return {}

    assignments: dict[str, str] = {}
    for observation in sorted(observations, key=lambda item: item.candidate_id):
        encoded = codec.encode(
            _params_for(observation.candidate_id, params_by_candidate_id)
        )
        chosen = min(
            active_regions,
            key=lambda region: (distance_sq(encoded, region.center), region.region_id),
        )
        assignments[observation.candidate_id] = chosen.region_id

    return assignments


def update_regions(
    regions: Sequence[RegionRecord],
    observations: Sequence[FrontierRecord],
    frontier: Sequence[FrontierRecord],
    assignments: Mapping[str, str],
    proposal_region_map: Mapping[str, str],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
    policy: RegionalPolicy,
) -> list[RegionRecord]:
    frontier_ids = {record.candidate_id for record in frontier}
    updated: list[RegionRecord] = []

    for region in sorted(regions, key=lambda item: item.region_id):
        assigned = [
            obs
            for obs in observations
            if assignments.get(obs.candidate_id) == region.region_id
        ]
        attributable = [
            obs
            for obs in observations
            if proposal_region_map.get(obs.candidate_id) == region.region_id
        ]
        processed = set(
            str(item) for item in region.metadata.get("processed_candidate_ids", [])
        )
        new_attributable = [
            obs for obs in attributable if obs.candidate_id not in processed
        ]

        observation_count = len(assigned)
        frontier_count = sum(1 for obs in assigned if obs.candidate_id in frontier_ids)
        metadata = dict(region.metadata)
        center = tuple(region.center)
        radius = region.radius
        success_streak = region.success_streak
        failure_streak = region.failure_streak
        status = region.status

        if status == ACTIVE:
            successful = [
                obs for obs in new_attributable if obs.candidate_id in frontier_ids
            ]
            if successful:
                success_streak += 1
                failure_streak = 0
                radius = min(region.radius * policy.expansion_factor, policy.max_radius)
                representative = choose_region_center(
                    successful,
                    current_center=center,
                    params_by_candidate_id=params_by_candidate_id,
                    codec=codec,
                    frontier_ids=frontier_ids,
                )
                center = tuple(
                    codec.encode(
                        _params_for(representative.candidate_id, params_by_candidate_id)
                    )
                )
                metadata["last_outcome"] = "success"
                metadata["center_candidate_id"] = representative.candidate_id
            elif new_attributable:
                success_streak = 0
                failure_streak += 1
                radius = max(
                    region.radius * policy.contraction_factor, policy.min_radius
                )
                metadata["last_outcome"] = "failure"

            if new_attributable:
                metadata["processed_candidate_ids"] = sorted(
                    processed | {obs.candidate_id for obs in new_attributable}
                )

            if (
                radius <= policy.min_radius
                or failure_streak >= policy.restart_failure_threshold
            ):
                status = RESTARTING

        updated.append(
            RegionRecord(
                region_id=region.region_id,
                status=status,
                center=center,
                radius=radius,
                observation_count=observation_count,
                frontier_count=frontier_count,
                success_streak=success_streak,
                failure_streak=failure_streak,
                restart_count=region.restart_count,
                generation=region.generation,
                metadata=metadata,
            )
        )

    return updated


def restart_regions(
    regions: Sequence[RegionRecord],
    observations: Sequence[FrontierRecord],
    frontier: Sequence[FrontierRecord],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
    policy: RegionalPolicy,
) -> list[RegionRecord]:
    available = reseeding_candidates(
        observations, frontier, regions, params_by_candidate_id, codec
    )
    used_seed_ids = {
        str(region.metadata.get("seed_candidate_id"))
        for region in regions
        if region.metadata.get("seed_candidate_id") is not None
    }

    output: list[RegionRecord] = []
    for region in sorted(regions, key=lambda item: item.region_id):
        if region.status != RESTARTING:
            output.append(region)
            continue

        seed = next(
            (item for item in available if item.candidate_id not in used_seed_ids), None
        )
        if seed is None:
            output.append(
                RegionRecord(
                    region_id=region.region_id,
                    status=INACTIVE,
                    center=region.center,
                    radius=region.radius,
                    observation_count=region.observation_count,
                    frontier_count=region.frontier_count,
                    success_streak=0,
                    failure_streak=region.failure_streak,
                    restart_count=region.restart_count,
                    generation=region.generation,
                    metadata={
                        **dict(region.metadata),
                        "last_outcome": "inactive_no_seed",
                    },
                )
            )
            continue

        used_seed_ids.add(seed.candidate_id)
        output.append(
            RegionRecord(
                region_id=region.region_id,
                status=ACTIVE,
                center=codec.encode(
                    _params_for(seed.candidate_id, params_by_candidate_id)
                ),
                radius=policy.initial_radius,
                observation_count=0,
                frontier_count=0,
                success_streak=0,
                failure_streak=0,
                restart_count=region.restart_count + 1,
                generation=region.generation + 1,
                metadata={
                    "seed_candidate_id": seed.candidate_id,
                    "center_candidate_id": seed.candidate_id,
                    "last_outcome": "restart",
                    "processed_candidate_ids": [],
                },
            )
        )

    return output


def reseeding_candidates(
    observations: Sequence[FrontierRecord],
    frontier: Sequence[FrontierRecord],
    regions: Sequence[RegionRecord],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
) -> list[FrontierRecord]:
    frontier_ids = {record.candidate_id for record in frontier}
    active_centers = [region.center for region in regions if region.status == ACTIVE]
    seeds = seed_observations(observations, frontier, params_by_candidate_id, codec)

    return sorted(
        seeds,
        key=lambda record: (
            0 if record.candidate_id in frontier_ids else 1,
            -distance_to_centers(
                codec.encode(_params_for(record.candidate_id, params_by_candidate_id)),
                active_centers,
            ),
            record.candidate_id,
        ),
    )


def choose_region_center(
    candidates: Sequence[FrontierRecord],
    current_center: Sequence[float],
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
    codec: SearchSpaceCodec,
    frontier_ids: set[str] | frozenset[str],
) -> FrontierRecord:
    if not candidates:
        raise ValueError("candidate observations must be non-empty")

    return sorted(
        candidates,
        key=lambda record: (
            0 if record.candidate_id in frontier_ids else 1,
            distance_sq(
                codec.encode(_params_for(record.candidate_id, params_by_candidate_id)),
                current_center,
            ),
            record.candidate_id,
        ),
    )[0]


def finalize_region_counts(
    regions: Sequence[RegionRecord],
    observations: Sequence[FrontierRecord],
    frontier: Sequence[FrontierRecord],
    assignments: Mapping[str, str],
) -> list[RegionRecord]:
    frontier_ids = {record.candidate_id for record in frontier}
    finalized: list[RegionRecord] = []

    for region in sorted(regions, key=lambda item: item.region_id):
        assigned = [
            obs
            for obs in observations
            if assignments.get(obs.candidate_id) == region.region_id
        ]
        finalized.append(
            RegionRecord(
                region_id=region.region_id,
                status=region.status,
                center=region.center,
                radius=region.radius,
                observation_count=len(assigned),
                frontier_count=sum(
                    1 for obs in assigned if obs.candidate_id in frontier_ids
                ),
                success_streak=region.success_streak,
                failure_streak=region.failure_streak,
                restart_count=region.restart_count,
                generation=region.generation,
                metadata=dict(region.metadata),
            )
        )

    return finalized


def distance_to_centers(
    encoded: Sequence[float], centers: Sequence[Sequence[float]]
) -> float:
    if not centers:
        return 0.0
    return min(distance_sq(encoded, center) for center in centers)


def _params_for(
    candidate_id: str,
    params_by_candidate_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if candidate_id not in params_by_candidate_id:
        raise KeyError(f"missing parameters for candidate_id '{candidate_id}'")
    return params_by_candidate_id[candidate_id]
