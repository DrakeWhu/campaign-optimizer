from __future__ import annotations

import unittest

from campaign_optimizer.morbo.frontier import FrontierRecord
from campaign_optimizer.morbo.regions import (
    ACTIVE,
    INACTIVE,
    RESTARTING,
    RegionRecord,
    RegionalPolicy,
    assign_observations_to_regions,
    choose_region_center,
    distance_sq,
    finalize_region_counts,
    initialize_regions,
    region_bounds,
    reseeding_candidates,
    restart_regions,
    update_regions,
)
from campaign_optimizer.morbo.search_space import Choice, FloatRange, SearchSpaceCodec


def _record(candidate_id: str, energy: float, emittance: float) -> FrontierRecord:
    return FrontierRecord(
        candidate_id=candidate_id,
        objective_values={
            "beam_energy": energy,
            "beam_emittance": emittance,
        },
        canonical_objective_values={
            "beam_energy": energy,
            "beam_emittance": -emittance,
        },
    )


class TestMorboRegions(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = SearchSpaceCodec(
            {
                "geom.n0": FloatRange(1.0, 10.0),
                "ion.state": Choice([1, 2]),
            }
        )
        self.params = {
            "case_001": {"geom.n0": 2.0, "ion.state": 1},
            "case_002": {"geom.n0": 9.0, "ion.state": 2},
            "case_003": {"geom.n0": 8.5, "ion.state": 2},
            "case_004": {"geom.n0": 2.2, "ion.state": 1},
        }
        self.records = [
            _record("case_001", 80.0, 5.0),
            _record("case_002", 140.0, 1.0),
            _record("case_003", 130.0, 2.0),
            _record("case_004", 70.0, 6.0),
        ]
        self.frontier = [self.records[1]]

    def test_region_policy_validation_and_roundtrip(self) -> None:
        policy = RegionalPolicy(max_regions=3, initial_radius=0.2)
        self.assertEqual(RegionalPolicy.from_dict(policy.as_dict()), policy)

        with self.assertRaisesRegex(ValueError, "max_regions"):
            RegionalPolicy(max_regions=0)
        with self.assertRaisesRegex(ValueError, "radii"):
            RegionalPolicy(min_radius=0.3, initial_radius=0.2)
        with self.assertRaisesRegex(ValueError, "contraction_factor"):
            RegionalPolicy(contraction_factor=1.0)

    def test_region_record_validation_and_roundtrip(self) -> None:
        region = RegionRecord(
            "region_000",
            ACTIVE,
            center=[0.5, 1.0, 0.0],
            radius=0.2,
            metadata={"seed_candidate_id": "case_001"},
        )

        self.assertEqual(region.center, (0.5, 1.0, 0.0))
        self.assertEqual(RegionRecord.from_dict(region.as_dict()), region)

        with self.assertRaisesRegex(ValueError, "unsupported region status"):
            RegionRecord("region_bad", "bad", [0.5], 0.2)
        with self.assertRaisesRegex(ValueError, r"inside \[0, 1\]"):
            RegionRecord("region_bad", ACTIVE, [1.2], 0.2)

    def test_initialize_regions_uses_frontier_seeds_first(self) -> None:
        policy = RegionalPolicy(max_regions=2, initial_radius=0.2)
        regions = initialize_regions(
            self.records,
            self.frontier,
            self.params,
            self.codec,
            policy,
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].region_id, "region_000")
        self.assertEqual(regions[0].metadata["seed_candidate_id"], "case_002")
        self.assertEqual(
            regions[0].center, tuple(self.codec.encode(self.params["case_002"]))
        )
        self.assertEqual(regions[1].metadata["seed_candidate_id"], "case_001")

    def test_assign_observations_to_nearest_active_region_is_deterministic(
        self,
    ) -> None:
        regions = [
            RegionRecord(
                "region_000", ACTIVE, self.codec.encode(self.params["case_001"]), 0.2
            ),
            RegionRecord(
                "region_001", ACTIVE, self.codec.encode(self.params["case_002"]), 0.2
            ),
        ]

        assignments = assign_observations_to_regions(
            self.records,
            self.params,
            self.codec,
            regions,
        )

        self.assertEqual(assignments["case_001"], "region_000")
        self.assertEqual(assignments["case_002"], "region_001")
        self.assertEqual(assignments["case_003"], "region_001")
        self.assertEqual(assignments["case_004"], "region_000")

    def test_region_bounds_are_clipped_to_unit_box(self) -> None:
        region = RegionRecord("region_000", ACTIVE, [0.05, 0.5, 0.95], 0.1)

        self.assertEqual(
            region_bounds(region),
            [(0.0, 0.15000000000000002), (0.4, 0.6), (0.85, 1.0)],
        )

    def test_region_success_uses_attributed_proposals_not_geometric_assignment(
        self,
    ) -> None:
        regions = [
            RegionRecord(
                "region_000", ACTIVE, self.codec.encode(self.params["case_001"]), 0.2
            ),
            RegionRecord(
                "region_001", ACTIVE, self.codec.encode(self.params["case_002"]), 0.2
            ),
        ]
        assignments = {"case_002": "region_000"}
        proposal_region_map = {"case_002": "region_001"}

        updated = update_regions(
            regions,
            [self.records[1]],
            self.frontier,
            assignments,
            proposal_region_map,
            self.params,
            self.codec,
            RegionalPolicy(max_regions=2, initial_radius=0.2, expansion_factor=1.5),
        )

        region0 = next(region for region in updated if region.region_id == "region_000")
        region1 = next(region for region in updated if region.region_id == "region_001")

        self.assertNotEqual(region0.metadata.get("last_outcome"), "success")
        self.assertEqual(region1.metadata.get("last_outcome"), "success")
        self.assertGreater(region1.radius, regions[1].radius)

    def test_region_failure_contracts_and_processed_ids_prevent_double_count(
        self,
    ) -> None:
        region = RegionRecord(
            "region_000",
            ACTIVE,
            self.codec.encode(self.params["case_001"]),
            0.2,
        )
        policy = RegionalPolicy(
            max_regions=1, initial_radius=0.2, contraction_factor=0.5
        )

        first = update_regions(
            [region],
            [self.records[3]],
            self.frontier,
            {"case_004": "region_000"},
            {"case_004": "region_000"},
            self.params,
            self.codec,
            policy,
        )[0]
        second = update_regions(
            [first],
            [self.records[3]],
            self.frontier,
            {"case_004": "region_000"},
            {"case_004": "region_000"},
            self.params,
            self.codec,
            policy,
        )[0]

        self.assertEqual(first.metadata["last_outcome"], "failure")
        self.assertLess(first.radius, region.radius)
        self.assertEqual(first.failure_streak, 1)
        self.assertEqual(second.failure_streak, 1)
        self.assertEqual(second.radius, first.radius)

    def test_region_restart_reseeds_frontier_aware_and_deterministic(self) -> None:
        regions = [
            RegionRecord(
                "region_000",
                RESTARTING,
                self.codec.encode(self.params["case_001"]),
                0.1,
                failure_streak=3,
                metadata={"seed_candidate_id": "case_001"},
            ),
            RegionRecord(
                "region_001",
                ACTIVE,
                self.codec.encode(self.params["case_002"]),
                0.2,
                metadata={"seed_candidate_id": "case_002"},
            ),
        ]
        frontier = [self.records[1], self.records[2]]

        candidates = reseeding_candidates(
            self.records, frontier, regions, self.params, self.codec
        )
        self.assertEqual(candidates[0].candidate_id, "case_003")

        restarted = restart_regions(
            regions,
            self.records,
            frontier,
            self.params,
            self.codec,
            RegionalPolicy(max_regions=2, initial_radius=0.25),
        )
        region0 = next(
            region for region in restarted if region.region_id == "region_000"
        )

        self.assertEqual(region0.status, ACTIVE)
        self.assertEqual(region0.restart_count, 1)
        self.assertEqual(region0.generation, 1)
        self.assertEqual(region0.metadata["seed_candidate_id"], "case_003")

    def test_restart_without_available_seed_marks_region_inactive(self) -> None:
        regions = [
            RegionRecord(
                "region_000",
                RESTARTING,
                self.codec.encode(self.params["case_001"]),
                0.1,
                metadata={"seed_candidate_id": "case_001"},
            )
        ]

        restarted = restart_regions(
            regions,
            [self.records[0]],
            [self.records[0]],
            self.params,
            self.codec,
            RegionalPolicy(max_regions=1),
        )

        self.assertEqual(restarted[0].status, INACTIVE)
        self.assertEqual(restarted[0].metadata["last_outcome"], "inactive_no_seed")

    def test_choose_region_center_prefers_frontier_then_closest(self) -> None:
        current_center = self.codec.encode({"geom.n0": 8.4, "ion.state": 2})
        chosen = choose_region_center(
            [self.records[1], self.records[2]],
            current_center,
            self.params,
            self.codec,
            frontier_ids={"case_002", "case_003"},
        )

        self.assertEqual(chosen.candidate_id, "case_003")

    def test_finalize_region_counts_refreshes_observation_and_frontier_counts(
        self,
    ) -> None:
        regions = [
            RegionRecord(
                "region_000", ACTIVE, self.codec.encode(self.params["case_001"]), 0.2
            ),
            RegionRecord(
                "region_001", ACTIVE, self.codec.encode(self.params["case_002"]), 0.2
            ),
        ]
        assignments = {
            "case_001": "region_000",
            "case_002": "region_001",
            "case_003": "region_001",
        }

        finalized = finalize_region_counts(
            regions, self.records[:3], self.frontier, assignments
        )
        by_id = {region.region_id: region for region in finalized}

        self.assertEqual(by_id["region_000"].observation_count, 1)
        self.assertEqual(by_id["region_000"].frontier_count, 0)
        self.assertEqual(by_id["region_001"].observation_count, 2)
        self.assertEqual(by_id["region_001"].frontier_count, 1)

    def test_distance_dimension_mismatch_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "different dimensions"):
            distance_sq([0.0, 1.0], [0.0])


if __name__ == "__main__":
    unittest.main()
