from __future__ import annotations

import math
import unittest

from campaign_optimizer.morbo.frontier import (
    FrontierRecord,
    compute_pareto_frontier,
    crowding_distance,
    dominates,
)


def _record(
    candidate_id: str,
    energy: float,
    emittance: float,
) -> FrontierRecord:
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


class TestMorboFrontier(unittest.TestCase):
    def test_dominates_uses_explicit_objective_order_not_dict_values(self) -> None:
        objective_names = ("beam_energy", "beam_emittance")

        lhs = {
            "beam_energy": 100.0,
            "beam_emittance": -1.0,
        }

        # Same logical objective names, deliberately inserted in reverse order.
        rhs = {
            "beam_emittance": -2.0,
            "beam_energy": 90.0,
        }

        self.assertTrue(dominates(lhs, rhs, objective_names))
        self.assertFalse(dominates(rhs, lhs, objective_names))

    def test_dominates_requires_at_least_one_strict_improvement(self) -> None:
        objective_names = ("a", "b")

        self.assertFalse(
            dominates({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}, objective_names)
        )
        self.assertTrue(
            dominates({"a": 1.0, "b": 3.0}, {"a": 1.0, "b": 2.0}, objective_names)
        )
        self.assertFalse(
            dominates({"a": 0.5, "b": 3.0}, {"a": 1.0, "b": 2.0}, objective_names)
        )

    def test_compute_pareto_frontier_filters_dominated_records(self) -> None:
        records = [
            _record("case_001", energy=100.0, emittance=4.0),
            _record("case_002", energy=120.0, emittance=2.0),
            _record("case_003", energy=90.0, emittance=5.0),
        ]

        frontier = compute_pareto_frontier(
            records,
            objective_names=("beam_energy", "beam_emittance"),
        )

        self.assertEqual([record.candidate_id for record in frontier], ["case_002"])
        self.assertEqual(
            frontier[0].canonical_objective_values,
            {"beam_energy": 120.0, "beam_emittance": -2.0},
        )

    def test_compute_pareto_frontier_keeps_tradeoff_points(self) -> None:
        records = [
            _record("case_001", energy=100.0, emittance=1.0),
            _record("case_002", energy=120.0, emittance=2.0),
            _record("case_003", energy=80.0, emittance=4.0),
        ]

        frontier = compute_pareto_frontier(
            records,
            objective_names=("beam_energy", "beam_emittance"),
        )

        # case_001 has lower energy but better emittance than case_002.
        # case_002 has higher energy but worse emittance than case_001.
        # case_003 is dominated.
        self.assertEqual(
            [record.candidate_id for record in frontier], ["case_001", "case_002"]
        )

    def test_dominated_point_with_better_scalar_sum_does_not_count_as_frontier(
        self,
    ) -> None:
        objective_names = ("beam_energy", "beam_emittance")

        dominant = FrontierRecord(
            candidate_id="dominant",
            objective_values={"beam_energy": 100.0, "beam_emittance": 1.0},
            canonical_objective_values={"beam_energy": 100.0, "beam_emittance": -1.0},
        )
        dominated = FrontierRecord(
            candidate_id="dominated",
            objective_values={"beam_energy": 99.5, "beam_emittance": 1.05},
            canonical_objective_values={"beam_energy": 99.5, "beam_emittance": -1.05},
        )

        # This protects the multi-objective invariant: success/frontier logic
        # must not collapse objectives into a scalar sum.
        self.assertTrue(
            dominates(
                dominant.canonical_objective_values,
                dominated.canonical_objective_values,
                objective_names,
            )
        )

        frontier = compute_pareto_frontier([dominant, dominated], objective_names)
        self.assertEqual([record.candidate_id for record in frontier], ["dominant"])

    def test_crowding_distance_returns_infinity_for_boundary_points(self) -> None:
        records = [
            _record("case_001", energy=100.0, emittance=4.0),
            _record("case_002", energy=110.0, emittance=3.0),
            _record("case_003", energy=120.0, emittance=2.0),
        ]

        self.assertTrue(
            math.isinf(
                crowding_distance(
                    records[0],
                    records,
                    objective_names=("beam_energy", "beam_emittance"),
                )
            )
        )
        self.assertTrue(
            math.isinf(
                crowding_distance(
                    records[2],
                    records,
                    objective_names=("beam_energy", "beam_emittance"),
                )
            )
        )

    def test_crowding_distance_is_finite_for_interior_point(self) -> None:
        records = [
            _record("case_001", energy=100.0, emittance=4.0),
            _record("case_002", energy=110.0, emittance=3.0),
            _record("case_003", energy=120.0, emittance=2.0),
        ]

        distance = crowding_distance(
            records[1],
            records,
            objective_names=("beam_energy", "beam_emittance"),
        )

        self.assertTrue(math.isfinite(distance))
        self.assertGreater(distance, 0.0)

    def test_frontier_validation_rejects_missing_nonfinite_and_duplicate_records(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "objective_names must be non-empty"):
            compute_pareto_frontier([], objective_names=())

        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            compute_pareto_frontier(
                [_record("case_001", 100.0, 2.0), _record("case_001", 120.0, 1.0)],
                objective_names=("beam_energy", "beam_emittance"),
            )

        with self.assertRaisesRegex(KeyError, "missing objective"):
            dominates(
                {"beam_energy": 100.0},
                {"beam_energy": 90.0, "beam_emittance": -2.0},
                objective_names=("beam_energy", "beam_emittance"),
            )

        with self.assertRaisesRegex(ValueError, "not finite"):
            dominates(
                {"beam_energy": float("nan"), "beam_emittance": -1.0},
                {"beam_energy": 90.0, "beam_emittance": -2.0},
                objective_names=("beam_energy", "beam_emittance"),
            )

    def test_crowding_distance_rejects_population_without_target_once(self) -> None:
        records = [
            _record("case_001", energy=100.0, emittance=4.0),
            _record("case_002", energy=110.0, emittance=3.0),
            _record("case_003", energy=120.0, emittance=2.0),
        ]
        missing = _record("case_missing", energy=105.0, emittance=3.5)

        with self.assertRaisesRegex(ValueError, "must appear exactly once"):
            crowding_distance(
                missing,
                records,
                objective_names=("beam_energy", "beam_emittance"),
            )


if __name__ == "__main__":
    unittest.main()
