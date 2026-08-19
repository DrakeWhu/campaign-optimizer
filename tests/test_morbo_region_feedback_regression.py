from __future__ import annotations

import unittest

from campaign_optimizer.morbo.backend import MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import TrialInput
from campaign_optimizer.morbo.regions import RegionalPolicy
from campaign_optimizer.morbo.search_space import FloatRange
from campaign_optimizer.morbo.state import OptimizerState
from campaign_optimizer.multichannel.recommend import _proposal_region_map_from_state


def _policy(*, max_regions: int = 1) -> RegionalPolicy:
    return RegionalPolicy(
        max_regions=max_regions,
        initial_radius=0.25,
        min_radius=0.05,
        max_radius=0.5,
        expansion_factor=1.2,
        contraction_factor=0.7,
        restart_failure_threshold=3,
        local_min_observations=1,
    )


def _single_spec():
    return load_objective_spec(
        spec_data={
            "objectives": [
                {"name": "score", "metric": "score", "sense": "max"},
            ]
        }
    )


def _multi_spec():
    return load_objective_spec(
        spec_data={
            "objectives": [
                {"name": "score_a", "metric": "score_a", "sense": "max"},
                {"name": "score_b", "metric": "score_b", "sense": "max"},
            ]
        }
    )


def _single_trial(candidate_id: str, x: float, score: float) -> TrialInput:
    return TrialInput(
        candidate_id=candidate_id,
        params={"x": x},
        raw_metrics={"score": score},
        simulation_status="finished",
    )


def _multi_trial(
    candidate_id: str,
    x: float,
    score_a: float,
    score_b: float,
) -> TrialInput:
    return TrialInput(
        candidate_id=candidate_id,
        params={"x": x},
        raw_metrics={"score_a": score_a, "score_b": score_b},
        simulation_status="finished",
    )


class TestMorboRegionalFeedbackRegression(unittest.TestCase):
    def _single_backend(self, *, max_regions: int = 1) -> MorboLikeBackend:
        return MorboLikeBackend(
            {"x": FloatRange(0.0, 1.0)},
            _single_spec(),
            seed=17,
            min_observations=2,
            regional_policy=_policy(max_regions=max_regions),
        )

    def _multi_backend(self) -> MorboLikeBackend:
        return MorboLikeBackend(
            {"x": FloatRange(0.0, 1.0)},
            _multi_spec(),
            seed=23,
            min_observations=2,
            regional_policy=_policy(),
        )

    def test_single_objective_dominated_proposal_contracts_region(self) -> None:
        backend = self._single_backend()
        baseline = [
            _single_trial("base_low", 0.1, 5.0),
            _single_trial("incumbent", 0.5, 10.0),
        ]
        initial = backend.sync(baseline)

        self.assertEqual(initial.regions[0].observation_count, 2)
        self.assertEqual(initial.regions[0].frontier_count, 1)

        result = backend.sync(
            [*baseline, _single_trial("dominated", 0.6, 8.0)],
            proposal_region_map={"dominated": "region_000"},
        )
        region = result.regions[0]

        self.assertAlmostEqual(region.radius, 0.25 * 0.7)
        self.assertEqual(region.success_streak, 0)
        self.assertEqual(region.failure_streak, 1)
        self.assertEqual(region.metadata["last_outcome"], "failure")
        self.assertEqual(region.metadata["processed_candidate_ids"], ["dominated"])
        self.assertEqual(region.observation_count, 3)
        self.assertEqual(region.frontier_count, 1)

    def test_single_objective_new_record_expands_and_moves_center(self) -> None:
        backend = self._single_backend()
        baseline = [
            _single_trial("base_low", 0.1, 5.0),
            _single_trial("incumbent", 0.5, 10.0),
        ]
        backend.sync(baseline)

        result = backend.sync(
            [*baseline, _single_trial("new_record", 0.7, 12.0)],
            proposal_region_map={"new_record": "region_000"},
        )
        region = result.regions[0]

        self.assertAlmostEqual(region.radius, 0.25 * 1.2)
        self.assertEqual(region.success_streak, 1)
        self.assertEqual(region.failure_streak, 0)
        self.assertEqual(region.metadata["last_outcome"], "success")
        self.assertEqual(region.metadata["center_candidate_id"], "new_record")
        self.assertEqual(
            [record.candidate_id for record in result.frontier], ["new_record"]
        )

    def test_processed_candidate_is_not_counted_twice(self) -> None:
        backend = self._single_backend()
        trials = [
            _single_trial("base_low", 0.1, 5.0),
            _single_trial("incumbent", 0.5, 10.0),
        ]
        backend.sync(trials)
        trials = [*trials, _single_trial("dominated", 0.6, 8.0)]

        first = backend.sync(
            trials,
            proposal_region_map={"dominated": "region_000"},
        ).regions[0]
        second = backend.sync(
            trials,
            proposal_region_map={"dominated": "region_000"},
        ).regions[0]

        self.assertEqual(second.failure_streak, first.failure_streak)
        self.assertAlmostEqual(second.radius, first.radius)
        self.assertEqual(second.metadata["processed_candidate_ids"], ["dominated"])

    def test_three_failed_batches_restart_region(self) -> None:
        backend = self._single_backend()
        trials = [
            _single_trial("base_low", 0.1, 5.0),
            _single_trial("incumbent", 0.5, 10.0),
        ]
        backend.sync(trials)

        for index, score in enumerate((9.0, 8.0, 7.0), start=1):
            candidate_id = f"failure_{index}"
            trials = [
                *trials,
                _single_trial(candidate_id, 0.5 + 0.05 * index, score),
            ]
            result = backend.sync(
                trials,
                proposal_region_map={candidate_id: "region_000"},
            )

        region = result.regions[0]
        self.assertEqual(region.restart_count, 1)
        self.assertEqual(region.generation, 1)
        self.assertEqual(region.failure_streak, 0)
        self.assertEqual(region.success_streak, 0)
        self.assertAlmostEqual(region.radius, 0.25)
        self.assertEqual(region.metadata["last_outcome"], "restart")

    def test_multiobjective_dominated_proposal_contracts_region(self) -> None:
        backend = self._multi_backend()
        baseline = [
            _multi_trial("pareto_a", 0.2, 10.0, 0.0),
            _multi_trial("pareto_b", 0.8, 0.0, 10.0),
        ]
        backend.sync(baseline)

        result = backend.sync(
            [*baseline, _multi_trial("dominated", 0.3, 4.0, 0.0)],
            proposal_region_map={"dominated": "region_000"},
        )
        region = result.regions[0]

        self.assertEqual(region.metadata["last_outcome"], "failure")
        self.assertEqual(region.failure_streak, 1)
        self.assertAlmostEqual(region.radius, 0.25 * 0.7)
        self.assertNotIn(
            "dominated", {record.candidate_id for record in result.frontier}
        )

    def test_multiobjective_nondominated_proposal_expands_region(self) -> None:
        backend = self._multi_backend()
        baseline = [
            _multi_trial("pareto_a", 0.2, 10.0, 0.0),
            _multi_trial("pareto_b", 0.8, 0.0, 10.0),
        ]
        backend.sync(baseline)

        result = backend.sync(
            [*baseline, _multi_trial("pareto_new", 0.5, 6.0, 6.0)],
            proposal_region_map={"pareto_new": "region_000"},
        )
        region = result.regions[0]

        self.assertEqual(region.metadata["last_outcome"], "success")
        self.assertEqual(region.success_streak, 1)
        self.assertAlmostEqual(region.radius, 0.25 * 1.2)
        self.assertIn(
            "pareto_new", {record.candidate_id for record in result.frontier}
        )

    def test_multiobjective_mixed_batch_is_success_if_any_proposal_is_pareto(self) -> None:
        backend = self._multi_backend()
        baseline = [
            _multi_trial("pareto_a", 0.2, 10.0, 0.0),
            _multi_trial("pareto_b", 0.8, 0.0, 10.0),
        ]
        backend.sync(baseline)

        result = backend.sync(
            [
                *baseline,
                _multi_trial("dominated", 0.3, 4.0, 0.0),
                _multi_trial("pareto_new", 0.5, 6.0, 6.0),
            ],
            proposal_region_map={
                "dominated": "region_000",
                "pareto_new": "region_000",
            },
        )
        region = result.regions[0]

        self.assertEqual(region.metadata["last_outcome"], "success")
        self.assertEqual(region.success_streak, 1)
        self.assertEqual(region.failure_streak, 0)
        self.assertEqual(
            region.metadata["processed_candidate_ids"],
            ["dominated", "pareto_new"],
        )

    def test_multiple_regions_use_proposal_attribution_independently(self) -> None:
        backend = self._single_backend(max_regions=2)
        baseline = [
            _single_trial("base_low", 0.1, 5.0),
            _single_trial("incumbent", 0.9, 10.0),
        ]
        initial = backend.sync(baseline)
        self.assertEqual(len(initial.regions), 2)

        result = backend.sync(
            [
                *baseline,
                _single_trial("region0_success", 0.8, 12.0),
                _single_trial("region1_failure", 0.2, 4.0),
            ],
            proposal_region_map={
                "region0_success": "region_000",
                "region1_failure": "region_001",
            },
        )
        regions = {region.region_id: region for region in result.regions}

        self.assertEqual(regions["region_000"].metadata["last_outcome"], "success")
        self.assertEqual(regions["region_000"].success_streak, 1)
        self.assertAlmostEqual(regions["region_000"].radius, 0.25 * 1.2)

        self.assertEqual(regions["region_001"].metadata["last_outcome"], "failure")
        self.assertEqual(regions["region_001"].failure_streak, 1)
        self.assertAlmostEqual(regions["region_001"].radius, 0.25 * 0.7)


class TestMultichannelRegionAttributionRegression(unittest.TestCase):
    def test_pending_candidate_signature_recovers_region_for_observed_trial(self) -> None:
        backend = MorboLikeBackend(
            {"x": FloatRange(0.0, 1.0)},
            _single_spec(),
            seed=31,
            min_observations=2,
            regional_policy=_policy(),
        )
        params = {"x": 0.42}
        signature = backend.space_codec.signature(params)
        state = OptimizerState(
            extra={
                "pending_proposals": [
                    {
                        "candidate_id": "mc_005_morbo_000",
                        "candidate_signature": signature,
                        "region_id": "region_002",
                        "strategy": "regional_model",
                        "params": params,
                    }
                ]
            }
        )
        trials = [
            TrialInput(
                candidate_id="materialized_case_name",
                params=params,
                raw_metrics={"score": 1.0},
                simulation_status="finished",
            ),
            TrialInput(
                candidate_id="unmatched_case",
                params={"x": 0.11},
                raw_metrics={"score": 0.5},
                simulation_status="finished",
            ),
        ]

        mapping = _proposal_region_map_from_state(
            state,
            backend=backend,
            trials=trials,
        )

        self.assertEqual(mapping, {"materialized_case_name": "region_002"})


if __name__ == "__main__":
    unittest.main()
