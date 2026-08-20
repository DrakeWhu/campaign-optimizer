from __future__ import annotations

import unittest

from campaign_optimizer.capillary.recommend import (
    _proposal_region_map_from_state as capillary_proposal_region_map,
)
from campaign_optimizer.morbo.backend import MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import TrialInput
from campaign_optimizer.morbo.regions import RegionalPolicy
from campaign_optimizer.morbo.search_space import Choice, FloatRange
from campaign_optimizer.multichannel.recommend import (
    _proposal_region_map_from_state as multichannel_proposal_region_map,
)


PROPOSED_FOCUS = 0.46876832819742553
OBSERVED_FOCUS = 0.4687683281974255
PROPOSED_PARAMS = {
    "laser_case": "f32",
    "n0_1e18cm3": 4.93481526543,
    "plateau_mm_num": 5.46856213611761,
    "diameter_um_num": 469.747987845673,
    "focus_mm_num": PROPOSED_FOCUS,
}
OBSERVED_PARAMS = {
    **PROPOSED_PARAMS,
    "focus_mm_num": OBSERVED_FOCUS,
}


def _spec():
    return load_objective_spec(
        spec_data={
            "objectives": [
                {"name": "score_a", "metric": "score_a", "sense": "max"},
                {"name": "score_b", "metric": "score_b", "sense": "max"},
            ]
        }
    )


def _policy() -> RegionalPolicy:
    return RegionalPolicy(
        max_regions=1,
        initial_radius=0.25,
        min_radius=0.05,
        max_radius=0.5,
        expansion_factor=1.2,
        contraction_factor=0.7,
        restart_failure_threshold=3,
        local_min_observations=1,
    )


def _backend() -> MorboLikeBackend:
    return MorboLikeBackend(
        {
            "laser_case": Choice(("f20", "f32", "f40")),
            "n0_1e18cm3": FloatRange(1.0, 10.0),
            "plateau_mm_num": FloatRange(1.0, 20.0),
            "diameter_um_num": FloatRange(100.0, 600.0),
            "focus_mm_num": FloatRange(-5.0, 5.0),
        },
        _spec(),
        seed=23,
        min_observations=2,
        regional_policy=_policy(),
    )


def _trial(
    candidate_id: str,
    params: dict[str, object],
    score_a: float,
    score_b: float,
    *,
    signature: str = "",
) -> TrialInput:
    metadata = {"candidate_signature": signature} if signature else {}
    return TrialInput(
        candidate_id=candidate_id,
        params=params,
        raw_metrics={"score_a": score_a, "score_b": score_b},
        simulation_status="finished",
        metadata=metadata,
    )


class CandidateSignatureMultiobjectiveRegressionTests(unittest.TestCase):
    def test_persisted_signature_preserves_multiobjective_region_success(self) -> None:
        for mapper in (
            capillary_proposal_region_map,
            multichannel_proposal_region_map,
        ):
            with self.subTest(mapper=mapper.__module__):
                backend = _backend()
                baseline = [
                    _trial(
                        "pareto_a",
                        {**PROPOSED_PARAMS, "focus_mm_num": -1.0},
                        10.0,
                        0.0,
                    ),
                    _trial(
                        "pareto_b",
                        {**PROPOSED_PARAMS, "focus_mm_num": 1.0},
                        0.0,
                        10.0,
                    ),
                ]
                backend.sync(baseline)

                original_signature = backend.space_codec.signature(PROPOSED_PARAMS)
                self.assertNotEqual(
                    backend.space_codec.signature(OBSERVED_PARAMS),
                    original_signature,
                )
                previous_state = backend.state.with_updates(
                    extra={
                        **dict(backend.state.extra),
                        "pending_proposals": [
                            {
                                "candidate_signature": original_signature,
                                "region_id": "region_000",
                            }
                        ],
                    }
                )
                new_pareto = _trial(
                    "pareto_new",
                    OBSERVED_PARAMS,
                    6.0,
                    6.0,
                    signature=original_signature,
                )
                trials = [*baseline, new_pareto]
                proposal_map = mapper(
                    previous_state,
                    backend=backend,
                    trials=trials,
                )
                self.assertEqual(proposal_map, {"pareto_new": "region_000"})

                result = backend.sync(
                    trials,
                    proposal_region_map=proposal_map,
                )
                region = result.regions[0]

                self.assertEqual(region.metadata["last_outcome"], "success")
                self.assertEqual(region.success_streak, 1)
                self.assertEqual(region.failure_streak, 0)
                self.assertAlmostEqual(region.radius, 0.25 * 1.2)
                self.assertIn(
                    "pareto_new",
                    {record.candidate_id for record in result.frontier},
                )


if __name__ == "__main__":
    unittest.main()
