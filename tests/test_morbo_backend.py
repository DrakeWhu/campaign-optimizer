from __future__ import annotations

import unittest

from campaign_optimizer.morbo.backend import CandidateProposal, MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import TrialInput
from campaign_optimizer.morbo.regions import ACTIVE, RegionalPolicy, region_bounds
from campaign_optimizer.morbo.search_space import Choice, FloatRange, SearchSpaceCodec
from campaign_optimizer.morbo.state import CandidateRegistry, OptimizerState


def _trial(
    candidate_id: str,
    n0: float,
    ion_state: int,
    energy: float,
    emittance: float,
    status: str = "finished",
) -> TrialInput:
    return TrialInput(
        candidate_id=candidate_id,
        params={"geom.n0": n0, "ion.state": ion_state},
        raw_metrics={"mean_energy": energy, "emittance_x": emittance},
        simulation_status=status,
    )


class TestMorboLikeBackendNoBotorch(unittest.TestCase):
    def setUp(self) -> None:
        self.space = {
            "geom.n0": FloatRange(1.0, 10.0),
            "ion.state": Choice([1, 2]),
        }
        self.spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                    {"name": "beam_emittance", "metric": "emittance_x", "sense": "min"},
                ]
            }
        )
        self.trials = [
            _trial("case_001", 2.0, 1, 80.0, 5.0),
            _trial("case_002", 9.0, 2, 140.0, 1.0),
            _trial("case_003", 8.5, 2, 130.0, 2.0),
            _trial("case_004", 2.2, 1, 70.0, 6.0),
        ]

    def test_sync_builds_observations_frontier_regions_and_state(self) -> None:
        backend = MorboLikeBackend(
            self.space,
            self.spec,
            seed=11,
            min_observations=2,
            regional_policy=RegionalPolicy(max_regions=1, initial_radius=0.2),
        )

        result = backend.sync(self.trials)

        self.assertEqual(result.observed_count, 4)
        self.assertEqual(result.skipped_count, 0)
        self.assertEqual(
            result.training_data.objective_names, ("beam_energy", "beam_emittance")
        )
        self.assertEqual(
            [record.candidate_id for record in result.frontier], ["case_002"]
        )
        self.assertEqual(len(result.regions), 1)
        self.assertEqual(result.regions[0].status, ACTIVE)
        self.assertEqual(result.regions[0].metadata["seed_candidate_id"], "case_002")
        self.assertEqual(result.state.regions, result.regions)

    def test_backend_falls_back_to_global_random_when_observations_are_insufficient(
        self,
    ) -> None:
        backend = MorboLikeBackend(
            self.space,
            self.spec,
            seed=5,
            min_observations=10,
        )
        backend.sync([])

        proposals = backend.suggest_proposals(2)

        self.assertEqual(len(proposals), 2)
        self.assertEqual(
            {proposal.strategy for proposal in proposals}, {"global_random"}
        )
        self.assertEqual(backend.last_strategy, "global_random")
        self.assertNotEqual(
            proposals[0].candidate_signature, proposals[1].candidate_signature
        )

    def test_pending_trials_are_not_resuggested(self) -> None:
        codec = SearchSpaceCodec({"ion.state": Choice([1, 2])})
        pending_signature = codec.signature({"ion.state": 1})
        state = OptimizerState(
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=(pending_signature,),
                pending_candidate_signatures=(pending_signature,),
            )
        )
        backend = MorboLikeBackend(
            {"ion.state": Choice([1, 2])},
            self.spec,
            seed=7,
            min_observations=10,
            state=state,
        )
        backend.sync([])

        proposals = backend.suggest_proposals(1)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].params, {"ion.state": 2})

    def test_register_pending_blocks_future_suggestions(self) -> None:
        backend = MorboLikeBackend(
            {"ion.state": Choice([1, 2])},
            self.spec,
            seed=7,
            min_observations=10,
        )
        backend.sync([])

        first = backend.suggest_proposals(1)
        backend.register_pending(first)
        second = backend.suggest_proposals(1)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].params, second[0].params)
        self.assertIn(
            first[0].candidate_signature,
            backend.state.candidate_registry.pending_candidate_signatures,
        )
        self.assertIn(
            first[0].candidate_signature,
            backend.state.candidate_registry.seen_candidate_signatures,
        )

    def test_regional_random_suggestions_respect_active_region_bounds(self) -> None:
        backend = MorboLikeBackend(
            self.space,
            self.spec,
            seed=13,
            min_observations=2,
            regional_policy=RegionalPolicy(max_regions=1, initial_radius=0.1),
        )
        backend.sync(self.trials)

        proposal = backend.suggest_proposals(1)[0]
        region = backend.regions[0]
        encoded = backend.space_codec.encode(proposal.params)
        bounds = region_bounds(region)

        self.assertEqual(proposal.strategy, "regional_random")
        self.assertEqual(proposal.region_id, region.region_id)
        for idx, value in enumerate(encoded):
            self.assertGreaterEqual(value, bounds[idx][0] - 1.0e-12)
            self.assertLessEqual(value, bounds[idx][1] + 1.0e-12)

    def test_regional_suggestion_fills_from_global_if_region_is_exhausted(self) -> None:
        # One categorical dimension gives only two possible candidates.
        # Both observed candidates consume the full regional/global space, so no
        # new candidate is possible.
        tiny_space = {"ion.state": Choice([1, 2])}
        trials = [
            TrialInput(
                "case_001",
                {"ion.state": 1},
                {"mean_energy": 80.0, "emittance_x": 5.0},
                "finished",
            ),
            TrialInput(
                "case_002",
                {"ion.state": 2},
                {"mean_energy": 140.0, "emittance_x": 1.0},
                "finished",
            ),
        ]

        backend = MorboLikeBackend(
            tiny_space,
            self.spec,
            seed=1,
            min_observations=2,
            regional_policy=RegionalPolicy(max_regions=1),
        )
        backend.sync(trials)

        proposals = backend.suggest_proposals(1)

        self.assertEqual(proposals, [])

    def test_sync_keeps_running_pending_signature_but_clears_finished_pending_signature(
        self,
    ) -> None:
        codec = SearchSpaceCodec(self.space)
        running_sig = codec.signature({"geom.n0": 3.0, "ion.state": 1})
        finished_sig = codec.signature({"geom.n0": 4.0, "ion.state": 2})

        state = OptimizerState(
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=(running_sig, finished_sig),
                pending_candidate_signatures=(running_sig, finished_sig),
            )
        )
        backend = MorboLikeBackend(
            self.space,
            self.spec,
            seed=3,
            min_observations=10,
            state=state,
        )

        backend.sync(
            [
                TrialInput(
                    "running_case",
                    {"geom.n0": 3.0, "ion.state": 1},
                    {"mean_energy": 50.0, "emittance_x": 5.0},
                    "running",
                ),
                TrialInput(
                    "finished_case",
                    {"geom.n0": 4.0, "ion.state": 2},
                    {"mean_energy": 100.0, "emittance_x": 2.0},
                    "finished",
                ),
            ]
        )

        self.assertIn(
            running_sig, backend.state.candidate_registry.pending_candidate_signatures
        )
        self.assertNotIn(
            finished_sig, backend.state.candidate_registry.pending_candidate_signatures
        )
        self.assertIn(
            running_sig, backend.state.candidate_registry.seen_candidate_signatures
        )
        self.assertIn(
            finished_sig, backend.state.candidate_registry.seen_candidate_signatures
        )

    def test_custom_success_status_is_used_for_sync_and_pending_clear(self) -> None:
        codec = SearchSpaceCodec(self.space)
        signature = codec.signature({"geom.n0": 4.0, "ion.state": 2})
        state = OptimizerState(
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=(signature,),
                pending_candidate_signatures=(signature,),
            )
        )
        backend = MorboLikeBackend(
            self.space,
            self.spec,
            seed=3,
            min_observations=10,
            state=state,
        )

        result = backend.sync(
            [
                TrialInput(
                    "case_001",
                    {"geom.n0": 4.0, "ion.state": 2},
                    {"mean_energy": 100.0, "emittance_x": 2.0},
                    "reduced_validated",
                )
            ],
            success_statuses={"reduced_validated"},
        )

        self.assertEqual(result.observed_count, 1)
        self.assertNotIn(
            signature, backend.state.candidate_registry.pending_candidate_signatures
        )

    def test_suggest_zero_or_negative_returns_empty_list(self) -> None:
        backend = MorboLikeBackend(self.space, self.spec, seed=1)
        self.assertEqual(backend.suggest_proposals(0), [])
        self.assertEqual(backend.suggest_proposals(-1), [])
        self.assertEqual(backend.suggest(0), [])

    def test_candidate_proposal_as_dict(self) -> None:
        proposal = CandidateProposal(
            params={"x": 1.0},
            candidate_signature="sig",
            region_id="region_000",
            strategy="regional_random",
        )

        self.assertEqual(
            proposal.as_dict(),
            {
                "params": {"x": 1.0},
                "candidate_signature": "sig",
                "region_id": "region_000",
                "strategy": "regional_random",
            },
        )


if __name__ == "__main__":
    unittest.main()
