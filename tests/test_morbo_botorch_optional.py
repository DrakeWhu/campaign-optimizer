from __future__ import annotations

import unittest

from campaign_optimizer.morbo.botorch_model import (
    BotorchRegionalConfig,
    suggest_botorch_regional_candidates,
)
from campaign_optimizer.morbo.backend import MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import TrialInput
from campaign_optimizer.morbo.regions import RegionalPolicy
from campaign_optimizer.morbo.search_space import Choice, FloatRange


def _trial(
    candidate_id: str,
    n0: float,
    laser: str,
    y1: float,
    y2: float,
) -> TrialInput:
    return TrialInput(
        candidate_id=candidate_id,
        params={"n0": n0, "laser": laser},
        raw_metrics={"score_a": y1, "score_b": y2},
        simulation_status="finished",
    )


class BotorchOptionalTests(unittest.TestCase):
    def test_botorch_config_from_dict_defaults_and_validation(self) -> None:
        cfg = BotorchRegionalConfig.from_dict(
            {
                "enabled": True,
                "candidate_pool_size_per_region": 12,
                "mc_samples": 8,
                "fit_maxiter": 5,
            }
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.candidate_pool_size_per_region, 12)
        self.assertEqual(cfg.mc_samples, 8)
        self.assertEqual(cfg.fit_maxiter, 5)
        self.assertTrue(cfg.fallback_to_random)

        with self.assertRaisesRegex(ValueError, "mc_samples"):
            BotorchRegionalConfig(mc_samples=0)

    def test_disabled_botorch_result_does_not_import_botorch(self) -> None:
        cfg = BotorchRegionalConfig(enabled=False)
        result = suggest_botorch_regional_candidates(
            encoded_X=((0.1, 1.0, 0.0), (0.9, 0.0, 1.0)),
            canonical_Y=((1.0, 2.0), (2.0, 1.0)),
            objective_names=("score_a", "score_b"),
            codec=None,  # type: ignore[arg-type]
            active_regions=(),
            rng=None,  # type: ignore[arg-type]
            n=1,
            blocked_signatures=set(),
            config=cfg,
        )

        self.assertEqual(result.status, "disabled")

    def test_backend_regional_model_mode_falls_back_when_disabled(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "score_a", "metric": "score_a", "sense": "max"},
                    {"name": "score_b", "metric": "score_b", "sense": "max"},
                ]
            }
        )
        backend = MorboLikeBackend(
            {
                "n0": FloatRange(1.0, 10.0),
                "laser": Choice(["f20", "f32"]),
            },
            spec,
            seed=2,
            min_observations=2,
            regional_policy=RegionalPolicy(max_regions=1, initial_radius=0.3),
            suggestion_mode="regional_model",
            botorch_config=BotorchRegionalConfig(enabled=False),
        )
        backend.sync(
            [
                _trial("case_001", 2.0, "f20", 1.0, 1.0),
                _trial("case_002", 8.0, "f32", 2.0, 2.0),
                _trial("case_003", 7.0, "f32", 1.5, 2.5),
            ]
        )

        proposals = backend.suggest_proposals(2)

        self.assertEqual(len(proposals), 2)
        self.assertEqual(
            {proposal.strategy for proposal in proposals}, {"regional_random"}
        )
        self.assertEqual(backend.last_model_diagnostics["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
