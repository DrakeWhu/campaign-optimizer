from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from campaign_optimizer.capillary import observations as capillary_observations
from campaign_optimizer.capillary.recommend import (
    _morbo_trials_from_history as capillary_morbo_trials,
)
from campaign_optimizer.capillary.recommend import (
    _proposal_region_map_from_state as capillary_proposal_region_map,
)
from campaign_optimizer.morbo.backend import MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import TrialInput
from campaign_optimizer.morbo.regions import RegionalPolicy
from campaign_optimizer.morbo.search_space import Choice, FloatRange
from campaign_optimizer.morbo.state import OptimizerState
from campaign_optimizer.multichannel import observations as multichannel_observations
from campaign_optimizer.multichannel.recommend import (
    _morbo_trials as multichannel_morbo_trials,
)
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
ORIGINAL_SIGNATURE = (
    '{"diameter_um_num":"469.747987845673",'
    '"focus_mm_num":"0.468768328197426",'
    '"laser_case":"f32",'
    '"n0_1e18cm3":"4.93481526543",'
    '"plateau_mm_num":"5.46856213611761"}'
)
RECOMPUTED_SIGNATURE = (
    '{"diameter_um_num":"469.747987845673",'
    '"focus_mm_num":"0.468768328197425",'
    '"laser_case":"f32",'
    '"n0_1e18cm3":"4.93481526543",'
    '"plateau_mm_num":"5.46856213611761"}'
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


def _single_spec():
    return load_objective_spec(
        spec_data={
            "objectives": [
                {"name": "score", "metric": "score", "sense": "max"},
            ]
        }
    )


def _capillary_parameter_space() -> dict[str, object]:
    return {
        "laser_cases": ["f20", "f32", "f40"],
        "ranges": {
            "n0_1e18cm3": [1.0, 10.0],
            "plateau_mm_num": [1.0, 20.0],
            "diameter_um_num": [100.0, 600.0],
            "focus_mm_num": [-5.0, 5.0],
        },
    }


def _backend() -> MorboLikeBackend:
    return MorboLikeBackend(
        {
            "laser_case": Choice(("f20", "f32", "f40")),
            "n0_1e18cm3": FloatRange(1.0, 10.0),
            "plateau_mm_num": FloatRange(1.0, 20.0),
            "diameter_um_num": FloatRange(100.0, 600.0),
            "focus_mm_num": FloatRange(-5.0, 5.0),
        },
        _single_spec(),
        seed=17,
        min_observations=2,
        regional_policy=_policy(),
    )


def _pending_state(*records: tuple[str, str]) -> OptimizerState:
    return OptimizerState(
        extra={
            "pending_proposals": [
                {"candidate_signature": signature, "region_id": region_id}
                for signature, region_id in records
            ]
        }
    )


def _trial(
    candidate_id: str,
    params: dict[str, object],
    *,
    score: float = 690.829069606,
    candidate_signature: str = "",
) -> TrialInput:
    metadata = {}
    if candidate_signature:
        metadata["candidate_signature"] = candidate_signature
    return TrialInput(
        candidate_id=candidate_id,
        params=params,
        raw_metrics={"score": score},
        simulation_status="finished",
        metadata=metadata,
    )


class CandidateSignatureRegionFeedbackTests(unittest.TestCase):
    def test_region_feedback_uses_persisted_signature_after_float_roundtrip(self) -> None:
        backend = _backend()
        original = backend.space_codec.signature(PROPOSED_PARAMS)
        recomputed = backend.space_codec.signature(OBSERVED_PARAMS)

        self.assertEqual(original, ORIGINAL_SIGNATURE)
        self.assertEqual(recomputed, RECOMPUTED_SIGNATURE)
        self.assertNotEqual(recomputed, original)
        self.assertEqual(OBSERVED_FOCUS - PROPOSED_FOCUS, -5.551115123125783e-17)

        state = _pending_state((original, "region_000"))
        trial = _trial(
            "campaign:morbo_001_004",
            OBSERVED_PARAMS,
            candidate_signature=original,
        )

        for mapper in (
            capillary_proposal_region_map,
            multichannel_proposal_region_map,
        ):
            with self.subTest(mapper=mapper.__module__):
                self.assertEqual(
                    mapper(state, backend=backend, trials=[trial]),
                    {"campaign:morbo_001_004": "region_000"},
                )

    def test_region_feedback_without_provenance_falls_back_to_recomputed_signature(
        self,
    ) -> None:
        backend = _backend()
        original = backend.space_codec.signature(PROPOSED_PARAMS)
        state = _pending_state((original, "region_000"))
        trial = _trial("historical:case", PROPOSED_PARAMS)

        for mapper in (
            capillary_proposal_region_map,
            multichannel_proposal_region_map,
        ):
            with self.subTest(mapper=mapper.__module__):
                self.assertEqual(
                    mapper(state, backend=backend, trials=[trial]),
                    {"historical:case": "region_000"},
                )

    def test_region_feedback_does_not_use_numeric_tolerance_without_provenance(
        self,
    ) -> None:
        backend = _backend()
        original = backend.space_codec.signature(PROPOSED_PARAMS)
        state = _pending_state((original, "region_000"))
        trial = _trial("roundtrip:no-provenance", OBSERVED_PARAMS)

        self.assertNotEqual(
            backend.space_codec.signature(trial.params),
            original,
        )
        for mapper in (
            capillary_proposal_region_map,
            multichannel_proposal_region_map,
        ):
            with self.subTest(mapper=mapper.__module__):
                self.assertEqual(
                    mapper(state, backend=backend, trials=[trial]),
                    {},
                )

    def test_distinct_persisted_signatures_are_not_merged(self) -> None:
        backend = _backend()
        first_params = dict(PROPOSED_PARAMS)
        second_params = {**PROPOSED_PARAMS, "focus_mm_num": PROPOSED_FOCUS + 1.0e-10}
        first_signature = backend.space_codec.signature(first_params)
        second_signature = backend.space_codec.signature(second_params)
        self.assertNotEqual(first_signature, second_signature)

        state = _pending_state(
            (first_signature, "region_000"),
            (second_signature, "region_001"),
        )
        trials = [
            _trial("first", OBSERVED_PARAMS, candidate_signature=first_signature),
            _trial("second", second_params, candidate_signature=second_signature),
        ]

        for mapper in (
            capillary_proposal_region_map,
            multichannel_proposal_region_map,
        ):
            with self.subTest(mapper=mapper.__module__):
                self.assertEqual(
                    mapper(state, backend=backend, trials=trials),
                    {"first": "region_000", "second": "region_001"},
                )

    def test_persisted_signature_feedback_is_not_double_counted(self) -> None:
        backend = _backend()
        baseline = [
            _trial(
                "base_low",
                {**PROPOSED_PARAMS, "focus_mm_num": -1.0},
                score=500.0,
            ),
            _trial(
                "incumbent",
                {**PROPOSED_PARAMS, "focus_mm_num": 0.0},
                score=767.44,
            ),
        ]
        backend.sync(baseline)

        original = backend.space_codec.signature(PROPOSED_PARAMS)
        state = backend.state.with_updates(
            extra={
                **dict(backend.state.extra),
                "pending_proposals": [
                    {
                        "candidate_signature": original,
                        "region_id": "region_000",
                    }
                ],
            }
        )
        champion = _trial(
            "new_champion",
            OBSERVED_PARAMS,
            score=800.0,
            candidate_signature=original,
        )
        trials = [*baseline, champion]
        proposal_map = capillary_proposal_region_map(
            state,
            backend=backend,
            trials=trials,
        )

        first = backend.sync(trials, proposal_region_map=proposal_map).regions[0]
        second = backend.sync(trials, proposal_region_map=proposal_map).regions[0]

        self.assertEqual(first.metadata["last_outcome"], "success")
        self.assertIn("new_champion", first.metadata["processed_candidate_ids"])
        self.assertEqual(second.success_streak, first.success_streak)
        self.assertEqual(second.failure_streak, first.failure_streak)
        self.assertAlmostEqual(second.radius, first.radius)
        self.assertEqual(
            second.metadata["processed_candidate_ids"],
            first.metadata["processed_candidate_ids"],
        )


class CandidateSignatureObservationPipelineTests(unittest.TestCase):
    def test_capillary_observation_pipeline_preserves_opt_candidate_signature(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_root = root / "campaign"
            case_dir = campaign_root / "case_004"
            case_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "CASE_ID": 4,
                        "CASE_NAME": "case_004",
                        "OPT_CANDIDATE_SIGNATURE": ORIGINAL_SIGNATURE,
                    }
                ]
            ).to_csv(campaign_root / "cases.tsv", sep="\t", index=False)

            config = MagicMock()
            config.base_dir = root
            config.data = {"allow_case_name_parameter_fallback": False}
            config.iteration_dir.return_value = root / "optimizer_runs" / "iter_002"
            config.parameter_space.return_value = _capillary_parameter_space()
            config.validation_config.return_value = {"require_case_validation": False}
            config.source_campaigns_for_iteration.return_value = [
                {
                    "campaign_root": "campaign",
                    "campaign_name": "campaign",
                    "reduced_outputs": {
                        "particle_summary": "",
                        "soft50_curves": "",
                        "guiding_metrics": "",
                        "guiding_singlecase_score": "",
                        "acceptance_curves": "",
                    },
                }
            ]

            parsed_params = {**OBSERVED_PARAMS, "plasma_kind": "chan"}
            with patch.object(
                capillary_observations,
                "canonical_parameters_from_case_row",
                return_value=parsed_params,
            ):
                path = capillary_observations.build_observations(config, 2)

            observations = pd.read_csv(path)
            self.assertEqual(observations.loc[0, "candidate_signature"], ORIGINAL_SIGNATURE)

            history = observations.copy()
            history["score"] = 690.829069606
            trials = capillary_morbo_trials(
                history,
                objective_names=("score",),
                parameter_space=_capillary_parameter_space(),
            )
            self.assertEqual(
                trials[0].metadata["candidate_signature"],
                ORIGINAL_SIGNATURE,
            )

            backend = _backend()
            self.assertEqual(
                capillary_proposal_region_map(
                    _pending_state((ORIGINAL_SIGNATURE, "region_000")),
                    backend=backend,
                    trials=trials,
                ),
                {trials[0].candidate_id: "region_000"},
            )

    def test_multichannel_observation_pipeline_preserves_opt_candidate_signature(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_root = root / "campaign"
            case_dir = campaign_root / "case_004"
            case_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "CASE_ID": 4,
                        "CASE_NAME": "case_004",
                        "OPT_CANDIDATE_SIGNATURE": ORIGINAL_SIGNATURE,
                    }
                ]
            ).to_csv(campaign_root / "cases.tsv", sep="\t", index=False)

            config = MagicMock()
            config.base_dir = root
            config.iteration_dir.return_value = root / "optimizer_runs" / "iter_002"
            config.parameter_space.return_value = {"test": True}
            config.validation_config.return_value = {"require_case_validation": False}
            config.source_campaigns_for_iteration.return_value = [
                {
                    "campaign_root": "campaign",
                    "campaign_name": "campaign",
                    "reduced_outputs": {},
                }
            ]

            with (
                patch.object(
                    multichannel_observations,
                    "parameter_names",
                    return_value=["x"],
                ),
                patch.object(
                    multichannel_observations,
                    "parameters_from_case_row",
                    return_value={"x": OBSERVED_FOCUS},
                ),
                patch.object(
                    multichannel_observations,
                    "read_first_row_csv",
                    return_value=({}, "ok", ""),
                ),
            ):
                path = multichannel_observations.build_observations(config, 2)

            observations = pd.read_csv(path)
            self.assertEqual(observations.loc[0, "candidate_signature"], ORIGINAL_SIGNATURE)

            history = observations.copy()
            history["score"] = 690.829069606
            with patch(
                "campaign_optimizer.multichannel.recommend.parameter_names",
                return_value=["x"],
            ):
                trials = multichannel_morbo_trials(
                    history,
                    parameter_space={"test": True},
                    objective_spec=_single_spec(),
                )
            self.assertEqual(
                trials[0].metadata["candidate_signature"],
                ORIGINAL_SIGNATURE,
            )

            backend = MorboLikeBackend(
                {"x": FloatRange(-5.0, 5.0)},
                _single_spec(),
                seed=17,
                min_observations=2,
                regional_policy=_policy(),
            )
            self.assertEqual(
                multichannel_proposal_region_map(
                    _pending_state((ORIGINAL_SIGNATURE, "region_000")),
                    backend=backend,
                    trials=trials,
                ),
                {trials[0].candidate_id: "region_000"},
            )


if __name__ == "__main__":
    unittest.main()
