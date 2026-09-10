from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from campaign_optimizer import problems
from campaign_optimizer.capillary.candidate_identity import (
    normalize_observation_candidate_signatures,
    normalize_persisted_candidate_signature,
)
from campaign_optimizer.capillary.parameters import capillary_search_space
from campaign_optimizer.capillary.recommend import _morbo_trials_from_history
from campaign_optimizer.morbo.backend import MorboLikeBackend
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.state import CandidateRegistry


PROPOSED_FOCUS = 0.46876832819742553
OBSERVED_FOCUS = 0.4687683281974255

PROPOSED_PARAMS_5D = {
    "laser_case": "f32",
    "n0_1e18cm3": 4.93481526543,
    "plateau_mm_num": 5.46856213611761,
    "diameter_um_num": 469.747987845673,
    "focus_mm_num": PROPOSED_FOCUS,
}
OBSERVED_PARAMS_5D = {
    **PROPOSED_PARAMS_5D,
    "focus_mm_num": OBSERVED_FOCUS,
}
REAL_PR2_SIGNATURE_5D = (
    '{"diameter_um_num":"469.747987845673",'
    '"focus_mm_num":"0.468768328197426",'
    '"laser_case":"f32",'
    '"n0_1e18cm3":"4.93481526543",'
    '"plateau_mm_num":"5.46856213611761"}'
)


def _parameter_space_6d() -> dict[str, object]:
    return {
        "laser_cases": ["f20", "f32", "f40"],
        "ranges": {
            "n0_1e18cm3": [0.7, 6.0],
            "plateau_mm_num": [5.0, 25.0],
            "diameter_um_num": [150.0, 500.0],
            "focus_mm_num": [-5.0, 5.0],
            "nitrogen_fraction": [0.0, 0.01],
        },
    }


def _parameter_space_5d() -> dict[str, object]:
    payload = _parameter_space_6d()
    payload["ranges"] = dict(payload["ranges"])
    del payload["ranges"]["nitrogen_fraction"]
    return payload


def _spec():
    return load_objective_spec(
        spec_data={
            "objectives": [
                {"name": "score", "metric": "score", "sense": "max"},
            ]
        }
    )


def _write_observation_csv(
    path: Path,
    *,
    signature: str,
    focus_text: str = "0.4687683281974255",
) -> None:
    fieldnames = [
        "observation_id",
        "candidate_signature",
        "laser_case",
        "n0_1e18cm3",
        "plateau_mm_num",
        "diameter_um_num",
        "focus_mm_num",
        "nitrogen_fraction",
        "score",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "observation_id": "historical:case",
                "candidate_signature": signature,
                "laser_case": "f32",
                "n0_1e18cm3": "4.93481526543",
                "plateau_mm_num": "5.46856213611761",
                "diameter_um_num": "469.747987845673",
                "focus_mm_num": focus_text,
                "nitrogen_fraction": "0",
                "score": "690.829069606",
            }
        )


class ClpuNitrogenCandidateIdentityTests(unittest.TestCase):
    def test_real_pr2_5d_signature_maps_to_equivalent_f0_6d_signature(self) -> None:
        parameter_space = _parameter_space_6d()
        codec = MorboLikeBackend(
            capillary_search_space(parameter_space),
            _spec(),
            seed=17,
            min_observations=10,
        ).space_codec
        expected = codec.signature({**PROPOSED_PARAMS_5D, "nitrogen_fraction": 0.0})

        migrated = normalize_persisted_candidate_signature(
            REAL_PR2_SIGNATURE_5D,
            parameter_space=parameter_space,
        )

        self.assertEqual(migrated, expected)
        self.assertIn('"nitrogen_fraction":"0"', migrated)
        self.assertNotEqual(migrated, REAL_PR2_SIGNATURE_5D)

    def test_observation_boundary_changes_only_signature_not_numeric_text(self) -> None:
        parameter_space = _parameter_space_6d()
        codec = MorboLikeBackend(
            capillary_search_space(parameter_space),
            _spec(),
            seed=17,
            min_observations=10,
        ).space_codec
        expected = codec.signature({**PROPOSED_PARAMS_5D, "nitrogen_fraction": 0.0})

        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "observations.csv"
            _write_observation_csv(path, signature=REAL_PR2_SIGNATURE_5D)
            before = path.read_text(encoding="utf-8")
            self.assertIn("0.4687683281974255", before)
            self.assertIn("469.747987845673", before)

            normalize_observation_candidate_signatures(
                path,
                parameter_space=parameter_space,
            )
            after = path.read_text(encoding="utf-8")
            with path.open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))

        self.assertIn("0.4687683281974255", after)
        self.assertIn("469.747987845673", after)
        self.assertIn('""nitrogen_fraction"":""0""', after)
        self.assertEqual(row["candidate_signature"], expected)

    def test_migrated_historical_identity_blocks_equivalent_current_f0_candidate(self) -> None:
        parameter_space = _parameter_space_6d()
        backend = MorboLikeBackend(
            capillary_search_space(parameter_space),
            _spec(),
            seed=17,
            min_observations=10,
        )
        expected = backend.space_codec.signature(
            {**PROPOSED_PARAMS_5D, "nitrogen_fraction": 0.0}
        )

        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "observations.csv"
            _write_observation_csv(path, signature=REAL_PR2_SIGNATURE_5D)
            normalize_observation_candidate_signatures(
                path,
                parameter_space=parameter_space,
            )
            history = pd.read_csv(path)

        trials = _morbo_trials_from_history(
            history,
            objective_names=("score",),
            parameter_space=parameter_space,
        )
        result = backend.sync(trials)

        self.assertEqual(trials[0].metadata["candidate_signature"], expected)
        self.assertIn(expected, result.state.candidate_registry.seen_candidate_signatures)
        self.assertNotIn(
            REAL_PR2_SIGNATURE_5D,
            result.state.candidate_registry.seen_candidate_signatures,
        )
        self.assertIn(expected, backend._blocked_signatures())

    def test_current_6d_persisted_signature_survives_float_roundtrip_byte_for_byte(self) -> None:
        parameter_space = _parameter_space_6d()
        backend = MorboLikeBackend(
            capillary_search_space(parameter_space),
            _spec(),
            seed=17,
            min_observations=10,
        )
        proposed = {**PROPOSED_PARAMS_5D, "nitrogen_fraction": 0.00432198765432123}
        observed = {**OBSERVED_PARAMS_5D, "nitrogen_fraction": 0.00432198765432123}
        original = backend.space_codec.signature(proposed)
        recomputed = backend.space_codec.signature(observed)
        self.assertNotEqual(original, recomputed)

        normalized = normalize_persisted_candidate_signature(
            original,
            parameter_space=parameter_space,
        )
        self.assertEqual(normalized, original)

        history = pd.DataFrame(
            [
                {
                    "observation_id": "current:roundtrip",
                    "candidate_signature": original,
                    **observed,
                    "score": 690.829069606,
                }
            ]
        )
        trials = _morbo_trials_from_history(
            history,
            objective_names=("score",),
            parameter_space=parameter_space,
        )
        backend.state = backend.state.with_updates(
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=(original,),
                pending_candidate_signatures=(original,),
            )
        )
        result = backend.sync(trials)

        registry = result.state.candidate_registry
        self.assertEqual(trials[0].metadata["candidate_signature"], original)
        self.assertIn(original, registry.seen_candidate_signatures)
        self.assertNotIn(recomputed, registry.seen_candidate_signatures)
        self.assertNotIn(original, registry.pending_candidate_signatures)

    def test_empty_signature_is_left_for_historical_recomputation_fallback(self) -> None:
        self.assertEqual(
            normalize_persisted_candidate_signature(
                "",
                parameter_space=_parameter_space_6d(),
            ),
            "",
        )
        self.assertEqual(
            normalize_persisted_candidate_signature(
                float("nan"),
                parameter_space=_parameter_space_6d(),
            ),
            "",
        )

    def test_incompatible_nonempty_signature_fails_closed_in_nitrogen_space(self) -> None:
        incompatible = '{"laser_case":"f32","focus_mm_num":"0"}'
        with self.assertRaisesRegex(ValueError, "keyset is incompatible"):
            normalize_persisted_candidate_signature(
                incompatible,
                parameter_space=_parameter_space_6d(),
            )
        with self.assertRaisesRegex(ValueError, "non-JSON"):
            normalize_persisted_candidate_signature(
                "not-json",
                parameter_space=_parameter_space_6d(),
            )

    def test_legacy_5d_space_preserves_persisted_signature_unchanged(self) -> None:
        self.assertEqual(
            normalize_persisted_candidate_signature(
                REAL_PR2_SIGNATURE_5D,
                parameter_space=_parameter_space_5d(),
            ),
            REAL_PR2_SIGNATURE_5D,
        )

    def test_problem_observation_entrypoint_applies_migration(self) -> None:
        parameter_space = _parameter_space_6d()
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "observations.csv"
            _write_observation_csv(path, signature=REAL_PR2_SIGNATURE_5D)
            cfg = MagicMock()
            cfg.problem_kind.return_value = "capillary"
            cfg.parameter_space.return_value = parameter_space

            with patch(
                "campaign_optimizer.problems._problem_callable",
                return_value=lambda _cfg, _iteration: path,
            ):
                result = problems.build_observations(cfg, 0)

            with result.open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))

        expected = MorboLikeBackend(
            capillary_search_space(parameter_space),
            _spec(),
            seed=17,
            min_observations=10,
        ).space_codec.signature({**PROPOSED_PARAMS_5D, "nitrogen_fraction": 0.0})
        self.assertEqual(row["candidate_signature"], expected)


if __name__ == "__main__":
    unittest.main()
