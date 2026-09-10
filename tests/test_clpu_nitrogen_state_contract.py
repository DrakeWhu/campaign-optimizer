from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.morbo.state import (
    CandidateRegistry,
    OptimizerState,
    load_optimizer_state,
    save_optimizer_state,
)
from campaign_optimizer.capillary.state_contract import (
    prepare_state_contract,
    state_fingerprint_payload,
    state_fingerprint_signature,
)
from campaign_optimizer.problems import propose_recommendations


SEMANTICS = "fraction_of_atomic_nuclei_nitrogen_equal_H2_N2_molecular_fraction"
PROFILE = "uniform_v1"


def _payload(*, strict: bool = True) -> dict:
    data = {
        "schema_version": 1,
        "problem": "capillary",
        "optimizer_run_root": "optimizer_runs",
        "source_campaigns": [
            {
                "campaign_name": "historical",
                "campaign_root": "historical",
                "cases_tsv": "cases.tsv",
            }
        ],
        "parameter_space": {
            "version": 3,
            "laser_cases": ["f20", "f32", "f40"],
            "ranges": {
                "n0_1e18cm3": [0.7, 6.0],
                "plateau_mm_num": [5.0, 25.0],
                "diameter_um_num": [150.0, 500.0],
                "focus_mm_num": [-5.0, 5.0],
                "nitrogen_fraction": [0.0, 0.01],
            },
        },
        "objective": {
            "schema_version": 3,
            "config_id": "clpu_n2_soft50_charge_plateau_v1",
            "required_scores_for_fit": ["score_charge_soft50_v1"],
            "derived_scores": ["score_charge_soft50_v1"],
            "soft50_v1": {
                "energy_low_MeV": 10.0,
                "energy_target_MeV": 50.0,
            },
        },
        "recommendation": {
            "backend": "morbo_like",
            "seed": 17,
            "n_candidates": 8,
            "objective_names": ["score_charge_soft50_v1"],
            "suggestion_mode": "regional_model",
        },
    }
    if strict:
        data["recommendation"]["state_contract"] = {
            "contract_id": "clpu_n2_fresh_state_v1",
            "fresh_bootstrap_iteration": 0,
            "expected_encoded_dim": 8,
            "nitrogen_fraction_semantics": SEMANTICS,
            "nitrogen_profile": PROFILE,
        }
    return data


def _config(tmp: Path, data: dict | None = None, name: str = "optimizer.json"):
    path = tmp / name
    path.write_text(json.dumps(data or _payload()), encoding="utf-8")
    return load_optimizer_config(path)


def _write_compatible_state(cfg, iteration: int = 0) -> Path:
    payload = state_fingerprint_payload(cfg)
    assert payload is not None
    signature = state_fingerprint_signature(payload)
    path = cfg.iteration_dir(iteration) / "outputs" / "morbo_optimizer_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_optimizer_state(
        path,
        OptimizerState(
            search_space_signature=signature,
            extra={
                "state_contract_id": "clpu_n2_fresh_state_v1",
                "state_fingerprint_signature": signature,
                "state_fingerprint_payload": payload,
            },
        ),
    )
    return path


class ClpuNitrogenStateContractTests(unittest.TestCase):
    def test_fingerprint_captures_six_physical_parameters_as_eight_encoded_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            cfg = _config(Path(tmp_name))
            payload = state_fingerprint_payload(cfg)

        self.assertIsNotNone(payload)
        search = payload["search_space"]
        self.assertEqual(
            search["parameter_names"],
            [
                "laser_case",
                "n0_1e18cm3",
                "plateau_mm_num",
                "diameter_um_num",
                "focus_mm_num",
                "nitrogen_fraction",
            ],
        )
        self.assertEqual(search["encoded_dim"], 8)
        self.assertEqual(search["parameters"][0]["type"], "choice")
        self.assertEqual(search["parameters"][0]["options"], ["f20", "f32", "f40"])
        self.assertEqual(search["parameters"][-1]["low"], 0.0)
        self.assertEqual(search["parameters"][-1]["high"], 0.01)
        self.assertEqual(payload["nitrogen"]["fraction_semantics"], SEMANTICS)
        self.assertEqual(payload["nitrogen"]["profile"], PROFILE)

    def test_clean_iter0_bootstrap_is_accepted_and_written_state_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = _config(tmp)
            prepare_state_contract(cfg, 0)

            output = cfg.iteration_dir(0) / "outputs" / "recommended_candidates.tsv"

            def fake_recommend(config, iteration):
                outputs = config.iteration_dir(iteration) / "outputs"
                outputs.mkdir(parents=True, exist_ok=True)
                output.write_text("candidate_id\n", encoding="utf-8")
                save_optimizer_state(
                    outputs / "morbo_optimizer_state.json",
                    OptimizerState(
                        search_space_signature=(
                            "{'parameter_names': ['laser_case'], 'encoded_dim': 8}"
                        ),
                        candidate_registry=CandidateRegistry(
                            seen_candidate_signatures=("new_6d_signature",),
                            pending_candidate_signatures=("new_6d_signature",),
                        ),
                    ),
                )
                return output

            with patch(
                "campaign_optimizer.problems._problem_callable",
                return_value=fake_recommend,
            ):
                result = propose_recommendations(cfg, 0)

            state = load_optimizer_state(
                cfg.iteration_dir(0) / "outputs" / "morbo_optimizer_state.json"
            )
            expected_payload = state_fingerprint_payload(cfg)
            assert expected_payload is not None
            expected_signature = state_fingerprint_signature(expected_payload)

        self.assertEqual(result, output)
        self.assertEqual(state.search_space_signature, expected_signature)
        self.assertEqual(state.extra["state_fingerprint_payload"], expected_payload)
        self.assertEqual(state.extra["state_fingerprint_signature"], expected_signature)
        self.assertEqual(
            state.candidate_registry.pending_candidate_signatures,
            ("new_6d_signature",),
        )

    def test_unexpected_state_in_iter0_outputs_is_rejected_before_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = _config(tmp)
            path = cfg.iteration_dir(0) / "outputs" / "morbo_optimizer_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_optimizer_state(path, OptimizerState())

            with patch("campaign_optimizer.problems._problem_callable") as callback:
                with self.assertRaisesRegex(ValueError, "unexpected optimizer state"):
                    propose_recommendations(cfg, 0)
                callback.assert_not_called()

    def test_old_weak_or_5d_state_is_rejected_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = _config(tmp)
            path = cfg.iteration_dir(0) / "outputs" / "morbo_optimizer_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_optimizer_state(
                path,
                OptimizerState(
                    search_space_signature=(
                        "{'parameter_names': ['laser_case', 'n0_1e18cm3', "
                        "'plateau_mm_num', 'diameter_um_num', 'focus_mm_num'], "
                        "'encoded_dim': 7}"
                    )
                ),
            )

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                prepare_state_contract(cfg, 1)

    def test_same_dimension_but_changed_bounds_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            base = _payload()
            cfg = _config(tmp, base)
            _write_compatible_state(cfg)

            changed = copy.deepcopy(base)
            changed["parameter_space"]["ranges"]["n0_1e18cm3"] = [0.8, 6.0]
            changed_cfg = _config(tmp, changed, "optimizer_changed.json")

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                prepare_state_contract(changed_cfg, 1)

    def test_same_dimension_but_changed_category_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            base = _payload()
            cfg = _config(tmp, base)
            _write_compatible_state(cfg)

            changed = copy.deepcopy(base)
            changed["parameter_space"]["laser_cases"] = ["f32", "f20", "f40"]
            changed_cfg = _config(tmp, changed, "optimizer_changed.json")

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                prepare_state_contract(changed_cfg, 1)

    def test_objective_semantics_and_profile_changes_are_rejected(self) -> None:
        mutations = (
            ("objective", lambda data: data["objective"].update({"config_id": "other"})),
            (
                "semantics",
                lambda data: data["recommendation"]["state_contract"].update(
                    {"nitrogen_fraction_semantics": "other_fraction_semantics"}
                ),
            ),
            (
                "profile",
                lambda data: data["recommendation"]["state_contract"].update(
                    {"nitrogen_profile": "other_profile"}
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_name:
                tmp = Path(tmp_name)
                base = _payload()
                cfg = _config(tmp, base)
                _write_compatible_state(cfg)
                changed = copy.deepcopy(base)
                mutate(changed)
                changed_cfg = _config(tmp, changed, "optimizer_changed.json")
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    prepare_state_contract(changed_cfg, 1)

    def test_strict_resume_without_prior_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            cfg = _config(Path(tmp_name))
            with self.assertRaisesRegex(ValueError, "requires a compatible prior state"):
                prepare_state_contract(cfg, 1)

    def test_legacy_configuration_remains_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = _config(tmp, _payload(strict=False))
            result_path = tmp / "legacy.tsv"

            def fake_recommend(_config, _iteration):
                result_path.write_text("candidate_id\n", encoding="utf-8")
                return result_path

            with patch(
                "campaign_optimizer.problems._problem_callable",
                return_value=fake_recommend,
            ):
                result = propose_recommendations(cfg, 0)

        self.assertEqual(result, result_path)


if __name__ == "__main__":
    unittest.main()
