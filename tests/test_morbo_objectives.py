from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from campaign_optimizer.morbo.objectives import (
    ObjectiveDefinition,
    ObjectiveResolutionError,
    ObjectiveSpec,
    ObjectiveSpecError,
    adapt_metrics_to_objectives,
    evaluate_trial_eligibility,
    load_objective_spec,
)


class TestMorboObjectives(unittest.TestCase):
    def test_adapt_metrics_to_objectives_applies_transform_and_sense(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                    {
                        "name": "beam_emittance",
                        "metric": "emittance_x",
                        "sense": "min",
                        "transform": "log10",
                    },
                ]
            }
        )

        derived = adapt_metrics_to_objectives(
            {"mean_energy": 120.0, "emittance_x": 1.0e-6},
            spec,
        )

        self.assertEqual(derived["beam_energy"]["value"], 120.0)
        self.assertEqual(derived["beam_energy"]["canonical_value"], 120.0)
        self.assertEqual(derived["beam_emittance"]["value"], -6.0)
        self.assertEqual(derived["beam_emittance"]["canonical_value"], 6.0)
        self.assertEqual(derived["beam_emittance"]["sense"], "min")
        self.assertEqual(derived["beam_emittance"]["metric"], "emittance_x")

    def test_evaluate_trial_eligibility_requires_terminal_success_and_all_objectives(
        self,
    ) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                    {"name": "beam_emittance", "metric": "emittance_x", "sense": "min"},
                ]
            }
        )

        result = evaluate_trial_eligibility(
            simulation_status="finished",
            raw_metrics={"mean_energy": 50.0},
            spec=spec,
            candidate_id=7,
        )

        self.assertFalse(result.eligible)
        self.assertEqual(result.candidate_id, "7")
        self.assertEqual(result.missing_objectives, ("beam_emittance",))
        self.assertIn("missing or invalid objectives", result.reason)
        self.assertEqual(result.objective_values, {"beam_energy": 50.0})
        self.assertEqual(result.canonical_objective_values, {"beam_energy": 50.0})

    def test_evaluate_trial_eligibility_rejects_non_terminal_status(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                ]
            }
        )

        result = evaluate_trial_eligibility(
            simulation_status="running",
            raw_metrics={"mean_energy": 120.0},
            spec=spec,
            candidate_id="case_001",
        )

        self.assertFalse(result.eligible)
        self.assertEqual(result.candidate_id, "case_001")
        self.assertEqual(result.objective_values, {})
        self.assertIn("not terminal-successful", result.reason)

    def test_evaluate_trial_eligibility_accepts_custom_success_statuses(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "score", "metric": "score", "sense": "max"},
                ]
            }
        )

        result = evaluate_trial_eligibility(
            simulation_status="reduced_validated",
            raw_metrics={"score": 0.75},
            spec=spec,
            candidate_id="case_001",
            success_statuses={"reduced_validated"},
        )

        self.assertTrue(result.eligible)
        self.assertEqual(result.objective_values, {"score": 0.75})
        self.assertEqual(result.canonical_objective_values, {"score": 0.75})

    def test_min_objective_is_canonicalized_by_sign_flip(self) -> None:
        objective = ObjectiveDefinition(
            name="beam_emittance",
            metric="emittance_x",
            sense="min",
        )

        self.assertEqual(objective.apply(2.5), 2.5)
        self.assertEqual(objective.canonicalize(2.5), -2.5)

    def test_scale_is_applied_before_canonicalization(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {
                        "name": "scaled_loss",
                        "metric": "loss",
                        "sense": "min",
                        "scale": 0.1,
                    },
                ]
            }
        )

        derived = adapt_metrics_to_objectives({"loss": 30.0}, spec)

        self.assertEqual(derived["scaled_loss"]["value"], 3.0)
        self.assertEqual(derived["scaled_loss"]["canonical_value"], -3.0)

    def test_transforms_are_available_and_checked(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {
                        "name": "abs_value",
                        "metric": "x",
                        "sense": "max",
                        "transform": "abs",
                    },
                    {
                        "name": "sqrt_value",
                        "metric": "y",
                        "sense": "max",
                        "transform": "sqrt",
                    },
                    {
                        "name": "square_value",
                        "metric": "z",
                        "sense": "max",
                        "transform": "square",
                    },
                    {
                        "name": "log10p_value",
                        "metric": "w",
                        "sense": "max",
                        "transform": "log10p",
                    },
                ]
            }
        )

        derived = adapt_metrics_to_objectives(
            {"x": -3.0, "y": 9.0, "z": 4.0, "w": 99.0},
            spec,
        )

        self.assertEqual(derived["abs_value"]["value"], 3.0)
        self.assertEqual(derived["sqrt_value"]["value"], 3.0)
        self.assertEqual(derived["square_value"]["value"], 16.0)
        self.assertEqual(derived["log10p_value"]["value"], 2.0)

    def test_invalid_transform_value_marks_objective_missing_in_eligibility(
        self,
    ) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                    {
                        "name": "bad_log",
                        "metric": "bad_metric",
                        "sense": "max",
                        "transform": "log10",
                    },
                ]
            }
        )

        result = evaluate_trial_eligibility(
            simulation_status="finished",
            raw_metrics={"mean_energy": 120.0, "bad_metric": -1.0},
            spec=spec,
            candidate_id="case_bad",
        )

        self.assertFalse(result.eligible)
        self.assertEqual(result.missing_objectives, ("bad_log",))
        self.assertEqual(result.objective_values, {"beam_energy": 120.0})

    def test_adapt_metrics_raises_for_missing_metric(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                ]
            }
        )

        with self.assertRaisesRegex(ObjectiveResolutionError, "Missing metrics"):
            adapt_metrics_to_objectives({}, spec)

    def test_apply_rejects_non_numeric_and_non_finite_values(self) -> None:
        objective = ObjectiveDefinition(
            name="beam_energy",
            metric="mean_energy",
            sense="max",
        )

        with self.assertRaisesRegex(ObjectiveResolutionError, "not numeric"):
            objective.apply("120")

        with self.assertRaisesRegex(ObjectiveResolutionError, "not numeric"):
            objective.apply(True)

        with self.assertRaisesRegex(ObjectiveResolutionError, "not finite"):
            objective.apply(float("nan"))

        with self.assertRaisesRegex(ObjectiveResolutionError, "not finite"):
            objective.apply(float("inf"))

    def test_load_objective_spec_accepts_mapping_list_and_file(self) -> None:
        mapping_spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "score", "metric": "score", "sense": "max"},
                ]
            }
        )
        self.assertEqual(mapping_spec.objective_names, ("score",))

        list_spec = load_objective_spec(
            spec_data=[
                {"name": "loss", "metric": "loss", "sense": "min"},
            ]
        )
        self.assertEqual(list_spec.objective_names, ("loss",))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "objectives.json"
            path.write_text(
                json.dumps(
                    {
                        "objectives": [
                            {"name": "energy", "metric": "energy", "sense": "max"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            file_spec = load_objective_spec(spec_path=path)

        self.assertEqual(file_spec.objective_names, ("energy",))

    def test_objective_spec_validation_rejects_bad_specs(self) -> None:
        with self.assertRaisesRegex(ObjectiveSpecError, "non-empty 'name'"):
            ObjectiveDefinition.from_dict({"metric": "x", "sense": "max"})

        with self.assertRaisesRegex(ObjectiveSpecError, "non-empty 'metric'"):
            ObjectiveDefinition.from_dict({"name": "x", "sense": "max"})

        with self.assertRaisesRegex(ObjectiveSpecError, "invalid sense"):
            ObjectiveDefinition.from_dict(
                {"name": "x", "metric": "x", "sense": "maximize"}
            )

        with self.assertRaisesRegex(ObjectiveSpecError, "unsupported transform"):
            ObjectiveDefinition.from_dict(
                {"name": "x", "metric": "x", "sense": "max", "transform": "cube"}
            )

        with self.assertRaisesRegex(ObjectiveSpecError, "scale must be finite"):
            ObjectiveDefinition.from_dict(
                {"name": "x", "metric": "x", "sense": "max", "scale": float("nan")}
            )

        with self.assertRaisesRegex(
            ObjectiveSpecError, "Objective names must be unique"
        ):
            ObjectiveSpec(
                (
                    ObjectiveDefinition("x", "metric_a", "max"),
                    ObjectiveDefinition("x", "metric_b", "min"),
                )
            )

    def test_objective_spec_roundtrip_and_get(self) -> None:
        spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {
                        "name": "beam_energy",
                        "metric": "mean_energy",
                        "sense": "max",
                        "transform": "identity",
                        "scale": 2.0,
                    },
                ]
            }
        )

        self.assertEqual(spec.get("beam_energy").metric, "mean_energy")
        self.assertEqual(spec.as_dict()["objectives"][0]["scale"], 2.0)

        with self.assertRaises(KeyError):
            spec.get("missing")


if __name__ == "__main__":
    unittest.main()
