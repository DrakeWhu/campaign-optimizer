from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from campaign_optimizer.morbo.frontier import FrontierRecord
from campaign_optimizer.morbo.objectives import load_objective_spec
from campaign_optimizer.morbo.observations import (
    ObservationBuildResult,
    ObservedTrial,
    SkippedTrial,
    TrialInput,
    build_observation,
    build_observations,
    build_training_data,
    frontier_records_from_observations,
    read_trial_inputs_from_table,
)
from campaign_optimizer.morbo.search_space import Choice, FloatRange, SearchSpaceCodec


class TestMorboObservations(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_objective_spec(
            spec_data={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                    {"name": "beam_emittance", "metric": "emittance_x", "sense": "min"},
                ]
            }
        )
        self.codec = SearchSpaceCodec(
            {
                "geom.n0": FloatRange(1.0, 10.0),
                "ion.state": Choice([1, 2]),
            }
        )

    def test_build_observation_returns_observed_trial_for_eligible_input(self) -> None:
        trial = TrialInput(
            candidate_id="case_001",
            params={"geom.n0": 2.0, "ion.state": 1},
            raw_metrics={"mean_energy": 100.0, "emittance_x": 3.0},
            simulation_status="finished",
            metadata={"batch_id": "batch_000"},
        )

        observed = build_observation(trial, self.spec)

        self.assertIsInstance(observed, ObservedTrial)
        self.assertEqual(observed.candidate_id, "case_001")
        self.assertEqual(
            observed.objective_values, {"beam_energy": 100.0, "beam_emittance": 3.0}
        )
        self.assertEqual(
            observed.canonical_objective_values,
            {"beam_energy": 100.0, "beam_emittance": -3.0},
        )
        self.assertEqual(
            observed.senses, {"beam_energy": "max", "beam_emittance": "min"}
        )
        self.assertEqual(observed.metadata, {"batch_id": "batch_000"})

    def test_build_observation_returns_skipped_trial_for_missing_objective(
        self,
    ) -> None:
        trial = TrialInput(
            candidate_id="case_002",
            params={"geom.n0": 4.0, "ion.state": 2},
            raw_metrics={"mean_energy": 100.0},
            simulation_status="finished",
        )

        skipped = build_observation(trial, self.spec)

        self.assertIsInstance(skipped, SkippedTrial)
        self.assertEqual(skipped.candidate_id, "case_002")
        self.assertEqual(skipped.missing_objectives, ("beam_emittance",))
        self.assertIn("missing or invalid objectives", skipped.reason)

    def test_build_observation_returns_skipped_trial_for_non_terminal_status(
        self,
    ) -> None:
        trial = TrialInput(
            candidate_id="case_003",
            params={"geom.n0": 4.0, "ion.state": 2},
            raw_metrics={"mean_energy": 100.0, "emittance_x": 2.0},
            simulation_status="running",
        )

        skipped = build_observation(trial, self.spec)

        self.assertIsInstance(skipped, SkippedTrial)
        self.assertEqual(
            skipped.objective_values if hasattr(skipped, "objective_values") else {}, {}
        )
        self.assertIn("not terminal-successful", skipped.reason)

    def test_build_observations_sorts_outputs_and_rejects_duplicates(self) -> None:
        trials = [
            TrialInput(
                "case_002",
                {"geom.n0": 4.0, "ion.state": 2},
                {"mean_energy": 120.0, "emittance_x": 2.0},
                "finished",
            ),
            TrialInput(
                "case_001",
                {"geom.n0": 2.0, "ion.state": 1},
                {"mean_energy": 100.0},
                "finished",
            ),
        ]

        result = build_observations(trials, self.spec)

        self.assertIsInstance(result, ObservationBuildResult)
        self.assertEqual(result.observed_count, 1)
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(result.observations[0].candidate_id, "case_002")
        self.assertEqual(result.skipped[0].candidate_id, "case_001")

        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            build_observations([trials[0], trials[0]], self.spec)

    def test_build_training_data_uses_projected_params_and_canonical_objective_order(
        self,
    ) -> None:
        trials = [
            TrialInput(
                "case_001",
                {"geom.n0": 2.0, "ion.state": 1},
                {"mean_energy": 100.0, "emittance_x": 4.0},
                "finished",
            ),
            TrialInput(
                "case_002",
                {"geom.n0": 5.0, "ion.state": 1.8},
                {"mean_energy": 120.0, "emittance_x": 2.5},
                "finished",
            ),
        ]
        observations = build_observations(trials, self.spec).observations

        training = build_training_data(
            observations,
            self.codec,
            objective_names=self.spec.objective_names,
        )

        self.assertEqual(len(training), 2)
        self.assertEqual(training.candidate_ids, ("case_001", "case_002"))
        self.assertEqual(training.objective_names, ("beam_energy", "beam_emittance"))
        self.assertEqual(training.params[1], {"geom.n0": 5.0, "ion.state": 2})
        self.assertEqual(training.canonical_Y[0], (100.0, -4.0))
        self.assertEqual(training.canonical_Y[1], (120.0, -2.5))
        self.assertEqual(
            training.encoded_X[0],
            tuple(self.codec.encode({"geom.n0": 2.0, "ion.state": 1})),
        )

    def test_build_training_data_rejects_missing_objective_name(self) -> None:
        observation = ObservedTrial(
            candidate_id="case_001",
            params={"geom.n0": 2.0, "ion.state": 1},
            objective_values={"beam_energy": 100.0},
            canonical_objective_values={"beam_energy": 100.0},
            senses={"beam_energy": "max"},
            raw_metrics={"mean_energy": 100.0},
            simulation_status="finished",
        )

        with self.assertRaisesRegex(KeyError, "missing objective"):
            build_training_data(
                [observation],
                self.codec,
                objective_names=("beam_energy", "beam_emittance"),
            )

    def test_frontier_records_from_observations_preserves_objective_values(
        self,
    ) -> None:
        observations = build_observations(
            [
                TrialInput(
                    "case_001",
                    {"geom.n0": 2.0, "ion.state": 1},
                    {"mean_energy": 100.0, "emittance_x": 4.0},
                    "finished",
                )
            ],
            self.spec,
        ).observations

        records = frontier_records_from_observations(observations)

        self.assertEqual(
            records,
            [
                FrontierRecord(
                    candidate_id="case_001",
                    objective_values={"beam_energy": 100.0, "beam_emittance": 4.0},
                    canonical_objective_values={
                        "beam_energy": 100.0,
                        "beam_emittance": -4.0,
                    },
                )
            ],
        )

    def test_read_trial_inputs_from_csv_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "candidate_id",
                        "simulation_status",
                        "geom.n0",
                        "ion.state",
                        "mean_energy",
                        "emittance_x",
                        "batch_id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_id": "case_001",
                        "simulation_status": "finished",
                        "geom.n0": "2.0",
                        "ion.state": "1",
                        "mean_energy": "100.0",
                        "emittance_x": "4.0",
                        "batch_id": "batch_000",
                    }
                )
                writer.writerow(
                    {
                        "candidate_id": "case_002",
                        "simulation_status": "finished",
                        "geom.n0": "5.0",
                        "ion.state": "2",
                        "mean_energy": "120.0",
                        "emittance_x": "",
                        "batch_id": "batch_000",
                    }
                )

            trials = read_trial_inputs_from_table(
                path,
                parameter_columns=("geom.n0", "ion.state"),
                metric_columns=("mean_energy", "emittance_x"),
            )

        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0].candidate_id, "case_001")
        self.assertEqual(trials[0].params, {"geom.n0": 2.0, "ion.state": 1})
        self.assertEqual(
            trials[0].raw_metrics, {"mean_energy": 100.0, "emittance_x": 4.0}
        )
        self.assertEqual(trials[0].metadata, {"batch_id": "batch_000"})
        self.assertEqual(trials[1].raw_metrics, {"mean_energy": 120.0})

        result = build_observations(trials, self.spec)
        self.assertEqual(result.observed_count, 1)
        self.assertEqual(result.skipped_count, 1)

    def test_read_trial_inputs_from_tsv_table_with_default_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.tsv"
            path.write_text(
                "\t".join(
                    [
                        "candidate_id",
                        "geom.n0",
                        "ion.state",
                        "mean_energy",
                        "emittance_x",
                    ]
                )
                + "\n"
                + "\t".join(["case_001", "2.0", "1", "100.0", "4.0"])
                + "\n",
                encoding="utf-8",
            )

            trials = read_trial_inputs_from_table(
                path,
                parameter_columns=("geom.n0", "ion.state"),
                metric_columns=("mean_energy", "emittance_x"),
                status_column=None,
                default_status="reduced_validated",
            )

        self.assertEqual(trials[0].simulation_status, "reduced_validated")
        result = build_observations(
            trials,
            self.spec,
            success_statuses={"reduced_validated"},
        )
        self.assertEqual(result.observed_count, 1)

    def test_read_trial_inputs_from_table_rejects_missing_columns_and_empty_candidate_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.csv"
            missing_path.write_text(
                "candidate_id,mean_energy\ncase_001,100.0\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_trial_inputs_from_table(
                    missing_path,
                    parameter_columns=("geom.n0",),
                    metric_columns=("mean_energy",),
                )

            empty_id_path = Path(tmpdir) / "empty_id.csv"
            empty_id_path.write_text(
                "candidate_id,simulation_status,geom.n0,mean_energy\n,finished,2.0,100.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "empty candidate_id"):
                read_trial_inputs_from_table(
                    empty_id_path,
                    parameter_columns=("geom.n0",),
                    metric_columns=("mean_energy",),
                )

    def test_trial_input_rejects_empty_candidate_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            TrialInput("", {}, {}, "finished")


if __name__ == "__main__":
    unittest.main()
