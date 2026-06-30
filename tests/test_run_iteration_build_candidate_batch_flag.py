from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from campaign_optimizer.cli.run_iteration import main as run_iteration_main


class FakeOptimizerConfig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = {"schema_version": 1}

    def iteration_dir(self, iteration: int) -> Path:
        return self.root / "optimizer_runs" / f"iter_{iteration:03d}"

    def objective_config(self) -> dict:
        return {
            "schema_version": 1,
            "config_id": "capillary_objectives_v1_test",
        }

    def parameter_space(self) -> dict:
        return {
            "version": 1,
            "laser_cases": ["f20", "f32", "f40"],
            "ranges": {
                "n0_1e18cm3": [0.7, 6.0],
                "plateau_mm_num": [5.0, 25.0],
                "diameter_um_num": [150.0, 500.0],
                "focus_mm_num": [-5.0, 5.0],
            },
        }


class RunIterationBuildCandidateBatchFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_ctx.name)
        self.config_path = self.root / "optimizer.json"
        self.config_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
        self.cfg = FakeOptimizerConfig(self.root)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def test_run_iteration_build_candidate_batch_flag_calls_builder_once(self) -> None:
        iter_dir = self.cfg.iteration_dir(1)
        observations = iter_dir / "inputs" / "observations.csv"
        objectives = iter_dir / "inputs" / "objective_table.csv"
        recommendations = iter_dir / "outputs" / "recommended_candidates.tsv"
        surrogate = iter_dir / "outputs" / "surrogate_summary.json"
        state = iter_dir / "optimizer_state.json"
        candidate_batch = iter_dir / "outputs" / "candidate_batch.tsv"
        batch_plan = iter_dir / "outputs" / "batch_campaign_plan.json"

        stdout = io.StringIO()
        with (
            mock.patch(
                "campaign_optimizer.cli.run_iteration.load_optimizer_config",
                return_value=self.cfg,
            ) as load_config,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.build_observations",
                return_value=observations,
            ) as build_observations,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.build_objectives",
                return_value=objectives,
            ) as build_objectives,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.read_table",
                return_value=pd.DataFrame(
                    [
                        {"fit_eligible": "true"},
                        {"fit_eligible": "false"},
                    ]
                ),
            ) as read_table,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.write_json",
                return_value=surrogate,
            ) as write_json,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.propose_recommendations",
                return_value=recommendations,
            ) as propose_recommendations,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.write_basic_plots",
            ) as write_basic_plots,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.write_optimizer_state",
                return_value=state,
            ) as write_optimizer_state,
            mock.patch(
                "campaign_optimizer.cli.run_iteration.build_candidate_batch",
                return_value=(candidate_batch, batch_plan),
            ) as build_candidate_batch,
            contextlib.redirect_stdout(stdout),
        ):
            rc = run_iteration_main(
                [
                    "--config",
                    str(self.config_path),
                    "--iteration",
                    "1",
                    "--build-candidate-batch",
                ]
            )

        self.assertEqual(rc, 0)

        load_config.assert_called_once_with(str(self.config_path))
        build_observations.assert_called_once_with(self.cfg, 1)
        build_objectives.assert_called_once_with(self.cfg, 1)
        read_table.assert_called_once_with(objectives)
        write_json.assert_called_once()
        propose_recommendations.assert_called_once_with(self.cfg, 1)
        write_basic_plots.assert_called_once_with(iter_dir)
        write_optimizer_state.assert_called_once()
        build_candidate_batch.assert_called_once_with(self.cfg, 1)

        text = stdout.getvalue()
        self.assertIn("[OK] observations", text)
        self.assertIn("[OK] objectives", text)
        self.assertIn("[OK] recommendations", text)
        self.assertIn("[OK] state", text)
        self.assertIn("[OK] candidate_batch", text)
        self.assertIn("[OK] batch_campaign_plan", text)

    def test_run_iteration_without_flag_does_not_build_candidate_batch(self) -> None:
        iter_dir = self.cfg.iteration_dir(1)

        stdout = io.StringIO()
        with (
            mock.patch(
                "campaign_optimizer.cli.run_iteration.load_optimizer_config",
                return_value=self.cfg,
            ),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.build_observations",
                return_value=iter_dir / "inputs" / "observations.csv",
            ),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.build_objectives",
                return_value=iter_dir / "inputs" / "objective_table.csv",
            ),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.read_table",
                return_value=pd.DataFrame([{"fit_eligible": "true"}]),
            ),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.write_json",
                return_value=iter_dir / "outputs" / "surrogate_summary.json",
            ),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.propose_recommendations",
                return_value=iter_dir / "outputs" / "recommended_candidates.tsv",
            ),
            mock.patch("campaign_optimizer.cli.run_iteration.write_basic_plots"),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.write_optimizer_state",
                return_value=iter_dir / "optimizer_state.json",
            ),
            mock.patch(
                "campaign_optimizer.cli.run_iteration.build_candidate_batch",
            ) as build_candidate_batch,
            contextlib.redirect_stdout(stdout),
        ):
            rc = run_iteration_main(
                [
                    "--config",
                    str(self.config_path),
                    "--iteration",
                    "1",
                ]
            )

        self.assertEqual(rc, 0)
        build_candidate_batch.assert_not_called()
        self.assertNotIn("[OK] candidate_batch", stdout.getvalue())
        self.assertNotIn("[OK] batch_campaign_plan", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
