from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from campaign_optimizer.capillary.batch import (
    CANDIDATE_BATCH_REQUIRED_COLUMNS,
    build_candidate_batch,
    compute_cap_rmax_um,
    is_safe_relative_case_name,
)
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.io import read_table


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestCapillaryCandidateBatchPhase2(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_ctx.name)
        self.config_path = self._make_fixture(self.tmp)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def _make_fixture(self, tmp: Path) -> Path:
        campaign_root = tmp / "source_campaign"
        campaign_root.mkdir()

        cfg = {
            "schema_version": 1,
            "optimizer_run_root": "optimizer_runs",
            "objective": {
                "schema_version": 1,
                "config_id": "capillary_objectives_v1_test",
            },
            "source_campaigns": [
                {
                    "campaign_name": "source_campaign_test",
                    "campaign_root": str(campaign_root),
                    "cases_tsv": "cases.tsv",
                    "campaign_json": "campaign.json",
                }
            ],
            "candidate_batch": {
                "case_id_start": 100,
                "cap_nr": 192,
                "cap_rmax_um": {
                    "policy": "diameter_factor",
                    "factor": 0.6,
                },
                "campaign_template": {
                    "campaign_json": "campaign.json",
                    "input_template": "input_template.py",
                },
            },
        }
        config_path = tmp / "optimizer.json"
        write_json(config_path, cfg)

        iter_dir = tmp / "optimizer_runs" / "iter_000"
        (iter_dir / "inputs").mkdir(parents=True)
        (iter_dir / "outputs").mkdir(parents=True)

        pd.DataFrame([{"observation_id": "obs0"}]).to_csv(
            iter_dir / "inputs" / "observations.csv", index=False
        )
        pd.DataFrame([{"observation_id": "obs0", "fit_eligible": True}]).to_csv(
            iter_dir / "inputs" / "objective_table.csv", index=False
        )
        write_json(
            iter_dir / "outputs" / "surrogate_summary.json",
            {"schema_version": 1},
        )

        pd.DataFrame(
            [
                {
                    "recommendation_id": "iter_000_rank_001",
                    "optimizer_iteration": 0,
                    "candidate_id": "opt_000_000",
                    "rank": 1,
                    "recommendation_status": "recommended_placeholder_no_launch",
                    "candidate_source": "unit_test",
                    "ranking_source": "score_balanced_conservative",
                    "acquisition_value": 1.25,
                    "laser_case": "f20",
                    "f_number": 20.0,
                    "n0_1e18cm3": 4.5,
                    "plateau_mm_num": 10.0,
                    "diameter_um_num": 300.0,
                    "focus_mm_num": -2.5,
                },
                {
                    "recommendation_id": "iter_000_rank_002",
                    "optimizer_iteration": 0,
                    "candidate_id": "opt_000_001",
                    "rank": 2,
                    "recommendation_status": "recommended_placeholder_no_launch",
                    "candidate_source": "unit_test",
                    "ranking_source": "score_balanced_conservative",
                    "acquisition_value": 0.75,
                    "laser_case": "f32",
                    "f_number": 32.0,
                    "n0_1e18cm3": 5.25,
                    "plateau_mm_num": 25.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                },
            ]
        ).to_csv(
            iter_dir / "outputs" / "recommended_candidates.tsv",
            sep="\t",
            index=False,
        )

        return config_path

    def test_candidate_batch_contains_required_columns_and_separate_files(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        candidate_path, plan_path = build_candidate_batch(cfg, 0)

        self.assertEqual(candidate_path.name, "candidate_batch.tsv")
        self.assertEqual(plan_path.name, "batch_campaign_plan.json")
        self.assertTrue(
            (cfg.iteration_dir(0) / "outputs" / "recommended_candidates.tsv").is_file()
        )
        self.assertNotEqual(candidate_path.name, "recommended_candidates.tsv")

        batch = read_table(candidate_path, sep="\t")
        for col in CANDIDATE_BATCH_REQUIRED_COLUMNS:
            self.assertIn(col, batch.columns)

    def test_laser_case_is_categorical_case_names_are_safe_and_unique(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        candidate_path, _ = build_candidate_batch(cfg, 0)
        batch = read_table(candidate_path, sep="\t")

        self.assertEqual(set(batch["LASER_CASE"]), {"f20", "f32"})
        self.assertEqual(set(batch["PLASMA_KIND"]), {"chan"})
        self.assertEqual(len(batch["CASE_NAME"]), len(set(batch["CASE_NAME"])))
        self.assertEqual(len(batch["CASE_ID"]), len(set(batch["CASE_ID"])))

        for case_name in batch["CASE_NAME"]:
            self.assertTrue(is_safe_relative_case_name(str(case_name)))
            self.assertNotIn("..", str(case_name))

    def test_units_radius_and_cap_rmax_are_calculated(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        candidate_path, _ = build_candidate_batch(cfg, 0)
        batch = read_table(candidate_path, sep="\t")

        first = batch.iloc[0]
        self.assertAlmostEqual(float(first["N0_CM3"]), 4.5e18)
        self.assertAlmostEqual(float(first["RADIUS_UM"]), 150.0)
        self.assertAlmostEqual(float(first["CAP_RMAX_UM"]), 180.0)
        self.assertEqual(int(first["CAP_NR"]), 192)

        self.assertAlmostEqual(
            compute_cap_rmax_um(
                500.0,
                250.0,
                {"policy": "radius_factor", "factor": 1.2},
            ),
            300.0,
        )
        self.assertAlmostEqual(
            compute_cap_rmax_um(
                500.0,
                250.0,
                {"policy": "fixed", "value": 300.0},
            ),
            300.0,
        )

    def test_no_campaign_root_is_created_and_no_slurm_or_warpx_is_called(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        forbidden = {"sbatch", "srun", "mpiexec", "mpirun", "warpx", "python input.py"}

        with (
            mock.patch("subprocess.run") as run_mock,
            mock.patch("subprocess.Popen") as popen_mock,
        ):
            build_candidate_batch(cfg, 0)

        run_mock.assert_not_called()
        popen_mock.assert_not_called()

        for name in forbidden:
            self.assertFalse((self.tmp / name).exists())

        self.assertFalse((self.tmp / "candidate_campaign").exists())
        self.assertFalse((self.tmp / "new_campaign_from_optimizer_iter_000").exists())

    def test_batch_campaign_plan_is_declarative_and_state_records_paths(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        build_candidate_batch(cfg, 0)

        iter_dir = cfg.iteration_dir(0)
        plan = json.loads(
            (iter_dir / "outputs" / "batch_campaign_plan.json").read_text()
        )
        self.assertEqual(plan["plan_type"], "optimizer_candidate_batch")
        self.assertEqual(plan["candidate_batch"], "outputs/candidate_batch.tsv")
        self.assertEqual(
            plan["recommended_candidates"],
            "outputs/recommended_candidates.tsv",
        )
        self.assertIn("does not submit jobs", plan["non_goals"])
        self.assertIn("does not read raw HDF5/openPMD", plan["non_goals"])

        state = json.loads((iter_dir / "optimizer_state.json").read_text())
        hist = state["history"][0]
        self.assertEqual(hist["candidate_batch"], "outputs/candidate_batch.tsv")
        self.assertEqual(
            hist["batch_campaign_plan"],
            "outputs/batch_campaign_plan.json",
        )

    def test_missing_required_recommended_column_fails_clearly(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        path = cfg.iteration_dir(0) / "outputs" / "recommended_candidates.tsv"
        rec = read_table(path, sep="\t")
        rec = rec.drop(columns=["diameter_um_num"])
        rec.to_csv(path, sep="\t", index=False)

        with self.assertRaisesRegex(
            ValueError,
            "missing required column.*diameter_um_num",
        ):
            build_candidate_batch(cfg, 0)


if __name__ == "__main__":
    unittest.main()
