from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from campaign_optimizer.cli.build_report import main as build_report_main
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.reporting import PlotRecord, build_report


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestReportingPhase5A(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_ctx.name)
        self.config_path = self._make_fixture(self.tmp)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def _make_fixture(self, tmp: Path) -> Path:
        cfg = {
            "schema_version": 1,
            "optimizer_run_root": "optimizer_runs",
            "objective": {
                "schema_version": 1,
                "config_id": "capillary_objectives_v1_test",
                "required_scores_for_fit": [
                    "score_guiding_v1",
                    "score_beamlike_v1",
                    "score_transverse_v1",
                ],
            },
            "source_campaigns": [
                {
                    "campaign_name": "dummy_source",
                    "campaign_root": str(tmp / "source_campaign"),
                    "cases_tsv": "cases.tsv",
                }
            ],
            "parameter_space": {
                "laser_cases": ["f20", "f32", "f40"],
                "ranges": {
                    "n0_1e18cm3": [0.7, 6.0],
                    "plateau_mm_num": [5.0, 25.0],
                    "diameter_um_num": [150.0, 500.0],
                    "focus_mm_num": [-5.0, 5.0],
                },
            },
        }
        config_path = tmp / "optimizer.json"
        write_json(config_path, cfg)

        iter_dir = tmp / "optimizer_runs" / "iter_000"
        (iter_dir / "inputs").mkdir(parents=True)
        (iter_dir / "outputs").mkdir(parents=True)

        pd.DataFrame(
            [
                {
                    "observation_id": "hist:case_a",
                    "source_campaign_name": "historical_campaign",
                    "laser_case": "f20",
                    "plasma_kind": "chan",
                    "n0_1e18cm3": 2.0,
                    "plateau_mm_num": 10.0,
                    "diameter_um_num": 250.0,
                    "focus_mm_num": 0.0,
                },
                {
                    "observation_id": "clpu_iter_000:case_b",
                    "source_campaign_name": "clpu_capillary_guiding_bo_001_iter_000",
                    "laser_case": "f20",
                    "plasma_kind": "chan",
                    "n0_1e18cm3": 4.0,
                    "plateau_mm_num": 20.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                },
                {
                    "observation_id": "clpu_iter_000:case_bad",
                    "source_campaign_name": "clpu_capillary_guiding_bo_001_iter_000",
                    "laser_case": "f20",
                    "plasma_kind": "chan",
                    "n0_1e18cm3": 5.0,
                    "plateau_mm_num": 24.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                },
            ]
        ).to_csv(iter_dir / "inputs" / "observations.csv", index=False)

        pd.DataFrame(
            [
                {
                    "observation_id": "hist:case_a",
                    "score_guiding_v1": 1.0,
                    "score_beamlike_v1": -2.0,
                    "score_transverse_v1": 0.5,
                    "fit_eligible": "true",
                },
                {
                    "observation_id": "clpu_iter_000:case_b",
                    "score_guiding_v1": 3.0,
                    "score_beamlike_v1": 4.0,
                    "score_transverse_v1": 5.0,
                    "fit_eligible": "true",
                },
                {
                    "observation_id": "clpu_iter_000:case_bad",
                    "score_guiding_v1": float("nan"),
                    "score_beamlike_v1": 100.0,
                    "score_transverse_v1": 100.0,
                    "fit_eligible": "false",
                },
            ]
        ).to_csv(iter_dir / "inputs" / "objective_table.csv", index=False)

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
                    "acquisition_value": 2.0,
                    "laser_case": "f20",
                    "f_number": 20.0,
                    "n0_1e18cm3": 6.0,
                    "plateau_mm_num": 25.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                    "nearest_known_scaled_dist": 0.05,
                    "score_balanced_conservative": 2.0,
                    "score_balanced_exploratory": 2.2,
                    "score_beamlike_guarded": 1.5,
                    "pred_score_guiding_sem": 0.4,
                    "pred_score_beamlike_sem": 0.0,
                    "pred_score_transverse_sem": 0.8,
                },
                {
                    "recommendation_id": "iter_000_rank_002",
                    "optimizer_iteration": 0,
                    "candidate_id": "opt_000_001",
                    "rank": 2,
                    "recommendation_status": "recommended_placeholder_no_launch",
                    "candidate_source": "unit_test",
                    "ranking_source": "score_balanced_conservative",
                    "acquisition_value": 1.5,
                    "laser_case": "f32",
                    "f_number": 32.0,
                    "n0_1e18cm3": 0.7,
                    "plateau_mm_num": 12.0,
                    "diameter_um_num": 350.0,
                    "focus_mm_num": -5.0,
                    "nearest_known_scaled_dist": 0.08,
                    "score_balanced_conservative": 1.5,
                    "score_balanced_exploratory": 1.7,
                    "score_beamlike_guarded": 1.1,
                    "pred_score_guiding_sem": 0.3,
                    "pred_score_beamlike_sem": 0.1,
                    "pred_score_transverse_sem": 0.9,
                },
            ]
        ).to_csv(
            iter_dir / "outputs" / "recommended_candidates.tsv", sep="\t", index=False
        )

        write_json(
            iter_dir / "outputs" / "surrogate_summary.json",
            {"schema_version": 1, "backend": "passive_nearest_observed", "fit_rows": 2},
        )
        return config_path

    def _build_report_fast(self, cfg):
        def fake_create_plot(name, plot_dir, records, reason, draw):
            if reason:
                records.append(PlotRecord(name=name, status="skipped", reason=reason))
                return
            plot_dir.mkdir(parents=True, exist_ok=True)
            path = plot_dir / name
            path.write_bytes(b"placeholder png for unittest\n")
            records.append(PlotRecord(name=name, status="created", path=str(path)))

        with mock.patch(
            "campaign_optimizer.reporting._create_plot", side_effect=fake_create_plot
        ):
            return build_report(cfg, 0)

    def test_build_report_creates_reports_and_plots(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        paths = self._build_report_fast(cfg)
        iter_dir = cfg.iteration_dir(0)

        self.assertTrue((iter_dir / "reports").is_dir())
        self.assertTrue((iter_dir / "plots").is_dir())
        self.assertTrue(paths["plot_manifest"].is_file())
        self.assertTrue((iter_dir / "reports" / "convergence_summary.csv").is_file())
        self.assertTrue((iter_dir / "reports" / "convergence_summary.json").is_file())
        self.assertTrue((iter_dir / "reports" / "parameter_convergence.csv").is_file())
        self.assertTrue((iter_dir / "reports" / "acquisition_summary.csv").is_file())
        self.assertTrue(
            (iter_dir / "reports" / "boundary_saturation_summary.csv").is_file()
        )
        self.assertTrue((iter_dir / "reports" / "surrogate_diagnostics.json").is_file())

    def test_plot_manifest_records_created_and_skipped_plots(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self._build_report_fast(cfg)
        manifest = json.loads(
            (cfg.iteration_dir(0) / "reports" / "plot_manifest.json").read_text()
        )

        created = [p for p in manifest["plots"] if p["status"] == "created"]
        skipped = [p for p in manifest["plots"] if p["status"] == "skipped"]
        self.assertGreaterEqual(len(created), 1)
        self.assertGreaterEqual(len(skipped), 1)
        self.assertTrue(any("score_acceptance_v1" in p["name"] for p in skipped))
        self.assertTrue(any("reason" in p for p in skipped))

    def test_candidate_plots_are_created_when_columns_exist(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self._build_report_fast(cfg)
        plot_dir = cfg.iteration_dir(0) / "plots"

        self.assertTrue((plot_dir / "candidate_parameters_by_rank.png").is_file())
        self.assertTrue((plot_dir / "candidate_distance_to_history.png").is_file())
        self.assertTrue((plot_dir / "candidate_acquisition_components.png").is_file())
        self.assertTrue((plot_dir / "surrogate_uncertainty_rank.png").is_file())

    def test_objective_tradeoff_and_single_iteration_traces_are_created(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self._build_report_fast(cfg)
        plot_dir = cfg.iteration_dir(0) / "plots"

        self.assertTrue(
            (plot_dir / "objective_tradeoff__guiding_vs_beamlike.png").is_file()
        )
        self.assertTrue(
            (plot_dir / "objective_tradeoff__beamlike_vs_transverse.png").is_file()
        )
        self.assertTrue(
            (plot_dir / "objective_tradeoff__guiding_vs_transverse.png").is_file()
        )
        self.assertTrue((plot_dir / "optimization_trace__best_scores.png").is_file())
        self.assertTrue((plot_dir / "optimization_trace__mean_scores.png").is_file())

    def test_fit_ineligible_rows_are_excluded_and_nan_is_not_zeroed(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self._build_report_fast(cfg)
        conv = pd.read_csv(cfg.iteration_dir(0) / "reports" / "convergence_summary.csv")

        beam = conv[conv["score_name"] == "score_beamlike_v1"]
        self.assertLess(float(beam["max"].max()), 100.0)

        guiding = conv[conv["score_name"] == "score_guiding_v1"]
        self.assertFalse((guiding["mean"] == 0.0).any())

    def test_boundary_summary_detects_domain_saturation(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self._build_report_fast(cfg)
        boundary = pd.read_csv(
            cfg.iteration_dir(0) / "reports" / "boundary_saturation_summary.csv"
        )

        n0 = boundary[boundary["parameter"] == "n0_1e18cm3"].iloc[0]
        focus = boundary[boundary["parameter"] == "focus_mm_num"].iloc[0]
        self.assertEqual(int(n0["n_saturated"]), 2)
        self.assertEqual(int(focus["n_saturated"]), 2)

    def test_missing_optional_acceptance_score_does_not_break(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self._build_report_fast(cfg)
        diag = json.loads(
            (
                cfg.iteration_dir(0) / "reports" / "surrogate_diagnostics.json"
            ).read_text()
        )

        self.assertFalse(diag["score_status"]["score_acceptance_v1"]["available"])

    def test_build_report_cli(self) -> None:
        rc = build_report_main(["--config", str(self.config_path), "--iteration", "0"])
        self.assertEqual(rc, 0)

    def test_no_subprocess_no_hdf5_no_campaign_workflow_import(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        seen_paths: list[str] = []

        def wrapped_read_table(path: str | Path, *args, **kwargs):
            seen_paths.append(str(path))
            from campaign_optimizer.io import read_table as real_read_table

            return real_read_table(path, *args, **kwargs)

        with (
            mock.patch("subprocess.run") as run_mock,
            mock.patch("subprocess.Popen") as popen_mock,
            mock.patch(
                "campaign_optimizer.reporting.read_table",
                side_effect=wrapped_read_table,
            ),
        ):
            self._build_report_fast(cfg)

        run_mock.assert_not_called()
        popen_mock.assert_not_called()
        self.assertFalse(any(path.endswith((".h5", ".hdf5")) for path in seen_paths))
        self.assertFalse(
            any("/diags/" in path.replace("\\", "/") for path in seen_paths)
        )
        self.assertFalse(
            any(name.startswith("campaign_workflow") for name in sys.modules)
        )


if __name__ == "__main__":
    unittest.main()
