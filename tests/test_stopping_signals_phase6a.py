from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.cli.build_stopping_signals import (
    main as build_stopping_signals_main,
)
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.reporting import build_report, build_stopping_signals


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class StoppingSignalsPhase6ATests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_ctx.name)
        self.config_path = self._make_fixture(self.tmp)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def test_build_stopping_signals_creates_json(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        path = build_stopping_signals(cfg, 1)

        self.assertTrue(path.is_file())
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["iteration"], 1)
        self.assertEqual(data["objective_config_id"], "capillary_objectives_v1_test")
        self.assertIn("signals", data)
        self.assertIn("recommendation", data)

    def test_stopping_signals_contains_best_scores_and_improvement(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        path = build_stopping_signals(cfg, 1)
        data = json.loads(path.read_text(encoding="utf-8"))

        signals = data["signals"]

        self.assertEqual(signals["n_fit_eligible_total"], 3)
        self.assertEqual(signals["n_new_valid_observations"], 2)

        self.assertEqual(signals["best_scores"]["score_guiding_v1"], 3.0)
        self.assertEqual(signals["best_scores"]["score_beamlike_v1"], 5.0)
        self.assertEqual(signals["best_scores"]["score_transverse_v1"], 6.0)

        self.assertEqual(signals["best_score_improvement"]["score_guiding_v1"], 2.0)
        self.assertEqual(
            signals["best_score_improvement_status"]["score_guiding_v1"], "ok"
        )

    def test_nan_is_not_silently_converted_to_zero(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        path = build_stopping_signals(cfg, 1)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertIsNone(data["signals"]["best_scores"]["score_acceptance_v1"])

    def test_candidate_novelty_and_boundary_saturation_are_included(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        path = build_stopping_signals(cfg, 1)
        data = json.loads(path.read_text(encoding="utf-8"))

        novelty = data["signals"]["candidate_novelty"]
        self.assertEqual(novelty["status"], "ok")
        self.assertAlmostEqual(novelty["min_nearest_known_scaled_dist"], 0.02)
        self.assertAlmostEqual(novelty["median_nearest_known_scaled_dist"], 0.035)

        boundary = data["signals"]["boundary_saturation"]
        self.assertEqual(boundary["status"], "ok")
        self.assertAlmostEqual(boundary["parameters"]["diameter_um_num"], 0.5)

    def test_build_report_also_writes_stopping_signals(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        paths = build_report(cfg, 1)

        self.assertIn("stopping_signals_json", paths)
        self.assertTrue(paths["stopping_signals_json"].is_file())

    def test_cli_build_stopping_signals(self) -> None:
        rc = build_stopping_signals_main(
            [
                "--config",
                str(self.config_path),
                "--iteration",
                "1",
            ]
        )

        self.assertEqual(rc, 0)

    def test_imports_do_not_pull_workflow_or_raw_dependencies(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "campaign_optimizer" / "reporting.py").read_text(
            encoding="utf-8"
        )
        import_lines = "\n".join(
            line for line in text.splitlines() if line.startswith(("import ", "from "))
        )

        forbidden = [
            "campaign_workflow",
            "h5py",
            "openpmd",
            "botorch",
            "torch",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, import_lines)

        forbidden_runtime_mechanisms = [
            "import subprocess",
            "from subprocess",
            "os.system",
            "os.popen",
            "subprocess.",
        ]

        for token in forbidden_runtime_mechanisms:
            with self.subTest(token=token):
                self.assertNotIn(token, text)

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
                "ranges": {
                    "n0_1e18cm3": [0.7, 6.0],
                    "plateau_mm_num": [5.0, 25.0],
                    "diameter_um_num": [150.0, 500.0],
                    "focus_mm_num": [-5.0, 5.0],
                }
            },
        }

        config_path = tmp / "optimizer.json"
        write_json(config_path, cfg)

        iter_dir = tmp / "optimizer_runs" / "iter_001"
        (iter_dir / "inputs").mkdir(parents=True)
        (iter_dir / "outputs").mkdir(parents=True)

        pd.DataFrame(
            [
                {
                    "observation_id": "hist:case_a",
                    "source_campaign_name": "historical_campaign",
                    "n0_1e18cm3": 2.0,
                    "plateau_mm_num": 10.0,
                    "diameter_um_num": 250.0,
                    "focus_mm_num": 0.0,
                },
                {
                    "observation_id": "clpu_iter_000:case_b",
                    "source_campaign_name": "clpu_capillary_guiding_bo_001_iter_000",
                    "n0_1e18cm3": 4.0,
                    "plateau_mm_num": 20.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                },
                {
                    "observation_id": "clpu_iter_000:case_c",
                    "source_campaign_name": "clpu_capillary_guiding_bo_001_iter_000",
                    "n0_1e18cm3": 4.5,
                    "plateau_mm_num": 25.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                },
                {
                    "observation_id": "clpu_iter_000:case_bad",
                    "source_campaign_name": "clpu_capillary_guiding_bo_001_iter_000",
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
                    "score_beamlike_v1": 2.0,
                    "score_transverse_v1": 3.0,
                    "score_acceptance_v1": float("nan"),
                    "fit_eligible": "true",
                },
                {
                    "observation_id": "clpu_iter_000:case_b",
                    "score_guiding_v1": 3.0,
                    "score_beamlike_v1": 4.0,
                    "score_transverse_v1": 5.0,
                    "score_acceptance_v1": float("nan"),
                    "fit_eligible": "true",
                },
                {
                    "observation_id": "clpu_iter_000:case_c",
                    "score_guiding_v1": 2.0,
                    "score_beamlike_v1": 5.0,
                    "score_transverse_v1": 6.0,
                    "score_acceptance_v1": float("nan"),
                    "fit_eligible": "true",
                },
                {
                    "observation_id": "clpu_iter_000:case_bad",
                    "score_guiding_v1": 999.0,
                    "score_beamlike_v1": 999.0,
                    "score_transverse_v1": 999.0,
                    "score_acceptance_v1": 999.0,
                    "fit_eligible": "false",
                },
            ]
        ).to_csv(iter_dir / "inputs" / "objective_table.csv", index=False)

        pd.DataFrame(
            [
                {
                    "rank": 1,
                    "n0_1e18cm3": 4.0,
                    "plateau_mm_num": 25.0,
                    "diameter_um_num": 500.0,
                    "focus_mm_num": 5.0,
                    "nearest_known_scaled_dist": 0.02,
                    "acquisition_value": 1.0,
                },
                {
                    "rank": 2,
                    "n0_1e18cm3": 3.0,
                    "plateau_mm_num": 20.0,
                    "diameter_um_num": 300.0,
                    "focus_mm_num": 0.0,
                    "nearest_known_scaled_dist": 0.05,
                    "acquisition_value": 0.8,
                },
            ]
        ).to_csv(
            iter_dir / "outputs" / "recommended_candidates.tsv",
            sep="\t",
            index=False,
        )

        write_json(
            iter_dir / "outputs" / "surrogate_summary.json",
            {
                "schema_version": 1,
                "backend": "passive_nearest_observed",
                "status": "ok",
                "fit_rows": 3,
            },
        )

        return config_path


if __name__ == "__main__":
    unittest.main()
