from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from campaign_optimizer.cli.run_iteration import main as run_iteration_main
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.multichannel.parameters import (
    build_codec,
    sobol_parameters,
)


class MultichannelOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        example = Path("examples/optimizer_multichannel_sunrise.json")
        self.config_path = self.root / "optimizer.json"
        self.config_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_completed_campaign(
        self, iteration: int, candidate_batch: pd.DataFrame
    ) -> None:
        campaign_root = self.root / "iterations" / f"iter_{iteration:03d}"
        campaign_root.mkdir(parents=True, exist_ok=True)
        candidate_batch.to_csv(
            campaign_root / "cases.tsv", sep="\t", index=False
        )

        for offset, row in candidate_batch.reset_index(drop=True).iterrows():
            case_dir = campaign_root / str(row["CASE_NAME"])
            post_dir = case_dir / "post"
            post_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "state.json").write_text(
                json.dumps({"state": "Raw_deleted"}), encoding="utf-8"
            )
            (case_dir / "validation.json").write_text(
                json.dumps(
                    {
                        "raw": {"particles_final_openpmd": {"ok": True}},
                        "reduced": {"particle_summary": {"ok": True}},
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "species": "electrons",
                        "n_macroparticles_selected": 1000 + offset,
                        "charge_selected_pC": 25.0 + offset,
                        "energy_p95_MeV": 75.0 + 2.0 * offset,
                        "theta_r_p95_mrad": 8.0 + 0.1 * offset,
                        "emitn_xy_um_rad": 0.8 + 0.01 * offset,
                        "forward_fraction": 0.9,
                        "pz_mean_MeV_c": 60.0 + offset,
                    }
                ]
            ).to_csv(post_dir / "particle_summary.csv", index=False)

    def test_periodic_parameters_expand_native_11d_space_to_13d(self) -> None:
        config = load_optimizer_config(self.config_path)
        self.assertEqual(config.problem_kind(), "multichannel")
        self.assertEqual(len(config.parameter_space()["parameters"]), 11)
        self.assertEqual(build_codec(config.parameter_space()).encoded_dim(), 13)

    def test_sobol_sequence_is_deterministic_and_continuable(self) -> None:
        parameter_space = load_optimizer_config(self.config_path).parameter_space()
        first = sobol_parameters(
            parameter_space, start_index=0, count=8, seed=20260713
        )
        second = sobol_parameters(
            parameter_space, start_index=8, count=8, seed=20260713
        )
        complete = sobol_parameters(
            parameter_space, start_index=0, count=16, seed=20260713
        )

        self.assertEqual(first + second, complete)
        self.assertEqual(
            first,
            sobol_parameters(
                parameter_space, start_index=0, count=8, seed=20260713
            ),
        )

    def test_iteration_zero_builds_reference_plus_eight_sobol_cases(self) -> None:
        rc = run_iteration_main(
            [
                "--config",
                str(self.config_path),
                "--iteration",
                "0",
                "--build-candidate-batch",
            ]
        )
        self.assertEqual(rc, 0)

        outputs = self.root / "optimizer_runs" / "iter_000" / "outputs"
        recommended = pd.read_csv(outputs / "recommended_candidates.tsv", sep="\t")
        candidate_batch = pd.read_csv(outputs / "candidate_batch.tsv", sep="\t")
        plan = json.loads(
            (outputs / "batch_campaign_plan.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(recommended), 9)
        self.assertEqual(len(candidate_batch), 9)
        self.assertEqual(recommended.iloc[0]["candidate_source"], "reference")
        self.assertEqual(
            list(recommended.iloc[1:]["sobol_index"].astype(int)), list(range(8))
        )
        self.assertEqual(candidate_batch.iloc[0]["WRITE_FIELD_DIAGNOSTIC"], 1)
        self.assertTrue(
            (candidate_batch.iloc[1:]["WRITE_FIELD_DIAGNOSTIC"] == 0).all()
        )
        self.assertIn("HONEYCOMB_ANGLE_DEG", candidate_batch.columns)
        self.assertIn("POLARIZATION_ANGLE_DEG", candidate_batch.columns)
        self.assertIn("ELLIPTICITY_ANGLE_DEG", candidate_batch.columns)
        self.assertEqual(plan["problem_kind"], "multichannel")
        self.assertIn("candidate_batch_contract", plan)
        self.assertEqual(
            plan["candidate_batch_contract"]["choices"]["WRITE_FIELD_DIAGNOSTIC"],
            ["0", "1"],
        )

    def test_design_continues_then_switches_to_four_morbo_candidates(self) -> None:
        for iteration in (0, 1):
            rc = run_iteration_main(
                [
                    "--config",
                    str(self.config_path),
                    "--iteration",
                    str(iteration),
                    "--build-candidate-batch",
                ]
            )
            self.assertEqual(rc, 0)

            outputs = (
                self.root
                / "optimizer_runs"
                / f"iter_{iteration:03d}"
                / "outputs"
            )
            recommended = pd.read_csv(
                outputs / "recommended_candidates.tsv", sep="\t"
            )
            candidate_batch = pd.read_csv(
                outputs / "candidate_batch.tsv", sep="\t"
            )

            if iteration == 1:
                self.assertEqual(len(recommended), 8)
                self.assertEqual(
                    list(recommended["sobol_index"].astype(int)), list(range(8, 16))
                )
            self._write_completed_campaign(iteration, candidate_batch)

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        payload["recommendation"]["botorch"]["enabled"] = False
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

        rc = run_iteration_main(
            [
                "--config",
                str(self.config_path),
                "--iteration",
                "2",
                "--build-candidate-batch",
            ]
        )
        self.assertEqual(rc, 0)

        outputs = self.root / "optimizer_runs" / "iter_002" / "outputs"
        recommended = pd.read_csv(outputs / "recommended_candidates.tsv", sep="\t")
        summary = json.loads(
            (outputs / "surrogate_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(recommended), 4)
        self.assertTrue((recommended["candidate_source"] == "morbo").all())
        self.assertEqual(summary["phase"], "morbo")
        self.assertEqual(summary["fit_rows"], 17)


if __name__ == "__main__":
    unittest.main()
