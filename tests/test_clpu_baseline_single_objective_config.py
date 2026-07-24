from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.problems import build_objectives


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "optimizer_clpu_baseline_soft50_single_objective_sunrise.json"
)


class ClpuBaselineSingleObjectiveConfigTests(unittest.TestCase):
    def test_example_uses_only_plateau_charge_for_fit(self) -> None:
        payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))

        reduced = payload["optimization_history"]["reduced_outputs"]
        self.assertEqual(
            reduced["particle_summary"],
            "particle_analysis/plateau_exit/particle_summary.csv",
        )
        self.assertNotIn("capillary_exit", json.dumps(reduced))

        objective = payload["objective"]
        recommendation = payload["recommendation"]
        expected = ["score_charge_soft50_direct_v1"]
        self.assertEqual(objective["required_scores_for_fit"], expected)
        self.assertEqual(recommendation["objective_names"], expected)
        self.assertEqual(
            objective["score_specs"][expected[0]],
            ["metric_particle_charge_soft50_pC"],
        )
        self.assertEqual(objective["derived_scores"], [])

        ranges = payload["parameter_space"]["ranges"]
        self.assertNotIn("nitrogen_fraction", ranges)
        self.assertNotIn("pulse_duration_factor", ranges)
        self.assertEqual(payload["candidate_batch"]["laser_duration_fwhm_fs"], 30.0)
        self.assertEqual(
            payload["candidate_batch"]["campaign_template"],
            {
                "campaign_json": "campaign_baseline_soft50.json",
                "input_template": "baseline_input_template.py",
            },
        )

    def test_objective_table_copies_raw_soft50_charge(self) -> None:
        payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            payload["optimizer_run_root"] = "optimizer_runs"
            config_path = tmp / "optimizer.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_optimizer_config(config_path)

            inputs = cfg.iteration_dir(0) / "inputs"
            inputs.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "observation_id": "case_a",
                        "plasma_kind": "chan",
                        "reduced_validation_status": "ok",
                        "metric_particle_charge_soft50_pC": 123.456,
                        "metric_guiding_singlecase_score_v1": 99.0,
                        "metric_particle_energy_p90_soft50_MeV": 500.0,
                    }
                ]
            ).to_csv(inputs / "observations.csv", index=False)

            table = pd.read_csv(build_objectives(cfg, 0))

        self.assertEqual(len(table), 1)
        self.assertTrue(bool(table.loc[0, "fit_eligible"]))
        self.assertAlmostEqual(
            table.loc[0, "score_charge_soft50_direct_v1"],
            123.456,
        )
        self.assertEqual(
            table.loc[0, "score_charge_soft50_direct_v1_source_metric"],
            "metric_particle_charge_soft50_pC",
        )
        self.assertNotIn("score_guiding_v1", payload["recommendation"]["objective_names"])


if __name__ == "__main__":
    unittest.main()
