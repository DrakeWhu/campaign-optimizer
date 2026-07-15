from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from campaign_optimizer.capillary.parameters import (
    capillary_sobol_parameters,
    resonant_duration_fwhm_fs,
)
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.problems import (
    build_candidate_batch,
    build_objectives,
    build_observations,
    propose_recommendations,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def campaign_config(tmp: Path) -> Path:
    payload = {
        "schema_version": 1,
        "problem": "capillary",
        "optimizer_run_root": "optimizer_runs",
        "source_campaigns": [],
        "optimization_history": {
            "enabled": True,
            "optimization_name": "corrected_capillary",
            "iterations_root": "iterations",
        },
        "parameter_space": {
            "version": 2,
            "laser_cases": ["f20", "f32", "f40"],
            "ranges": {
                "n0_1e18cm3": [0.7, 6.0],
                "plateau_mm_num": [5.0, 25.0],
                "diameter_um_num": [150.0, 500.0],
                "focus_mm_num": [-5.0, 5.0],
                "nitrogen_fraction": [0.0, 0.01],
                "pulse_duration_factor": [0.8, 1.4],
            },
            "references": [
                {
                    "reference_id": f"carlos_D300_N2_{fraction:g}",
                    "laser_case": "f32",
                    "n0_1e18cm3": 4.0,
                    "plateau_mm_num": 5.0,
                    "diameter_um_num": 300.0,
                    "focus_mm_num": 0.0,
                    "nitrogen_fraction": fraction,
                    "pulse_duration_factor": 1.0774395377251684,
                }
                for fraction in [0.0, 0.005, 0.01]
            ],
        },
        "objective": {
            "schema_version": 3,
            "config_id": "soft50_test",
            "required_scores_for_fit": [
                "score_guiding_v1",
                "score_charge_soft50_v1",
                "score_energy_soft50_v2",
            ],
            "derived_scores": [
                "score_charge_soft50_v1",
                "score_energy_soft50_v1",
                "score_energy_soft50_v2",
                "score_transverse_soft50_v1",
            ],
            "soft50_v1": {
                "energy_low_MeV": 10.0,
                "energy_target_MeV": 50.0,
                "energy_ref_MeV": 200.0,
                "energy_relative_spread_ref": 0.25,
                "energy_spread_weight": 0.2,
                "effective_count_reference": 200.0,
            },
        },
        "recommendation": {
            "backend": "sobol_then_morbo",
            "seed": 17,
            "n_candidates": 6,
            "min_observations": 8,
            "suggestion_mode": "regional_random",
            "objective_names": [
                "score_guiding_v1",
                "score_charge_soft50_v1",
                "score_energy_soft50_v2",
            ],
            "initial_design": {
                "seed": 17,
                "sobol_points": 64,
                "initial_batch_size": 32,
                "continuation_batch_size": 32,
                "include_references": True,
            },
        },
        "candidate_batch": {
            "plasma_kind": "chan",
            "cap_rmax_um": {
                "policy": "radius_plus_margin",
                "margin_um": 30.0,
            },
            "cap_nr": 192,
        },
    }
    path = tmp / "optimizer.json"
    write_json(path, payload)
    return path


def materialize_fake_results(tmp: Path, iteration: int, batch: pd.DataFrame) -> None:
    root = tmp / "iterations" / f"iter_{iteration:03d}"
    root.mkdir(parents=True, exist_ok=True)
    batch.to_csv(root / "cases.tsv", sep="\t", index=False)
    for offset, case in batch.reset_index(drop=True).iterrows():
        case_dir = root / str(case["CASE_NAME"])
        (case_dir / "particle_analysis").mkdir(parents=True)
        write_json(case_dir / "state.json", {"state": "Reduced_validated"})
        pd.DataFrame(
            [
                {
                    "metric_guiding_singlecase_status": "ok",
                    "metric_guiding_singlecase_score_v1": 20.0 + offset,
                }
            ]
        ).to_csv(case_dir / "guiding_singlecase_score.csv", index=False)
        pd.DataFrame([{"iteration": 1, "a0_peak": 1.0}]).to_csv(
            case_dir / "guiding_metrics.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "species_scope": "all_electrons",
                    "charge_soft50_pC": 20.0 + offset,
                    "n_effective_soft50": 300.0 + offset,
                    "reliability_soft50": 0.9,
                    "energy_p90_soft50_MeV": 30.0 + offset,
                    "energy_relative_spread_rms_soft50": 0.2,
                    "theta_r_p95_soft50_mrad": 8.0,
                    "emitn_xy_soft50_um_rad": 1.5,
                }
            ]
        ).to_csv(
            case_dir / "particle_analysis" / "particle_summary.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "species_scope": "all_electrons",
                    "soft50_energy_low_MeV": 5.0,
                    "charge_soft50_pC": 777.0,
                    "n_effective_soft50": 888.0,
                    "energy_p90_soft50_MeV": 45.0,
                    "energy_relative_spread_rms_soft50": 0.2,
                    "theta_r_p95_soft50_mrad": 7.0,
                    "emitn_xy_soft50_um_rad": 1.0,
                },
                {
                    "species_scope": "all_electrons",
                    "soft50_energy_low_MeV": 10.0,
                    "charge_soft50_pC": 20.0 + offset,
                    "n_effective_soft50": 300.0 + offset,
                    "energy_p90_soft50_MeV": 30.0 + offset,
                    "energy_relative_spread_rms_soft50": 0.2,
                    "theta_r_p95_soft50_mrad": 8.0,
                    "emitn_xy_soft50_um_rad": 1.5,
                },
            ]
        ).to_csv(
            case_dir / "particle_analysis" / "particle_soft50_curves.csv",
            index=False,
        )


class CapillarySoft50SobolTests(unittest.TestCase):
    def test_resonant_duration_matches_reference_density(self) -> None:
        resonant = resonant_duration_fwhm_fs(4.0)
        self.assertAlmostEqual(resonant, 27.843789790135165)
        self.assertAlmostEqual(30.0 / resonant, 1.0774395377251684)

    def test_sobol_sequence_is_deterministic_continuable_and_in_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            cfg = load_optimizer_config(campaign_config(Path(tmp_name)))
            space = cfg.parameter_space()
            first = capillary_sobol_parameters(
                space, start_index=0, count=8, seed=17
            )
            second = capillary_sobol_parameters(
                space, start_index=8, count=8, seed=17
            )
            complete = capillary_sobol_parameters(
                space, start_index=0, count=16, seed=17
            )
        self.assertEqual(first + second, complete)
        self.assertTrue({row["laser_case"] for row in complete} <= {"f20", "f32", "f40"})
        self.assertTrue(all(0.0 <= row["nitrogen_fraction"] <= 0.01 for row in complete))
        self.assertTrue(all(0.8 <= row["pulse_duration_factor"] <= 1.4 for row in complete))

    def test_empty_campaign_builds_three_controls_plus_first_sobol_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = load_optimizer_config(campaign_config(tmp))
            build_observations(cfg, 0)
            build_objectives(cfg, 0)
            propose_recommendations(cfg, 0)
            candidate_path, _ = build_candidate_batch(cfg, 0)
            recommended = pd.read_csv(
                cfg.iteration_dir(0) / "outputs" / "recommended_candidates.tsv",
                sep="\t",
            )
            batch = pd.read_csv(candidate_path, sep="\t")

        self.assertEqual(len(recommended), 35)
        self.assertEqual((recommended["candidate_source"] == "reference").sum(), 3)
        self.assertEqual(
            list(recommended.loc[recommended["candidate_source"] == "sobol", "sobol_index"]),
            list(range(32)),
        )
        self.assertEqual(
            set(batch.loc[batch["OPT_SAMPLE_SOURCE"] == "reference", "NITROGEN_DOPANT_FRACTION"]),
            {0.0, 0.005, 0.01},
        )
        self.assertTrue((batch["CAP_RMAX_UM"] > batch["RADIUS_UM"]).all())
        self.assertTrue((batch["CAP_NR"] == 192).all())
        self.assertTrue((batch["LASER_DURATION_FWHM_FS"] > 0.0).all())
        self.assertTrue(batch["CASE_NAME"].str.contains("N2pct").all())

    def test_second_batch_continues_sobol_then_third_iteration_uses_morbo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = load_optimizer_config(campaign_config(tmp))

            build_observations(cfg, 0)
            build_objectives(cfg, 0)
            propose_recommendations(cfg, 0)
            path0, _ = build_candidate_batch(cfg, 0)
            batch0 = pd.read_csv(path0, sep="\t")
            materialize_fake_results(tmp, 0, batch0)

            build_observations(cfg, 1)
            build_objectives(cfg, 1)
            propose_recommendations(cfg, 1)
            path1, _ = build_candidate_batch(cfg, 1)
            rec1 = pd.read_csv(
                cfg.iteration_dir(1) / "outputs" / "recommended_candidates.tsv",
                sep="\t",
            )
            batch1 = pd.read_csv(path1, sep="\t")
            materialize_fake_results(tmp, 1, batch1)

            build_observations(cfg, 2)
            build_objectives(cfg, 2)
            propose_recommendations(cfg, 2)
            rec2 = pd.read_csv(
                cfg.iteration_dir(2) / "outputs" / "recommended_candidates.tsv",
                sep="\t",
            )

        self.assertEqual(list(rec1["sobol_index"].astype(int)), list(range(32, 64)))
        self.assertEqual(len(rec2), 6)
        self.assertEqual(set(rec2["candidate_source"]), {"morbo"})
        self.assertIn("nitrogen_fraction", rec2.columns)
        self.assertIn("pulse_duration_factor", rec2.columns)

    def test_soft_energy_has_credit_below_50_and_reliability_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = load_optimizer_config(campaign_config(tmp))
            inputs = cfg.iteration_dir(0) / "inputs"
            inputs.mkdir(parents=True)
            rows = []
            for name, energy, n_eff, charge in [
                ("below", 30.0, 1000.0, 100.0),
                ("target", 50.0, 1000.0, 100.0),
                ("high", 200.0, 1000.0, 100.0),
                ("one_macro", 200.0, 1.0, 100.0),
                ("empty", np.nan, 0.0, 0.0),
            ]:
                rows.append(
                    {
                        "observation_id": name,
                        "plasma_kind": "chan",
                        "reduced_validation_status": "ok",
                        "metric_guiding_singlecase_score_v1": 50.0,
                        "metric_particle_charge_soft50_pC": charge,
                        "metric_particle_n_effective_soft50": n_eff,
                        # Deliberately constant: the objective must recompute it.
                        "metric_particle_reliability_soft50": 1.0,
                        "metric_particle_energy_p90_soft50_MeV": energy,
                        "metric_particle_energy_relative_spread_rms_soft50": 0.2,
                        "metric_particle_theta_r_p95_soft50_mrad": 8.0,
                        "metric_particle_emitn_xy_soft50_um_rad": 1.5,
                    }
                )
            pd.DataFrame(rows).to_csv(inputs / "observations.csv", index=False)
            objectives = pd.read_csv(build_objectives(cfg, 0))

        by_id = objectives.set_index("observation_id")
        below = by_id.loc["below", "score_energy_soft50_v1"]
        target = by_id.loc["target", "score_energy_soft50_v1"]
        high = by_id.loc["high", "score_energy_soft50_v1"]
        self.assertGreater(below, 0.0)
        self.assertLess(below, target)
        self.assertLess(target, high)
        self.assertLess(
            by_id.loc["one_macro", "score_energy_soft50_v1"],
            0.1 * high,
        )
        self.assertEqual(by_id.loc["empty", "score_energy_soft50_v1"], 0.0)
        self.assertEqual(by_id.loc["empty", "score_energy_soft50_v2"], 0.0)
        self.assertEqual(str(by_id.loc["empty", "fit_eligible"]).lower(), "true")

    def test_energy_v2_softly_prefers_lower_relative_spread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = load_optimizer_config(campaign_config(tmp))
            inputs = cfg.iteration_dir(0) / "inputs"
            inputs.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "observation_id": name,
                        "plasma_kind": "chan",
                        "reduced_validation_status": "ok",
                        "metric_guiding_singlecase_score_v1": 50.0,
                        "metric_particle_charge_soft50_pC": 100.0,
                        "metric_particle_n_effective_soft50": 1000.0,
                        "metric_particle_energy_p90_soft50_MeV": 100.0,
                        "metric_particle_energy_relative_spread_rms_soft50": spread,
                    }
                    for name, spread in [("narrow", 0.05), ("broad", 0.60)]
                ]
            ).to_csv(inputs / "observations.csv", index=False)
            objectives = pd.read_csv(build_objectives(cfg, 0)).set_index(
                "observation_id"
            )

        narrow = objectives.loc["narrow", "score_energy_soft50_v2"]
        broad = objectives.loc["broad", "score_energy_soft50_v2"]
        self.assertGreater(narrow, broad)
        self.assertGreater(broad, 0.8 * narrow)

    def test_observations_select_configured_energy_low_from_persisted_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            config_path = campaign_config(tmp)
            cfg = load_optimizer_config(config_path)
            build_observations(cfg, 0)
            build_objectives(cfg, 0)
            propose_recommendations(cfg, 0)
            batch_path, _ = build_candidate_batch(cfg, 0)
            materialize_fake_results(tmp, 0, pd.read_csv(batch_path, sep="\t"))

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["objective"]["soft50_v1"]["energy_low_MeV"] = 5.0
            write_json(config_path, payload)
            cfg = load_optimizer_config(config_path)
            observations = pd.read_csv(build_observations(cfg, 1))

        self.assertTrue(observations["soft50_curves_status"].eq("ok").all())
        self.assertTrue(
            np.isclose(
                observations["metric_particle_soft50_energy_low_MeV"], 5.0
            ).all()
        )
        self.assertTrue(
            np.isclose(observations["metric_particle_charge_soft50_pC"], 777.0).all()
        )


if __name__ == "__main__":
    unittest.main()
