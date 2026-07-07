from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.io import assert_not_raw_diagnostic_path, read_table
from campaign_optimizer.observations import build_observations
from campaign_optimizer.objectives import build_objectives
from campaign_optimizer.recommend import propose_recommendations


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_case(
    root: Path,
    name: str,
    state: str = "Reduced_validated",
    *,
    transverse: float | None = 3.0,
    singlecase_score: float | None = None,
) -> None:
    case_dir = root / name
    (case_dir / "particle_analysis").mkdir(parents=True, exist_ok=True)

    write_json(case_dir / "state.json", {"state": state})

    write_json(
        case_dir / "validation.json",
        {
            "raw": {
                "fields_openpmd": {"ok": True},
            },
            "reduced": {
                "guiding_metrics": {"ok": True},
                "particle_summary": {"ok": True},
            },
        },
    )

    pd.DataFrame(
        [
            {
                "iteration": 1,
                "a0_peak": 1.2,
                "waist_um": 40.0,
            }
        ]
    ).to_csv(case_dir / "guiding_metrics.csv", index=False)

    if singlecase_score is not None:
        pd.DataFrame(
            [
                {
                    "metric_guiding_singlecase_status": "ok",
                    "metric_guiding_singlecase_score_v1": singlecase_score,
                    "metric_guiding_waist_growth_component_v1": 0.9,
                    "metric_guiding_a0_retention_component_v1": 0.8,
                }
            ]
        ).to_csv(case_dir / "guiding_singlecase_score.csv", index=False)

    row = {
        "n_macroparticles_total": 1000,
        "n_macroparticles_valid": 1000,
        "n_macroparticles_hot": 300,
        "charge_hot_pC": 150.0,
        "hot_energy_threshold_MeV": 10.0,
        "Emax_MeV": 160.0,
        "E95_MeV": 120.0,
        "Emax_hot_MeV": 160.0,
        "E95_hot_MeV": 120.0,
        "eligible_beamlike": True,
        "beamlike_status": "ok",
        "beamlike_score": 10.0,
        "beam_yield_score": 20.0,
        "mono_proxy_E95_over_Emax": 0.75,
        "n_macroparticles_transverse": 300,
        "weight_transverse": 1.0,
    }

    if transverse is not None:
        row["beam_transverse_quality_score"] = transverse

    pd.DataFrame([row]).to_csv(
        case_dir / "particle_analysis" / "particle_summary.csv",
        index=False,
    )


class TestCapillaryOptimizerPhase1(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_ctx.name)
        self.optimizer_config_path = self._make_optimizer_fixture(self.tmp)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def _make_optimizer_fixture(self, tmp_path: Path) -> Path:
        root = tmp_path / "campaign"
        root.mkdir()

        cases = pd.DataFrame(
            [
                {
                    "CASE_ID": 0,
                    "CASE_NAME": "000_f20_chan_n4e18cm3_L5mm_d300um_foc0mm_rz",
                    "LASER_CASE": "f20",
                    "PLASMA_KIND": "chan",
                    "N0_CM3": 4.0e18,
                    "PLATEAU_LENGTH_MM": 5.0,
                    "DIAMETER_UM": 300.0,
                    "RADIUS_UM": 150.0,
                    "FOCUS_OFFSET_FROM_PLATEAU_START_MM": 0.0,
                    "CAP_RMAX_UM": 180.0,
                    "CAP_NR": 192,
                },
                {
                    "CASE_ID": 1,
                    "CASE_NAME": "001_f32_chan_n5e18cm3_L10mm_d400um_focp5mm_rz",
                    "LASER_CASE": "f32",
                    "PLASMA_KIND": "chan",
                    "N0_CM3": 5.0e18,
                    "PLATEAU_LENGTH_MM": 10.0,
                    "DIAMETER_UM": 400.0,
                    "RADIUS_UM": 200.0,
                    "FOCUS_OFFSET_FROM_PLATEAU_START_MM": 5.0,
                    "CAP_RMAX_UM": 240.0,
                    "CAP_NR": 256,
                },
                {
                    "CASE_ID": 2,
                    "CASE_NAME": "002_f40_chan_n3e18cm3_L25mm_d500um_focm5mm_rz",
                    "LASER_CASE": "f40",
                    "PLASMA_KIND": "chan",
                    "N0_CM3": 3.0e18,
                    "PLATEAU_LENGTH_MM": 25.0,
                    "DIAMETER_UM": 500.0,
                    "RADIUS_UM": 250.0,
                    "FOCUS_OFFSET_FROM_PLATEAU_START_MM": -5.0,
                    "CAP_RMAX_UM": 300.0,
                    "CAP_NR": 320,
                },
            ]
        )

        cases.to_csv(root / "cases.tsv", sep="\t", index=False)

        make_case(
            root,
            str(cases.iloc[0]["CASE_NAME"]),
            transverse=4.0,
            singlecase_score=42.0,
        )
        make_case(root, str(cases.iloc[1]["CASE_NAME"]), transverse=2.0)
        make_case(root, str(cases.iloc[2]["CASE_NAME"]), transverse=None)

        joint_dir = root / "analysis_outputs" / "joint"
        joint_dir.mkdir(parents=True)

        pd.DataFrame(
            [
                {
                    "channel_case_id": cases.iloc[0]["CASE_NAME"],
                    "uniform_case_id": "u0",
                    "guiding_final_score": 1.0,
                    "beam_beamlike_gain_score": 8.0,
                    "beam_beam_transverse_quality_score_channel": 4.0,
                },
                {
                    "channel_case_id": cases.iloc[1]["CASE_NAME"],
                    "uniform_case_id": "u1",
                    "guiding_final_score": 2.0,
                    "beam_beamlike_gain_score": 4.0,
                    "beam_beam_transverse_quality_score_channel": 2.0,
                },
                {
                    "channel_case_id": cases.iloc[2]["CASE_NAME"],
                    "uniform_case_id": "u2",
                    "guiding_final_score": 3.0,
                    "beam_beamlike_gain_score": 6.0,
                },
            ]
        ).to_csv(joint_dir / "guiding_beam_transverse_joined.csv", index=False)

        cfg = {
            "schema_version": 1,
            "optimizer_run_root": "optimizer_runs",
            "validation": {
                "require_case_validation": True,
            },
            "source_campaigns": [
                {
                    "campaign_name": "test_campaign",
                    "campaign_root": str(root),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": {
                        "guiding_metrics": "guiding_metrics.csv",
                        "particle_summary": "particle_analysis/particle_summary.csv",
                    },
                    "capillary": {
                        "global_tables": {
                            "joint_table": "analysis_outputs/joint/guiding_beam_transverse_joined.csv",
                        }
                    },
                }
            ],
            "recommendation": {
                "seed": 7,
                "n_candidates": 5,
                "n_random": 200,
                "min_known_scaled_dist": 0.0,
                "nearest_k": 2,
            },
        }

        path = tmp_path / "optimizer.json"
        write_json(path, cfg)
        return path

    def test_build_observations_prefers_explicit_case_columns(self) -> None:
        cfg = load_optimizer_config(self.optimizer_config_path)
        path = build_observations(cfg, 0)

        obs = read_table(path)

        self.assertEqual(len(obs), 3)
        self.assertEqual(set(obs["laser_case"]), {"f20", "f32", "f40"})
        self.assertEqual(set(obs["f_number"]), {20.0, 32.0, 40.0})
        self.assertIn("metric_guiding_final_score", obs.columns)
        self.assertTrue(obs["reduced_validation_status"].eq("ok").all())

    def test_build_objectives_keeps_missing_as_nan_and_fit_ineligible(self) -> None:
        cfg = load_optimizer_config(self.optimizer_config_path)

        build_observations(cfg, 0)
        path = build_objectives(cfg, 0)

        obj = read_table(path)

        self.assertIn("objective_config_hash", obj.columns)

        missing = obj[obj["score_transverse_v1_status"] != "ok"]

        self.assertEqual(len(missing), 1)
        self.assertTrue(pd.isna(missing.iloc[0]["score_transverse_v1"]))
        self.assertEqual(str(missing.iloc[0]["fit_eligible"]).lower(), "false")

    def test_recommendations_are_not_launchable_candidate_batch_and_laser_is_categorical(
        self,
    ) -> None:
        cfg = load_optimizer_config(self.optimizer_config_path)

        build_observations(cfg, 0)
        build_objectives(cfg, 0)

        path = propose_recommendations(cfg, 0)
        rec = read_table(path, sep="\t")

        self.assertEqual(len(rec), 5)
        self.assertTrue(set(rec["laser_case"]).issubset({"f20", "f32", "f40"}))
        self.assertTrue(set(rec["f_number"]).issubset({20.0, 32.0, 40.0}))
        self.assertTrue(
            rec["recommendation_status"].eq("recommended_placeholder_no_launch").all()
        )
        self.assertFalse(
            (cfg.iteration_dir(0) / "outputs" / "candidate_batch.tsv").exists()
        )

    def test_refuses_raw_hdf5_paths(self) -> None:
        with self.assertRaises(ValueError):
            assert_not_raw_diagnostic_path("case/diags/fields/openpmd_000001.h5")

    def test_objectives_fall_back_past_nan_candidate_metrics(self) -> None:
        cfg = load_optimizer_config(self.optimizer_config_path)

        build_observations(cfg, 0)

        obs_path = cfg.iteration_dir(0) / "inputs" / "observations.csv"
        obs = read_table(obs_path)

        target = obs["source_case_id"].astype(str) == "2"

        # Simulate the CLPU situation:
        # the preferred comparison/global metric column exists but is NaN,
        # while the direct per-case metric is valid.
        obs.loc[target, "metric_particle_beam_transverse_quality_score_channel"] = (
            float("nan")
        )
        obs.loc[target, "metric_particle_beam_transverse_quality_score"] = 0.25

        obs.loc[target, "metric_comparison_beam_beamlike_gain_score"] = float("nan")
        obs.loc[target, "metric_particle_beamlike_score"] = 0.75

        obs.to_csv(obs_path, index=False)

        obj_path = build_objectives(cfg, 0)
        obj = read_table(obj_path)

        row = obj[obj["observation_id"].str.contains("002_f40_chan")].iloc[0]

        self.assertEqual(row["score_transverse_v1_status"], "ok")
        self.assertEqual(
            row["score_transverse_v1_source_metric"],
            "metric_particle_beam_transverse_quality_score",
        )
        self.assertAlmostEqual(float(row["score_transverse_v1"]), 0.25)

        self.assertEqual(row["score_beamlike_v1_status"], "ok")
        self.assertEqual(
            row["score_beamlike_v1_source_metric"],
            "metric_particle_beamlike_score",
        )
        self.assertAlmostEqual(float(row["score_beamlike_v1"]), 0.75)

    def test_objective_score_specs_can_be_configured(self) -> None:
        cfg_path = self.optimizer_config_path
        data = json.loads(cfg_path.read_text())

        data["objective"] = {
            "schema_version": 1,
            "config_id": "test_configurable_score_specs",
            "required_scores_for_fit": [
                "score_guiding_v1",
                "score_beamlike_v1",
                "score_transverse_v1",
            ],
            "score_specs": {
                "score_guiding_v1": ["metric_guiding_a0_peak"],
                "score_beamlike_v1": ["metric_particle_charge_hot_pC"],
                "score_transverse_v1": [
                    "metric_particle_beam_transverse_quality_score"
                ],
            },
        }

        cfg_path.write_text(json.dumps(data), encoding="utf-8")

        cfg = load_optimizer_config(cfg_path)
        build_observations(cfg, 0)
        obj_path = build_objectives(cfg, 0)
        obj = read_table(obj_path)

        eligible = obj[obj["fit_eligible"].astype(str).str.lower() == "true"]
        self.assertGreaterEqual(len(eligible), 2)

        row = eligible.iloc[0]
        self.assertEqual(
            row["score_guiding_v1_source_metric"], "metric_guiding_a0_peak"
        )
        self.assertEqual(
            row["score_beamlike_v1_source_metric"], "metric_particle_charge_hot_pC"
        )

    def test_singlecase_guiding_sidecar_is_ingested_and_preferred(self) -> None:
        cfg = load_optimizer_config(self.optimizer_config_path)

        build_observations(cfg, 0)
        obs = read_table(cfg.iteration_dir(0) / "inputs" / "observations.csv")

        row = obs[obs["observation_id"].str.contains("000_f20_chan")].iloc[0]
        self.assertEqual(row["guiding_singlecase_score_status"], "ok")
        self.assertIn("metric_guiding_singlecase_score_v1", obs.columns)
        self.assertAlmostEqual(
            float(row["metric_guiding_singlecase_score_v1"]),
            42.0,
        )

        obj_path = build_objectives(cfg, 0)
        obj = read_table(obj_path)
        obj_row = obj[obj["observation_id"].str.contains("000_f20_chan")].iloc[0]

        self.assertEqual(obj_row["score_guiding_v1_status"], "ok")
        self.assertEqual(
            obj_row["score_guiding_v1_source_metric"],
            "metric_guiding_singlecase_score_v1",
        )
        self.assertAlmostEqual(float(obj_row["score_guiding_v1"]), 42.0)

    def test_beam_v2_hard_gates_no_electron_cases_even_with_high_guiding(self) -> None:
        cfg_path = self.optimizer_config_path
        data = json.loads(cfg_path.read_text())
        data["objective"] = {
            "schema_version": 2,
            "config_id": "test_capillary_objectives_guiding_longitudinal_transverse_v2",
            "required_scores_for_fit": [
                "score_guiding_v1",
                "score_beam_longitudinal_v2",
                "score_beam_transverse_v2",
            ],
            "derived_scores": [
                "score_beam_longitudinal_v2",
                "score_beam_transverse_v2",
            ],
            "beam_gated_v2": {
                "min_charge_hot_pC": 100.0,
                "min_n_macroparticles_hot": 200.0,
                "min_n_macroparticles_transverse": 200.0,
                "score_scale": 100.0,
            },
        }
        cfg_path.write_text(json.dumps(data), encoding="utf-8")

        cfg = load_optimizer_config(cfg_path)
        build_observations(cfg, 0)

        obs_path = cfg.iteration_dir(0) / "inputs" / "observations.csv"
        obs = read_table(obs_path)

        bad = obs["source_case_id"].astype(str) == "0"
        good = obs["source_case_id"].astype(str) == "1"

        obs.loc[bad, "metric_guiding_singlecase_score_v1"] = 90.0
        obs.loc[bad, "metric_particle_beamlike_score"] = 0.0
        obs.loc[bad, "metric_particle_beam_yield_score"] = 0.0
        obs.loc[bad, "metric_particle_charge_hot_pC"] = 0.153
        obs.loc[bad, "metric_particle_n_macroparticles_hot"] = 14.0
        obs.loc[bad, "metric_particle_n_macroparticles_transverse"] = 1.0
        obs.loc[bad, "metric_particle_E95_hot_MeV"] = 100.0
        obs.loc[bad, "metric_particle_Emax_hot_MeV"] = 120.0
        obs.loc[bad, "metric_particle_beam_transverse_quality_score"] = 1000.0

        obs.loc[good, "metric_guiding_singlecase_score_v1"] = 70.0
        obs.loc[good, "metric_particle_beamlike_score"] = 10.0
        obs.loc[good, "metric_particle_beam_yield_score"] = 20.0
        obs.loc[good, "metric_particle_charge_hot_pC"] = 140.0
        obs.loc[good, "metric_particle_n_macroparticles_hot"] = 300.0
        obs.loc[good, "metric_particle_n_macroparticles_transverse"] = 300.0
        obs.loc[good, "metric_particle_E95_hot_MeV"] = 120.0
        obs.loc[good, "metric_particle_Emax_hot_MeV"] = 150.0
        obs.loc[good, "metric_particle_beam_transverse_quality_score"] = 200.0

        obs.to_csv(obs_path, index=False)

        obj = read_table(build_objectives(cfg, 0))
        bad_row = obj[obj["observation_id"].str.contains("000_f20_chan")].iloc[0]
        good_row = obj[obj["observation_id"].str.contains("001_f32_chan")].iloc[0]

        self.assertEqual(bad_row["score_beam_longitudinal_v2_status"], "ok")
        self.assertEqual(bad_row["score_beam_transverse_v2_status"], "ok")
        self.assertAlmostEqual(float(bad_row["score_beam_longitudinal_v2"]), 0.0)
        self.assertAlmostEqual(float(bad_row["score_beam_transverse_v2"]), 0.0)

        self.assertGreater(
            float(good_row["score_beam_longitudinal_v2"]),
            float(bad_row["score_beam_longitudinal_v2"]),
        )
        self.assertGreater(
            float(good_row["score_beam_transverse_v2"]),
            float(bad_row["score_beam_transverse_v2"]),
        )

    def test_derived_scores_must_be_declared_as_derived_not_score_specs(self) -> None:
        cfg_path = self.optimizer_config_path
        data = json.loads(cfg_path.read_text())
        data["objective"] = {
            "required_scores_for_fit": ["score_beam_longitudinal_v2"],
            "score_specs": {
                "score_beam_longitudinal_v2": ["metric_particle_beam_yield_score"],
            },
        }
        cfg_path.write_text(json.dumps(data), encoding="utf-8")

        cfg = load_optimizer_config(cfg_path)
        build_observations(cfg, 0)

        with self.assertRaisesRegex(ValueError, "cannot override a derived score"):
            build_objectives(cfg, 0)

    def test_default_config_uses_morbo_like_and_beam_gated_v2(self) -> None:
        cfg = load_optimizer_config(self.optimizer_config_path)

        objective = cfg.objective_config()
        recommendation = cfg.recommendation_config()

        self.assertEqual(
            objective["config_id"],
            "capillary_objectives_guiding_longitudinal_transverse_v2",
        )
        self.assertEqual(
            objective["required_scores_for_fit"],
            [
                "score_guiding_v1",
                "score_beam_longitudinal_v2",
                "score_beam_transverse_v2",
            ],
        )
        self.assertEqual(
            objective["derived_scores"],
            [
                "score_beam_longitudinal_v2",
                "score_beam_transverse_v2",
            ],
        )
        self.assertEqual(recommendation["backend"], "morbo_like")
        self.assertEqual(recommendation["suggestion_mode"], "regional_random")
        self.assertEqual(float(objective["beam_gated_v2"]["transverse_ref"]), 0.02)


if __name__ == "__main__":
    unittest.main()
