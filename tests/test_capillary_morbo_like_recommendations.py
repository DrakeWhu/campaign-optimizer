from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.capillary.batch import recommended_candidates_to_candidate_batch
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.io import read_table
from campaign_optimizer.objectives import build_objectives
from campaign_optimizer.observations import build_observations
from campaign_optimizer.recommend import propose_recommendations


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_case(
    root: Path,
    name: str,
    *,
    guiding_score: float,
    beamlike_score: float,
    transverse_score: float,
) -> None:
    case_dir = root / name
    (case_dir / "particle_analysis").mkdir(parents=True, exist_ok=True)

    _write_json(case_dir / "state.json", {"state": "Reduced_validated"})
    _write_json(
        case_dir / "validation.json",
        {
            "raw": {"fields_openpmd": {"ok": True}},
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
    pd.DataFrame(
        [
            {
                "metric_guiding_singlecase_status": "ok",
                "metric_guiding_singlecase_score_v1": guiding_score,
            }
        ]
    ).to_csv(case_dir / "guiding_singlecase_score.csv", index=False)
    pd.DataFrame(
        [
            {
                "charge_hot_pC": beamlike_score,
                "E95_hot_MeV": 100.0,
                "beam_transverse_quality_score": transverse_score,
            }
        ]
    ).to_csv(case_dir / "particle_analysis" / "particle_summary.csv", index=False)


class CapillaryMorboLikeRecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_ctx.name)
        self.config_path = self._make_fixture(self.tmp)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def _make_fixture(self, tmp_path: Path) -> Path:
        root = tmp_path / "campaign"
        root.mkdir()

        cases = pd.DataFrame(
            [
                {
                    "CASE_ID": 0,
                    "CASE_NAME": "000_f20_chan_n2e18cm3_L5mm_d300um_foc0mm_rz",
                    "LASER_CASE": "f20",
                    "PLASMA_KIND": "chan",
                    "N0_CM3": 2.0e18,
                    "PLATEAU_LENGTH_MM": 5.0,
                    "DIAMETER_UM": 300.0,
                    "RADIUS_UM": 150.0,
                    "FOCUS_OFFSET_FROM_PLATEAU_START_MM": 0.0,
                    "CAP_RMAX_UM": 180.0,
                    "CAP_NR": 192,
                },
                {
                    "CASE_ID": 1,
                    "CASE_NAME": "001_f32_chan_n4e18cm3_L10mm_d350um_foc1mm_rz",
                    "LASER_CASE": "f32",
                    "PLASMA_KIND": "chan",
                    "N0_CM3": 4.0e18,
                    "PLATEAU_LENGTH_MM": 10.0,
                    "DIAMETER_UM": 350.0,
                    "RADIUS_UM": 175.0,
                    "FOCUS_OFFSET_FROM_PLATEAU_START_MM": 1.0,
                    "CAP_RMAX_UM": 210.0,
                    "CAP_NR": 192,
                },
                {
                    "CASE_ID": 2,
                    "CASE_NAME": "002_f40_chan_n5e18cm3_L20mm_d450um_focm2mm_rz",
                    "LASER_CASE": "f40",
                    "PLASMA_KIND": "chan",
                    "N0_CM3": 5.0e18,
                    "PLATEAU_LENGTH_MM": 20.0,
                    "DIAMETER_UM": 450.0,
                    "RADIUS_UM": 225.0,
                    "FOCUS_OFFSET_FROM_PLATEAU_START_MM": -2.0,
                    "CAP_RMAX_UM": 270.0,
                    "CAP_NR": 192,
                },
            ]
        )
        cases.to_csv(root / "cases.tsv", sep="\t", index=False)

        _make_case(
            root,
            str(cases.iloc[0]["CASE_NAME"]),
            guiding_score=1.0,
            beamlike_score=2.0,
            transverse_score=3.0,
        )
        _make_case(
            root,
            str(cases.iloc[1]["CASE_NAME"]),
            guiding_score=2.0,
            beamlike_score=3.0,
            transverse_score=2.0,
        )
        _make_case(
            root,
            str(cases.iloc[2]["CASE_NAME"]),
            guiding_score=3.0,
            beamlike_score=1.0,
            transverse_score=4.0,
        )

        cfg = {
            "schema_version": 1,
            "optimizer_run_root": "optimizer_runs",
            "validation": {"require_case_validation": True},
            "source_campaigns": [
                {
                    "campaign_name": "test_campaign",
                    "campaign_root": str(root),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": {
                        "guiding_metrics": "guiding_metrics.csv",
                        "guiding_singlecase_score": "guiding_singlecase_score.csv",
                        "particle_summary": "particle_analysis/particle_summary.csv",
                    },
                }
            ],
            "objective": {
                "required_scores_for_fit": [
                    "score_guiding_v1",
                    "score_beamlike_v1",
                    "score_transverse_v1",
                ],
                "score_specs": {
                    "score_guiding_v1": ["metric_guiding_singlecase_score_v1"],
                    "score_beamlike_v1": ["metric_particle_charge_hot_pC"],
                    "score_transverse_v1": [
                        "metric_particle_beam_transverse_quality_score"
                    ],
                },
            },
            "recommendation": {
                "backend": "morbo_like",
                "seed": 7,
                "n_candidates": 2,
                "min_observations": 2,
                "regional_policy": {
                    "max_regions": 1,
                    "initial_radius": 0.2,
                    "local_min_observations": 2,
                },
            },
            "candidate_batch": {
                "case_id_start": 10,
                "plasma_kind": "chan",
                "cap_nr": 192,
                "cap_rmax_um": {"policy": "diameter_factor", "factor": 1.2},
            },
        }
        path = tmp_path / "optimizer.json"
        _write_json(path, cfg)
        return path

    def test_propose_recommendations_supports_morbo_like_backend(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        build_observations(cfg, 0)
        build_objectives(cfg, 0)

        path = propose_recommendations(cfg, 0)
        rec = read_table(path, sep="\t")
        outputs_dir = cfg.iteration_dir(0) / "outputs"

        self.assertEqual(len(rec), 2)
        self.assertEqual(set(rec["recommendation_backend"]), {"morbo_like"})
        self.assertEqual(set(rec["surrogate_backend"]), {"morbo_like"})
        self.assertTrue(
            set(rec["morbo_strategy"]).issubset({"global_random", "regional_random"})
        )
        self.assertIn("candidate_signature", rec.columns)
        self.assertIn("region_id", rec.columns)
        self.assertTrue((outputs_dir / "morbo_optimizer_state.json").is_file())
        self.assertTrue((outputs_dir / "morbo_frontier.json").is_file())
        self.assertTrue((outputs_dir / "morbo_regions.json").is_file())

        summary = json.loads(
            (outputs_dir / "surrogate_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["backend"], "morbo_like")
        self.assertEqual(summary["surrogate_backend"], "morbo_like_no_botorch")
        self.assertEqual(summary["candidate_rows"], 2)

        batch = recommended_candidates_to_candidate_batch(
            rec,
            iteration=0,
            objective_config_id=cfg.objective_config()["config_id"],
            parameter_space=cfg.parameter_space(),
            batch_config=cfg.candidate_batch_config(),
        )
        self.assertEqual(len(batch), 2)
        self.assertEqual(list(batch["CASE_ID"]), [10, 11])
        self.assertTrue(
            batch["OPT_CANDIDATE_ID"].astype(str).str.startswith("morbo_000_").all()
        )

    def test_morbo_like_backend_is_selectable_without_new_cli_command(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        self.assertEqual(cfg.recommendation_config()["backend"], "morbo_like")


if __name__ == "__main__":
    unittest.main()
