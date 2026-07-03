from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.morbo.audit import (
    audit_morbo_outputs,
    render_morbo_audit_text,
    write_morbo_audit_json,
)
from campaign_optimizer.morbo.search_space import Choice, FloatRange, SearchSpaceCodec


class MorboAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp_ctx.name) / "outputs"
        self.out.mkdir(parents=True)
        self.codec = SearchSpaceCodec(
            {
                "laser_case": Choice(["f20", "f32", "f40"]),
                "n0_1e18cm3": FloatRange(0.7, 6.0),
                "plateau_mm_num": FloatRange(5.0, 25.0),
                "diameter_um_num": FloatRange(150.0, 500.0),
                "focus_mm_num": FloatRange(-5.0, 5.0),
            }
        )
        self._write_fixture()

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def _write_fixture(self) -> None:
        pd.DataFrame(
            [
                {
                    "candidate_id": "morbo_001_000",
                    "rank": 1,
                    "candidate_source": "regional_model",
                    "morbo_strategy": "regional_model",
                    "acquisition_value": 10.0,
                    "region_id": "region_001",
                    "laser_case": "f20",
                    "n0_1e18cm3": 5.0,
                    "plateau_mm_num": 20.0,
                    "diameter_um_num": 300.0,
                    "focus_mm_num": 0.0,
                },
                {
                    "candidate_id": "morbo_001_001",
                    "rank": 2,
                    "candidate_source": "regional_model",
                    "morbo_strategy": "regional_model",
                    "acquisition_value": 8.0,
                    "region_id": "region_001",
                    "laser_case": "f20",
                    "n0_1e18cm3": 5.5,
                    "plateau_mm_num": 21.0,
                    "diameter_um_num": 320.0,
                    "focus_mm_num": 1.0,
                },
            ]
        ).to_csv(self.out / "recommended_candidates.tsv", sep="\t", index=False)

        regions = [
            {
                "region_id": "region_000",
                "status": "active",
                "center": self.codec.encode(
                    {
                        "laser_case": "f20",
                        "n0_1e18cm3": 3.0,
                        "plateau_mm_num": 25.0,
                        "diameter_um_num": 500.0,
                        "focus_mm_num": 5.0,
                    }
                ),
                "radius": 0.25,
                "observation_count": 5,
                "frontier_count": 3,
                "metadata": {"seed_candidate_id": "campaign:case_f20"},
            },
            {
                "region_id": "region_001",
                "status": "active",
                "center": self.codec.encode(
                    {
                        "laser_case": "f20",
                        "n0_1e18cm3": 5.0,
                        "plateau_mm_num": 20.0,
                        "diameter_um_num": 300.0,
                        "focus_mm_num": 0.0,
                    }
                ),
                "radius": 0.25,
                "observation_count": 4,
                "frontier_count": 2,
                "metadata": {"seed_candidate_id": "campaign:case_f20_b"},
            },
        ]
        (self.out / "morbo_regions.json").write_text(
            json.dumps(regions), encoding="utf-8"
        )

        frontier = [
            {
                "candidate_id": "campaign:053_f20_case",
                "objective_values": {"score_a": 1.0, "score_b": 2.0},
                "canonical_objective_values": {"score_a": 1.0, "score_b": 2.0},
            },
            {
                "candidate_id": "campaign:100_f32_case",
                "objective_values": {"score_a": 3.0, "score_b": 1.0},
                "canonical_objective_values": {"score_a": 3.0, "score_b": 1.0},
            },
        ]
        (self.out / "morbo_frontier.json").write_text(
            json.dumps(frontier), encoding="utf-8"
        )

        state = {
            "schema_version": "morbo_like_state_v1",
            "backend": "morbo_like",
            "last_strategy": "regional_model",
            "candidate_registry": {
                "seen_candidate_signatures": ["a", "b", "c"],
                "pending_candidate_signatures": ["b", "c"],
            },
            "regions": regions,
            "ref_point_raw": {"score_a": -1.0},
            "ref_point_model_units": {"score_a": -0.5},
            "y_transform": {"type": "botorch_outcome_standardize"},
        }
        (self.out / "morbo_optimizer_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )

        surrogate = {
            "backend": "morbo_like",
            "surrogate_backend": "morbo_like_botorch_qnehvi",
            "last_strategy": "regional_model",
            "suggestion_mode": "regional_model",
            "model_diagnostics": {
                "status": "ok",
                "reason": None,
                "acquisition_function": "qLogNoisyExpectedHypervolumeImprovement",
                "train_rows": 241,
                "candidate_pool_rows": 64,
                "selected_rows": 2,
                "objective_names": ["score_a", "score_b"],
                "candidate_pool_categorical_counts": {
                    "laser_case": [
                        {"value": "f20", "count": 8, "fraction": 0.8},
                        {"value": "f32", "count": 2, "fraction": 0.2},
                    ]
                },
                "selected_categorical_counts": {
                    "laser_case": [
                        {"value": "f20", "count": 2, "fraction": 1.0},
                    ]
                },
            },
        }
        (self.out / "surrogate_summary.json").write_text(
            json.dumps(surrogate), encoding="utf-8"
        )

    def test_audit_morbo_outputs_summarizes_recommendations_regions_and_frontier(
        self,
    ) -> None:
        summary = audit_morbo_outputs(self.out, codec=self.codec)

        self.assertEqual(summary["recommendations"]["row_count"], 2)
        self.assertEqual(
            summary["recommendations"]["numeric"]["acquisition_value"],
            {"count": 2, "min": 8.0, "max": 10.0, "mean": 9.0},
        )
        self.assertEqual(summary["regions"]["row_count"], 2)
        self.assertEqual(summary["regions"]["active_count"], 2)
        self.assertEqual(
            summary["regions"]["records"][0]["decoded_center"]["laser_case"], "f20"
        )
        self.assertEqual(summary["frontier"]["row_count"], 2)
        self.assertEqual(
            summary["frontier"]["candidate_id_token_counts"]["laser_case"][0],
            {"value": "f20", "count": 1, "fraction": 0.5},
        )
        self.assertIn(
            "All recommendations share laser_case=f20",
            "\n".join(summary["warnings"]),
        )
        self.assertIn(
            "Model pool explored multiple laser_case values",
            "\n".join(summary["warnings"]),
        )

    def test_write_json_and_render_text(self) -> None:
        path = write_morbo_audit_json(self.out, codec=self.codec)
        self.assertTrue(path.is_file())

        loaded = json.loads(path.read_text(encoding="utf-8"))
        text = render_morbo_audit_text(loaded)

        self.assertIn("MORBO-like audit", text)
        self.assertIn("surrogate_backend: morbo_like_botorch_qnehvi", text)
        self.assertIn("laser_case: f20=2", text)
        self.assertIn("Warnings", text)
        self.assertIn("Model candidate pool", text)
        self.assertIn("pool laser_case: f20=8, f32=2", text)
        self.assertIn("selected laser_case: f20=2", text)

    def test_audit_handles_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = audit_morbo_outputs(tmpdir, codec=self.codec)

        self.assertEqual(summary["recommendations"]["row_count"], 0)
        self.assertIn("Missing MORBO output file", "\n".join(summary["warnings"]))


if __name__ == "__main__":
    unittest.main()
