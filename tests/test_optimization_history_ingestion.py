from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.io import read_json_optional, read_table
from campaign_optimizer.observations import build_observations


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_case(iter_root: Path, case_id: int, case_name: str, state: str) -> None:
    case_dir = iter_root / case_name
    (case_dir / "particle_analysis").mkdir(parents=True, exist_ok=True)

    write_json(case_dir / "state.json", {"state": state})
    write_json(
        case_dir / "validation.json",
        {
            "raw": {"fields_openpmd": {"ok": True}},
            "reduced": {
                "guiding_metrics": {"ok": True},
                "particle_summary": {"ok": True},
            },
        },
    )

    pd.DataFrame([{"iteration": case_id, "a0_peak": 1.0 + case_id}]).to_csv(
        case_dir / "guiding_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "charge_hot_pC": 0.1 + case_id,
                "E95_hot_MeV": 10.0 + case_id,
                "beam_transverse_quality_score": 0.5 + case_id,
            }
        ]
    ).to_csv(case_dir / "particle_analysis" / "particle_summary.csv", index=False)


def make_iteration(root: Path, iteration: int) -> None:
    iter_root = root / "iterations" / f"iter_{iteration:03d}"
    iter_root.mkdir(parents=True, exist_ok=True)

    case_name = f"{iteration:03d}_f20_chan_n4e18cm3_L5mm_d300um_foc0mm_rz"
    pd.DataFrame(
        [
            {
                "CASE_ID": iteration,
                "CASE_NAME": case_name,
                "LASER_CASE": "f20",
                "PLASMA_KIND": "chan",
                "N0_CM3": 4.0e18,
                "PLATEAU_LENGTH_MM": 5.0,
                "DIAMETER_UM": 300.0,
                "RADIUS_UM": 150.0,
                "FOCUS_OFFSET_FROM_PLATEAU_START_MM": 0.0,
                "CAP_RMAX_UM": 180.0,
                "CAP_NR": 192,
            }
        ]
    ).to_csv(iter_root / "cases.tsv", sep="\t", index=False)

    make_case(iter_root, iteration, case_name, "Raw_delete_eligible")


class TestOptimizationHistoryIngestion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_ctx.name)
        self.root = self.tmp / "clpu_capillary_guiding_bo_001"
        self.root.mkdir()

        make_iteration(self.root, 0)
        make_iteration(self.root, 1)
        make_iteration(self.root, 2)
        make_iteration(self.root, 901)

        config = {
            "schema_version": 1,
            "optimizer_run_root": "optimizer_runs",
            "validation": {"require_case_validation": True},
            "source_campaigns": [
                {
                    "campaign_name": "external_baseline",
                    "campaign_root": str(self.root / "iterations" / "iter_000"),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": {
                        "guiding_metrics": "guiding_metrics.csv",
                        "particle_summary": "particle_analysis/particle_summary.csv",
                    },
                }
            ],
        }
        self.config_path = self.root / "optimizer.json"
        write_json(self.config_path, config)

    def tearDown(self) -> None:
        self.tmp_ctx.cleanup()

    def test_iteration_two_includes_previous_history_only_once(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        path = build_observations(cfg, 2)
        obs = read_table(path)

        campaigns = set(obs["source_campaign_name"])

        self.assertIn("external_baseline", campaigns)
        self.assertIn("clpu_capillary_guiding_bo_001_iter_001", campaigns)
        self.assertNotIn("clpu_capillary_guiding_bo_001_iter_002", campaigns)
        self.assertNotIn("clpu_capillary_guiding_bo_001_iter_901", campaigns)

        self.assertEqual(len(obs), 2)

        roots = [Path(p).resolve() for p in obs["source_campaign_root"]]
        self.assertEqual(len(roots), len(set(map(str, roots))))

    def test_effective_source_manifest_is_written(self) -> None:
        cfg = load_optimizer_config(self.config_path)
        build_observations(cfg, 2)

        manifest = read_json_optional(
            self.root
            / "optimizer_runs"
            / "iter_002"
            / "inputs"
            / "source_campaigns_effective.json"
        )

        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["iteration"], 2)

        names = {source["campaign_name"] for source in manifest["source_campaigns"]}
        self.assertIn("external_baseline", names)
        self.assertIn("clpu_capillary_guiding_bo_001_iter_001", names)


if __name__ == "__main__":
    unittest.main()
