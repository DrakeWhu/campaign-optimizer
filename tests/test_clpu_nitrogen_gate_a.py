from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.capillary.gate_a import (
    DIRECTED_SCAN_CASE_NAMES,
    DIRECTED_SCAN_NAME,
    EXPECTED_ENCODED_DIM,
    EXPECTED_INITIAL_OBSERVATIONS,
    EXPECTED_PARAMETER_NAMES,
    EXPECTED_REDUCED_OUTPUTS,
    EXPECTED_STATIC_SOURCES,
    GATE_A_CONTRACT_ID,
    _assert_batch_plan,
    _check_config_contract,
    freeze_warm_start_sources,
    gate_a_optimizer_config,
    prepare_gate_a_staging,
)
from campaign_optimizer.capillary.parameters import capillary_search_space
from campaign_optimizer.config import OptimizerConfig, load_optimizer_config
from campaign_optimizer.morbo.search_space import SearchSpaceCodec


def _write_cases(path: Path, count: int, names: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = [f"case_{index:03d}" for index in range(count)]
    pd.DataFrame(
        {
            "CASE_ID": list(range(count)),
            "CASE_NAME": names,
        }
    ).to_csv(path, sep="\t", index=False)


def _source_rows(runs: Path) -> tuple[list[dict], list[dict]]:
    source_rows: list[dict] = []
    evidence: list[dict] = []
    families = (
        ("clpu_capillary_guiding_baseline_soft50", 13),
        ("clpu_capillary_guiding_baseline_soft50_trfix_v1", 6),
        ("clpu_capillary_guiding_baseline_soft50_trfix_v2", 20),
    )
    for family, count in families:
        for iteration in range(count):
            rows = 33 if family == families[0][0] and iteration == 0 else 8
            root = runs / family / "iterations" / f"iter_{iteration:03d}"
            _write_cases(root / "cases.tsv", rows)
            name = f"{family}_iter_{iteration:03d}"
            source_rows.append(
                {
                    "campaign_name": name,
                    "campaign_root": str(root),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": dict(EXPECTED_REDUCED_OUTPUTS),
                    "source_kind": "warm_start_historical",
                    "history_iteration": iteration,
                }
            )
            evidence.append({"campaign_name": name, "rows": rows})
    return source_rows, evidence


def _fixture(tmp: Path) -> dict[str, Path]:
    runs = tmp / "warpx_runs"
    sources, _ = _source_rows(runs)
    ref_dir = (
        runs
        / "clpu_capillary_guiding_baseline_soft50_trfix_v2"
        / "optimizer_runs"
        / "iter_020"
        / "inputs"
    )
    ref_dir.mkdir(parents=True, exist_ok=True)
    manifest = ref_dir / "source_campaigns_effective.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": 20,
                "source_campaigns": sources,
            }
        ),
        encoding="utf-8",
    )
    observations = ref_dir / "observations.csv"
    objectives = ref_dir / "objective_table.csv"
    ids = [f"historical:{index:03d}" for index in range(337)]
    pd.DataFrame({"observation_id": ids}).to_csv(observations, index=False)
    pd.DataFrame(
        {
            "observation_id": ids,
            "objective_status": ["ok"] * 337,
            "fit_eligible": ["true"] * 337,
        }
    ).to_csv(objectives, index=False)

    scan = runs / DIRECTED_SCAN_NAME
    _write_cases(scan / "cases.tsv", 7, list(DIRECTED_SCAN_CASE_NAMES))
    for case_name in DIRECTED_SCAN_CASE_NAMES:
        for relative in EXPECTED_REDUCED_OUTPUTS.values():
            path = scan / case_name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n", encoding="utf-8")

    return {
        "runs": runs,
        "manifest": manifest,
        "observations": observations,
        "objectives": objectives,
        "scan": scan,
    }


def _config_object(tmp: Path, data: dict) -> OptimizerConfig:
    path = tmp / "optimizer.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_optimizer_config(path)


class ClpuNitrogenGateATests(unittest.TestCase):
    def test_gate_a_config_is_six_physical_parameters_eight_encoded_and_no_sobol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = [
                {
                    "campaign_name": f"source_{index:02d}",
                    "campaign_root": str(tmp / f"source_{index:02d}"),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": dict(EXPECTED_REDUCED_OUTPUTS),
                }
                for index in range(EXPECTED_STATIC_SOURCES)
            ]
            data = gate_a_optimizer_config(sources)
            config = _config_object(tmp, data)
            _check_config_contract(config)
            parameter_space = config.parameter_space()
            codec = SearchSpaceCodec(capillary_search_space(parameter_space))

        self.assertEqual(
            tuple(parameter_space["ranges"]),
            EXPECTED_PARAMETER_NAMES[1:],
        )
        self.assertEqual(codec.encoded_dim(), EXPECTED_ENCODED_DIM)
        self.assertNotIn("pulse_duration_factor", parameter_space["ranges"])
        self.assertNotIn("initial_design", data["recommendation"])
        self.assertFalse(data["recommendation"]["botorch"]["fallback_to_random"])
        self.assertEqual(
            data["candidate_batch"]["cap_rmax_um"],
            {"policy": "radius_plus_margin", "margin_um": 30.0},
        )
        self.assertEqual(data["candidate_batch"]["laser_duration_fwhm_fs"], 30.0)

    def test_freeze_warm_start_accepts_exact_audited_39_plus_7(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            paths = _fixture(Path(tmp_name))
            sources, evidence = freeze_warm_start_sources(
                reference_manifest_path=paths["manifest"],
                directed_scan_root=paths["scan"],
                warpx_runs_root=paths["runs"],
            )

        self.assertEqual(len(sources), 40)
        self.assertEqual(len({row["campaign_name"] for row in sources}), 40)
        self.assertEqual(sum(int(row["case_rows"]) for row in evidence), 344)
        self.assertEqual(sources[-1]["campaign_name"], DIRECTED_SCAN_NAME)

    def test_freeze_warm_start_rejects_mutated_genealogy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            paths = _fixture(Path(tmp_name))
            payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            payload["source_campaigns"][0]["campaign_name"] = "wrong_alias"
            paths["manifest"].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "name mismatch"):
                freeze_warm_start_sources(
                    reference_manifest_path=paths["manifest"],
                    directed_scan_root=paths["scan"],
                    warpx_runs_root=paths["runs"],
                )

    def test_freeze_warm_start_rejects_wrong_directed_scan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            paths = _fixture(Path(tmp_name))
            scan_cases = pd.read_csv(paths["scan"] / "cases.tsv", sep="\t")
            scan_cases.loc[6, "CASE_NAME"] = "wrong_case"
            scan_cases.to_csv(paths["scan"] / "cases.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "directed scan case identity mismatch"):
                freeze_warm_start_sources(
                    reference_manifest_path=paths["manifest"],
                    directed_scan_root=paths["scan"],
                    warpx_runs_root=paths["runs"],
                )

    def test_prepare_staging_freezes_hashes_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            paths = _fixture(tmp)
            staging = tmp / "gate_a"
            result = prepare_gate_a_staging(
                staging_root=staging,
                reference_manifest_path=paths["manifest"],
                reference_observations_path=paths["observations"],
                reference_objectives_path=paths["objectives"],
                directed_scan_root=paths["scan"],
                warpx_runs_root=paths["runs"],
            )
            config = load_optimizer_config(staging / "optimizer.json")
            _check_config_contract(config)

            self.assertEqual(result["contract_id"], GATE_A_CONTRACT_ID)
            self.assertEqual(result["source_count"], 40)
            self.assertEqual(result["source_case_rows"], EXPECTED_INITIAL_OBSERVATIONS)
            self.assertTrue(result["optimizer_json"]["sha256"])
            self.assertTrue(result["reference_manifest"]["sha256"])
            self.assertTrue((staging / "provenance" / "gate_a_bootstrap.json").is_file())

            with self.assertRaisesRegex(ValueError, "refusing to reuse existing Gate A artifact"):
                prepare_gate_a_staging(
                    staging_root=staging,
                    reference_manifest_path=paths["manifest"],
                    reference_observations_path=paths["observations"],
                    reference_objectives_path=paths["objectives"],
                    directed_scan_root=paths["scan"],
                    warpx_runs_root=paths["runs"],
                )

    def test_config_contract_rejects_random_fallback_and_seventh_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = [
                {
                    "campaign_name": f"source_{index:02d}",
                    "campaign_root": str(tmp / f"source_{index:02d}"),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": dict(EXPECTED_REDUCED_OUTPUTS),
                }
                for index in range(40)
            ]
            data = gate_a_optimizer_config(sources)
            data["recommendation"]["botorch"]["fallback_to_random"] = True
            with self.assertRaisesRegex(ValueError, "random fallback"):
                _check_config_contract(_config_object(tmp, data))

            data = gate_a_optimizer_config(sources)
            data["parameter_space"]["ranges"]["pulse_duration_factor"] = [0.8, 1.2]
            with self.assertRaisesRegex(ValueError, "numeric parameter ranges mismatch"):
                _check_config_contract(_config_object(tmp, data))

    def test_batch_plan_uses_real_linkage_contract_not_fictitious_count_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = [
                {
                    "campaign_name": f"source_{index:02d}",
                    "campaign_root": str(tmp / f"source_{index:02d}"),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": dict(EXPECTED_REDUCED_OUTPUTS),
                }
                for index in range(40)
            ]
            config = _config_object(tmp, gate_a_optimizer_config(sources))
            plan = {
                "schema_version": 1,
                "plan_type": "optimizer_candidate_batch",
                "optimizer_iteration": 0,
                "objective_config_id": "clpu_baseline_charge_soft50_plateau_exit_v1",
                "candidate_batch": "outputs/candidate_batch.tsv",
                "recommended_candidates": "outputs/recommended_candidates.tsv",
                "campaign_template": config.candidate_batch_config()["campaign_template"],
                "source_campaigns": [
                    {
                        "campaign_name": source["campaign_name"],
                        "campaign_root": source["campaign_root"],
                        "cases_tsv": "cases.tsv",
                        "campaign_json": "campaign.json",
                    }
                    for source in sources
                ],
                "non_goals": [
                    "does not submit jobs",
                    "does not launch WarpX",
                    "does not delete data",
                    "does not create campaign roots",
                ],
            }
            _assert_batch_plan(config, plan)
            self.assertNotIn("candidate_count", plan)
            self.assertNotIn("case_count", plan)

            plan["candidate_batch"] = "outputs/wrong.tsv"
            with self.assertRaisesRegex(ValueError, "candidate_batch linkage mismatch"):
                _assert_batch_plan(config, plan)


if __name__ == "__main__":
    unittest.main()
