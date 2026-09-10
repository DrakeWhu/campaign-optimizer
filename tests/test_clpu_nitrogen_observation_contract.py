from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.problems import build_objectives, build_observations


CONTRACT = "clpu_n2_plateau_all_electrons_v1"
SCOPES = [
    "all_electrons",
    "preionized_background_electrons",
    "nitrogen_ionized_electrons",
]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ClpuNitrogenObservationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.campaign = self.tmp / "source_campaign"
        self.case_name = "000_f32_chan_n4e18cm3_L10mm_d300um_foc0mm_rz"
        self.case_dir = self.campaign / self.case_name
        self.plateau = self.case_dir / "particle_analysis" / "plateau_exit"
        self.plateau.mkdir(parents=True)
        self._write_case_table(nitrogen_fraction=0.005)
        self._write_state_and_validation()
        self._write_summary()
        self._write_curve()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_case_table(self, *, nitrogen_fraction: float | None) -> None:
        row: dict[str, object] = {
            "CASE_ID": 0,
            "CASE_NAME": self.case_name,
            "LASER_CASE": "f32",
            "PLASMA_KIND": "chan",
            "N0_CM3": 4.0e18,
            "PLATEAU_LENGTH_MM": 10.0,
            "DIAMETER_UM": 300.0,
            "RADIUS_UM": 150.0,
            "FOCUS_OFFSET_FROM_PLATEAU_START_MM": 0.0,
            "CAP_RMAX_UM": 180.0,
            "CAP_NR": 192,
            "LASER_DURATION_FWHM_FS": 30.0,
        }
        if nitrogen_fraction is not None:
            row["NITROGEN_DOPANT_FRACTION"] = nitrogen_fraction
        pd.DataFrame([row]).to_csv(self.campaign / "cases.tsv", sep="\t", index=False)

    def _write_state_and_validation(self) -> None:
        write_json(self.case_dir / "state.json", {"state": "Reduced_validated"})
        write_json(
            self.case_dir / "validation.json",
            {
                "raw": {"fields_openpmd": {"ok": True}},
                "reduced": {
                    "particle_summary": {"ok": True},
                    "particle_soft50_curves": {"ok": True},
                },
            },
        )

    def _summary_rows(self) -> list[dict[str, object]]:
        counts = {
            "all_electrons": 120,
            "preionized_background_electrons": 100,
            "nitrogen_ionized_electrons": 20,
        }
        charge = {
            "all_electrons": 123.0,
            "preionized_background_electrons": 80.0,
            "nitrogen_ionized_electrons": 43.0,
        }
        hard = {
            "all_electrons": 101.0,
            "preionized_background_electrons": 70.0,
            "nitrogen_ionized_electrons": 31.0,
        }
        rows: list[dict[str, object]] = []
        for scope in SCOPES:
            rows.append(
                {
                    "species_scope": scope,
                    "selection_mode": "exit",
                    "particle_exit_selection_policy": "exact_resolved_v1",
                    "selected_particle_iteration": 60973,
                    "soft50_schema_version": "soft50_v2",
                    "soft50_energy_low_MeV": 10.0,
                    "soft50_energy_target_MeV": 50.0,
                    "charge_soft50_pC": charge[scope],
                    "charge_Ege50MeV_pC": hard[scope],
                    "n_effective_soft50": 300.0,
                    "energy_p90_soft50_MeV": 80.0,
                    "forward_only": True,
                    "n_macroparticles_total": counts[scope],
                }
            )
        return rows

    def _write_summary(self, rows: list[dict[str, object]] | None = None) -> None:
        pd.DataFrame(rows or self._summary_rows()).to_csv(
            self.plateau / "particle_summary.csv",
            index=False,
        )

    def _curve_rows(self) -> list[dict[str, object]]:
        charge = {
            "all_electrons": 123.0,
            "preionized_background_electrons": 80.0,
            "nitrogen_ionized_electrons": 43.0,
        }
        hard = {
            "all_electrons": 101.0,
            "preionized_background_electrons": 70.0,
            "nitrogen_ionized_electrons": 31.0,
        }
        rows: list[dict[str, object]] = []
        for scope in SCOPES:
            rows.append(
                {
                    "species_scope": scope,
                    "selection_mode": "exit",
                    "selected_particle_iteration": 60973,
                    "soft50_schema_version": "soft50_v2",
                    "soft50_energy_low_MeV": 10.0,
                    "soft50_energy_target_MeV": 50.0,
                    "charge_soft50_pC": charge[scope],
                    "charge_Ege50MeV_pC": hard[scope],
                    "n_effective_soft50": 300.0,
                    "energy_p90_soft50_MeV": 999.0,
                    "forward_only": True,
                }
            )
        return rows

    def _write_curve(self, rows: list[dict[str, object]] | None = None) -> None:
        pd.DataFrame(rows or self._curve_rows()).to_csv(
            self.plateau / "particle_soft50_curves.csv",
            index=False,
        )

    def _config(self, *, nitrogen_contract: bool) -> Path:
        source: dict[str, object] = {
            "campaign_name": "fixture_source",
            "campaign_root": str(self.campaign),
            "cases_tsv": "cases.tsv",
            "reduced_outputs": {
                "guiding_metrics": None,
                "guiding_singlecase_score": None,
                "particle_summary": "particle_analysis/plateau_exit/particle_summary.csv",
                "acceptance_curves": None,
                "soft50_curves": "particle_analysis/plateau_exit/particle_soft50_curves.csv",
            },
        }
        if nitrogen_contract:
            source["particle_observation_contract"] = CONTRACT
        payload = {
            "schema_version": 1,
            "problem": "capillary",
            "optimizer_run_root": "optimizer_runs",
            "source_campaigns": [source],
            "optimization_history": {"enabled": False},
            "parameter_space": {
                "version": 2,
                "laser_cases": ["f20", "f32", "f40"],
                "ranges": {
                    "n0_1e18cm3": [0.7, 6.0],
                    "plateau_mm_num": [5.0, 25.0],
                    "diameter_um_num": [150.0, 500.0],
                    "focus_mm_num": [-5.0, 5.0],
                    "nitrogen_fraction": [0.0, 0.01],
                },
            },
            "objective": {
                "schema_version": 1,
                "config_id": "clpu_n2_direct_soft50_test",
                "required_scores_for_fit": ["score_charge_soft50_direct_v1"],
                "score_specs": {
                    "score_charge_soft50_direct_v1": [
                        "metric_particle_charge_soft50_pC"
                    ]
                },
                "derived_scores": [],
                "soft50_v1": {
                    "energy_low_MeV": 10.0,
                    "energy_target_MeV": 50.0,
                },
            },
            "validation": {"require_case_validation": True},
        }
        path = self.tmp / "optimizer.json"
        write_json(path, payload)
        return path

    def _build(self, *, nitrogen_contract: bool) -> tuple[pd.Series, pd.Series]:
        cfg = load_optimizer_config(self._config(nitrogen_contract=nitrogen_contract))
        observations = pd.read_csv(build_observations(cfg, 0))
        objectives = pd.read_csv(build_objectives(cfg, 0))
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(objectives), 1)
        return observations.iloc[0], objectives.iloc[0]

    def test_historical_single_species_missing_all_curve_remains_fit_eligible(self) -> None:
        self._write_case_table(nitrogen_fraction=None)
        summary = self._summary_rows()[1].copy()
        summary["species_scope"] = "preionized_background_electrons"
        self._write_summary([summary])
        curve = self._curve_rows()[1].copy()
        curve["species_scope"] = "preionized_background_electrons"
        curve["charge_soft50_pC"] = 999.0
        self._write_curve([curve])

        observation, objective = self._build(nitrogen_contract=False)

        self.assertEqual(observation["soft50_curves_status"], "missing_metric")
        self.assertEqual(observation["reduced_validation_status"], "ok")
        self.assertEqual(observation["particle_metric_origin"], "particle_summary:first_row_legacy")
        self.assertAlmostEqual(float(observation["metric_particle_charge_soft50_pC"]), 80.0)
        self.assertEqual(str(objective["fit_eligible"]).lower(), "true")
        self.assertAlmostEqual(float(objective["score_charge_soft50_direct_v1"]), 80.0)
        self.assertAlmostEqual(float(observation["nitrogen_fraction"]), 0.0)

    def test_nitrogen_summary_is_authoritative_and_matching_curve_is_concordance_only(self) -> None:
        observation, objective = self._build(nitrogen_contract=True)

        self.assertEqual(observation["particle_observation_contract"], CONTRACT)
        self.assertEqual(observation["particle_metric_origin"], "particle_summary:first_row_all_electrons")
        self.assertEqual(observation["soft50_curve_concordance_status"], "ok")
        self.assertAlmostEqual(float(observation["metric_particle_charge_soft50_pC"]), 123.0)
        self.assertAlmostEqual(float(observation["metric_particle_charge_Ege50MeV_pC"]), 101.0)
        self.assertAlmostEqual(float(observation["metric_particle_energy_p90_soft50_MeV"]), 80.0)
        self.assertEqual(str(objective["fit_eligible"]).lower(), "true")
        self.assertAlmostEqual(float(objective["score_charge_soft50_direct_v1"]), 123.0)

    def test_nitrogen_curve_charge_disagreement_rejects_without_overwriting_summary(self) -> None:
        rows = self._curve_rows()
        rows[0]["charge_soft50_pC"] = 999.0
        self._write_curve(rows)

        observation, objective = self._build(nitrogen_contract=True)

        self.assertAlmostEqual(float(observation["metric_particle_charge_soft50_pC"]), 123.0)
        self.assertEqual(
            observation["soft50_curve_concordance_status"],
            "particle_observation_contract_failed",
        )
        self.assertEqual(
            observation["reduced_validation_status"],
            "particle_observation_contract_failed",
        )
        self.assertIn("charge_soft50_pC differs", observation["failure_reason"])
        self.assertEqual(str(objective["fit_eligible"]).lower(), "false")

    def test_nitrogen_hard50_disagreement_rejects_observation(self) -> None:
        rows = self._curve_rows()
        rows[0]["charge_Ege50MeV_pC"] = 100.5
        self._write_curve(rows)

        observation, objective = self._build(nitrogen_contract=True)

        self.assertIn("charge_Ege50MeV_pC differs", observation["failure_reason"])
        self.assertEqual(str(objective["fit_eligible"]).lower(), "false")

    def test_nitrogen_reordered_summary_scopes_are_rejected(self) -> None:
        rows = self._summary_rows()
        rows[0], rows[1] = rows[1], rows[0]
        self._write_summary(rows)

        observation, objective = self._build(nitrogen_contract=True)

        self.assertEqual(observation["particle_summary_status"], "particle_observation_contract_failed")
        self.assertEqual(observation["failure_kind"], "particle_observation_contract")
        self.assertEqual(str(objective["fit_eligible"]).lower(), "false")

    def test_nitrogen_duplicate_summary_scope_is_rejected(self) -> None:
        rows = self._summary_rows()
        rows.append(dict(rows[-1]))
        self._write_summary(rows)

        observation, objective = self._build(nitrogen_contract=True)

        self.assertEqual(observation["particle_summary_status"], "particle_observation_contract_failed")
        self.assertEqual(str(objective["fit_eligible"]).lower(), "false")

    def test_nitrogen_duplicate_all_electrons_curve_row_is_rejected(self) -> None:
        rows = self._curve_rows()
        rows.insert(1, dict(rows[0]))
        self._write_curve(rows)

        observation, objective = self._build(nitrogen_contract=True)

        self.assertEqual(observation["soft50_curves_status"], "particle_observation_contract_failed")
        self.assertIn("exactly one all_electrons row", observation["failure_reason"])
        self.assertEqual(str(objective["fit_eligible"]).lower(), "false")

    def test_nitrogen_curve_iteration_or_threshold_mismatch_is_rejected(self) -> None:
        rows = self._curve_rows()
        rows[0]["selected_particle_iteration"] = 60974
        rows[0]["soft50_energy_target_MeV"] = 55.0
        self._write_curve(rows)

        observation, objective = self._build(nitrogen_contract=True)

        self.assertEqual(
            observation["soft50_curve_concordance_status"],
            "particle_observation_contract_failed",
        )
        self.assertEqual(str(objective["fit_eligible"]).lower(), "false")


if __name__ == "__main__":
    unittest.main()
