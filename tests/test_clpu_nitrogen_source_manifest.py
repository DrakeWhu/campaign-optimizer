from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from campaign_optimizer.config import load_optimizer_config


REDUCED = {
    "guiding_metrics": "guiding_metrics.csv",
    "guiding_singlecase_score": "guiding_singlecase_score.csv",
    "particle_summary": "particle_analysis/plateau_exit/particle_summary.csv",
    "acceptance_curves": "particle_analysis/plateau_exit/particle_acceptance_curves.csv",
    "soft50_curves": "particle_analysis/plateau_exit/particle_soft50_curves.csv",
}


def _sources(tmp: Path, count: int = 40) -> list[dict]:
    return [
        {
            "campaign_name": f"historical_{index:02d}",
            "campaign_root": str(tmp / "historical" / f"source_{index:02d}"),
            "cases_tsv": "cases.tsv",
            "reduced_outputs": dict(REDUCED),
            "source_kind": "warm_start_historical",
        }
        for index in range(count)
    ]


def _config(
    tmp: Path,
    sources: list[dict],
    *,
    strict: bool = True,
    history_name_template: str = "clpu_n2_iter_{iteration:03d}",
) -> Path:
    payload = {
        "schema_version": 1,
        "problem": "capillary",
        "optimizer_run_root": "optimizer_runs",
        "source_campaigns": sources,
        "optimization_history": {
            "enabled": True,
            "optimization_name": "clpu_n2",
            "iterations_root": "iterations",
            "cases_tsv": "cases.tsv",
            "campaign_name_template": history_name_template,
            "particle_observation_contract": "clpu_n2_plateau_all_electrons_v1",
            "reduced_outputs": dict(REDUCED),
        },
        "recommendation": {
            "backend": "morbo_like",
            "n_candidates": 8,
        },
    }
    if strict:
        payload["source_manifest_contract"] = {
            "contract_id": "strict_static_sources_v1",
            "expected_static_source_count": 40,
        }
    path = tmp / "optimizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ClpuNitrogenSourceManifestTests(unittest.TestCase):
    def test_strict_manifest_accepts_exactly_40_unique_static_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = load_optimizer_config(_config(tmp, _sources(tmp)))
            effective = cfg.source_campaigns_for_iteration(0)

        self.assertEqual(len(effective), 40)
        self.assertEqual(len({row["campaign_name"] for row in effective}), 40)
        self.assertEqual(len({row["campaign_root"] for row in effective}), 40)

    def test_repeated_static_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = _sources(tmp)
            sources[-1]["campaign_name"] = sources[0]["campaign_name"]
            cfg = load_optimizer_config(_config(tmp, sources))
            with self.assertRaisesRegex(ValueError, "duplicate campaign_name"):
                cfg.source_campaigns_for_iteration(0)

    def test_repeated_resolved_root_is_rejected_instead_of_silently_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = _sources(tmp)
            sources[-1]["campaign_root"] = sources[0]["campaign_root"]
            cfg = load_optimizer_config(_config(tmp, sources))
            with self.assertRaisesRegex(ValueError, "duplicate resolved campaign_root"):
                cfg.source_campaigns_for_iteration(0)

    def test_accidental_consolidated_source_41_is_rejected_by_static_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = _sources(tmp)
            sources.append(
                {
                    "campaign_name": "consolidated_reference_must_not_be_a_source",
                    "campaign_root": str(tmp / "consolidated_reference"),
                    "cases_tsv": "cases.tsv",
                    "reduced_outputs": dict(REDUCED),
                }
            )
            cfg = load_optimizer_config(_config(tmp, sources))
            with self.assertRaisesRegex(ValueError, "expected 40 static sources, got 41"):
                cfg.source_campaigns_for_iteration(0)

    def test_own_history_is_empty_at_k0_and_added_once_at_k1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            cfg = load_optimizer_config(_config(tmp, _sources(tmp)))

            at_k0 = cfg.source_campaigns_for_iteration(0)
            self.assertEqual(len(at_k0), 40)
            self.assertFalse(
                any(row.get("source_kind") == "optimization_history" for row in at_k0)
            )

            own = tmp / "iterations" / "iter_000"
            own.mkdir(parents=True)
            (own / "cases.tsv").write_text("CASE_ID\tCASE_NAME\n", encoding="utf-8")

            at_k1 = cfg.source_campaigns_for_iteration(1)
            self.assertEqual(len(at_k1), 41)
            history = [
                row
                for row in at_k1
                if row.get("source_kind") == "optimization_history"
            ]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["history_iteration"], 0)
            self.assertEqual(history[0]["campaign_name"], "clpu_n2_iter_000")
            self.assertEqual(
                history[0]["particle_observation_contract"],
                "clpu_n2_plateau_all_electrons_v1",
            )
            self.assertEqual(Path(history[0]["campaign_root"]).resolve(), own.resolve())

    def test_own_history_alias_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = _sources(tmp)
            cfg = load_optimizer_config(
                _config(
                    tmp,
                    sources,
                    history_name_template=sources[0]["campaign_name"],
                )
            )
            own = tmp / "iterations" / "iter_000"
            own.mkdir(parents=True)
            (own / "cases.tsv").write_text("CASE_ID\tCASE_NAME\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate campaign_name"):
                cfg.source_campaigns_for_iteration(1)

    def test_static_source_cannot_alias_the_new_campaign_own_history_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            own = tmp / "iterations" / "iter_000"
            own.mkdir(parents=True)
            (own / "cases.tsv").write_text("CASE_ID\tCASE_NAME\n", encoding="utf-8")

            sources = _sources(tmp)
            sources[0]["campaign_root"] = str(own)
            cfg = load_optimizer_config(_config(tmp, sources))

            with self.assertRaisesRegex(ValueError, "duplicate resolved campaign_root"):
                cfg.source_campaigns_for_iteration(1)

    def test_legacy_source_resolution_keeps_historical_silent_root_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            sources = _sources(tmp, count=2)
            sources[1]["campaign_root"] = sources[0]["campaign_root"]
            cfg = load_optimizer_config(_config(tmp, sources, strict=False))
            effective = cfg.source_campaigns_for_iteration(0)

        self.assertEqual(len(effective), 1)
        self.assertEqual(effective[0]["campaign_name"], "historical_00")


if __name__ == "__main__":
    unittest.main()
