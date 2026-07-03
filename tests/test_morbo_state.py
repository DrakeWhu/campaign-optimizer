from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from campaign_optimizer.morbo.frontier import FrontierRecord
from campaign_optimizer.morbo.regions import ACTIVE, RegionRecord
from campaign_optimizer.morbo.state import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    CandidateRegistry,
    OptimizerState,
    load_candidate_registry,
    load_frontier_records,
    load_optimizer_state,
    load_region_records,
    save_candidate_registry,
    save_frontier_records,
    save_optimizer_state,
    save_region_records,
    write_json_atomic,
)


class TestMorboState(unittest.TestCase):
    def test_candidate_registry_dedupes_sorts_and_tracks_pending_as_seen(self) -> None:
        registry = CandidateRegistry(
            seen_candidate_signatures=("b", "a", "a"),
            pending_candidate_signatures=("c", "b", "c"),
        )

        self.assertEqual(registry.seen_candidate_signatures, ("a", "b"))
        self.assertEqual(registry.pending_candidate_signatures, ("b", "c"))

        updated = registry.mark_pending(("d", "a"))
        self.assertEqual(updated.seen_candidate_signatures, ("a", "b", "d"))
        self.assertEqual(updated.pending_candidate_signatures, ("a", "b", "c", "d"))

        cleared = updated.clear_pending(("b", "d"))
        self.assertEqual(cleared.seen_candidate_signatures, ("a", "b", "d"))
        self.assertEqual(cleared.pending_candidate_signatures, ("a", "c"))

    def test_optimizer_state_roundtrip_preserves_regions_and_registry(self) -> None:
        region = RegionRecord(
            region_id="region_000",
            status=ACTIVE,
            center=[0.5, 1.0, 0.0],
            radius=0.2,
            metadata={"seed_candidate_id": "case_001"},
        )
        state = OptimizerState(
            seed=123,
            objective_spec={
                "objectives": [
                    {"name": "beam_energy", "metric": "mean_energy", "sense": "max"},
                ]
            },
            search_space_signature="space-v1",
            candidate_registry=CandidateRegistry(
                seen_candidate_signatures=("sig_a",),
                pending_candidate_signatures=("sig_b",),
            ),
            ref_point_policy={"name": "observed_min_margin"},
            ref_point_raw={"beam_energy": 10.0},
            ref_point_model_units={"beam_energy": -1.0},
            y_transform={"beam_energy": {"mean": 100.0, "std": 20.0}},
            regions=(region,),
            last_strategy="regional_random",
            extra={"note": "unit-test"},
        )

        payload = state.as_dict()
        restored = OptimizerState.from_dict(payload)

        self.assertEqual(restored.schema_version, SCHEMA_VERSION)
        self.assertEqual(restored.backend, BACKEND_NAME)
        self.assertEqual(restored.seed, 123)
        self.assertEqual(restored.search_space_signature, "space-v1")
        self.assertEqual(
            restored.candidate_registry.seen_candidate_signatures, ("sig_a",)
        )
        self.assertEqual(
            restored.candidate_registry.pending_candidate_signatures, ("sig_b",)
        )
        self.assertEqual(restored.regions, (region,))
        self.assertEqual(restored.last_strategy, "regional_random")
        self.assertEqual(restored.extra, {"note": "unit-test"})

    def test_optimizer_state_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optimizer_state.json"
            state = OptimizerState(seed=7, search_space_signature="abc")

            save_optimizer_state(path, state)
            restored = load_optimizer_state(path)

            self.assertEqual(restored.seed, 7)
            self.assertEqual(restored.search_space_signature, "abc")
            self.assertEqual(restored.schema_version, SCHEMA_VERSION)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["backend"], BACKEND_NAME)
            self.assertIn("candidate_registry", payload)
            self.assertIn("created_at", payload)
            self.assertIn("updated_at", payload)

    def test_optimizer_state_with_updates_refreshes_updated_at(self) -> None:
        state = OptimizerState(
            seed=1, created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z"
        )
        updated = state.with_updates(last_strategy="global_random")

        self.assertEqual(updated.created_at, "2026-01-01T00:00:00Z")
        self.assertEqual(updated.last_strategy, "global_random")
        self.assertNotEqual(updated.updated_at, state.updated_at)

    def test_frontier_records_file_roundtrip(self) -> None:
        records = [
            FrontierRecord(
                candidate_id="case_001",
                objective_values={"beam_energy": 120.0, "beam_emittance": 2.0},
                canonical_objective_values={
                    "beam_energy": 120.0,
                    "beam_emittance": -2.0,
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frontier.json"
            save_frontier_records(path, records)
            restored = load_frontier_records(path)

        self.assertEqual(restored, records)

    def test_region_records_file_roundtrip(self) -> None:
        records = [
            RegionRecord(
                region_id="region_000",
                status=ACTIVE,
                center=[0.1, 0.9],
                radius=0.25,
                metadata={"seed_candidate_id": "case_001"},
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regions.json"
            save_region_records(path, records)
            restored = load_region_records(path)

        self.assertEqual(restored, records)

    def test_candidate_registry_file_roundtrip(self) -> None:
        registry = CandidateRegistry(
            seen_candidate_signatures=("sig_b", "sig_a"),
            pending_candidate_signatures=("sig_b",),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidate_registry.json"
            save_candidate_registry(path, registry)
            restored = load_candidate_registry(path)

        self.assertEqual(restored.seen_candidate_signatures, ("sig_a", "sig_b"))
        self.assertEqual(restored.pending_candidate_signatures, ("sig_b",))

    def test_write_json_atomic_creates_parent_directories_and_replaces_existing_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "state.json"

            write_json_atomic(path, {"value": 1})
            write_json_atomic(path, {"value": 2})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_loaders_reject_wrong_json_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            registry_path = Path(tmpdir) / "registry.json"
            frontier_path = Path(tmpdir) / "frontier.json"
            regions_path = Path(tmpdir) / "regions.json"

            state_path.write_text("[]", encoding="utf-8")
            registry_path.write_text("[]", encoding="utf-8")
            frontier_path.write_text("{}", encoding="utf-8")
            regions_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "optimizer state JSON must be an object"
            ):
                load_optimizer_state(state_path)
            with self.assertRaisesRegex(
                ValueError, "candidate registry JSON must be an object"
            ):
                load_candidate_registry(registry_path)
            with self.assertRaisesRegex(ValueError, "frontier JSON must be a list"):
                load_frontier_records(frontier_path)
            with self.assertRaisesRegex(ValueError, "regions JSON must be a list"):
                load_region_records(regions_path)

    def test_state_validation_rejects_bad_schema_backend_duplicate_regions_and_nonfinite_refs(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, "unsupported optimizer state schema_version"
        ):
            OptimizerState(schema_version="bad")

        with self.assertRaisesRegex(ValueError, "unsupported optimizer backend"):
            OptimizerState(backend="other")

        with self.assertRaisesRegex(ValueError, "seed must be"):
            OptimizerState(seed=-1)

        with self.assertRaisesRegex(ValueError, "region IDs must be unique"):
            OptimizerState(
                regions=(
                    RegionRecord("region_000", ACTIVE, [0.1], 0.2),
                    RegionRecord("region_000", ACTIVE, [0.2], 0.2),
                )
            )

        with self.assertRaisesRegex(ValueError, "ref_point_raw"):
            OptimizerState(ref_point_raw={"beam_energy": float("nan")})

    def test_frontier_record_loader_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frontier.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "candidate_id": "case_001",
                            "objective_values": {"beam_energy": float("nan")},
                            "canonical_objective_values": {"beam_energy": 1.0},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "objective_values"):
                load_frontier_records(path)


if __name__ == "__main__":
    unittest.main()
