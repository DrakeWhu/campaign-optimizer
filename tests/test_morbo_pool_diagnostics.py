from __future__ import annotations

import unittest

from campaign_optimizer.morbo.botorch_model import _categorical_pool_counts
from campaign_optimizer.morbo.search_space import Choice, FloatRange, SearchSpaceCodec


class MorboPoolDiagnosticsTests(unittest.TestCase):
    def test_categorical_pool_counts_counts_only_choice_parameters(self) -> None:
        codec = SearchSpaceCodec(
            {
                "laser_case": Choice(["f20", "f32", "f40"]),
                "gas_mix": Choice(["he", "he_n2"]),
                "n0": FloatRange(1.0, 10.0),
            }
        )
        rows = [
            {"params": {"laser_case": "f20", "gas_mix": "he", "n0": 2.0}},
            {"params": {"laser_case": "f20", "gas_mix": "he_n2", "n0": 3.0}},
            {"params": {"laser_case": "f32", "gas_mix": "he_n2", "n0": 4.0}},
        ]

        counts = _categorical_pool_counts(rows, codec=codec)

        self.assertEqual(
            counts["laser_case"],
            [
                {"value": "f20", "count": 2, "fraction": 2 / 3},
                {"value": "f32", "count": 1, "fraction": 1 / 3},
            ],
        )
        self.assertEqual(
            counts["gas_mix"],
            [
                {"value": "he_n2", "count": 2, "fraction": 2 / 3},
                {"value": "he", "count": 1, "fraction": 1 / 3},
            ],
        )
        self.assertNotIn("n0", counts)

    def test_categorical_pool_counts_handles_empty_rows(self) -> None:
        codec = SearchSpaceCodec(
            {
                "laser_case": Choice(["f20", "f32", "f40"]),
                "n0": FloatRange(1.0, 10.0),
            }
        )

        counts = _categorical_pool_counts([], codec=codec)

        self.assertEqual(counts, {"laser_case": []})


if __name__ == "__main__":
    unittest.main()
