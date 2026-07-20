from __future__ import annotations

import json
from pathlib import Path
import unittest


CONFIG = Path("examples/optimizer_clpu_corrected_soft50_sunrise.json")
EXPECTED_NAME = "clpu_capillary_guiding_bo_004_corrected_n2_soft50_v2"


class CorrectedClpuSunriseExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_rebuilt_campaign_has_a_fresh_optimization_identity(self) -> None:
        self.assertEqual(
            self.config["optimization_history"]["optimization_name"],
            EXPECTED_NAME,
        )

    def test_rebuild_preserves_the_reviewed_sobol_design(self) -> None:
        recommendation = self.config["recommendation"]
        initial = recommendation["initial_design"]
        self.assertEqual(recommendation["seed"], 20260715)
        self.assertEqual(initial["seed"], 20260715)
        self.assertEqual(initial["sobol_points"], 64)
        self.assertEqual(initial["initial_batch_size"], 32)
        self.assertEqual(len(self.config["parameter_space"]["references"]), 3)


if __name__ == "__main__":
    unittest.main()
