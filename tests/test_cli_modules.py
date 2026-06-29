from __future__ import annotations

import importlib
import unittest


class TestCliModules(unittest.TestCase):
    def test_expected_cli_modules_are_importable(self) -> None:
        modules = [
            "campaign_optimizer.cli.build_observations",
            "campaign_optimizer.cli.build_objectives",
            "campaign_optimizer.cli.fit_surrogates",
            "campaign_optimizer.cli.propose_recommendations",
            "campaign_optimizer.cli.run_iteration",
            "campaign_optimizer.cli.build_candidate_batch",
        ]

        for name in modules:
            with self.subTest(name=name):
                module = importlib.import_module(name)
                self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
