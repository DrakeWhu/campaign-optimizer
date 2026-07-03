from __future__ import annotations

import random
import unittest

from campaign_optimizer.morbo.categorical import (
    CategoricalRegionalPolicy,
    apply_categorical_regional_policy,
    categorical_choice_parameter_names,
)
from campaign_optimizer.morbo.search_space import Choice, FloatRange, SearchSpaceCodec


class CategoricalRegionalPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = SearchSpaceCodec(
            {
                "laser_case": Choice(["f20", "f32", "f40"]),
                "gas_mix": Choice(["he", "he_n2", "he_ar"]),
                "n0": FloatRange(1.0, 10.0),
            }
        )

    def test_choice_parameter_names_detects_only_categorical_choices(self) -> None:
        self.assertEqual(
            categorical_choice_parameter_names(self.codec),
            ("laser_case", "gas_mix"),
        )
        self.assertEqual(
            categorical_choice_parameter_names(
                self.codec,
                parameter_names=("gas_mix",),
            ),
            ("gas_mix",),
        )

    def test_fixed_policy_preserves_categories(self) -> None:
        params = {"laser_case": "f20", "gas_mix": "he", "n0": 5.0}

        out = apply_categorical_regional_policy(
            params,
            codec=self.codec,
            rng=random.Random(1),
            policy=CategoricalRegionalPolicy(mode="fixed"),
        )

        self.assertEqual(out["laser_case"], "f20")
        self.assertEqual(out["gas_mix"], "he")
        self.assertEqual(out["n0"], 5.0)

    def test_epsilon_zero_preserves_categories(self) -> None:
        params = {"laser_case": "f20", "gas_mix": "he", "n0": 5.0}

        out = apply_categorical_regional_policy(
            params,
            codec=self.codec,
            rng=random.Random(1),
            policy=CategoricalRegionalPolicy(mode="epsilon", epsilon=0.0),
        )

        self.assertEqual(out["laser_case"], "f20")
        self.assertEqual(out["gas_mix"], "he")

    def test_epsilon_one_uses_alternative_categories_by_default(self) -> None:
        params = {"laser_case": "f20", "gas_mix": "he", "n0": 5.0}

        out = apply_categorical_regional_policy(
            params,
            codec=self.codec,
            rng=random.Random(2),
            policy=CategoricalRegionalPolicy(mode="epsilon", epsilon=1.0),
        )

        self.assertIn(out["laser_case"], {"f32", "f40"})
        self.assertIn(out["gas_mix"], {"he_n2", "he_ar"})
        self.assertEqual(out["n0"], 5.0)

    def test_always_mode_can_be_limited_to_selected_parameters(self) -> None:
        params = {"laser_case": "f20", "gas_mix": "he", "n0": 5.0}

        out = apply_categorical_regional_policy(
            params,
            codec=self.codec,
            rng=random.Random(3),
            policy=CategoricalRegionalPolicy(
                mode="always",
                parameter_names=("gas_mix",),
            ),
        )

        self.assertEqual(out["laser_case"], "f20")
        self.assertIn(out["gas_mix"], {"he_n2", "he_ar"})

    def test_invalid_policy_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            CategoricalRegionalPolicy(mode="bad")

        with self.assertRaisesRegex(ValueError, "epsilon"):
            CategoricalRegionalPolicy(mode="epsilon", epsilon=1.5)

        with self.assertRaisesRegex(ValueError, "resample_strategy"):
            CategoricalRegionalPolicy(resample_strategy="bad")


if __name__ == "__main__":
    unittest.main()
