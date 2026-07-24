from __future__ import annotations

import unittest

from campaign_optimizer.morbo.botorch_model import _build_acquisition_function


class _CaptureAcquisition:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class SingleObjectiveAcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.imports = {
            "single_acqf_class": _CaptureAcquisition,
            "single_acqf_name": "qLogNoisyExpectedImprovement",
            "multi_acqf_class": _CaptureAcquisition,
            "multi_acqf_name": "qLogNoisyExpectedHypervolumeImprovement",
        }

    def test_one_objective_uses_noisy_expected_improvement(self) -> None:
        acqf, name, family = _build_acquisition_function(
            imports=self.imports,
            model="model",
            train_X="baseline",
            sampler="sampler",
            objective_names=("charge",),
            ref_point_raw=None,
            prune_baseline=True,
        )

        self.assertEqual(name, "qLogNoisyExpectedImprovement")
        self.assertEqual(family, "single_objective")
        self.assertEqual(
            acqf.kwargs,
            {
                "model": "model",
                "X_baseline": "baseline",
                "prune_baseline": True,
                "sampler": "sampler",
            },
        )
        self.assertNotIn("ref_point", acqf.kwargs)

    def test_multiple_objectives_preserve_hypervolume_acquisition(self) -> None:
        acqf, name, family = _build_acquisition_function(
            imports=self.imports,
            model="model",
            train_X="baseline",
            sampler="sampler",
            objective_names=("charge", "energy"),
            ref_point_raw={"charge": 1.0, "energy": 2.0},
            prune_baseline=False,
        )

        self.assertEqual(name, "qLogNoisyExpectedHypervolumeImprovement")
        self.assertEqual(family, "multi_objective")
        self.assertEqual(acqf.kwargs["ref_point"], [1.0, 2.0])

    def test_multiobjective_requires_reference_point(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a reference point"):
            _build_acquisition_function(
                imports=self.imports,
                model="model",
                train_X="baseline",
                sampler="sampler",
                objective_names=("charge", "energy"),
                ref_point_raw=None,
                prune_baseline=True,
            )


if __name__ == "__main__":
    unittest.main()
