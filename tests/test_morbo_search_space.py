from __future__ import annotations

import random
import unittest

from campaign_optimizer.morbo.search_space import (
    Choice,
    FloatRange,
    IntRange,
    SearchSpaceCodec,
)


class TestMorboSearchSpaceCodec(unittest.TestCase):
    def setUp(self) -> None:
        self.space = {
            "geom.n0": FloatRange(1.0, 10.0),
            "ion.state": Choice([1, 2]),
        }
        self.codec = SearchSpaceCodec(self.space)

    def test_categorical_codec_projection_and_roundtrip_are_deterministic(self) -> None:
        encoded = self.codec.encode({"geom.n0": 3.0, "ion.state": 2})
        decoded = self.codec.decode(encoded)

        self.assertEqual(decoded, {"geom.n0": 3.0, "ion.state": 2})
        self.assertEqual(
            self.codec.project_params({"geom.n0": 3.0, "ion.state": 1.6}),
            {"geom.n0": 3.0, "ion.state": 2},
        )
        self.assertEqual(
            self.codec.decode([0.25, 0.5, 0.5]),
            {"geom.n0": 3.25, "ion.state": 1},
        )

    def test_duplicate_signature_canonicalization_collapses_equivalent_floats(
        self,
    ) -> None:
        sig_a = self.codec.signature({"geom.n0": 0.1 + 0.2, "ion.state": 1})
        sig_b = self.codec.signature({"geom.n0": 0.3, "ion.state": 1.0})

        self.assertEqual(sig_a, sig_b)

    def test_project_params_clips_continuous_and_integer_values(self) -> None:
        codec = SearchSpaceCodec(
            {
                "x": FloatRange(-1.0, 1.0),
                "cells": IntRange(8, 64),
            }
        )

        self.assertEqual(
            codec.project_params({"x": 5.0, "cells": 17.7}), {"x": 1.0, "cells": 18}
        )
        self.assertEqual(
            codec.project_params({"x": -5.0, "cells": 1}), {"x": -1.0, "cells": 8}
        )

    def test_int_and_log_roundtrip_stability(self) -> None:
        codec = SearchSpaceCodec(
            {
                "geom.n0": FloatRange(1.0e24, 1.0e26, log=True),
                "mesh.cells": IntRange(8, 64),
            }
        )

        params = {"geom.n0": 1.0e25, "mesh.cells": 17.7}
        projected = codec.project_params(params)
        roundtrip = codec.decode(codec.encode(projected))

        self.assertEqual(projected, {"geom.n0": 1.0e25, "mesh.cells": 18})
        self.assertEqual(roundtrip, {"geom.n0": 1.0e25, "mesh.cells": 18})

    def test_sample_within_bounds_respects_encoded_bounds(self) -> None:
        rng = random.Random(123)
        bounds = [(0.2, 0.4), (0.0, 0.25), (0.75, 1.0)]

        for _ in range(20):
            params = self.codec.sample_within_bounds(bounds, rng)
            encoded = self.codec.encode(params)
            for idx, value in enumerate(encoded):
                self.assertGreaterEqual(value, bounds[idx][0] - 1.0e-12)
                self.assertLessEqual(value, bounds[idx][1] + 1.0e-12)

    def test_missing_parameter_and_bad_encoded_length_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(KeyError, "Missing parameter"):
            self.codec.project_params({"geom.n0": 3.0})

        with self.assertRaisesRegex(ValueError, "expected 3"):
            self.codec.decode([0.1, 0.2])

    def test_validation_rejects_invalid_spaces(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            SearchSpaceCodec({})

        with self.assertRaisesRegex(ValueError, "positive"):
            SearchSpaceCodec({"bad_log": FloatRange(0.0, 1.0, log=True)})

        with self.assertRaisesRegex(ValueError, "non-empty"):
            SearchSpaceCodec({"bad_choice": Choice([])})


if __name__ == "__main__":
    unittest.main()
