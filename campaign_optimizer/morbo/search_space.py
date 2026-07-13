from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FloatRange:
    """Continuous parameter range.

    Values are encoded into [0, 1]. When ``log`` is true, encoding and
    sampling are performed in log10-space and both bounds must be positive.
    """

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        low = float(self.low)
        high = float(self.high)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("FloatRange bounds must be finite")
        if high < low:
            raise ValueError("FloatRange high must be >= low")
        if self.log and (low <= 0.0 or high <= 0.0):
            raise ValueError("log FloatRange bounds must be positive")


@dataclass(frozen=True)
class IntRange:
    """Integer parameter range.

    Values are projected by rounding to the nearest integer and clipping to
    the inclusive [low, high] bounds.
    """

    low: int
    high: int

    def __post_init__(self) -> None:
        if isinstance(self.low, bool) or isinstance(self.high, bool):
            raise ValueError("IntRange bounds must be integers, not booleans")
        if int(self.high) < int(self.low):
            raise ValueError("IntRange high must be >= low")


@dataclass(frozen=True)
class PeriodicRange:
    """Continuous periodic parameter encoded on the unit circle.

    ``low`` is inclusive and ``high`` is the equivalent wrapped endpoint. A
    periodic parameter contributes two encoded coordinates while remaining one
    native optimization variable.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        low = float(self.low)
        high = float(self.high)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("PeriodicRange bounds must be finite")
        if high <= low:
            raise ValueError("PeriodicRange high must be greater than low")

    @property
    def period(self) -> float:
        return float(self.high) - float(self.low)


@dataclass(frozen=True)
class Choice:
    """Categorical parameter with a non-empty ordered set of options."""

    options: Sequence[Any]

    def __post_init__(self) -> None:
        if isinstance(self.options, (str, bytes)):
            raise ValueError(
                "Choice options must be a sequence of values, not a string"
            )
        if not list(self.options):
            raise ValueError("Choice options must be non-empty")


class SearchSpaceCodec:
    """Encode, decode, sample and sign candidates in a stable search space.

    The codec keeps the insertion order of the input mapping as the canonical
    parameter order. Candidate signatures are JSON strings built from projected
    values; they are intended for duplicate detection across seen and pending
    candidates.
    """

    def __init__(self, space: Mapping[str, Any]):
        if not space:
            raise ValueError("search space must be non-empty")

        self.space = dict(space)
        self.parameter_names = tuple(self.space.keys())

        for key, spec in self.space.items():
            self._validate_spec(key, spec)

    def canonical_signature_payload(self, params: Mapping[str, Any]) -> dict[str, Any]:
        projected = self.project_params(params)
        payload: dict[str, Any] = {}

        for key in self.parameter_names:
            value = projected[key]
            if isinstance(value, bool):
                payload[key] = value
            elif isinstance(value, int):
                payload[key] = value
            elif isinstance(value, float):
                payload[key] = format(value, ".15g")
            else:
                payload[key] = value

        return payload

    def signature(self, params: Mapping[str, Any]) -> str:
        return json.dumps(
            self.canonical_signature_payload(params),
            sort_keys=True,
            separators=(",", ":"),
        )

    def project_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        projected: dict[str, Any] = {}

        for key, spec in self.space.items():
            if key not in params:
                raise KeyError(f"Missing parameter '{key}'")
            projected[key] = self._project_value(spec, params[key])

        return projected

    def encoded_dim(self) -> int:
        dim = 0
        for spec in self.space.values():
            if self._is_choice(spec):
                dim += len(list(spec.options))
            elif self._is_periodic_range(spec):
                dim += 2
            else:
                dim += 1
        return dim

    def encode(self, params: Mapping[str, Any]) -> list[float]:
        projected = self.project_params(params)
        encoded: list[float] = []

        for key, spec in self.space.items():
            value = projected[key]

            if self._is_choice(spec):
                options = list(spec.options)
                encoded.extend(1.0 if value == option else 0.0 for option in options)
                continue

            if self._is_periodic_range(spec):
                phase = (float(value) - float(spec.low)) / float(spec.period)
                angle = 2.0 * math.pi * phase
                encoded.extend(
                    [
                        0.5 + 0.5 * math.cos(angle),
                        0.5 + 0.5 * math.sin(angle),
                    ]
                )
                continue

            if self._is_int_range(spec):
                low = float(int(spec.low))
                high = float(int(spec.high))
                value_f = float(int(value))
            elif self._is_float_range(spec):
                low = float(spec.low)
                high = float(spec.high)
                value_f = float(value)
                if bool(getattr(spec, "log", False)):
                    low = math.log10(low)
                    high = math.log10(high)
                    value_f = math.log10(value_f)
            else:
                raise ValueError(f"Unsupported search-space spec for '{key}'")

            denom = high - low
            encoded.append(0.0 if abs(denom) < 1.0e-15 else (value_f - low) / denom)

        return encoded

    def decode(self, encoded: Sequence[float]) -> dict[str, Any]:
        values = [float(value) for value in encoded]
        expected_dim = self.encoded_dim()
        if len(values) != expected_dim:
            raise ValueError(
                f"encoded vector has length {len(values)}; expected {expected_dim}"
            )

        out: dict[str, Any] = {}
        idx = 0

        for key, spec in self.space.items():
            if self._is_choice(spec):
                options = list(spec.options)
                window = values[idx : idx + len(options)]
                idx += len(options)
                best_idx = max(range(len(options)), key=lambda pos: (window[pos], -pos))
                out[key] = options[best_idx]
                continue

            if self._is_periodic_range(spec):
                x = 2.0 * min(max(values[idx], 0.0), 1.0) - 1.0
                y = 2.0 * min(max(values[idx + 1], 0.0), 1.0) - 1.0
                idx += 2
                if abs(x) < 1.0e-15 and abs(y) < 1.0e-15:
                    phase = 0.0
                else:
                    phase = (math.atan2(y, x) / (2.0 * math.pi)) % 1.0
                out[key] = float(spec.low) + phase * float(spec.period)
                continue

            raw = min(max(values[idx], 0.0), 1.0)
            idx += 1

            if self._is_int_range(spec):
                low = int(spec.low)
                high = int(spec.high)
                out[key] = int(round(float(low) + raw * (float(high) - float(low))))
            elif self._is_float_range(spec):
                low = float(spec.low)
                high = float(spec.high)
                if bool(getattr(spec, "log", False)):
                    lo_log = math.log10(low)
                    hi_log = math.log10(high)
                    out[key] = float(10.0 ** (lo_log + raw * (hi_log - lo_log)))
                else:
                    out[key] = float(low + raw * (high - low))
            else:
                raise ValueError(f"Unsupported search-space spec for '{key}'")

        return self.project_params(out)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        params: dict[str, Any] = {}

        for key, spec in self.space.items():
            if self._is_choice(spec):
                params[key] = rng.choice(list(spec.options))
            elif self._is_periodic_range(spec):
                params[key] = rng.uniform(float(spec.low), float(spec.high))
            elif self._is_int_range(spec):
                params[key] = rng.randint(int(spec.low), int(spec.high))
            elif self._is_float_range(spec):
                low = float(spec.low)
                high = float(spec.high)
                if bool(getattr(spec, "log", False)):
                    params[key] = float(
                        10.0 ** rng.uniform(math.log10(low), math.log10(high))
                    )
                else:
                    params[key] = float(rng.uniform(low, high))
            else:
                raise ValueError(f"Unsupported search-space spec for '{key}'")

        return self.project_params(params)

    def sample_within_bounds(
        self,
        bounds: Sequence[tuple[float, float]],
        rng: random.Random,
    ) -> dict[str, Any]:
        if len(bounds) != self.encoded_dim():
            raise ValueError(
                f"bounds has length {len(bounds)}; expected {self.encoded_dim()}"
            )

        encoded = []
        for low, high in bounds:
            lo = min(max(float(low), 0.0), 1.0)
            hi = min(max(float(high), 0.0), 1.0)
            if hi < lo:
                raise ValueError(f"invalid encoded bound ({low}, {high})")
            encoded.append(rng.uniform(lo, hi))

        return self.decode(encoded)

    def bounds01(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.encoded_dim()

    def _project_value(self, spec: Any, value: Any) -> Any:
        if self._is_choice(spec):
            options = list(spec.options)
            for option in options:
                if value == option:
                    return option

            numeric_options = options and all(
                isinstance(option, (int, float)) and not isinstance(option, bool)
                for option in options
            )
            if (
                numeric_options
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return min(
                    options,
                    key=lambda option: (
                        abs(float(option) - float(value)),
                        options.index(option),
                    ),
                )

            return options[0]

        if self._is_periodic_range(spec):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("PeriodicRange value must be finite")
            return float(spec.low) + (
                (numeric - float(spec.low)) % float(spec.period)
            )

        if self._is_int_range(spec):
            low = int(spec.low)
            high = int(spec.high)
            return min(max(int(round(float(value))), low), high)

        if self._is_float_range(spec):
            numeric = float(value)
            low = float(spec.low)
            high = float(spec.high)
            return float(min(max(numeric, low), high))

        raise ValueError("Unsupported search-space spec")

    def _validate_spec(self, key: str, spec: Any) -> None:
        if self._is_choice(spec):
            if not list(spec.options):
                raise ValueError(
                    f"Choice parameter '{key}' must have at least one option"
                )
            return

        if self._is_periodic_range(spec):
            low = float(spec.low)
            high = float(spec.high)
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(
                    f"PeriodicRange parameter '{key}' bounds must be finite"
                )
            if high <= low:
                raise ValueError(
                    f"PeriodicRange parameter '{key}' has high <= low"
                )
            return

        if self._is_int_range(spec):
            if isinstance(spec.low, bool) or isinstance(spec.high, bool):
                raise ValueError(
                    f"IntRange parameter '{key}' cannot use boolean bounds"
                )
            if int(spec.high) < int(spec.low):
                raise ValueError(f"IntRange parameter '{key}' has high < low")
            return

        if self._is_float_range(spec):
            low = float(spec.low)
            high = float(spec.high)
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"FloatRange parameter '{key}' bounds must be finite")
            if high < low:
                raise ValueError(f"FloatRange parameter '{key}' has high < low")
            if bool(getattr(spec, "log", False)) and (low <= 0.0 or high <= 0.0):
                raise ValueError(
                    f"log FloatRange parameter '{key}' bounds must be positive"
                )
            return

        raise ValueError(f"Unsupported search-space spec for '{key}'")

    @staticmethod
    def _is_choice(spec: Any) -> bool:
        return hasattr(spec, "options")

    @staticmethod
    def _is_periodic_range(spec: Any) -> bool:
        return isinstance(spec, PeriodicRange)

    @staticmethod
    def _is_int_range(spec: Any) -> bool:
        return (
            hasattr(spec, "low")
            and hasattr(spec, "high")
            and isinstance(spec.low, int)
            and isinstance(spec.high, int)
            and not isinstance(spec.low, bool)
            and not isinstance(spec.high, bool)
            and not bool(getattr(spec, "log", False))
        )

    @staticmethod
    def _is_float_range(spec: Any) -> bool:
        return (
            hasattr(spec, "low")
            and hasattr(spec, "high")
            and not isinstance(spec, PeriodicRange)
        )
