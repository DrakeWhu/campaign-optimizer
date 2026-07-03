from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from .search_space import SearchSpaceCodec


@dataclass(frozen=True)
class CategoricalRegionalPolicy:
    """Policy for categorical Choice parameters inside regional sampling.

    Continuous/integer dimensions remain trust-region bounded. This policy only
    controls whether Choice parameters are allowed to escape the one-hot region
    center during regional sampling.
    """

    mode: str = "fixed"
    epsilon: float = 0.0
    resample_strategy: str = "alternative"
    parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = str(self.mode or "fixed").strip()
        aliases = {
            "none": "fixed",
            "off": "fixed",
            "keep": "fixed",
            "current": "fixed",
            "eps": "epsilon",
            "epsilon_uniform": "epsilon",
            "always_uniform": "always",
            "resample": "always",
        }
        mode = aliases.get(mode, mode)

        if mode not in {"fixed", "epsilon", "always"}:
            raise ValueError(
                "categorical regional policy mode must be one of: "
                "fixed, epsilon, always"
            )

        epsilon = float(self.epsilon)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("categorical epsilon must be in [0, 1]")

        strategy = str(self.resample_strategy or "alternative").strip()
        if strategy not in {"alternative", "any"}:
            raise ValueError(
                "categorical resample_strategy must be 'alternative' or 'any'"
            )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "resample_strategy", strategy)
        object.__setattr__(
            self,
            "parameter_names",
            tuple(str(item) for item in self.parameter_names),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "CategoricalRegionalPolicy":
        data = dict(payload or {})
        return cls(
            mode=str(data.get("mode", "fixed")),
            epsilon=float(data.get("epsilon", 0.0)),
            resample_strategy=str(data.get("resample_strategy", "alternative")),
            parameter_names=tuple(data.get("parameter_names", ()) or ()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "epsilon": self.epsilon,
            "resample_strategy": self.resample_strategy,
            "parameter_names": list(self.parameter_names),
        }


def categorical_choice_parameter_names(
    codec: SearchSpaceCodec,
    *,
    parameter_names: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return SearchSpaceCodec parameters that behave like categorical choices."""

    allowed = {str(item) for item in parameter_names}
    names: list[str] = []

    for name, spec in codec.space.items():
        options = getattr(spec, "options", None)
        if options is None:
            continue
        if allowed and name not in allowed:
            continue
        names.append(str(name))

    return tuple(names)


def apply_categorical_regional_policy(
    params: Mapping[str, Any],
    *,
    codec: SearchSpaceCodec,
    rng: random.Random,
    policy: CategoricalRegionalPolicy,
) -> dict[str, Any]:
    """Apply categorical escape/resampling to projected regional parameters."""

    out = dict(params)
    if policy.mode == "fixed":
        return codec.project_params(out)

    names = categorical_choice_parameter_names(
        codec,
        parameter_names=policy.parameter_names,
    )

    for name in names:
        if policy.mode == "epsilon" and rng.random() >= policy.epsilon:
            continue

        spec = codec.space[name]
        options = list(getattr(spec, "options", ()))
        if not options:
            continue

        current = out.get(name)
        if policy.resample_strategy == "alternative":
            candidates = [option for option in options if option != current]
            if not candidates:
                candidates = options
        else:
            candidates = options

        out[name] = rng.choice(candidates)

    return codec.project_params(out)
