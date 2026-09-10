from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from campaign_optimizer.config import OptimizerConfig


def _problem_callable(config: OptimizerConfig, module: str, name: str) -> Callable[..., Any]:
    problem_kind = config.problem_kind()
    imported = import_module(f"campaign_optimizer.{problem_kind}.{module}")
    callback = getattr(imported, name, None)
    if callback is None or not callable(callback):
        raise ValueError(
            f"optimizer problem {problem_kind!r} does not implement {module}.{name}"
        )
    return callback


def build_observations(config: OptimizerConfig, iteration: int) -> Path:
    result = _problem_callable(config, "observations", "build_observations")(
        config, iteration
    )
    if config.problem_kind() == "capillary":
        from campaign_optimizer.capillary.candidate_identity import (
            normalize_observation_candidate_signatures,
        )

        return normalize_observation_candidate_signatures(
            result,
            parameter_space=config.parameter_space(),
        )
    return result


def build_objectives(config: OptimizerConfig, iteration: int) -> Path:
    return _problem_callable(config, "objectives", "build_objectives")(
        config, iteration
    )


def propose_recommendations(config: OptimizerConfig, iteration: int) -> Path:
    if config.problem_kind() == "capillary":
        from campaign_optimizer.capillary.state_contract import (
            finalize_state_contract,
            prepare_state_contract,
        )

        prepare_state_contract(config, iteration)
        result = _problem_callable(config, "recommend", "propose_recommendations")(
            config, iteration
        )
        finalize_state_contract(config, iteration)
        return result

    return _problem_callable(config, "recommend", "propose_recommendations")(
        config, iteration
    )


def build_candidate_batch(
    config: OptimizerConfig, iteration: int
) -> tuple[Path, Path]:
    return _problem_callable(config, "batch", "build_candidate_batch")(
        config, iteration
    )


def write_basic_plots(config: OptimizerConfig, iter_dir: Path) -> None:
    if config.problem_kind() == "multichannel":
        return
    callback = _problem_callable(config, "plotting", "write_basic_plots")
    callback(iter_dir)
