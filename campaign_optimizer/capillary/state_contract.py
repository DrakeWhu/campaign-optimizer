from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.morbo.search_space import (
    Choice,
    FloatRange,
    IntRange,
    PeriodicRange,
    SearchSpaceCodec,
)
from campaign_optimizer.morbo.state import load_optimizer_state, save_optimizer_state

from .parameters import capillary_search_space


STATE_CONTRACT_ID = "clpu_n2_fresh_state_v1"
FINGERPRINT_SCHEMA_VERSION = "morbo_state_fingerprint_v1"


def _state_contract(config: OptimizerConfig) -> dict[str, Any] | None:
    raw = config.recommendation_config().get("state_contract")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("recommendation.state_contract must be an object")

    contract = dict(raw)
    contract_id = str(contract.get("contract_id", "")).strip()
    if contract_id != STATE_CONTRACT_ID:
        raise ValueError(
            "unsupported recommendation.state_contract.contract_id="
            f"{contract_id!r}"
        )

    fresh_iteration = int(contract.get("fresh_bootstrap_iteration", 0))
    if fresh_iteration != 0:
        raise ValueError(
            "clpu_n2_fresh_state_v1 requires fresh_bootstrap_iteration=0"
        )

    expected_encoded_dim = int(contract.get("expected_encoded_dim", 8))
    if expected_encoded_dim != 8:
        raise ValueError(
            "clpu_n2_fresh_state_v1 requires expected_encoded_dim=8"
        )

    for key in ("nitrogen_fraction_semantics", "nitrogen_profile"):
        value = str(contract.get(key, "")).strip()
        if not value:
            raise ValueError(f"recommendation.state_contract.{key} must be non-empty")
        contract[key] = value

    contract["contract_id"] = contract_id
    contract["fresh_bootstrap_iteration"] = fresh_iteration
    contract["expected_encoded_dim"] = expected_encoded_dim
    return contract


def _serialize_parameter_spec(name: str, spec: Any) -> dict[str, Any]:
    if isinstance(spec, Choice):
        return {
            "name": name,
            "type": "choice",
            "encoding": "one_hot_ordered",
            "options": list(spec.options),
        }
    if isinstance(spec, PeriodicRange):
        return {
            "name": name,
            "type": "periodic_range",
            "encoding": "unit_circle",
            "low": float(spec.low),
            "high": float(spec.high),
        }
    if isinstance(spec, IntRange):
        return {
            "name": name,
            "type": "int_range",
            "encoding": "linear_unit_interval_rounded",
            "low": int(spec.low),
            "high": int(spec.high),
        }
    if isinstance(spec, FloatRange):
        return {
            "name": name,
            "type": "float_range",
            "encoding": (
                "log10_unit_interval" if bool(spec.log) else "linear_unit_interval"
            ),
            "low": float(spec.low),
            "high": float(spec.high),
            "log": bool(spec.log),
        }
    raise ValueError(f"unsupported search-space spec for {name!r}: {type(spec)!r}")


def state_fingerprint_payload(config: OptimizerConfig) -> dict[str, Any] | None:
    """Return the complete strict-state identity for the CLPU N2 optimizer."""

    contract = _state_contract(config)
    if contract is None:
        return None

    parameter_space = config.parameter_space()
    space = capillary_search_space(parameter_space)
    codec = SearchSpaceCodec(space)
    expected_encoded_dim = int(contract["expected_encoded_dim"])
    if codec.encoded_dim() != expected_encoded_dim:
        raise ValueError(
            "CLPU N2 optimizer encoded dimension mismatch: "
            f"expected {expected_encoded_dim}, got {codec.encoded_dim()}"
        )

    rec_cfg = config.recommendation_config()
    objective_config = config.objective_config()
    objective_names = rec_cfg.get("objective_names") or rec_cfg.get("objectives")
    if objective_names is None:
        objective_names = objective_config.get("required_scores_for_fit", [])
    objective_names = [str(item) for item in objective_names]

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "contract_id": contract["contract_id"],
        "search_space": {
            "codec": "SearchSpaceCodec_v1",
            "parameter_names": list(codec.parameter_names),
            "encoded_dim": codec.encoded_dim(),
            "parameters": [
                _serialize_parameter_spec(name, space[name])
                for name in codec.parameter_names
            ],
        },
        "objective": {
            "objective_names": objective_names,
            "config": objective_config,
        },
        "nitrogen": {
            "fraction_semantics": contract["nitrogen_fraction_semantics"],
            "profile": contract["nitrogen_profile"],
        },
    }


def state_fingerprint_signature(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _state_path_for_resume(
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
) -> Path | None:
    current = outputs_dir / "morbo_optimizer_state.json"
    if current.is_file():
        return current

    for previous_iteration in range(iteration - 1, -1, -1):
        previous = (
            config.iteration_dir(previous_iteration)
            / "outputs"
            / "morbo_optimizer_state.json"
        )
        if previous.is_file():
            return previous
    return None


def _assert_state_matches_fingerprint(
    path: Path,
    *,
    expected_payload: Mapping[str, Any],
    expected_signature: str,
) -> None:
    state = load_optimizer_state(path)
    if state.search_space_signature != expected_signature:
        raise ValueError(
            "optimizer state fingerprint mismatch before resume: "
            f"path={path}; expected={expected_signature}; "
            f"observed={state.search_space_signature!r}"
        )

    extra = dict(state.extra)
    observed_payload = extra.get("state_fingerprint_payload")
    if observed_payload != dict(expected_payload):
        raise ValueError(
            "optimizer state fingerprint payload mismatch before resume: "
            f"path={path}"
        )


def prepare_state_contract(config: OptimizerConfig, iteration: int) -> None:
    """Fail closed before recommendation code can load incompatible state."""

    payload = state_fingerprint_payload(config)
    if payload is None:
        return

    contract = _state_contract(config)
    assert contract is not None
    fresh_iteration = int(contract["fresh_bootstrap_iteration"])
    outputs_dir = config.iteration_dir(iteration) / "outputs"
    current = outputs_dir / "morbo_optimizer_state.json"

    if iteration == fresh_iteration:
        if current.is_file():
            raise ValueError(
                "unexpected optimizer state in fresh bootstrap outputs: "
                f"{current}"
            )
        return

    if iteration < fresh_iteration:
        raise ValueError(
            f"iteration {iteration} precedes fresh bootstrap iteration {fresh_iteration}"
        )

    state_path = _state_path_for_resume(config, iteration, outputs_dir)
    if state_path is None:
        raise ValueError(
            "strict optimizer state contract requires a compatible prior state "
            f"for iteration {iteration}"
        )

    _assert_state_matches_fingerprint(
        state_path,
        expected_payload=payload,
        expected_signature=state_fingerprint_signature(payload),
    )


def finalize_state_contract(config: OptimizerConfig, iteration: int) -> None:
    """Seal the just-written MORBO state with the complete strict fingerprint."""

    payload = state_fingerprint_payload(config)
    if payload is None:
        return

    state_path = (
        config.iteration_dir(iteration) / "outputs" / "morbo_optimizer_state.json"
    )
    if not state_path.is_file():
        raise ValueError(
            "strict optimizer state contract expected recommendation to write state: "
            f"{state_path}"
        )

    signature = state_fingerprint_signature(payload)
    state = load_optimizer_state(state_path)
    state = state.with_updates(
        search_space_signature=signature,
        extra={
            **dict(state.extra),
            "state_contract_id": STATE_CONTRACT_ID,
            "state_fingerprint_signature": signature,
            "state_fingerprint_payload": payload,
        },
    )
    save_optimizer_state(state_path, state)
