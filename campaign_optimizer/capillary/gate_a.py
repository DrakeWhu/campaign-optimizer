from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig, load_optimizer_config
from campaign_optimizer.io import read_table, write_json
from campaign_optimizer.morbo.search_space import SearchSpaceCodec
from campaign_optimizer.morbo.state import load_optimizer_state

from .candidate_identity import normalize_persisted_candidate_signature
from .parameters import active_optimizer_parameter_columns, capillary_search_space


GATE_A_SCHEMA_VERSION = 1
GATE_A_CONTRACT_ID = "clpu_n2_warm_start_344_gate_a_v1"
OBJECTIVE_CONFIG_ID = "clpu_baseline_charge_soft50_plateau_exit_v1"
OBJECTIVE_NAME = "score_charge_soft50_direct_v1"
OBJECTIVE_SOURCE_METRIC = "metric_particle_charge_soft50_pC"
HARD50_METRIC = "metric_particle_charge_Ege50MeV_pC"
PARTICLE_OBSERVATION_CONTRACT = "clpu_n2_plateau_all_electrons_v1"
STATE_CONTRACT_ID = "clpu_n2_fresh_state_v1"
NITROGEN_FRACTION_SEMANTICS = (
    "fraction_of_atomic_nuclei_nitrogen_equal_H2_N2_molecular_fraction"
)
NITROGEN_PROFILE_ID = "uniform_nitrogen_fraction_v1"

EXPECTED_STATIC_SOURCES = 40
EXPECTED_REFERENCE_OBSERVATIONS = 337
EXPECTED_DIRECTED_SCAN_OBSERVATIONS = 7
EXPECTED_INITIAL_OBSERVATIONS = 344
EXPECTED_CANDIDATES = 8
EXPECTED_ENCODED_DIM = 8
EXPECTED_MODEL_TRAIN_ROWS = 256
EXPECTED_SOBOL_INDICES = tuple(range(32))
EXPECTED_LASER_CASES = ("f20", "f32", "f40")
EXPECTED_PARAMETER_NAMES = (
    "laser_case",
    "n0_1e18cm3",
    "plateau_mm_num",
    "diameter_um_num",
    "focus_mm_num",
    "nitrogen_fraction",
)
EXPECTED_RANGES = {
    "n0_1e18cm3": (0.7, 6.0),
    "plateau_mm_num": (5.0, 25.0),
    "diameter_um_num": (150.0, 500.0),
    "focus_mm_num": (-5.0, 5.0),
    "nitrogen_fraction": (0.0, 0.01),
}
EXPECTED_REDUCED_OUTPUTS = {
    "acceptance_curves": "particle_analysis/plateau_exit/particle_acceptance_curves.csv",
    "guiding_metrics": "guiding_metrics.csv",
    "guiding_singlecase_score": "guiding_singlecase_score.csv",
    "particle_summary": "particle_analysis/plateau_exit/particle_summary.csv",
    "soft50_curves": "particle_analysis/plateau_exit/particle_soft50_curves.csv",
}
DIRECTED_SCAN_NAME = "clpu_f32_radius_density_a0_7case_20260826"
DIRECTED_SCAN_CASE_NAMES = (
    "000_f32_D200_n1p5_foc0",
    "001_f32_D300_n1p5_foc0",
    "002_f32_D200_n3p5_foc0",
    "003_f32_D300_n3p5_foc0",
    "004_f32_D250_n2p4_foc0",
    "005_f32_D250_n2p4_focm2",
    "006_f32_D250_n2p4_focp2",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON file: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _historical_specs(warpx_runs_root: Path) -> list[tuple[str, Path, int]]:
    families = (
        ("clpu_capillary_guiding_baseline_soft50", 13),
        ("clpu_capillary_guiding_baseline_soft50_trfix_v1", 6),
        ("clpu_capillary_guiding_baseline_soft50_trfix_v2", 20),
    )
    specs: list[tuple[str, Path, int]] = []
    for family, count in families:
        for iteration in range(count):
            rows = 33 if family == families[0][0] and iteration == 0 else 8
            specs.append(
                (
                    f"{family}_iter_{iteration:03d}",
                    warpx_runs_root / family / "iterations" / f"iter_{iteration:03d}",
                    rows,
                )
            )
    return specs


def _source_identity(source: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(source.get("campaign_name", "")).strip(),
        str(_resolved(source.get("campaign_root", ""))),
        str(source.get("cases_tsv", "cases.tsv")),
    )


def freeze_warm_start_sources(
    *,
    reference_manifest_path: str | Path,
    directed_scan_root: str | Path,
    warpx_runs_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify the audited 39+1 genealogy and freeze it as forty static sources."""

    manifest_path = _resolved(reference_manifest_path)
    scan_root = _resolved(directed_scan_root)
    runs_root = _resolved(warpx_runs_root)
    manifest = _read_json(manifest_path)

    _require(manifest.get("schema_version") == 1, "reference manifest schema_version must be 1")
    _require(int(manifest.get("iteration", -1)) == 20, "reference manifest must be final V2 iter_020")
    raw_sources = manifest.get("source_campaigns")
    _require(isinstance(raw_sources, list), "reference manifest source_campaigns must be a list")
    _require(len(raw_sources) == 39, f"reference manifest expected 39 sources, got {len(raw_sources)}")

    expected = _historical_specs(runs_root)
    _require(len(expected) == 39, "internal historical specification must contain 39 roots")

    frozen: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, (source, spec) in enumerate(zip(raw_sources, expected)):
        _require(isinstance(source, dict), f"reference source[{index}] must be an object")
        expected_name, expected_root, expected_rows = spec
        name = str(source.get("campaign_name", "")).strip()
        root = _resolved(source.get("campaign_root", ""))
        cases_tsv = str(source.get("cases_tsv", "cases.tsv"))
        reduced = dict(source.get("reduced_outputs", {}) or {})

        _require(name == expected_name, f"source[{index}] name mismatch: {name!r} != {expected_name!r}")
        _require(root == _resolved(expected_root), f"source[{index}] root mismatch: {root} != {_resolved(expected_root)}")
        _require(reduced == EXPECTED_REDUCED_OUTPUTS, f"source[{index}] reduced-output contract mismatch")
        case_rows = int(len(read_table(root / cases_tsv)))
        _require(case_rows == expected_rows, f"source[{index}] expected {expected_rows} cases, got {case_rows}")

        item = dict(source)
        item["campaign_root"] = str(root)
        item["cases_tsv"] = cases_tsv
        item["reduced_outputs"] = dict(EXPECTED_REDUCED_OUTPUTS)
        frozen.append(item)
        evidence.append(
            {
                "index": index,
                "campaign_name": name,
                "campaign_root": str(root),
                "case_rows": case_rows,
            }
        )

    _require(scan_root.is_dir(), f"directed scan root is missing: {scan_root}")
    scan_cases = read_table(scan_root / "cases.tsv")
    _require(len(scan_cases) == EXPECTED_DIRECTED_SCAN_OBSERVATIONS, f"directed scan expected 7 cases, got {len(scan_cases)}")
    _require("CASE_NAME" in scan_cases.columns, "directed scan cases.tsv lacks CASE_NAME")
    observed_names = tuple(str(value) for value in scan_cases["CASE_NAME"].tolist())
    _require(observed_names == DIRECTED_SCAN_CASE_NAMES, f"directed scan case identity mismatch: {observed_names}")
    for case_name in DIRECTED_SCAN_CASE_NAMES:
        case_dir = scan_root / case_name
        for relative in EXPECTED_REDUCED_OUTPUTS.values():
            _require((case_dir / relative).is_file(), f"directed scan missing reduced output: {case_dir / relative}")

    frozen.append(
        {
            "campaign_name": DIRECTED_SCAN_NAME,
            "campaign_root": str(scan_root),
            "cases_tsv": "cases.tsv",
            "reduced_outputs": dict(EXPECTED_REDUCED_OUTPUTS),
            "source_kind": "warm_start_historical",
        }
    )
    evidence.append(
        {
            "index": 39,
            "campaign_name": DIRECTED_SCAN_NAME,
            "campaign_root": str(scan_root),
            "case_rows": EXPECTED_DIRECTED_SCAN_OBSERVATIONS,
        }
    )

    identities = [_source_identity(source) for source in frozen]
    _require(len(frozen) == EXPECTED_STATIC_SOURCES, "frozen manifest must contain exactly 40 roots")
    _require(len(set(identities)) == EXPECTED_STATIC_SOURCES, "frozen manifest contains duplicate source identities")
    _require(len({item[0] for item in identities}) == EXPECTED_STATIC_SOURCES, "frozen manifest contains duplicate aliases")
    _require(len({item[1] for item in identities}) == EXPECTED_STATIC_SOURCES, "frozen manifest contains duplicate roots")
    _require(sum(int(row["case_rows"]) for row in evidence) == EXPECTED_INITIAL_OBSERVATIONS, "frozen source case count must sum to 344")
    return frozen, evidence


def gate_a_optimizer_config(source_campaigns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the explicit six-parameter, no-Sobol, fail-closed Gate A config."""

    _require(len(source_campaigns) == EXPECTED_STATIC_SOURCES, "Gate A requires exactly 40 source campaigns")
    return {
        "schema_version": 1,
        "problem": "capillary",
        "optimizer_run_root": "optimizer_runs",
        "source_campaigns": [dict(source) for source in source_campaigns],
        "source_manifest_contract": {
            "contract_id": "strict_static_sources_v1",
            "expected_static_source_count": EXPECTED_STATIC_SOURCES,
        },
        "optimization_history": {
            "enabled": True,
            "optimization_name": "clpu_capillary_guiding_n2_uniform_soft50",
            "iterations_root": "iterations",
            "cases_tsv": "cases.tsv",
            "campaign_name_template": "{optimization_name}_iter_{iteration:03d}",
            "particle_observation_contract": PARTICLE_OBSERVATION_CONTRACT,
            "reduced_outputs": dict(EXPECTED_REDUCED_OUTPUTS),
        },
        "parameter_space": {
            "version": 3,
            "laser_cases": list(EXPECTED_LASER_CASES),
            "ranges": {name: list(bounds) for name, bounds in EXPECTED_RANGES.items()},
        },
        "objective": {
            "schema_version": 1,
            "config_id": OBJECTIVE_CONFIG_ID,
            "required_scores_for_fit": [OBJECTIVE_NAME],
            "derived_scores": [],
            "score_specs": {OBJECTIVE_NAME: [OBJECTIVE_SOURCE_METRIC]},
        },
        "recommendation": {
            "backend": "morbo_like",
            "suggestion_mode": "regional_model",
            "seed": 20260724,
            "n_candidates": EXPECTED_CANDIDATES,
            "min_observations": 32,
            "objective_names": [OBJECTIVE_NAME],
            "state_contract": {
                "contract_id": STATE_CONTRACT_ID,
                "fresh_bootstrap_iteration": 0,
                "expected_encoded_dim": EXPECTED_ENCODED_DIM,
                "nitrogen_fraction_semantics": NITROGEN_FRACTION_SEMANTICS,
                "nitrogen_profile": NITROGEN_PROFILE_ID,
            },
            "regional_policy": {
                "max_regions": 3,
                "initial_radius": 0.25,
                "min_radius": 0.05,
                "max_radius": 0.5,
                "expansion_factor": 1.2,
                "contraction_factor": 0.7,
                "restart_failure_threshold": 3,
                "local_min_observations": 3,
            },
            "categorical_policy": {"mode": "epsilon", "epsilon": 0.2},
            "botorch": {
                "enabled": True,
                "fallback_to_random": False,
                "max_train_rows": EXPECTED_MODEL_TRAIN_ROWS,
                "candidate_pool_size_per_region": 256,
                "mc_samples": 64,
                "fit_maxiter": 75,
                "seed": 20260724,
            },
        },
        "candidate_batch": {
            "case_id_start": 0,
            "plasma_kind": "chan",
            "laser_duration_fwhm_fs": 30.0,
            "cap_nr": 192,
            "cap_rmax_um": {
                "policy": "radius_plus_margin",
                "margin_um": 30.0,
            },
            "campaign_template": {
                "campaign_json": "campaign.json",
                "input_template": "input_template.py",
            },
        },
        "validation": {"require_case_validation": True},
        "gate_a_contract": {
            "contract_id": GATE_A_CONTRACT_ID,
            "expected_initial_observations": EXPECTED_INITIAL_OBSERVATIONS,
            "expected_reference_observations": EXPECTED_REFERENCE_OBSERVATIONS,
            "expected_directed_scan_observations": EXPECTED_DIRECTED_SCAN_OBSERVATIONS,
            "expected_candidates": EXPECTED_CANDIDATES,
            "expected_encoded_dim": EXPECTED_ENCODED_DIM,
            "fixed_laser_duration_fwhm_fs": 30.0,
        },
    }


def _check_config_contract(config: OptimizerConfig) -> None:
    gate = dict(config.data.get("gate_a_contract", {}) or {})
    _require(gate.get("contract_id") == GATE_A_CONTRACT_ID, "optimizer config is not the CLPU N2 Gate A contract")
    _require(len(config.source_campaigns()) == EXPECTED_STATIC_SOURCES, "Gate A optimizer config must freeze exactly 40 static sources")

    parameter_space = config.parameter_space()
    _require(tuple(parameter_space.get("laser_cases", [])) == EXPECTED_LASER_CASES, "Gate A laser-case order mismatch")
    ranges = dict(parameter_space.get("ranges", {}) or {})
    _require(set(ranges) == set(EXPECTED_RANGES), f"Gate A numeric parameter ranges mismatch: {sorted(ranges)}")
    for name, bounds in EXPECTED_RANGES.items():
        _require(tuple(float(v) for v in ranges[name]) == bounds, f"Gate A range mismatch for {name}")
    _require(active_optimizer_parameter_columns(parameter_space) == list(EXPECTED_PARAMETER_NAMES), "Gate A must have exactly six physical optimizer parameters")
    codec = SearchSpaceCodec(capillary_search_space(parameter_space))
    _require(codec.encoded_dim() == EXPECTED_ENCODED_DIM, "Gate A six-parameter space must encode to 8 columns")

    objective = config.objective_config()
    _require(objective.get("config_id") == OBJECTIVE_CONFIG_ID, "Gate A objective config ID mismatch")
    _require(objective.get("required_scores_for_fit") == [OBJECTIVE_NAME], "Gate A must fit only direct Soft50 charge")
    _require(dict(objective.get("score_specs", {}) or {}) == {OBJECTIVE_NAME: [OBJECTIVE_SOURCE_METRIC]}, "Gate A objective source metric mismatch")
    _require(not (objective.get("derived_scores") or []), "Gate A direct objective must not use derived scores")

    rec = config.recommendation_config()
    _require(rec.get("backend") == "morbo_like", "Gate A backend must be morbo_like")
    _require(rec.get("suggestion_mode") == "regional_model", "Gate A must use regional_model")
    _require(int(rec.get("n_candidates", -1)) == EXPECTED_CANDIDATES, "Gate A must request 8 candidates")
    _require("initial_design" not in config.data.get("recommendation", {}), "Gate A must not contain an initial Sobol design")
    botorch = dict(rec.get("botorch", {}) or {})
    _require(botorch.get("enabled") is True, "Gate A BoTorch must be enabled")
    _require(botorch.get("fallback_to_random") is False, "Gate A random fallback must be disabled")
    _require(int(botorch.get("max_train_rows", -1)) == EXPECTED_MODEL_TRAIN_ROWS, "Gate A max_train_rows mismatch")
    state_contract = dict(rec.get("state_contract", {}) or {})
    _require(state_contract.get("contract_id") == STATE_CONTRACT_ID, "Gate A fresh-state contract mismatch")
    _require(int(state_contract.get("expected_encoded_dim", -1)) == EXPECTED_ENCODED_DIM, "Gate A state encoded dimension mismatch")
    _require(state_contract.get("nitrogen_fraction_semantics") == NITROGEN_FRACTION_SEMANTICS, "Gate A nitrogen-fraction semantics mismatch")
    _require(state_contract.get("nitrogen_profile") == NITROGEN_PROFILE_ID, "Gate A nitrogen profile mismatch")

    batch = config.candidate_batch_config()
    _require(float(batch.get("laser_duration_fwhm_fs", float("nan"))) == 30.0, "Gate A laser duration must stay fixed at 30 fs")
    _require(int(batch.get("cap_nr", -1)) == 192, "Gate A CAP_NR must remain 192")
    _require(dict(batch.get("cap_rmax_um", {}) or {}) == {"policy": "radius_plus_margin", "margin_um": 30.0}, "Gate A radial-domain policy mismatch")


def prepare_gate_a_staging(
    *,
    staging_root: str | Path,
    reference_manifest_path: str | Path,
    reference_observations_path: str | Path,
    reference_objectives_path: str | Path,
    directed_scan_root: str | Path,
    warpx_runs_root: str | Path,
) -> dict[str, Any]:
    """Create a new non-destructive Gate A staging root and provenance receipt."""

    staging = _resolved(staging_root)
    ref_manifest = _resolved(reference_manifest_path)
    ref_obs = _resolved(reference_observations_path)
    ref_obj = _resolved(reference_objectives_path)
    scan_root = _resolved(directed_scan_root)
    for path in (ref_manifest, ref_obs, ref_obj):
        _require(path.is_file(), f"missing immutable Gate A reference: {path}")

    sources, source_evidence = freeze_warm_start_sources(
        reference_manifest_path=ref_manifest,
        directed_scan_root=scan_root,
        warpx_runs_root=warpx_runs_root,
    )
    reference_observations = read_table(ref_obs)
    reference_objectives = read_table(ref_obj)
    _require(len(reference_observations) == EXPECTED_REFERENCE_OBSERVATIONS, f"reference observations expected 337 rows, got {len(reference_observations)}")
    _require(reference_observations["observation_id"].nunique() == EXPECTED_REFERENCE_OBSERVATIONS, "reference observations contain duplicate IDs")
    _require(len(reference_objectives) == EXPECTED_REFERENCE_OBSERVATIONS, f"reference objectives expected 337 rows, got {len(reference_objectives)}")
    _require(reference_objectives["observation_id"].nunique() == EXPECTED_REFERENCE_OBSERVATIONS, "reference objectives contain duplicate IDs")
    _require(reference_objectives["objective_status"].astype(str).eq("ok").all(), "reference objective table is no longer 337/337 status=ok")
    _require(reference_objectives["fit_eligible"].astype(str).str.lower().eq("true").all(), "reference objective table is no longer 337/337 fit-eligible")

    for source in sources:
        root = _resolved(source["campaign_root"])
        _require(staging != root and root not in staging.parents, f"staging root may not be inside historical source {root}")

    optimizer_json = staging / "optimizer.json"
    optimizer_runs = staging / "optimizer_runs"
    iterations = staging / "iterations"
    receipt_path = staging / "provenance" / "gate_a_bootstrap.json"
    for path in (optimizer_json, optimizer_runs, iterations, receipt_path):
        _require(not path.exists(), f"refusing to reuse existing Gate A artifact: {path}")

    staging.mkdir(parents=True, exist_ok=True)
    write_json(optimizer_json, gate_a_optimizer_config(sources))
    receipt = {
        "schema_version": GATE_A_SCHEMA_VERSION,
        "contract_id": GATE_A_CONTRACT_ID,
        "status": "prepared",
        "staging_root": str(staging),
        "optimizer_json": {"path": str(optimizer_json), "sha256": _sha256(optimizer_json)},
        "reference_manifest": {"path": str(ref_manifest), "sha256": _sha256(ref_manifest)},
        "reference_observations": {
            "path": str(ref_obs),
            "sha256": _sha256(ref_obs),
            "rows": EXPECTED_REFERENCE_OBSERVATIONS,
        },
        "reference_objectives": {
            "path": str(ref_obj),
            "sha256": _sha256(ref_obj),
            "rows": EXPECTED_REFERENCE_OBSERVATIONS,
        },
        "directed_scan_root": str(scan_root),
        "source_count": len(sources),
        "source_case_rows": sum(int(row["case_rows"]) for row in source_evidence),
        "sources": source_evidence,
        "note": "Gate A only: no campaign materialization, WarpX execution, cleanup, or sbatch.",
    }
    write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def _finite_column(frame: pd.DataFrame, column: str, *, label: str) -> np.ndarray:
    _require(column in frame.columns, f"{label} missing column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    _require(np.isfinite(values).all(), f"{label}.{column} must be finite for every row")
    return values


def _assert_numeric_match(left: pd.DataFrame, right: pd.DataFrame, *, column: str, label: str) -> None:
    _require(column in left.columns and column in right.columns, f"{label} missing comparison column {column!r}")
    a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
    b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
    _require(np.isfinite(a).all() and np.isfinite(b).all(), f"{label}.{column} contains non-finite values")
    _require(np.allclose(a, b, rtol=1.0e-12, atol=1.0e-12), f"{label}.{column} differs from immutable 337-row reference")


def _effective_identity_rows(
    observations: pd.DataFrame,
    *,
    parameter_space: dict[str, Any],
) -> tuple[pd.DataFrame, set[str]]:
    names = active_optimizer_parameter_columns(parameter_space)
    codec = SearchSpaceCodec(capillary_search_space(parameter_space))
    records: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for _, row in observations.iterrows():
        params = {name: row[name] for name in names}
        persisted = normalize_persisted_candidate_signature(
            row.get("candidate_signature", ""),
            parameter_space=parameter_space,
        )
        signature = persisted or codec.signature(params)
        origin = "persisted_current_6d" if persisted else "recomputed_missing_historical_provenance"
        _require(signature not in signatures, f"duplicate effective historical candidate identity: {signature}")
        signatures.add(signature)
        record: dict[str, Any] = {
            "observation_id": str(row["observation_id"]),
            "identity_origin": origin,
            "candidate_signature": signature,
        }
        for index, value in enumerate(codec.encode(params)):
            record[f"encoded_{index}"] = float(value)
        records.append(record)
    return pd.DataFrame(records), signatures


def _assert_effective_sources(config: OptimizerConfig, source_manifest: Mapping[str, Any]) -> None:
    actual = source_manifest.get("source_campaigns")
    _require(isinstance(actual, list) and len(actual) == EXPECTED_STATIC_SOURCES, "Gate A effective manifest must contain exactly 40 sources")
    expected = config.source_campaigns_for_iteration(0)
    _require(len(expected) == EXPECTED_STATIC_SOURCES, "Gate A config did not resolve exactly 40 iter_000 sources")
    _require([_source_identity(item) for item in actual] == [_source_identity(item) for item in expected], "Gate A effective source identity/order differs from frozen config")
    for index, item in enumerate(actual):
        _require(dict(item.get("reduced_outputs", {}) or {}) == EXPECTED_REDUCED_OUTPUTS, f"Gate A effective source[{index}] reduced-output contract mismatch")


def _assert_batch_plan(config: OptimizerConfig, batch_plan: Mapping[str, Any]) -> None:
    _require(batch_plan.get("schema_version") == 1, "Gate A batch plan schema_version mismatch")
    _require(batch_plan.get("plan_type") == "optimizer_candidate_batch", "Gate A batch plan type mismatch")
    _require(int(batch_plan.get("optimizer_iteration", -1)) == 0, "Gate A batch plan iteration mismatch")
    _require(batch_plan.get("objective_config_id") == OBJECTIVE_CONFIG_ID, "Gate A batch plan objective mismatch")
    _require(batch_plan.get("candidate_batch") == "outputs/candidate_batch.tsv", "Gate A batch plan candidate_batch linkage mismatch")
    _require(batch_plan.get("recommended_candidates") == "outputs/recommended_candidates.tsv", "Gate A batch plan recommendation linkage mismatch")
    _require(dict(batch_plan.get("campaign_template", {}) or {}) == dict(config.candidate_batch_config().get("campaign_template", {}) or {}), "Gate A batch plan campaign-template mismatch")
    plan_sources = batch_plan.get("source_campaigns")
    _require(isinstance(plan_sources, list) and len(plan_sources) == EXPECTED_STATIC_SOURCES, "Gate A batch plan must record all 40 sources")
    _require([_source_identity(item) for item in plan_sources] == [_source_identity(item) for item in config.source_campaigns_for_iteration(0)], "Gate A batch plan source linkage mismatch")
    non_goals = {str(item) for item in (batch_plan.get("non_goals") or [])}
    for required in ("does not submit jobs", "does not launch WarpX", "does not delete data", "does not create campaign roots"):
        _require(required in non_goals, f"Gate A batch plan lost non-goal {required!r}")


def validate_gate_a(
    *,
    config_path: str | Path,
    bootstrap_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the complete real-data CLPU N2 Gate A contract passes."""

    config_path = _resolved(config_path)
    config = load_optimizer_config(config_path)
    _check_config_contract(config)
    staging = config.base_dir
    receipt_path = _resolved(bootstrap_receipt_path or staging / "provenance" / "gate_a_bootstrap.json")
    receipt = _read_json(receipt_path)
    _require(receipt.get("contract_id") == GATE_A_CONTRACT_ID, "Gate A bootstrap receipt contract mismatch")

    optimizer_receipt = dict(receipt.get("optimizer_json", {}) or {})
    _require(_resolved(optimizer_receipt.get("path", "")) == config_path, "Gate A config path differs from bootstrap receipt")
    _require(_sha256(config_path) == optimizer_receipt.get("sha256"), "Gate A optimizer.json changed after preparation")
    for key in ("reference_manifest", "reference_observations", "reference_objectives"):
        item = dict(receipt.get(key, {}) or {})
        path = _resolved(item.get("path", ""))
        _require(path.is_file(), f"Gate A immutable reference disappeared: {path}")
        _require(_sha256(path) == item.get("sha256"), f"Gate A immutable reference changed after preparation: {key}")

    reference_obs_path = _resolved(receipt["reference_observations"]["path"])
    reference_obj_path = _resolved(receipt["reference_objectives"]["path"])
    iter_dir = config.iteration_dir(0)
    inputs = iter_dir / "inputs"
    outputs = iter_dir / "outputs"
    paths = {
        "source_manifest": inputs / "source_campaigns_effective.json",
        "observations": inputs / "observations.csv",
        "objectives": inputs / "objective_table.csv",
        "recommendations": outputs / "recommended_candidates.tsv",
        "surrogate_summary": outputs / "surrogate_summary.json",
        "morbo_state": outputs / "morbo_optimizer_state.json",
        "candidate_batch": outputs / "candidate_batch.tsv",
        "batch_plan": outputs / "batch_campaign_plan.json",
        "optimizer_state": iter_dir / "optimizer_state.json",
    }
    for label, path in paths.items():
        _require(path.is_file(), f"Gate A missing {label}: {path}")

    _assert_effective_sources(config, _read_json(paths["source_manifest"]))
    observations = read_table(paths["observations"])
    objectives = read_table(paths["objectives"])
    reference_obs = read_table(reference_obs_path).sort_values("observation_id", kind="stable").reset_index(drop=True)
    reference_obj = read_table(reference_obj_path).sort_values("observation_id", kind="stable").reset_index(drop=True)

    _require(len(observations) == EXPECTED_INITIAL_OBSERVATIONS, f"Gate A expected 344 observations, got {len(observations)}")
    _require(observations["observation_id"].nunique() == EXPECTED_INITIAL_OBSERVATIONS, "Gate A observations contain duplicate IDs")
    _require(len(objectives) == EXPECTED_INITIAL_OBSERVATIONS, f"Gate A expected 344 objective rows, got {len(objectives)}")
    _require(objectives["observation_id"].nunique() == EXPECTED_INITIAL_OBSERVATIONS, "Gate A objectives contain duplicate IDs")
    _require(objectives["objective_status"].astype(str).eq("ok").all(), "Gate A requires objective_status=ok for all 344 rows")
    _require(objectives["fit_eligible"].astype(str).str.lower().eq("true").all(), "Gate A requires 344/344 fit_eligible=true")
    _require(objectives["objective_config_id"].astype(str).eq(OBJECTIVE_CONFIG_ID).all(), "Gate A objective config ID mismatch")
    _require(objectives[f"{OBJECTIVE_NAME}_source_metric"].astype(str).eq(OBJECTIVE_SOURCE_METRIC).all(), "Gate A objective source metric mismatch")
    _finite_column(objectives, OBJECTIVE_NAME, label="objective_table")

    nitrogen = _finite_column(observations, "nitrogen_fraction", label="observations")
    _require(np.allclose(nitrogen, 0.0, rtol=0.0, atol=1.0e-15), "all 344 warm-start observations must have nitrogen_fraction=0")
    _finite_column(observations, OBJECTIVE_SOURCE_METRIC, label="observations")
    _finite_column(observations, HARD50_METRIC, label="observations")
    source_counts = observations.groupby("source_campaign_name").size().to_dict()
    _require(int(source_counts.get(DIRECTED_SCAN_NAME, 0)) == EXPECTED_DIRECTED_SCAN_OBSERVATIONS, "directed scan contribution must be exactly 7 rows")

    reference_ids = set(reference_obs["observation_id"].astype(str))
    current_reference = observations.loc[observations["observation_id"].astype(str).isin(reference_ids)].copy()
    _require(len(current_reference) == EXPECTED_REFERENCE_OBSERVATIONS, "Gate A must reproduce all 337 immutable observation IDs")
    current_reference = current_reference.sort_values("observation_id", kind="stable").reset_index(drop=True)
    for column in ("source_campaign_name", "source_case_id", "source_case_name", "laser_case", "sample_source"):
        _require(current_reference[column].fillna("").astype(str).tolist() == reference_obs[column].fillna("").astype(str).tolist(), f"Gate A 337-row reference mismatch in {column}")
    for column in (
        "n0_1e18cm3",
        "plateau_mm_num",
        "diameter_um_num",
        "focus_mm_num",
        OBJECTIVE_SOURCE_METRIC,
        HARD50_METRIC,
    ):
        _assert_numeric_match(current_reference, reference_obs, column=column, label="reference_observations")

    objective_reference_ids = set(reference_obj["observation_id"].astype(str))
    current_obj_ref = objectives.loc[objectives["observation_id"].astype(str).isin(objective_reference_ids)].copy()
    current_obj_ref = current_obj_ref.sort_values("observation_id", kind="stable").reset_index(drop=True)
    _require(len(current_obj_ref) == EXPECTED_REFERENCE_OBSERVATIONS, "Gate A must reproduce all 337 immutable objective IDs")
    _assert_numeric_match(current_obj_ref, reference_obj, column=OBJECTIVE_NAME, label="reference_objectives")

    _require("sobol_index" in observations.columns, "Gate A observations lost sobol_index provenance")
    sobol = pd.to_numeric(observations["sobol_index"], errors="coerce")
    sobol_indices = tuple(sorted({int(value) for value in sobol.dropna() if float(value) >= 0.0}))
    _require(sobol_indices == EXPECTED_SOBOL_INDICES, f"Gate A historical Sobol identity mismatch: {sobol_indices}")

    parameter_space = config.parameter_space()
    codec = SearchSpaceCodec(capillary_search_space(parameter_space))
    identity_matrix, historical_signatures = _effective_identity_rows(observations, parameter_space=parameter_space)
    _require(len(historical_signatures) == EXPECTED_INITIAL_OBSERVATIONS, "Gate A expected 344 unique effective historical signatures")

    surrogate = _read_json(paths["surrogate_summary"])
    _require(surrogate.get("status") == "ok", "Gate A surrogate status must be ok")
    _require(surrogate.get("backend") == "morbo_like", "Gate A surrogate backend must be morbo_like")
    _require(int(surrogate.get("fit_rows", -1)) == EXPECTED_INITIAL_OBSERVATIONS, "Gate A surrogate must receive all 344 fit rows")
    _require(int(surrogate.get("observed_count", -1)) == EXPECTED_INITIAL_OBSERVATIONS, "Gate A MORBO observed_count must be 344")
    _require(int(surrogate.get("skipped_count", -1)) == 0, "Gate A MORBO skipped_count must be zero")
    _require(int(surrogate.get("candidate_rows", -1)) == EXPECTED_CANDIDATES, "Gate A must produce exactly 8 candidates")
    _require(surrogate.get("last_strategy") == "regional_model", "Gate A must not fall back from regional_model")
    _require(surrogate.get("suggestion_mode") == "regional_model", "Gate A suggestion_mode mismatch")
    diagnostics = dict(surrogate.get("model_diagnostics", {}) or {})
    _require(diagnostics.get("status") == "ok", f"Gate A real model failed: {diagnostics.get('reason')}")
    _require(diagnostics.get("acquisition_family") == "single_objective", "Gate A acquisition family must be single_objective")
    _require(diagnostics.get("acquisition_function") == "qLogNoisyExpectedImprovement", "Gate A requires qLogNoisyExpectedImprovement in the certified SUNRISE environment")
    _require(int(diagnostics.get("encoded_dim", -1)) == EXPECTED_ENCODED_DIM, "Gate A model encoded_dim must be 8")
    _require(int(diagnostics.get("train_rows", -1)) == EXPECTED_MODEL_TRAIN_ROWS, "Gate A BoTorch training subset must contain 256 rows")
    _require(int(diagnostics.get("selected_rows", -1)) == EXPECTED_CANDIDATES, "Gate A model must select exactly 8 rows")
    diag_config = dict(diagnostics.get("config", {}) or {})
    _require(diag_config.get("enabled") is True and diag_config.get("fallback_to_random") is False, "Gate A diagnostics must certify no random fallback")

    recommendations = read_table(paths["recommendations"], sep="\t")
    _require(len(recommendations) == EXPECTED_CANDIDATES, f"Gate A expected 8 recommendations, got {len(recommendations)}")
    recommended_signatures = recommendations["candidate_signature"].astype(str).str.strip().tolist()
    _require(all(recommended_signatures), "Gate A recommendations require non-empty signatures")
    _require(len(set(recommended_signatures)) == EXPECTED_CANDIDATES, "Gate A recommendations contain duplicate signatures")
    _require(not (set(recommended_signatures) & historical_signatures), "Gate A recommendation duplicates a historical 6D identity")
    _require(recommendations["morbo_strategy"].astype(str).eq("regional_model").all(), "Gate A recommendations must all originate from regional_model")
    _finite_column(recommendations, "acquisition_value", label="recommended_candidates")
    for _, row in recommendations.iterrows():
        _require(str(row["laser_case"]) in EXPECTED_LASER_CASES, f"candidate laser_case out of bounds: {row['laser_case']}")
        for name, (low, high) in EXPECTED_RANGES.items():
            value = float(row[name])
            _require(math.isfinite(value) and low <= value <= high, f"candidate {name}={value} outside [{low}, {high}]")

    batch = read_table(paths["candidate_batch"], sep="\t")
    _require(len(batch) == EXPECTED_CANDIDATES, f"Gate A candidate_batch expected 8 rows, got {len(batch)}")
    batch_signatures = batch["OPT_CANDIDATE_SIGNATURE"].astype(str).str.strip().tolist()
    _require(set(batch_signatures) == set(recommended_signatures), "Gate A candidate_batch identity differs from recommendations")
    _require(batch["PLASMA_KIND"].astype(str).str.lower().eq("chan").all(), "Gate A candidate_batch must contain only channel cases")
    duration = _finite_column(batch, "LASER_DURATION_FWHM_FS", label="candidate_batch")
    _require(np.allclose(duration, 30.0, rtol=0.0, atol=1.0e-12), "Gate A candidate_batch must keep laser duration fixed at 30 fs")
    batch_nitrogen = _finite_column(batch, "NITROGEN_DOPANT_FRACTION", label="candidate_batch")
    _require(((batch_nitrogen >= 0.0) & (batch_nitrogen <= 0.01)).all(), "Gate A candidate_batch nitrogen fraction outside [0, 0.01]")
    if "OPT_SOBOL_INDEX" in batch.columns:
        _require(batch["OPT_SOBOL_INDEX"].fillna("").astype(str).str.strip().eq("").all(), "Gate A candidates must not be emitted as Sobol points")

    state = load_optimizer_state(paths["morbo_state"])
    fingerprint = dict(dict(state.extra).get("state_fingerprint_payload", {}) or {})
    search_space = dict(fingerprint.get("search_space", {}) or {})
    _require(int(search_space.get("encoded_dim", -1)) == EXPECTED_ENCODED_DIM, "Gate A MORBO fingerprint encoded_dim mismatch")
    _require(search_space.get("parameter_names") == list(EXPECTED_PARAMETER_NAMES), "Gate A MORBO fingerprint parameter order mismatch")
    registry = state.candidate_registry
    pending = set(registry.pending_candidate_signatures)
    seen = set(registry.seen_candidate_signatures)
    _require(pending == set(recommended_signatures), "Gate A pending registry must contain exactly the 8 recommendations")
    _require(historical_signatures.issubset(seen), "Gate A seen registry is missing historical identities")
    _require(set(recommended_signatures).issubset(seen), "Gate A seen registry is missing pending identities")
    _require(len(seen) == EXPECTED_INITIAL_OBSERVATIONS + EXPECTED_CANDIDATES, f"Gate A expected 352 seen identities, got {len(seen)}")

    _assert_batch_plan(config, _read_json(paths["batch_plan"]))
    optimizer_state = _read_json(paths["optimizer_state"])
    _require(optimizer_state.get("optimizer_backend") == "morbo_like", "Gate A optimizer_state backend mismatch")
    _require(int(optimizer_state.get("latest_iteration", -1)) == 0, "Gate A optimizer_state iteration mismatch")
    history = optimizer_state.get("history")
    _require(isinstance(history, list) and len(history) == 1, "Gate A optimizer_state must contain one iter_000 history record")
    record = history[0]
    expected_links = {
        "observations": "inputs/observations.csv",
        "objectives": "inputs/objective_table.csv",
        "recommendations": "outputs/recommended_candidates.tsv",
        "surrogate_summary": "outputs/surrogate_summary.json",
        "candidate_batch": "outputs/candidate_batch.tsv",
        "batch_campaign_plan": "outputs/batch_campaign_plan.json",
    }
    for key, value in expected_links.items():
        _require(record.get(key) == value, f"Gate A optimizer_state linkage mismatch for {key}")

    report_dir = iter_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    identity_path = report_dir / "clpu_n2_gate_a_identity_matrix.csv"
    identity_matrix.to_csv(identity_path, index=False)
    result = {
        "schema_version": GATE_A_SCHEMA_VERSION,
        "contract_id": GATE_A_CONTRACT_ID,
        "status": "pass",
        "staging_root": str(staging),
        "source_count": EXPECTED_STATIC_SOURCES,
        "observation_rows": len(observations),
        "fit_eligible_rows": int(objectives["fit_eligible"].astype(str).str.lower().eq("true").sum()),
        "historical_nitrogen_fraction_unique": sorted(float(v) for v in np.unique(nitrogen)),
        "sobol_indices": list(sobol_indices),
        "effective_historical_signature_count": len(historical_signatures),
        "physical_parameter_names": list(EXPECTED_PARAMETER_NAMES),
        "encoded_dim": codec.encoded_dim(),
        "surrogate": {
            "fit_rows": int(surrogate["fit_rows"]),
            "model_train_rows": int(diagnostics["train_rows"]),
            "model_status": diagnostics["status"],
            "acquisition_function": diagnostics["acquisition_function"],
            "candidate_rows": int(surrogate["candidate_rows"]),
            "last_strategy": surrogate["last_strategy"],
        },
        "candidate_count": len(recommendations),
        "pending_count": len(pending),
        "seen_count_after_registration": len(seen),
        "candidate_batch_fixed_duration_fwhm_fs": 30.0,
        "identity_matrix": str(identity_path),
        "artifacts": {label: str(path) for label, path in paths.items()},
        "note": "Gate A passed. No materialization, WarpX execution, cleanup, or sbatch is authorized; Gate B remains required.",
    }
    result_path = report_dir / "clpu_n2_gate_a.json"
    write_json(result_path, result)
    return {**result, "result_path": str(result_path)}
