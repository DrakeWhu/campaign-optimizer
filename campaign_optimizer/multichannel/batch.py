from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_json, write_tsv
from campaign_optimizer.state import now_utc, write_optimizer_state

from .parameters import parameter_definitions


EXPECTED_WORKFLOW = [
    "materialize case directories and case.env",
    "run WarpX 3D through the case-cycle array",
    "validate final openPMD/HDF5 particle diagnostic",
    "run multichannel-lfmetrics on the final frame",
    "validate post/particle_summary.csv and plots",
    "delete raw HDF5 files through cleanup manifests",
]

NON_GOALS = [
    "does not submit jobs",
    "does not call sbatch/srun/mpiexec/mpirun",
    "does not launch WarpX",
    "does not read raw HDF5/openPMD",
    "does not delete data",
    "does not mutate previous campaigns",
]

_SAFE_TOKEN = re.compile(r"[^a-zA-Z0-9_.-]+")


def _finite(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite: {value!r}")
    return numeric


def _case_name(iteration: int, case_id: int, source: str) -> str:
    token = _SAFE_TOKEN.sub("_", str(source).strip().lower()).strip("._-")
    token = token or "candidate"
    return f"mc_i{iteration:03d}_c{case_id:03d}_{token}"


def _source_campaigns_for_plan(
    config: OptimizerConfig, iteration: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in config.source_campaigns_for_iteration(iteration):
        out.append(
            {
                key: source[key]
                for key in (
                    "campaign_name",
                    "campaign_root",
                    "cases_tsv",
                    "source_kind",
                    "history_iteration",
                )
                if key in source
            }
        )
    return out


def recommended_candidates_to_candidate_batch(
    recommended: pd.DataFrame,
    *,
    config: OptimizerConfig,
    iteration: int,
) -> pd.DataFrame:
    parameter_space = config.parameter_space()
    definitions = parameter_definitions(parameter_space)
    required = [
        "recommendation_id",
        "optimizer_iteration",
        "candidate_id",
        "rank",
        "candidate_source",
        "candidate_signature",
        *[item["name"] for item in definitions],
    ]
    missing = [column for column in required if column not in recommended.columns]
    if missing:
        raise ValueError(f"recommended_candidates.tsv is missing {missing}")

    batch_cfg = config.candidate_batch_config()
    case_id_start = int(batch_cfg.get("case_id_start", 0))
    rows: list[dict[str, Any]] = []

    ordered = recommended.sort_values("rank", kind="stable").reset_index(drop=True)
    for offset, recommendation in ordered.iterrows():
        case_id = case_id_start + int(offset)
        source = str(recommendation.get("candidate_source", "candidate"))
        row: dict[str, Any] = {
            "CASE_ID": case_id,
            "CASE_NAME": _case_name(iteration, case_id, source),
        }
        for item in definitions:
            value = _finite(
                recommendation[item["name"]], label=item["name"]
            )
            low, high = item["bounds"]
            if item["type"] == "periodic":
                value = low + ((value - low) % (high - low))
            elif value < low or value > high:
                raise ValueError(
                    f"{item['name']}={value} is outside [{low}, {high}]"
                )
            row[item["column"]] = value

        row.update(
            {
                "WRITE_FIELD_DIAGNOSTIC": (
                    1
                    if bool(batch_cfg.get("write_fields_for_reference", True))
                    and source == "reference"
                    else 0
                ),
                "OPT_ITERATION": int(iteration),
                "OPT_CANDIDATE_ID": str(recommendation.get("candidate_id", "")),
                "OPT_RECOMMENDATION_ID": str(
                    recommendation.get("recommendation_id", "")
                ),
                "OPT_OBJECTIVE_CONFIG_ID": str(
                    config.objective_config().get(
                        "config_id", "multichannel_beam_objectives_v1"
                    )
                ),
                "OPT_SAMPLE_SOURCE": source,
                "OPT_SOBOL_INDEX": recommendation.get("sobol_index", ""),
                "OPT_CANDIDATE_SIGNATURE": str(
                    recommendation.get("candidate_signature", "")
                ),
                "OPT_RANKING_SOURCE": str(
                    recommendation.get("ranking_source", source)
                ),
                "OPT_ACQUISITION_VALUE": recommendation.get(
                    "acquisition_value", ""
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _candidate_batch_contract(config: OptimizerConfig) -> dict[str, Any]:
    definitions = parameter_definitions(config.parameter_space())
    parameter_columns = [item["column"] for item in definitions]
    return {
        "schema_version": 1,
        "required_columns": [
            "CASE_ID",
            "CASE_NAME",
            *parameter_columns,
            "WRITE_FIELD_DIAGNOSTIC",
        ],
        "numeric_columns": [*parameter_columns, "WRITE_FIELD_DIAGNOSTIC"],
        "integer_columns": ["CASE_ID", "WRITE_FIELD_DIAGNOSTIC"],
        "choices": {"WRITE_FIELD_DIAGNOSTIC": ["0", "1"]},
        "bounds": {
            item["column"]: item["bounds"]
            for item in definitions
            if item["type"] != "periodic"
        },
        "summary_columns": ["OPT_SAMPLE_SOURCE"],
    }


def write_batch_campaign_plan(
    *,
    config: OptimizerConfig,
    iteration: int,
    candidate_batch_path: Path,
    recommended_candidates_path: Path,
) -> Path:
    iter_dir = config.iteration_dir(iteration)
    batch_cfg = config.candidate_batch_config()

    def relative(path: Path) -> str:
        return str(path.relative_to(iter_dir)).replace("\\", "/")

    plan = {
        "schema_version": 1,
        "plan_type": "optimizer_candidate_batch",
        "problem_kind": "multichannel",
        "created_at": now_utc(),
        "optimizer_iteration": int(iteration),
        "objective_config_id": config.objective_config().get(
            "config_id", "multichannel_beam_objectives_v1"
        ),
        "candidate_batch": relative(candidate_batch_path),
        "recommended_candidates": relative(recommended_candidates_path),
        "source_campaigns": _source_campaigns_for_plan(config, iteration),
        "campaign_template": dict(batch_cfg.get("campaign_template", {}) or {}),
        "candidate_batch_contract": _candidate_batch_contract(config),
        "expected_workflow": EXPECTED_WORKFLOW,
        "non_goals": NON_GOALS,
    }
    return write_json(iter_dir / "outputs" / "batch_campaign_plan.json", plan)


def build_candidate_batch(
    config: OptimizerConfig, iteration: int
) -> tuple[Path, Path]:
    iter_dir = config.iteration_dir(iteration)
    outputs_dir = iter_dir / "outputs"
    recommended_path = outputs_dir / "recommended_candidates.tsv"
    recommended = read_table(recommended_path, sep="\t")

    candidate_batch = recommended_candidates_to_candidate_batch(
        recommended,
        config=config,
        iteration=iteration,
    )
    candidate_batch_path = write_tsv(
        outputs_dir / "candidate_batch.tsv", candidate_batch
    )
    plan_path = write_batch_campaign_plan(
        config=config,
        iteration=iteration,
        candidate_batch_path=candidate_batch_path,
        recommended_candidates_path=recommended_path,
    )

    write_optimizer_state(
        iter_dir=iter_dir,
        iteration=iteration,
        config=config.data,
        objective_config=config.objective_config(),
        parameter_space=config.parameter_space(),
        observations_path=iter_dir / "inputs" / "observations.csv",
        objectives_path=iter_dir / "inputs" / "objective_table.csv",
        recommendations_path=recommended_path,
        surrogate_summary_path=outputs_dir / "surrogate_summary.json",
        candidate_batch_path=candidate_batch_path,
        batch_campaign_plan_path=plan_path,
        backend="sobol_then_morbo",
    )
    return candidate_batch_path, plan_path
