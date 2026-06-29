from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import write_json


def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def relative_to_iter(iter_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None

    try:
        return str(path.relative_to(iter_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_optimizer_state(
    *,
    iter_dir: Path,
    iteration: int,
    config: dict[str, Any],
    objective_config: dict[str, Any],
    parameter_space: dict[str, Any],
    observations_path: Path | None = None,
    objectives_path: Path | None = None,
    recommendations_path: Path | None = None,
    surrogate_summary_path: Path | None = None,
    backend: str = "passive_nearest_observed",
) -> Path:
    objective_hash = stable_hash(objective_config)

    state = {
        "schema_version": 1,
        "created_at": now_utc(),
        "optimizer_backend": backend,
        "optimizer_backend_version": "0.1.0",
        "surrogate_backend": backend,
        "objective_schema_version": objective_config.get("schema_version", 1),
        "objective_config_id": objective_config.get(
            "config_id", "capillary_objectives_v1"
        ),
        "objective_config_hash": objective_hash,
        "parameter_space_version": parameter_space.get("version", 1),
        "random_seed": (config.get("recommendation", {}) or {}).get("seed"),
        "latest_iteration": int(iteration),
        "non_goals": [
            "does not launch WarpX",
            "does not call sbatch/srun/mpiexec/mpirun",
            "does not read raw HDF5/openPMD",
            "does not create campaign roots",
            "does not mutate campaign-workflow state",
        ],
        "history": [
            {
                "iteration": int(iteration),
                "observations": relative_to_iter(iter_dir, observations_path),
                "objectives": relative_to_iter(iter_dir, objectives_path),
                "recommendations": relative_to_iter(iter_dir, recommendations_path),
                "surrogate_summary": relative_to_iter(iter_dir, surrogate_summary_path),
                "candidate_batch": None,
                "batch_campaign_plan": None,
            }
        ],
    }

    return write_json(iter_dir / "optimizer_state.json", state)
