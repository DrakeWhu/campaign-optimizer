from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_OBJECTIVE_CONFIG = {
    "schema_version": 1,
    "config_id": "capillary_objectives_v1",
    "required_scores_for_fit": [
        "score_guiding_v1",
        "score_beamlike_v1",
        "score_transverse_v1",
    ],
}

DEFAULT_PARAMETER_SPACE = {
    "version": 1,
    "laser_cases": ["f20", "f32", "f40"],
    "ranges": {
        "n0_1e18cm3": [0.7, 6.0],
        "plateau_mm_num": [5.0, 25.0],
        "diameter_um_num": [150.0, 500.0],
        "focus_mm_num": [-5.0, 5.0],
    },
}

DEFAULT_RECOMMENDATION = {
    "seed": 12345,
    "n_candidates": 30,
    "n_random": 2000,
    "min_known_scaled_dist": 0.035,
    "nearest_k": 8,
}

DEFAULT_CANDIDATE_BATCH = {
    "case_id_start": 0,
    "plasma_kind": "chan",
    "cap_nr": 192,
    "cap_rmax_um": None,
    "campaign_template": {
        "campaign_json": "campaign.json",
        "input_template": "input_template.py",
    },
}

DEFAULT_REDUCED_OUTPUTS = {
    "guiding_metrics": "guiding_metrics.csv",
    "particle_summary": "particle_analysis/particle_summary.csv",
    "acceptance_curves": "particle_analysis/particle_acceptance_curves.csv",
}

DEFAULT_OPTIMIZATION_HISTORY = {
    "enabled": "auto",
    "optimization_name": None,
    "iterations_root": "iterations",
    "cases_tsv": "cases.tsv",
    "campaign_name_template": "{optimization_name}_iter_{iteration:03d}",
    "reduced_outputs": dict(DEFAULT_REDUCED_OUTPUTS),
}


@dataclass(frozen=True)
class OptimizerConfig:
    path: Path
    base_dir: Path
    data: dict[str, Any]

    @property
    def optimizer_run_root(self) -> Path:
        root = Path(str(self.data.get("optimizer_run_root", "optimizer_runs")))
        if not root.is_absolute():
            root = self.base_dir / root
        return root

    def iteration_dir(self, iteration: int) -> Path:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        return self.optimizer_run_root / f"iter_{iteration:03d}"

    def source_campaigns(self) -> list[dict[str, Any]]:
        value = self.data.get("source_campaigns")
        if not isinstance(value, list) or not value:
            raise ValueError(
                "optimizer.json must define a non-empty source_campaigns list"
            )
        return [dict(source) for source in value]

    def optimization_history_config(self) -> dict[str, Any]:
        value = self.data.get("optimization_history", {}) or {}
        if not isinstance(value, dict):
            raise ValueError("optimizer.json optimization_history must be an object")

        cfg = {
            "enabled": DEFAULT_OPTIMIZATION_HISTORY["enabled"],
            "optimization_name": self.base_dir.name,
            "iterations_root": DEFAULT_OPTIMIZATION_HISTORY["iterations_root"],
            "cases_tsv": DEFAULT_OPTIMIZATION_HISTORY["cases_tsv"],
            "campaign_name_template": DEFAULT_OPTIMIZATION_HISTORY[
                "campaign_name_template"
            ],
            "reduced_outputs": dict(DEFAULT_OPTIMIZATION_HISTORY["reduced_outputs"]),
        }
        cfg.update({k: v for k, v in value.items() if k != "reduced_outputs"})

        reduced_outputs = dict(cfg["reduced_outputs"])
        reduced_outputs.update(value.get("reduced_outputs", {}) or {})
        cfg["reduced_outputs"] = reduced_outputs

        if not cfg.get("optimization_name"):
            cfg["optimization_name"] = self.base_dir.name

        return cfg

    def optimization_history_enabled(self) -> bool:
        cfg = self.optimization_history_config()
        enabled = cfg.get("enabled", "auto")

        if isinstance(enabled, bool):
            return enabled

        if str(enabled).strip().lower() == "auto":
            iterations_root = resolve_path(self.base_dir, cfg["iterations_root"])
            return iterations_root.is_dir()

        return str(enabled).strip().lower() in {"1", "true", "yes", "on"}

    def optimization_history_source_campaigns(
        self, iteration: int
    ) -> list[dict[str, Any]]:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")

        if not self.optimization_history_enabled():
            return []

        cfg = self.optimization_history_config()
        iterations_root = resolve_path(self.base_dir, cfg["iterations_root"])
        optimization_name = str(cfg["optimization_name"])
        name_template = str(cfg["campaign_name_template"])
        cases_tsv = str(cfg["cases_tsv"])
        reduced_outputs = dict(cfg["reduced_outputs"])

        sources: list[dict[str, Any]] = []

        for previous_iteration in range(iteration):
            campaign_root = iterations_root / f"iter_{previous_iteration:03d}"

            if not campaign_root.is_dir():
                continue

            if not (campaign_root / cases_tsv).is_file():
                continue

            campaign_name = name_template.format(
                optimization_name=optimization_name,
                iteration=previous_iteration,
                iteration_name=f"iter_{previous_iteration:03d}",
            )

            sources.append(
                {
                    "campaign_name": campaign_name,
                    "campaign_root": str(campaign_root),
                    "cases_tsv": cases_tsv,
                    "reduced_outputs": dict(reduced_outputs),
                    "source_kind": "optimization_history",
                    "history_iteration": previous_iteration,
                }
            )

        return sources

    def source_campaigns_for_iteration(self, iteration: int) -> list[dict[str, Any]]:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")

        merged: list[dict[str, Any]] = []
        seen_roots: set[str] = set()

        for source in [
            *self.source_campaigns(),
            *self.optimization_history_source_campaigns(iteration),
        ]:
            campaign_root = resolve_path(self.base_dir, source["campaign_root"])
            root_key = str(campaign_root.resolve())

            if root_key in seen_roots:
                continue

            seen_roots.add(root_key)
            merged.append(dict(source))

        return merged

    def objective_config(self) -> dict[str, Any]:
        value = dict(DEFAULT_OBJECTIVE_CONFIG)
        value.update(self.data.get("objective", {}) or {})
        return value

    def parameter_space(self) -> dict[str, Any]:
        value = {
            "version": DEFAULT_PARAMETER_SPACE["version"],
            "laser_cases": list(DEFAULT_PARAMETER_SPACE["laser_cases"]),
            "ranges": dict(DEFAULT_PARAMETER_SPACE["ranges"]),
        }
        override = self.data.get("parameter_space", {}) or {}
        value.update({k: v for k, v in override.items() if k != "ranges"})

        ranges = dict(value["ranges"])
        ranges.update(override.get("ranges", {}) or {})
        value["ranges"] = ranges
        return value

    def recommendation_config(self) -> dict[str, Any]:
        value = dict(DEFAULT_RECOMMENDATION)
        value.update(self.data.get("recommendation", {}) or {})
        return value

    def candidate_batch_config(self) -> dict[str, Any]:
        value = {
            "case_id_start": DEFAULT_CANDIDATE_BATCH["case_id_start"],
            "plasma_kind": DEFAULT_CANDIDATE_BATCH["plasma_kind"],
            "cap_nr": DEFAULT_CANDIDATE_BATCH["cap_nr"],
            "cap_rmax_um": DEFAULT_CANDIDATE_BATCH["cap_rmax_um"],
            "campaign_template": dict(DEFAULT_CANDIDATE_BATCH["campaign_template"]),
        }

        override = self.data.get("candidate_batch", {}) or {}
        value.update(
            {
                k: v
                for k, v in override.items()
                if k not in {"cap_rmax_um", "campaign_template"}
            }
        )

        if override.get("cap_rmax_um") is not None:
            value["cap_rmax_um"] = dict(override.get("cap_rmax_um") or {})

        campaign_template = dict(value["campaign_template"])
        campaign_template.update(override.get("campaign_template", {}) or {})
        value["campaign_template"] = campaign_template

        return value

    def validation_config(self) -> dict[str, Any]:
        return dict(self.data.get("validation", {}) or {})


def load_optimizer_config(path: str | Path) -> OptimizerConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f_in:
        data = json.load(f_in)

    if data.get("schema_version") != 1:
        raise ValueError("optimizer.json schema_version must be 1")

    return OptimizerConfig(
        path=config_path,
        base_dir=config_path.parent,
        data=data,
    )


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path
