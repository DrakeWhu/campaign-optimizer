from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .parameters import DISTANCE_COLUMNS


DEFAULT_AX_OBJECTIVES = [
    "score_guiding_v1",
    "score_beamlike_v1",
    "score_transverse_v1",
]


def _prediction_stem(score_name: str) -> str:
    """Map score_guiding_v1 -> score_guiding for legacy output columns."""

    if score_name.endswith("_v1"):
        return score_name[: -len("_v1")]
    return score_name


def _finite_objective_rows(
    history: pd.DataFrame,
    objectives: list[str],
) -> pd.DataFrame:
    required = list(DISTANCE_COLUMNS) + list(objectives)
    missing = [c for c in required if c not in history.columns]
    if missing:
        raise ValueError(f"AxModelManager backend missing required columns: {missing}")

    fit = history.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    for col in required:
        fit[col] = pd.to_numeric(fit[col], errors="coerce")

    fit = fit.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return fit


def make_varying_parameters(parameter_space: dict[str, Any]):
    """Create optimas VaryingParameter objects from optimizer parameter_space."""

    from optimas.core import VaryingParameter

    ranges = parameter_space["ranges"]

    return [
        VaryingParameter("f_number", 20.0, 40.0),
        VaryingParameter(
            "n0_1e18cm3",
            float(ranges["n0_1e18cm3"][0]),
            float(ranges["n0_1e18cm3"][1]),
        ),
        VaryingParameter(
            "plateau_mm_num",
            float(ranges["plateau_mm_num"][0]),
            float(ranges["plateau_mm_num"][1]),
        ),
        VaryingParameter(
            "diameter_um_num",
            float(ranges["diameter_um_num"][0]),
            float(ranges["diameter_um_num"][1]),
        ),
        VaryingParameter(
            "focus_mm_num",
            float(ranges["focus_mm_num"][0]),
            float(ranges["focus_mm_num"][1]),
        ),
    ]


def make_objectives(objective_names: list[str]):
    """Create maximization objectives for Optimas/AxModelManager."""

    from optimas.core import Objective

    return [Objective(name, minimize=False) for name in objective_names]


def predict_with_ax_model_manager(
    *,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    parameter_space: dict[str, Any],
    rec_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit Optimas/AxModelManager GP surrogate(s) and predict candidate scores.

    This backend is intentionally file-contract compatible with the previous
    passive recommender: it returns pred_score_*_mean/sem columns that the
    existing ranking code can scalarize.

    It does not launch simulations, does not mutate campaigns, and does not
    read raw diagnostics.
    """

    from optimas.utils import AxModelManager

    logging.getLogger("ax").setLevel(logging.ERROR)

    objective_names = list(rec_cfg.get("objectives", DEFAULT_AX_OBJECTIVES))
    min_fit_rows = int(rec_cfg.get("min_fit_rows", 20))
    fit_out_of_design = bool(rec_cfg.get("fit_out_of_design", False))

    fit = _finite_objective_rows(history, objective_names)

    if len(fit) < min_fit_rows:
        raise ValueError(
            "Not enough fit rows for AxModelManager backend: "
            f"{len(fit)} < {min_fit_rows}"
        )

    sample = candidates[DISTANCE_COLUMNS].copy()
    for col in DISTANCE_COLUMNS:
        sample[col] = pd.to_numeric(sample[col], errors="coerce")

    bad_sample = sample.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    if bad_sample.any():
        raise ValueError(
            "Candidate cloud contains non-finite AxModelManager sample rows: "
            f"{int(bad_sample.sum())}"
        )

    model = AxModelManager(
        fit,
        varying_parameters=make_varying_parameters(parameter_space),
        objectives=make_objectives(objective_names),
        fit_out_of_design=fit_out_of_design,
    )

    out = candidates.copy()

    for objective in objective_names:
        mean, sem = model.evaluate_model(
            sample=sample,
            metric_name=objective,
        )

        stem = _prediction_stem(objective)
        out[f"pred_{stem}_mean"] = np.asarray(mean, dtype=float).reshape(-1)
        out[f"pred_{stem}_sem"] = np.asarray(sem, dtype=float).reshape(-1)

    summary = {
        "schema_version": 1,
        "backend": "optimas_ax_model_manager",
        "surrogate_backend": "optimas.utils.AxModelManager",
        "status": "ok",
        "fit_rows": int(len(fit)),
        "candidate_rows": int(len(candidates)),
        "objectives": objective_names,
        "fit_out_of_design": fit_out_of_design,
        "min_fit_rows": min_fit_rows,
        "note": (
            "Optimas/AxModelManager GP surrogate backend. Candidate ranking is "
            "a scalarized multi-objective acquisition over surrogate means/SEMs."
        ),
    }

    return out, summary
