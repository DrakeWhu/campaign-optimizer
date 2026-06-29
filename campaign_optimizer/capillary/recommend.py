from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_tsv

from .parameters import DISTANCE_COLUMNS, scaled_parameter_array
from .parsing import f_number_from_laser_case


PREDICTED_SCORES = [
    "score_guiding_v1",
    "score_beamlike_v1",
    "score_transverse_v1",
    "score_acceptance_v1",
]


def robust01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return np.zeros_like(arr)

    lo, hi = np.nanpercentile(finite, [5.0, 95.0])

    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1.0e-12:
        return np.zeros_like(arr)

    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _sample_uniform(
    rng: np.random.Generator,
    lo_hi: list[float],
    n: int,
) -> np.ndarray:
    lo, hi = [float(v) for v in lo_hi]
    return rng.uniform(lo, hi, size=n)


def build_candidate_cloud(
    obs: pd.DataFrame,
    objectives: pd.DataFrame,
    parameter_space: dict[str, Any],
    rec_cfg: dict[str, Any],
) -> pd.DataFrame:
    rng = np.random.default_rng(int(rec_cfg.get("seed", 12345)))
    n_random = int(rec_cfg.get("n_random", 2000))

    lasers = list(parameter_space.get("laser_cases", ["f20", "f32", "f40"]))
    ranges = parameter_space["ranges"]

    random = pd.DataFrame(
        {
            "laser_case": rng.choice(lasers, size=n_random),
            "n0_1e18cm3": _sample_uniform(rng, ranges["n0_1e18cm3"], n_random),
            "plateau_mm_num": _sample_uniform(rng, ranges["plateau_mm_num"], n_random),
            "diameter_um_num": _sample_uniform(
                rng, ranges["diameter_um_num"], n_random
            ),
            "focus_mm_num": _sample_uniform(rng, ranges["focus_mm_num"], n_random),
        }
    )
    random["candidate_source"] = "global_random_discrete_laser"

    fit = objectives[
        objectives["fit_eligible"].astype(str).str.lower() == "true"
    ].copy()

    joined = obs.merge(
        fit[
            [
                "observation_id",
                "score_guiding_v1",
                "score_beamlike_v1",
                "score_transverse_v1",
            ]
        ],
        on="observation_id",
        how="inner",
    )

    anchors = []
    for score in ["score_guiding_v1", "score_beamlike_v1", "score_transverse_v1"]:
        if score in joined.columns:
            anchors.append(joined.nlargest(min(8, len(joined)), score))

    if anchors:
        anchor_df = pd.concat(anchors, ignore_index=True).drop_duplicates(
            subset=DISTANCE_COLUMNS
        )
    else:
        anchor_df = joined.head(0)

    jitter_rows = []
    sigmas = {
        "n0_1e18cm3": 0.45,
        "plateau_mm_num": 2.5,
        "diameter_um_num": 45.0,
        "focus_mm_num": 1.2,
    }

    for _, row in anchor_df.iterrows():
        for _ in range(100):
            cand = {"laser_case": str(row["laser_case"])}

            for p, sigma in sigmas.items():
                lo, hi = [float(v) for v in ranges[p]]
                cand[p] = float(np.clip(float(row[p]) + rng.normal(0.0, sigma), lo, hi))

            cand["candidate_source"] = "jitter_best_observed_discrete_laser"
            jitter_rows.append(cand)

    jitter = pd.DataFrame(jitter_rows)

    candidates = pd.concat([random, jitter], ignore_index=True)

    candidates["f_number"] = candidates["laser_case"].map(f_number_from_laser_case)
    candidates["n0_cm3"] = candidates["n0_1e18cm3"] * 1.0e18
    candidates["plateau_length_mm"] = candidates["plateau_mm_num"]
    candidates["diameter_um"] = candidates["diameter_um_num"]
    candidates["radius_um"] = 0.5 * candidates["diameter_um_num"]
    candidates["focus_offset_from_plateau_start_mm"] = candidates["focus_mm_num"]

    candidates = candidates.drop_duplicates(
        subset=[
            "laser_case",
            "n0_1e18cm3",
            "plateau_mm_num",
            "diameter_um_num",
            "focus_mm_num",
        ]
    ).reset_index(drop=True)

    return candidates


def _predict_from_nearest(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    parameter_space: dict[str, Any],
    nearest_k: int,
) -> pd.DataFrame:
    out = candidates.copy()

    x_hist = scaled_parameter_array(history, parameter_space)
    x_cand = scaled_parameter_array(candidates, parameter_space)

    distances = np.sqrt(((x_cand[:, None, :] - x_hist[None, :, :]) ** 2).sum(axis=2))

    nearest = np.argsort(distances, axis=1)[:, : max(1, nearest_k)]
    nearest_dist = np.take_along_axis(distances, nearest, axis=1)

    out["nearest_known_scaled_dist"] = nearest_dist[:, 0]

    weights = 1.0 / np.maximum(nearest_dist, 1.0e-9)
    weights = weights / weights.sum(axis=1, keepdims=True)

    for score in PREDICTED_SCORES:
        out_mean = f"pred_{score.replace('_v1', '')}_mean"
        out_sem = f"pred_{score.replace('_v1', '')}_sem"

        if score not in history.columns:
            out[out_mean] = np.nan
            out[out_sem] = np.nan
            continue

        vals = pd.to_numeric(history[score], errors="coerce").to_numpy(float)
        v_near = vals[nearest]

        out[out_mean] = np.nansum(v_near * weights, axis=1)
        out[out_sem] = np.nanstd(v_near, axis=1)

    return out


def _add_ranking_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for score in [
        "score_guiding",
        "score_beamlike",
        "score_transverse",
        "score_acceptance",
    ]:
        mean_col = f"pred_{score}_mean"
        sem_col = f"pred_{score}_sem"

        if mean_col in out.columns:
            out[f"norm_{score}"] = robust01(out[mean_col].to_numpy(float))

        if sem_col in out.columns:
            out[f"norm_{score}_sem"] = robust01(out[sem_col].to_numpy(float))

    out["score_balanced_conservative"] = (
        out.get("norm_score_guiding", 0.0)
        + out.get("norm_score_beamlike", 0.0)
        + out.get("norm_score_transverse", 0.0)
        - 0.15
        * (
            out.get("norm_score_guiding_sem", 0.0)
            + out.get("norm_score_beamlike_sem", 0.0)
            + out.get("norm_score_transverse_sem", 0.0)
        )
    )

    out["score_balanced_exploratory"] = (
        out.get("norm_score_guiding", 0.0)
        + out.get("norm_score_beamlike", 0.0)
        + out.get("norm_score_transverse", 0.0)
        + 0.25
        * (
            out.get("norm_score_guiding_sem", 0.0)
            + out.get("norm_score_beamlike_sem", 0.0)
            + out.get("norm_score_transverse_sem", 0.0)
        )
    )

    out["score_beamlike_guarded"] = (
        out.get("norm_score_beamlike", 0.0)
        + 0.55 * out.get("norm_score_transverse", 0.0)
        + 0.30 * out.get("norm_score_guiding", 0.0)
    )

    return out


def _take_diverse_top(
    df: pd.DataFrame,
    parameter_space: dict[str, Any],
    score_col: str,
    n: int,
    min_dist: float,
) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    selected = []
    selected_x = []

    for _, row in ranked.iterrows():
        x = scaled_parameter_array(pd.DataFrame([row]), parameter_space)[0]

        if selected_x:
            d = np.sqrt(((np.vstack(selected_x) - x) ** 2).sum(axis=1)).min()
            if d < min_dist:
                continue

        selected.append(row)
        selected_x.append(x)

        if len(selected) >= n:
            break

    return pd.DataFrame(selected) if selected else ranked.head(n)


def propose_recommendations(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    outputs_dir = iter_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    obs = read_table(iter_dir / "inputs" / "observations.csv")
    obj = read_table(iter_dir / "inputs" / "objective_table.csv")

    parameter_space = config.parameter_space()
    rec_cfg = config.recommendation_config()

    fit = obj[obj["fit_eligible"].astype(str).str.lower() == "true"].copy()

    if fit.empty:
        raise ValueError("No fit-eligible objective rows available for recommendations")

    history = obs.merge(fit, on="observation_id", how="inner")
    history = history.dropna(subset=DISTANCE_COLUMNS).reset_index(drop=True)

    if history.empty:
        raise ValueError("No fit-eligible rows with finite parameter columns")

    candidates = build_candidate_cloud(obs, obj, parameter_space, rec_cfg)

    pred = _predict_from_nearest(
        history,
        candidates,
        parameter_space,
        int(rec_cfg.get("nearest_k", 8)),
    )

    pred = pred[
        pred["nearest_known_scaled_dist"]
        >= float(rec_cfg.get("min_known_scaled_dist", 0.035))
    ].reset_index(drop=True)

    pred = _add_ranking_scores(pred)

    n = int(rec_cfg.get("n_candidates", 30))

    recommended = _take_diverse_top(
        pred,
        parameter_space,
        "score_balanced_conservative",
        n=n,
        min_dist=0.045,
    ).reset_index(drop=True)

    recommended.insert(0, "rank", np.arange(1, len(recommended) + 1, dtype=int))
    recommended.insert(
        0,
        "candidate_id",
        [f"opt_{iteration:03d}_{i:03d}" for i in range(len(recommended))],
    )
    recommended.insert(0, "optimizer_iteration", int(iteration))
    recommended.insert(
        0,
        "recommendation_id",
        [f"iter_{iteration:03d}_rank_{i:03d}" for i in recommended["rank"]],
    )

    recommended["recommendation_status"] = "recommended_placeholder_no_launch"
    recommended["ranking_source"] = "score_balanced_conservative"
    recommended["acquisition_value"] = recommended["score_balanced_conservative"]

    first = [
        "recommendation_id",
        "optimizer_iteration",
        "candidate_id",
        "rank",
        "recommendation_status",
        "candidate_source",
        "ranking_source",
        "acquisition_value",
        "laser_case",
        "f_number",
        "n0_1e18cm3",
        "n0_cm3",
        "plateau_mm_num",
        "diameter_um_num",
        "focus_mm_num",
        "nearest_known_scaled_dist",
        "score_balanced_conservative",
        "score_balanced_exploratory",
        "score_beamlike_guarded",
        "pred_score_guiding_mean",
        "pred_score_guiding_sem",
        "pred_score_beamlike_mean",
        "pred_score_beamlike_sem",
        "pred_score_transverse_mean",
        "pred_score_transverse_sem",
        "pred_score_acceptance_mean",
        "pred_score_acceptance_sem",
    ]

    cols = [c for c in first if c in recommended.columns]
    cols += [c for c in recommended.columns if c not in cols]

    return write_tsv(outputs_dir / "recommended_candidates.tsv", recommended[cols])
