from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_json, write_tsv
from campaign_optimizer.morbo import (
    Choice,
    FloatRange,
    MorboLikeBackend,
    ObjectiveDefinition,
    ObjectiveSpec,
    OptimizerState,
    RegionalPolicy,
    TrialInput,
    load_optimizer_state,
    save_frontier_records,
    save_optimizer_state as save_morbo_optimizer_state,
    save_region_records,
    write_recommended_candidates_tsv,
    BotorchRegionalConfig,
)

from .parameters import DISTANCE_COLUMNS, scaled_parameter_array
from .parsing import f_number_from_laser_case
from .ax_backend import predict_with_ax_model_manager


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

    if "nearest_known_scaled_dist" not in out.columns:
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


def _morbo_suggestion_mode(rec_cfg: dict[str, Any]) -> str:
    return str(
        rec_cfg.get(
            "suggestion_mode",
            rec_cfg.get("morbo_suggestion_mode", "regional_random"),
        )
    )


def _morbo_botorch_config(rec_cfg: dict[str, Any]) -> BotorchRegionalConfig:
    payload = dict(rec_cfg.get("botorch", {}) or {})
    suggestion_mode = _morbo_suggestion_mode(rec_cfg)

    if suggestion_mode in {"regional_model", "botorch", "qlognehvi", "qnehvi"}:
        payload["enabled"] = bool(payload.get("enabled", True))

    payload.setdefault("seed", int(rec_cfg.get("seed", 12345)))
    return BotorchRegionalConfig.from_dict(payload)


def _recommendation_backend_name(rec_cfg: dict[str, Any]) -> str:
    return str(rec_cfg.get("backend", "passive_nearest_observed")).strip().lower()


def _add_nearest_known_scaled_distance(
    *,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    parameter_space: dict[str, Any],
) -> pd.DataFrame:
    """Add nearest-known distance in scaled parameter space.

    This is backend-independent metadata used for novelty filtering and
    reporting. Passive nearest-observed prediction also computes distances
    internally for weights, but AxModelManager does not, so the column must
    exist before backend-specific prediction.
    """

    out = candidates.copy()

    x_hist = scaled_parameter_array(history, parameter_space)
    x_cand = scaled_parameter_array(candidates, parameter_space)

    if len(x_hist) == 0:
        raise ValueError("Cannot compute nearest-known distance with empty history")

    distances = np.sqrt(((x_cand[:, None, :] - x_hist[None, :, :]) ** 2).sum(axis=2))
    out["nearest_known_scaled_dist"] = distances.min(axis=1)

    return out


def _predict_candidates(
    *,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    parameter_space: dict[str, Any],
    rec_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    backend = _recommendation_backend_name(rec_cfg)

    if backend in {
        "passive",
        "passive_nearest",
        "passive_nearest_observed",
    }:
        pred = _predict_from_nearest(
            history,
            candidates,
            parameter_space,
            int(rec_cfg.get("nearest_k", 8)),
        )
        summary = {
            "schema_version": 1,
            "backend": "passive_nearest_observed",
            "surrogate_backend": "passive_nearest_observed",
            "status": "ok",
            "fit_rows": int(len(history)),
            "candidate_rows": int(len(candidates)),
            "nearest_k": int(rec_cfg.get("nearest_k", 8)),
            "note": "Passive nearest-observed baseline; no GP surrogate fitted.",
        }
        return pred, summary

    if backend in {
        "optimas_ax",
        "optimas_ax_model_manager",
        "ax_model_manager",
    }:
        return predict_with_ax_model_manager(
            history=history,
            candidates=candidates,
            parameter_space=parameter_space,
            rec_cfg=rec_cfg,
        )

    raise ValueError(f"Unknown recommendation backend: {backend!r}")


def _is_morbo_like_backend(rec_cfg: dict[str, Any]) -> bool:
    return _recommendation_backend_name(rec_cfg) in {
        "morbo_like",
        "morbo-like",
        "morbo_random",
        "morbo_like_random",
    }


def _morbo_search_space(parameter_space: dict[str, Any]) -> dict[str, Any]:
    ranges = parameter_space["ranges"]
    return {
        "laser_case": Choice(
            list(parameter_space.get("laser_cases", ["f20", "f32", "f40"]))
        ),
        "n0_1e18cm3": FloatRange(*[float(v) for v in ranges["n0_1e18cm3"]]),
        "plateau_mm_num": FloatRange(*[float(v) for v in ranges["plateau_mm_num"]]),
        "diameter_um_num": FloatRange(*[float(v) for v in ranges["diameter_um_num"]]),
        "focus_mm_num": FloatRange(*[float(v) for v in ranges["focus_mm_num"]]),
    }


def _morbo_required_parameter_columns() -> tuple[str, ...]:
    return (
        "laser_case",
        "n0_1e18cm3",
        "plateau_mm_num",
        "diameter_um_num",
        "focus_mm_num",
    )


def _morbo_objective_names(
    *,
    objective_config: dict[str, Any],
    objective_table: pd.DataFrame,
    rec_cfg: dict[str, Any],
) -> tuple[str, ...]:
    configured = rec_cfg.get("objective_names") or rec_cfg.get("objectives")
    if configured:
        names = tuple(str(item) for item in configured)
    else:
        names = tuple(
            str(item) for item in objective_config.get("required_scores_for_fit", [])
        )

    if not names:
        names = tuple(
            score for score in PREDICTED_SCORES if score in objective_table.columns
        )

    missing = [name for name in names if name not in objective_table.columns]
    if missing:
        raise ValueError(
            "MORBO-like backend objective(s) missing from objective_table.csv: "
            + ", ".join(missing)
        )

    if not names:
        raise ValueError("MORBO-like backend requires at least one objective")

    return names


def _morbo_objective_spec(objective_names: tuple[str, ...]) -> ObjectiveSpec:
    return ObjectiveSpec(
        tuple(
            ObjectiveDefinition(name=name, metric=name, sense="max")
            for name in objective_names
        )
    )


def _morbo_trials_from_history(
    history: pd.DataFrame,
    *,
    objective_names: tuple[str, ...],
) -> list[TrialInput]:
    required_params = _morbo_required_parameter_columns()
    trials: list[TrialInput] = []

    for _, row in history.sort_values("observation_id", kind="stable").iterrows():
        params = {column: row[column] for column in required_params}
        raw_metrics = {
            objective_name: row[objective_name]
            for objective_name in objective_names
            if objective_name in row.index
        }
        metadata = {
            "observation_id": str(row.get("observation_id", "")),
            "source_case_id": str(row.get("source_case_id", "")),
            "source_case_name": str(row.get("source_case_name", "")),
        }
        trials.append(
            TrialInput(
                candidate_id=str(row["observation_id"]),
                params=params,
                raw_metrics=raw_metrics,
                simulation_status="finished",
                metadata=metadata,
            )
        )

    return trials


def _morbo_regional_policy(rec_cfg: dict[str, Any]) -> RegionalPolicy:
    payload = rec_cfg.get("regional_policy", {}) or {}
    if not isinstance(payload, dict):
        raise ValueError("recommendation.regional_policy must be an object")
    return RegionalPolicy.from_dict(payload)


def _load_morbo_state_for_iteration(
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
) -> OptimizerState | None:
    current = outputs_dir / "morbo_optimizer_state.json"
    if current.is_file():
        return load_optimizer_state(current)

    for previous_iteration in range(iteration - 1, -1, -1):
        previous = (
            config.iteration_dir(previous_iteration)
            / "outputs"
            / "morbo_optimizer_state.json"
        )
        if previous.is_file():
            return load_optimizer_state(previous)

    return None


def _proposal_region_map_from_state(
    state: OptimizerState | None,
    *,
    backend: MorboLikeBackend,
    trials: list[TrialInput],
) -> dict[str, str]:
    """Map observed trial IDs back to the region that proposed them.

    Candidate IDs in recommended_candidates.tsv are not guaranteed to survive as
    observation IDs after materialization/campaign execution, so attribution is
    done by stable candidate signature instead of by candidate_id.
    """

    if state is None:
        return {}

    pending = (
        state.extra.get("pending_proposals", [])
        if isinstance(state.extra, dict)
        else []
    )
    if not isinstance(pending, list):
        return {}

    region_by_signature: dict[str, str] = {}
    for item in pending:
        if not isinstance(item, dict):
            continue
        signature = str(item.get("candidate_signature", "")).strip()
        region_id = str(item.get("region_id", "")).strip()
        if signature and region_id:
            region_by_signature[signature] = region_id

    out: dict[str, str] = {}
    for trial in trials:
        try:
            signature = backend.space_codec.signature(trial.params)
        except Exception:
            continue
        region_id = region_by_signature.get(signature)
        if region_id:
            out[trial.candidate_id] = region_id

    return out


def _morbo_pending_proposal_records(
    proposals: list[Any],
    *,
    iteration: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, proposal in enumerate(proposals):
        records.append(
            {
                "candidate_id": f"morbo_{iteration:03d}_{idx:03d}",
                "candidate_signature": proposal.candidate_signature,
                "region_id": proposal.region_id,
                "strategy": proposal.strategy,
                "params": dict(proposal.params),
            }
        )
    return records


def _propose_morbo_like_recommendations(
    *,
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
    history: pd.DataFrame,
    objective_table: pd.DataFrame,
    parameter_space: dict[str, Any],
    rec_cfg: dict[str, Any],
) -> Path:
    objective_names = _morbo_objective_names(
        objective_config=config.objective_config(),
        objective_table=objective_table,
        rec_cfg=rec_cfg,
    )
    objective_spec = _morbo_objective_spec(objective_names)
    previous_state = _load_morbo_state_for_iteration(config, iteration, outputs_dir)

    backend = MorboLikeBackend(
        _morbo_search_space(parameter_space),
        objective_spec,
        seed=int(rec_cfg.get("seed", 12345)),
        min_observations=rec_cfg.get("min_observations"),
        regional_policy=_morbo_regional_policy(rec_cfg),
        state=previous_state,
        suggestion_mode=_morbo_suggestion_mode(rec_cfg),
        botorch_config=_morbo_botorch_config(rec_cfg),
    )
    trials = _morbo_trials_from_history(history, objective_names=objective_names)
    sync_result = backend.sync(
        trials,
        proposal_region_map=_proposal_region_map_from_state(
            previous_state,
            backend=backend,
            trials=trials,
        ),
    )

    n = int(rec_cfg.get("n_candidates", 30))
    proposals = backend.suggest_proposals(n)
    backend.register_pending(proposals)

    pending_records = _morbo_pending_proposal_records(proposals, iteration=iteration)
    backend.state = backend.state.with_updates(
        last_strategy=backend.last_strategy,
        extra={
            **dict(backend.state.extra),
            "objective_names": list(objective_names),
            "observed_count": sync_result.observed_count,
            "skipped_count": sync_result.skipped_count,
            "frontier_count": len(sync_result.frontier),
            "region_count": len(sync_result.regions),
            "pending_proposals": pending_records,
        },
    )

    out_path = write_recommended_candidates_tsv(
        outputs_dir / "recommended_candidates.tsv",
        proposals,
        iteration=iteration,
        candidate_id_prefix=f"morbo_{iteration:03d}",
        required_parameter_columns=_morbo_required_parameter_columns(),
    )

    save_morbo_optimizer_state(
        outputs_dir / "morbo_optimizer_state.json", backend.state
    )
    save_frontier_records(
        outputs_dir / "morbo_frontier.json", list(sync_result.frontier)
    )
    save_region_records(outputs_dir / "morbo_regions.json", list(sync_result.regions))

    write_json(
        outputs_dir / "surrogate_summary.json",
        {
            "schema_version": 1,
            "backend": "morbo_like",
            "surrogate_backend": (
                "morbo_like_botorch_qnehvi"
                if backend.last_strategy == "regional_model"
                else "morbo_like_no_botorch"
            ),
            "status": "ok",
            "fit_rows": int(len(history)),
            "candidate_rows": int(len(proposals)),
            "objective_names": list(objective_names),
            "observed_count": sync_result.observed_count,
            "skipped_count": sync_result.skipped_count,
            "frontier_count": len(sync_result.frontier),
            "region_count": len(sync_result.regions),
            "last_strategy": backend.last_strategy,
            "min_observations": backend.min_observations,
            "regional_policy": backend.regional_policy.as_dict(),
            "suggestion_mode": backend.suggestion_mode,
            "botorch_config": backend.botorch_config.as_dict(),
            "model_diagnostics": backend.last_model_diagnostics,
            "note": (
                "MORBO-like regional random backend; BoTorch model mode is not "
                "enabled in this implementation."
            ),
        },
    )

    return out_path


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

    if _is_morbo_like_backend(rec_cfg):
        return _propose_morbo_like_recommendations(
            config=config,
            iteration=iteration,
            outputs_dir=outputs_dir,
            history=history,
            objective_table=obj,
            parameter_space=parameter_space,
            rec_cfg=rec_cfg,
        )

    candidates = build_candidate_cloud(obs, obj, parameter_space, rec_cfg)
    candidates = _add_nearest_known_scaled_distance(
        history=history,
        candidates=candidates,
        parameter_space=parameter_space,
    )

    pred, surrogate_summary = _predict_candidates(
        history=history,
        candidates=candidates,
        parameter_space=parameter_space,
        rec_cfg=rec_cfg,
    )

    write_json(outputs_dir / "surrogate_summary.json", surrogate_summary)

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
    recommended["recommendation_backend"] = surrogate_summary["backend"]
    recommended["surrogate_backend"] = surrogate_summary.get(
        "surrogate_backend",
        surrogate_summary["backend"],
    )
    recommended["ranking_source"] = "score_balanced_conservative"
    recommended["acquisition_value"] = recommended["score_balanced_conservative"]

    first = [
        "recommendation_id",
        "optimizer_iteration",
        "candidate_id",
        "rank",
        "recommendation_status",
        "candidate_source",
        "recommendation_backend",
        "surrogate_backend",
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
