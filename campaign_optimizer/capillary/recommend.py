from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_table, write_json, write_tsv
from campaign_optimizer.morbo import (
    MorboLikeBackend,
    ObjectiveDefinition,
    ObjectiveSpec,
    OptimizerState,
    RegionalPolicy,
    SearchSpaceCodec,
    TrialInput,
    load_optimizer_state,
    save_frontier_records,
    save_optimizer_state as save_morbo_optimizer_state,
    save_region_records,
    write_recommended_candidates_tsv,
    BotorchRegionalConfig,
    CategoricalRegionalPolicy,
)

from .parameters import (
    DISTANCE_COLUMNS,
    active_optimizer_parameter_columns,
    capillary_reference_parameters,
    capillary_search_space,
    capillary_sobol_parameters,
    scaled_parameter_array,
)
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
    if "nitrogen_fraction" in ranges:
        random["nitrogen_fraction"] = _sample_uniform(
            rng, ranges["nitrogen_fraction"], n_random
        )
    if "pulse_duration_factor" in ranges:
        random["pulse_duration_factor"] = _sample_uniform(
            rng, ranges["pulse_duration_factor"], n_random
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
            subset=active_optimizer_parameter_columns(parameter_space)
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
    if "nitrogen_fraction" in ranges:
        sigmas["nitrogen_fraction"] = 0.001
    if "pulse_duration_factor" in ranges:
        sigmas["pulse_duration_factor"] = 0.08

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
        subset=active_optimizer_parameter_columns(parameter_space)
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


def _morbo_categorical_policy(rec_cfg: dict[str, Any]) -> CategoricalRegionalPolicy:
    payload = rec_cfg.get("categorical_policy", {}) or {}
    if not isinstance(payload, dict):
        raise ValueError("recommendation.categorical_policy must be an object")
    return CategoricalRegionalPolicy.from_dict(payload)


def _morbo_botorch_config(rec_cfg: dict[str, Any]) -> BotorchRegionalConfig:
    payload = dict(rec_cfg.get("botorch", {}) or {})
    suggestion_mode = _morbo_suggestion_mode(rec_cfg)

    if suggestion_mode in {"regional_model", "botorch", "qlognehvi", "qnehvi"}:
        payload["enabled"] = bool(payload.get("enabled", True))

    payload.setdefault("seed", int(rec_cfg.get("seed", 12345)))
    return BotorchRegionalConfig.from_dict(payload)


def _recommendation_backend_name(rec_cfg: dict[str, Any]) -> str:
    return str(rec_cfg.get("backend", "morbo_like")).strip().lower()


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
    return capillary_search_space(parameter_space)


def _morbo_required_parameter_columns(
    parameter_space: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(active_optimizer_parameter_columns(parameter_space))


def _add_capillary_derived_candidate_columns(recommended: pd.DataFrame) -> pd.DataFrame:
    """Add capillary-specific derived columns to recommendation tables.

    MORBO-like uses laser_case as the categorical optimization variable.  The
    legacy capillary recommendation contract also exposes f_number and n0_cm3
    as derived/audit columns.  They must not be used as independent optimizer
    variables.
    """

    out = recommended.copy()

    if "laser_case" in out.columns:
        out["f_number"] = out["laser_case"].map(f_number_from_laser_case)

    if "n0_1e18cm3" in out.columns:
        out["n0_cm3"] = (
            pd.to_numeric(
                out["n0_1e18cm3"],
                errors="coerce",
            )
            * 1.0e18
        )

    return out


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
    parameter_space: dict[str, Any],
) -> list[TrialInput]:
    required_params = _morbo_required_parameter_columns(parameter_space)
    trials: list[TrialInput] = []

    for _, row in history.sort_values("observation_id", kind="stable").iterrows():
        params = {column: row[column] for column in required_params}
        raw_metrics = {
            objective_name: row[objective_name]
            for objective_name in objective_names
            if objective_name in row.index
        }
        candidate_signature = row.get("candidate_signature", "")
        if pd.isna(candidate_signature):
            candidate_signature = ""
        metadata = {
            "observation_id": str(row.get("observation_id", "")),
            "source_case_id": str(row.get("source_case_id", "")),
            "source_case_name": str(row.get("source_case_name", "")),
            "candidate_signature": str(candidate_signature).strip(),
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
    done by stable candidate signature instead of by candidate_id.  When the
    persisted observation provenance is unavailable (historical campaigns), the
    legacy reconstruction from trial.params remains the compatibility fallback.
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
        signature = str(trial.metadata.get("candidate_signature", "")).strip()
        if not signature:
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
        categorical_policy=_morbo_categorical_policy(rec_cfg),
    )
    trials = _morbo_trials_from_history(
        history,
        objective_names=objective_names,
        parameter_space=parameter_space,
    )
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
        required_parameter_columns=_morbo_required_parameter_columns(parameter_space),
    )

    recommended = read_table(out_path, sep="\t")
    recommended = _add_capillary_derived_candidate_columns(recommended)
    if _is_sobol_then_morbo_backend(rec_cfg):
        recommended["candidate_source"] = "morbo"
        recommended["recommendation_backend"] = "sobol_then_morbo"

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
        "candidate_signature",
        "region_id",
        "morbo_strategy",
        "laser_case",
        "f_number",
        "n0_1e18cm3",
        "n0_cm3",
        "plateau_mm_num",
        "diameter_um_num",
        "focus_mm_num",
        "nitrogen_fraction",
    ]
    cols = [column for column in first if column in recommended.columns]
    cols += [column for column in recommended.columns if column not in cols]
    out_path = write_tsv(out_path, recommended[cols])

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
            "backend": (
                "sobol_then_morbo"
                if _is_sobol_then_morbo_backend(rec_cfg)
                else "morbo_like"
            ),
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
            "categorical_policy": backend.categorical_policy.as_dict(),
            "note": (
                "MORBO-like regional random backend; BoTorch model mode is not "
                "enabled in this implementation."
            ),
        },
    )

    return out_path


def _is_sobol_then_morbo_backend(rec_cfg: dict[str, Any]) -> bool:
    return _recommendation_backend_name(rec_cfg) in {
        "sobol_then_morbo",
        "sobol-to-morbo",
        "sobol_to_morbo",
    }


def _fit_history(
    observations: pd.DataFrame,
    objectives: pd.DataFrame,
    *,
    parameter_space: dict[str, Any],
) -> pd.DataFrame:
    if observations.empty or objectives.empty or "fit_eligible" not in objectives:
        return observations.head(0).copy()
    fit = objectives[
        objectives["fit_eligible"].astype(str).str.lower() == "true"
    ].copy()
    if fit.empty:
        return observations.head(0).copy()
    history = observations.merge(fit, on="observation_id", how="inner")
    required = active_optimizer_parameter_columns(parameter_space)
    return history.dropna(subset=required).reset_index(drop=True)


def _used_sobol_indices(observations: pd.DataFrame) -> list[int]:
    if "sobol_index" not in observations.columns:
        return []
    values = pd.to_numeric(observations["sobol_index"], errors="coerce")
    return sorted({int(value) for value in values.dropna() if value >= 0})


def _initial_recommendation_row(
    *,
    iteration: int,
    rank: int,
    candidate_id: str,
    source: str,
    signature: str,
    params: dict[str, Any],
    sobol_index: int | str = "",
    reference_id: str = "",
) -> dict[str, Any]:
    return {
        "recommendation_id": f"iter_{iteration:03d}_rank_{rank:03d}",
        "optimizer_iteration": int(iteration),
        "candidate_id": candidate_id,
        "rank": int(rank),
        "recommendation_status": "proposed",
        "candidate_source": source,
        "recommendation_backend": "sobol_then_morbo",
        "surrogate_backend": "none_sobol_design",
        "ranking_source": source,
        "acquisition_value": "",
        "candidate_signature": signature,
        "region_id": "",
        "morbo_strategy": "",
        "sobol_index": sobol_index,
        "reference_id": reference_id,
        **params,
    }


def _write_capillary_sobol_recommendations(
    *,
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
    observations: pd.DataFrame,
    eligible_count: int,
    minimum_observations: int,
) -> Path:
    parameter_space = config.parameter_space()
    rec_cfg = config.recommendation_config()
    design = dict(rec_cfg.get("initial_design", {}) or {})
    seed = int(design.get("seed", rec_cfg.get("seed", 12345)))
    target = int(design.get("sobol_points", 64))
    initial_batch_size = int(design.get("initial_batch_size", 32))
    continuation_size = int(
        design.get("continuation_batch_size", rec_cfg.get("n_candidates", 6))
    )
    include_references = bool(design.get("include_references", True))
    if target <= 0 or initial_batch_size <= 0 or continuation_size <= 0:
        raise ValueError("Sobol point and batch counts must be positive")

    used = _used_sobol_indices(observations)
    next_index = max(used) + 1 if used else 0
    remaining = max(target - len(used), 0)
    if remaining:
        batch_size = initial_batch_size if not used else continuation_size
        count = min(batch_size, remaining)
        phase = "balanced_sobol"
    else:
        count = continuation_size
        phase = "sobol_until_model_ready"

    codec = SearchSpaceCodec(capillary_search_space(parameter_space))
    known_signatures: set[str] = set()
    names = active_optimizer_parameter_columns(parameter_space)
    for _, row in observations.iterrows():
        try:
            known_signatures.add(codec.signature({name: row[name] for name in names}))
        except Exception:
            continue

    rows: list[dict[str, Any]] = []
    rank = 1
    if iteration == 0 and include_references:
        for reference_index, (reference_id, params) in enumerate(
            capillary_reference_parameters(parameter_space)
        ):
            signature = codec.signature(params)
            if signature in known_signatures:
                continue
            rows.append(
                _initial_recommendation_row(
                    iteration=iteration,
                    rank=rank,
                    candidate_id=f"cap_{iteration:03d}_reference_{reference_index:03d}",
                    source="reference",
                    signature=signature,
                    params=params,
                    reference_id=reference_id,
                )
            )
            known_signatures.add(signature)
            rank += 1

    accepted = 0
    candidate_index = next_index
    while accepted < count:
        params = capillary_sobol_parameters(
            parameter_space,
            start_index=candidate_index,
            count=1,
            seed=seed,
        )[0]
        signature = codec.signature(params)
        if signature not in known_signatures:
            rows.append(
                _initial_recommendation_row(
                    iteration=iteration,
                    rank=rank,
                    candidate_id=f"cap_{iteration:03d}_sobol_{candidate_index:06d}",
                    source="sobol",
                    signature=signature,
                    params=params,
                    sobol_index=candidate_index,
                )
            )
            known_signatures.add(signature)
            accepted += 1
            rank += 1
        candidate_index += 1

    recommended = _add_capillary_derived_candidate_columns(pd.DataFrame(rows))
    path = write_tsv(outputs_dir / "recommended_candidates.tsv", recommended)
    write_json(
        outputs_dir / "surrogate_summary.json",
        {
            "schema_version": 1,
            "problem_kind": "capillary",
            "backend": "sobol_then_morbo",
            "status": "initial_design",
            "phase": phase,
            "seed": seed,
            "sobol_target": target,
            "sobol_indices_already_used": used,
            "sobol_indices_proposed": [
                int(row["sobol_index"])
                for row in rows
                if row["candidate_source"] == "sobol"
            ],
            "reference_ids_proposed": [
                row["reference_id"]
                for row in rows
                if row["candidate_source"] == "reference"
            ],
            "eligible_observations": int(eligible_count),
            "minimum_observations": int(minimum_observations),
            "candidate_rows": len(rows),
        },
    )
    return path


def _propose_sobol_then_morbo(
    *,
    config: OptimizerConfig,
    iteration: int,
    outputs_dir: Path,
    observations: pd.DataFrame,
    objectives: pd.DataFrame,
    parameter_space: dict[str, Any],
    rec_cfg: dict[str, Any],
) -> Path:
    history = _fit_history(
        observations,
        objectives,
        parameter_space=parameter_space,
    )
    objective_names = tuple(
        str(name)
        for name in (
            rec_cfg.get("objective_names")
            or config.objective_config().get("required_scores_for_fit", [])
        )
    )
    encoded_dimensions = SearchSpaceCodec(
        capillary_search_space(parameter_space)
    ).encoded_dim()
    minimum_observations = int(
        rec_cfg.get("min_observations")
        or max(4, len(objective_names) + 1, encoded_dimensions + 1)
    )
    design = dict(rec_cfg.get("initial_design", {}) or {})
    target = int(design.get("sobol_points", 64))
    used = _used_sobol_indices(observations)

    if len(used) < target or len(history) < minimum_observations:
        return _write_capillary_sobol_recommendations(
            config=config,
            iteration=iteration,
            outputs_dir=outputs_dir,
            observations=observations,
            eligible_count=len(history),
            minimum_observations=minimum_observations,
        )

    return _propose_morbo_like_recommendations(
        config=config,
        iteration=iteration,
        outputs_dir=outputs_dir,
        history=history,
        objective_table=objectives,
        parameter_space=parameter_space,
        rec_cfg=rec_cfg,
    )


def propose_recommendations(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    outputs_dir = iter_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    obs = read_table(iter_dir / "inputs" / "observations.csv")
    obj = read_table(iter_dir / "inputs" / "objective_table.csv")

    parameter_space = config.parameter_space()
    rec_cfg = config.recommendation_config()

    if _is_sobol_then_morbo_backend(rec_cfg):
        return _propose_sobol_then_morbo(
            config=config,
            iteration=iteration,
            outputs_dir=outputs_dir,
            observations=obs,
            objectives=obj,
            parameter_space=parameter_space,
            rec_cfg=rec_cfg,
        )

    fit = obj[obj["fit_eligible"].astype(str).str.lower() == "true"].copy()

    if fit.empty:
        raise ValueError("No fit-eligible objective rows available for recommendations")

    history = obs.merge(fit, on="observation_id", how="inner")
    required_parameters = active_optimizer_parameter_columns(parameter_space)
    history = history.dropna(subset=required_parameters).reset_index(drop=True)

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