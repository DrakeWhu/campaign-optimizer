from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from campaign_optimizer.config import OptimizerConfig
from campaign_optimizer.io import read_json_optional, read_table, write_csv, write_json
from campaign_optimizer.state import now_utc

SCORES = [
    "score_guiding_v1",
    "score_beamlike_v1",
    "score_transverse_v1",
    "score_acceptance_v1",
]
PARAMS = ["n0_1e18cm3", "plateau_mm_num", "diameter_um_num", "focus_mm_num"]
TRADEOFFS = [
    ("score_guiding_v1", "score_beamlike_v1"),
    ("score_beamlike_v1", "score_transverse_v1"),
    ("score_guiding_v1", "score_transverse_v1"),
]


class PlotRecord:
    def __init__(
        self, name: str, status: str, path: str = "", reason: str = ""
    ) -> None:
        self.name = name
        self.status = status
        self.path = path
        self.reason = reason

    def as_dict(self) -> dict[str, str]:
        out = {"name": self.name, "status": self.status}
        if self.path:
            out["path"] = self.path
        if self.reason:
            out["reason"] = self.reason
        return out


def _optional_table(path: Path, *, sep: str | None = None) -> pd.DataFrame | None:
    return read_table(path, sep=sep) if path.is_file() else None


def _fit(df: pd.DataFrame) -> pd.DataFrame:
    if "fit_eligible" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["fit_eligible"].astype(str).str.lower().eq("true")].copy()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return (
        pd.to_numeric(df[col], errors="coerce")
        if col in df.columns
        else pd.Series(dtype=float)
    )


def _iter_label(row: pd.Series) -> str:
    for col in [
        "optimizer_iteration",
        "source_campaign_name",
        "source_campaign_root",
        "observation_id",
    ]:
        if col not in row.index or pd.isna(row[col]):
            continue
        text = str(row[col])
        m = re.search(r"iter[_-]?(\d+)", text)
        if m:
            return f"iter_{int(m.group(1)):03d}"
        if col == "source_campaign_name" and text:
            return text
    return "all"


def _iter_number(label: str) -> float:
    m = re.search(r"iter[_-]?(\d+)", str(label))
    return float(int(m.group(1))) if m else float("nan")


def _history(obs: pd.DataFrame | None, obj: pd.DataFrame | None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if obs is not None and "observation_id" in obs.columns:
        out = obs[[c for c in obs.columns if c != "fit_eligible"]].merge(
            obj, on="observation_id", how="right"
        )
    else:
        out = obj.copy()
    out["history_iteration_label"] = out.apply(_iter_label, axis=1)
    out["history_iteration"] = out["history_iteration_label"].map(_iter_number)
    return out


def _stats(values: pd.Series) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "n_finite": int(finite.count()),
        "n_missing": int(values.isna().sum()),
        "mean": float(finite.mean()) if not finite.empty else float("nan"),
        "median": float(finite.median()) if not finite.empty else float("nan"),
        "min": float(finite.min()) if not finite.empty else float("nan"),
        "max": float(finite.max()) if not finite.empty else float("nan"),
    }


def build_convergence_summary(history: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "history_iteration_label",
        "history_iteration",
        "score_name",
        "n_fit_eligible",
        "n_finite",
        "n_missing",
        "mean",
        "median",
        "min",
        "max",
        "best_so_far",
    ]
    fit = _fit(history)
    rows: list[dict[str, Any]] = []
    labels = list(
        dict.fromkeys(
            fit.get("history_iteration_label", pd.Series(dtype=str)).astype(str)
        )
    )
    for score in [s for s in SCORES if s in fit.columns]:
        best = float("nan")
        for label in labels:
            group = fit[fit["history_iteration_label"].astype(str).eq(label)]
            values = _num(group, score)
            st = _stats(values)
            if np.isfinite(st["max"]):
                best = (
                    st["max"]
                    if not np.isfinite(best)
                    else max(float(best), float(st["max"]))
                )
            rows.append(
                {
                    "history_iteration_label": label,
                    "history_iteration": _iter_number(label),
                    "score_name": score,
                    "n_fit_eligible": int(len(group)),
                    **st,
                    "best_so_far": best,
                }
            )
    return pd.DataFrame(rows, columns=cols)


def build_parameter_convergence(history: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "history_iteration_label",
        "history_iteration",
        "parameter",
        "n_fit_eligible",
        "n_finite",
        "n_missing",
        "mean",
        "median",
        "min",
        "max",
        "std",
    ]
    fit = _fit(history)
    rows: list[dict[str, Any]] = []
    labels = list(
        dict.fromkeys(
            fit.get("history_iteration_label", pd.Series(dtype=str)).astype(str)
        )
    )
    for label in labels:
        group = fit[fit["history_iteration_label"].astype(str).eq(label)]
        for param in [p for p in PARAMS if p in group.columns]:
            values = _num(group, param)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "history_iteration_label": label,
                    "history_iteration": _iter_number(label),
                    "parameter": param,
                    "n_fit_eligible": int(len(group)),
                    **_stats(values),
                    "std": float(finite.std(ddof=0))
                    if len(finite) > 1
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def build_acquisition_summary(rec: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["column", "n", "n_finite", "n_missing", "mean", "median", "min", "max"]
    if rec is None or rec.empty:
        return pd.DataFrame(columns=cols)
    keep = [
        c
        for c in rec.columns
        if c
        in {"acquisition_value", "nearest_known_scaled_dist", "score_beamlike_guarded"}
        or c.startswith(("score_balanced_", "pred_", "norm_"))
    ]
    rows = []
    for col in keep:
        values = _num(rec, col)
        rows.append({"column": col, "n": int(len(values)), **_stats(values)})
    return pd.DataFrame(rows, columns=cols)


def build_boundary_saturation_summary(
    rec: pd.DataFrame | None,
    parameter_space: dict[str, Any],
    *,
    tol_fraction: float = 1.0e-6,
) -> pd.DataFrame:
    cols = [
        "parameter",
        "lower",
        "upper",
        "tolerance",
        "n_candidates",
        "n_finite",
        "n_lower_saturated",
        "n_upper_saturated",
        "n_saturated",
        "fraction_lower_saturated",
        "fraction_upper_saturated",
        "fraction_saturated",
    ]
    rows = []
    ranges = parameter_space.get("ranges", {}) or {}
    for param in PARAMS:
        if param not in ranges:
            continue
        lo, hi = [float(v) for v in ranges[param]]
        tol = max(abs(hi - lo) * tol_fraction, 1.0e-12)
        values = (
            _num(rec, param)
            if rec is not None and param in rec.columns
            else pd.Series(dtype=float)
        )
        finite = values[np.isfinite(values)]
        lower = finite <= lo + tol
        upper = finite >= hi - tol
        sat = lower | upper
        denom = int(finite.count())
        frac = lambda mask: float(mask.mean()) if denom else float("nan")
        rows.append(
            {
                "parameter": param,
                "lower": lo,
                "upper": hi,
                "tolerance": tol,
                "n_candidates": int(len(values)),
                "n_finite": denom,
                "n_lower_saturated": int(lower.sum()),
                "n_upper_saturated": int(upper.sum()),
                "n_saturated": int(sat.sum()),
                "fraction_lower_saturated": frac(lower),
                "fraction_upper_saturated": frac(upper),
                "fraction_saturated": frac(sat),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _plot(
    name: str,
    plot_dir: Path,
    manifest: list[PlotRecord],
    reason: str | None,
    draw: Callable[[Any, Any], None],
) -> None:
    if reason:
        manifest.append(PlotRecord(name, "skipped", reason=reason))
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
        draw(fig, ax)
        fig.tight_layout()
        path = plot_dir / name
        fig.savefig(path)
        plt.close(fig)
        manifest.append(PlotRecord(name, "created", path=str(path)))
    except Exception as exc:
        manifest.append(PlotRecord(name, "skipped", reason=str(exc)))


_create_plot = _plot


def _missing(df: pd.DataFrame | None, cols: list[str]) -> str | None:
    if df is None:
        return "required table is missing"
    bad = [c for c in cols if c not in df.columns]
    return "missing column(s): " + ", ".join(bad) if bad else None


def _plot_histograms(
    obj: pd.DataFrame | None, plot_dir: Path, manifest: list[PlotRecord]
) -> None:
    for score in SCORES:
        reason = _missing(obj, [score])
        values = _num(obj, score) if reason is None else pd.Series(dtype=float)
        values = values[np.isfinite(values)]
        if reason is None and values.empty:
            reason = f"no finite values in {score}"
        _create_plot(
            f"observed_{score}.png",
            plot_dir,
            manifest,
            reason,
            lambda fig, ax, v=values, s=score: (
                ax.hist(v.to_numpy(), bins=min(30, max(5, len(v) // 5))),
                ax.set_xlabel(s),
                ax.set_ylabel("count"),
                ax.set_title(f"Observed {s}"),
            ),
        )


def _plot_candidates(
    rec: pd.DataFrame | None,
    ps: dict[str, Any],
    plot_dir: Path,
    manifest: list[PlotRecord],
) -> None:
    ranges = ps.get("ranges", {}) or {}
    reason = _missing(rec, ["rank", *PARAMS]) or (
        "missing parameter ranges" if any(p not in ranges for p in PARAMS) else None
    )

    def by_rank(fig: Any, ax: Any) -> None:
        r = rec.sort_values("rank")  # type: ignore[union-attr]
        x = _num(r, "rank")
        for p in PARAMS:
            lo, hi = [float(v) for v in ranges[p]]
            ax.plot(x, (_num(r, p) - lo) / (hi - lo), marker="o", label=p)
        ax.set_xlabel("rank")
        ax.set_ylabel("normalized parameter value")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Candidate parameters by rank")
        ax.legend(fontsize="small")

    _create_plot(
        "candidate_parameters_by_rank.png", plot_dir, manifest, reason, by_rank
    )

    def parallel(fig: Any, ax: Any) -> None:
        x = np.arange(len(PARAMS))
        for _, row in rec.iterrows():  # type: ignore[union-attr]
            y = []
            for p in PARAMS:
                lo, hi = [float(v) for v in ranges[p]]
                y.append(
                    (float(row[p]) - lo) / (hi - lo) if pd.notna(row[p]) else np.nan
                )
            ax.plot(x, y, alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(PARAMS, rotation=25, ha="right")
        ax.set_ylabel("normalized parameter value")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Candidate parallel coordinates")

    _create_plot(
        "candidate_parallel_coordinates.png", plot_dir, manifest, reason, parallel
    )

    reason_dist = _missing(rec, ["rank", "nearest_known_scaled_dist"])
    _create_plot(
        "candidate_distance_to_history.png",
        plot_dir,
        manifest,
        reason_dist,
        lambda fig, ax: (
            ax.plot(
                _num(rec.sort_values("rank"), "rank"),
                _num(rec.sort_values("rank"), "nearest_known_scaled_dist"),
                marker="o",
            ),  # type: ignore[union-attr]
            ax.set_xlabel("rank"),
            ax.set_ylabel("nearest_known_scaled_dist"),
            ax.set_title("Candidate distance to fit history"),
        ),
    )

    comp = [
        c
        for c in [
            "acquisition_value",
            "score_balanced_conservative",
            "score_balanced_exploratory",
            "score_beamlike_guarded",
        ]
        if rec is not None and c in rec.columns
    ]
    reason_comp = _missing(rec, ["rank", "acquisition_value"])
    if reason_comp is None and len(comp) < 2:
        reason_comp = "fewer than two acquisition/component columns are available"

    def components(fig: Any, ax: Any) -> None:
        r = rec.sort_values("rank")  # type: ignore[union-attr]
        for c in comp:
            ax.plot(_num(r, "rank"), _num(r, c), marker="o", label=c)
        ax.set_xlabel("rank")
        ax.set_ylabel("score/acquisition value")
        ax.set_title("Candidate acquisition components")
        ax.legend(fontsize="small")

    _create_plot(
        "candidate_acquisition_components.png",
        plot_dir,
        manifest,
        reason_comp,
        components,
    )

    sem = [
        c
        for c in (rec.columns if rec is not None else [])
        if c.startswith("pred_") and c.endswith("_sem")
    ]
    reason_sem = _missing(rec, ["rank"])
    if reason_sem is None and not sem:
        reason_sem = "no pred_*_sem columns available"

    def uncertainty(fig: Any, ax: Any) -> None:
        r = rec.sort_values("rank")  # type: ignore[union-attr]
        for c in sem:
            ax.plot(_num(r, "rank"), _num(r, c), marker="o", label=c)
        ax.set_xlabel("rank")
        ax.set_ylabel("surrogate SEM")
        ax.set_title("Surrogate uncertainty by candidate rank")
        ax.legend(fontsize="small")

    _create_plot(
        "surrogate_uncertainty_rank.png", plot_dir, manifest, reason_sem, uncertainty
    )


def _plot_tradeoffs(
    history: pd.DataFrame, plot_dir: Path, manifest: list[PlotRecord]
) -> None:
    fit = _fit(history)
    for xcol, ycol in TRADEOFFS:
        tag = f"{xcol.replace('score_', '').replace('_v1', '')}_vs_{ycol.replace('score_', '').replace('_v1', '')}"
        reason = None
        if fit.empty:
            reason = "no fit-eligible rows available"
        elif xcol not in fit.columns or ycol not in fit.columns:
            reason = f"missing required score column(s): {xcol}, {ycol}"
        elif not (np.isfinite(_num(fit, xcol)) & np.isfinite(_num(fit, ycol))).any():
            reason = "no finite score pairs available"

        def draw(fig: Any, ax: Any, x=xcol, y=ycol) -> None:
            finite = np.isfinite(_num(fit, x)) & np.isfinite(_num(fit, y))
            labels = fit.loc[finite, "history_iteration_label"].astype(str)
            for label in list(dict.fromkeys(labels)):
                mask = finite & fit["history_iteration_label"].astype(str).eq(label)
                ax.scatter(
                    _num(fit, x)[mask], _num(fit, y)[mask], label=label, alpha=0.7
                )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"Objective tradeoff: {x} vs {y}")
            if labels.nunique() > 1:
                ax.legend(fontsize="small")

        _create_plot(f"objective_tradeoff__{tag}.png", plot_dir, manifest, reason, draw)


def _plot_traces(
    conv: pd.DataFrame, plot_dir: Path, manifest: list[PlotRecord]
) -> None:
    for col, filename, ylabel in [
        ("best_so_far", "optimization_trace__best_scores.png", "best-so-far score"),
        ("mean", "optimization_trace__mean_scores.png", "mean fit-eligible score"),
    ]:
        reason = "convergence_summary is empty" if conv.empty else None

        def draw(fig: Any, ax: Any, metric=col, yl=ylabel) -> None:
            labels = list(dict.fromkeys(conv["history_iteration_label"].astype(str)))
            x = np.arange(len(labels))
            for score in list(dict.fromkeys(conv["score_name"].astype(str))):
                sub = conv[conv["score_name"].astype(str).eq(score)]
                y = [
                    float(
                        sub[sub["history_iteration_label"].astype(str).eq(l)][
                            metric
                        ].iloc[0]
                    )
                    if not sub[sub["history_iteration_label"].astype(str).eq(l)].empty
                    else np.nan
                    for l in labels
                ]
                ax.plot(x, y, marker="o", label=score)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_xlabel("history iteration/source")
            ax.set_ylabel(yl)
            ax.set_title(filename.replace(".png", ""))
            ax.legend(fontsize="small")

        _create_plot(filename, plot_dir, manifest, reason, draw)


def _surrogate_diagnostics(
    history: pd.DataFrame,
    obj: pd.DataFrame | None,
    rec: pd.DataFrame | None,
    surrogate: dict[str, Any] | None,
    objective_config: dict[str, Any],
) -> dict[str, Any]:
    required = list(objective_config.get("required_scores_for_fit", []))
    score_status = {}
    for s in SCORES:
        if obj is None or s not in obj.columns:
            score_status[s] = {"available": False}
            continue
        values = _num(obj, s)
        score_status[s] = {
            "available": True,
            "n_total": int(len(values)),
            "n_finite": int(np.isfinite(values).sum()),
            "n_missing": int(values.isna().sum()),
            "required_for_fit": s in required,
        }

    sem_summary = {}
    for c in [
        c
        for c in (rec.columns if rec is not None else [])
        if c.startswith("pred_") and c.endswith("_sem")
    ]:
        finite = _num(rec, c)[np.isfinite(_num(rec, c))]  # type: ignore[arg-type]
        sem_summary[c] = {
            "n_finite": int(finite.count()),
            "mean": float(finite.mean()) if not finite.empty else float("nan"),
            "max": float(finite.max()) if not finite.empty else float("nan"),
        }

    return {
        "schema_version": 1,
        "created_at": now_utc(),
        "surrogate_summary": surrogate or {},
        "cv_available": False,
        "cv_reason": "current passive nearest-observed backend has no explicit cross-validation artifact",
        "n_fit_eligible": int(len(_fit(history))),
        "required_scores_for_fit": required,
        "score_status": score_status,
        "candidate_uncertainty": sem_summary,
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value) if value is not None else False:
        return None
    return value


def _finite_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    values = _num(df, column)
    return values[np.isfinite(values)]


def _max_or_none(df: pd.DataFrame, column: str) -> float | None:
    values = _finite_numeric(df, column)
    if values.empty:
        return None
    return float(values.max())


def _median_or_none(values: pd.Series) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return None
    return float(finite.median())


def _min_or_none(values: pd.Series) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return None
    return float(finite.min())


def _target_history_iteration(iteration: int) -> int | None:
    # Optimizer iteration N normally consumes observations available up to
    # campaign iteration N-1. For N=0 there is no previous campaign iteration.
    return iteration - 1 if iteration > 0 else None


def _build_best_score_signals(
    fit: pd.DataFrame,
    *,
    iteration: int,
    objective_config: dict[str, Any],
) -> dict[str, Any]:
    required = list(objective_config.get("required_scores_for_fit", []))
    score_names = [s for s in SCORES if s in fit.columns]

    best_scores: dict[str, float | None] = {}
    best_score_improvement: dict[str, float | None] = {}
    best_score_improvement_status: dict[str, str] = {}

    target_iter = _target_history_iteration(iteration)
    if target_iter is not None and "history_iteration" in fit.columns:
        new_mask = _num(fit, "history_iteration").eq(float(target_iter))
    else:
        new_mask = pd.Series(False, index=fit.index)

    new_fit = fit[new_mask].copy()
    previous_fit = fit[~new_mask].copy()

    for score in score_names:
        best_all = _max_or_none(fit, score)
        best_new = _max_or_none(new_fit, score)
        best_previous = _max_or_none(previous_fit, score)

        best_scores[score] = best_all

        if best_new is None:
            best_score_improvement[score] = None
            best_score_improvement_status[score] = "no_new_finite_score"
        elif best_previous is None:
            best_score_improvement[score] = None
            best_score_improvement_status[score] = "no_previous_finite_score"
        else:
            best_score_improvement[score] = float(best_new - best_previous)
            best_score_improvement_status[score] = "ok"

    return {
        "required_scores_for_fit": required,
        "score_names": score_names,
        "target_history_iteration": target_iter,
        "n_fit_eligible_total": int(len(fit)),
        "n_new_valid_observations": int(len(new_fit)),
        "best_scores": best_scores,
        "best_score_improvement": best_score_improvement,
        "best_score_improvement_status": best_score_improvement_status,
    }


def _build_improvement_window_signal(
    fit: pd.DataFrame,
    *,
    window: int,
) -> dict[str, Any]:
    if "history_iteration" not in fit.columns:
        return {
            "window": int(window),
            "status": "unknown",
            "reason": "history_iteration column is missing",
            "values": {},
        }

    hist_iter = _num(fit, "history_iteration")
    finite_iters = sorted(
        int(v) for v in hist_iter[np.isfinite(hist_iter)].drop_duplicates().tolist()
    )

    if len(finite_iters) < 2:
        return {
            "window": int(window),
            "status": "skipped",
            "reason": "fewer than two finite optimizer/campaign iterations",
            "values": {},
        }

    latest = finite_iters[-1]
    older = finite_iters[max(0, len(finite_iters) - int(window) - 1) : -1]

    latest_fit = fit[hist_iter.eq(float(latest))]
    older_fit = fit[hist_iter.isin([float(v) for v in older])]

    values: dict[str, float | None] = {}
    statuses: dict[str, str] = {}

    for score in [s for s in SCORES if s in fit.columns]:
        best_latest = _max_or_none(latest_fit, score)
        best_older = _max_or_none(older_fit, score)

        if best_latest is None or best_older is None:
            values[score] = None
            statuses[score] = "missing_finite_score"
        else:
            values[score] = float(best_latest - best_older)
            statuses[score] = "ok"

    return {
        "window": int(window),
        "status": "ok",
        "latest_history_iteration": latest,
        "comparison_history_iterations": older,
        "values": values,
        "statuses": statuses,
    }


def _build_candidate_novelty_signal(rec: pd.DataFrame | None) -> dict[str, Any]:
    if rec is None or rec.empty:
        return {
            "status": "unknown",
            "reason": "recommended_candidates.tsv is missing or empty",
        }

    if "nearest_known_scaled_dist" not in rec.columns:
        return {
            "status": "unknown",
            "reason": "nearest_known_scaled_dist column is missing",
        }

    values = _num(rec, "nearest_known_scaled_dist")
    return {
        "status": "ok",
        "n_candidates": int(len(values)),
        "n_finite": int(np.isfinite(values).sum()),
        "min_nearest_known_scaled_dist": _min_or_none(values),
        "median_nearest_known_scaled_dist": _median_or_none(values),
    }


def _build_boundary_signal(boundary: pd.DataFrame) -> dict[str, Any]:
    if boundary.empty:
        return {
            "status": "unknown",
            "reason": "boundary_saturation_summary is empty",
            "fraction_candidates_near_boundary": None,
            "parameters": {},
        }

    parameters: dict[str, float | None] = {}
    for _, row in boundary.iterrows():
        name = str(row.get("parameter", ""))
        if not name:
            continue
        value = pd.to_numeric(
            pd.Series([row.get("fraction_saturated")]), errors="coerce"
        ).iloc[0]
        parameters[name] = float(value) if np.isfinite(value) else None

    finite = [v for v in parameters.values() if v is not None]
    return {
        "status": "ok" if finite else "unknown",
        "reason": "" if finite else "no finite boundary saturation fractions",
        "fraction_candidates_near_boundary": max(finite) if finite else None,
        "parameters": parameters,
    }


def _build_surrogate_reliability_signal(
    surrogate: dict[str, Any] | None,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    summary = surrogate or {}
    backend = str(summary.get("backend", "unknown"))
    status = str(summary.get("status", "unknown"))

    if not summary:
        return {
            "status": "unknown",
            "reason": "surrogate_summary.json is missing",
            "backend": backend,
        }

    if backend == "passive_nearest_observed":
        return {
            "status": "unknown",
            "reason": "passive nearest-observed backend has no explicit CV/reliability estimate",
            "backend": backend,
            "surrogate_status": status,
        }

    if status == "ok":
        return {
            "status": "ok",
            "reason": "surrogate summary status is ok",
            "backend": backend,
            "surrogate_status": status,
        }

    return {
        "status": "weak",
        "reason": f"surrogate summary status is {status!r}",
        "backend": backend,
        "surrogate_status": status,
    }


def _optimizer_recommendation_from_signals(signals: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []

    if int(signals.get("n_fit_eligible_total", 0)) <= 0:
        return {
            "optimizer_view": "unknown",
            "reasons": ["no fit-eligible observations are available"],
        }

    novelty = signals.get("candidate_novelty", {}) or {}
    boundary = signals.get("boundary_saturation", {}) or {}
    surrogate = signals.get("surrogate_reliability", {}) or {}

    median_dist = novelty.get("median_nearest_known_scaled_dist")
    max_boundary = boundary.get("fraction_candidates_near_boundary")
    improvements = signals.get("best_score_improvement", {}) or {}

    finite_improvements = [
        float(v)
        for v in improvements.values()
        if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    any_positive_improvement = any(v > 0.0 for v in finite_improvements)

    if isinstance(max_boundary, (int, float)) and max_boundary >= 0.85:
        reasons.append(
            "candidate batch is highly saturated near parameter-space boundary"
        )
        return {
            "optimizer_view": "needs_human_review",
            "reasons": reasons,
        }

    if (
        isinstance(median_dist, (int, float))
        and median_dist < 0.025
        and finite_improvements
        and not any_positive_improvement
    ):
        reasons.append(
            "candidate novelty is low and no positive best-score improvement is visible"
        )
        return {
            "optimizer_view": "stop_converged",
            "reasons": reasons,
        }

    if surrogate.get("status") == "weak":
        reasons.append("surrogate reliability is weak")
        return {
            "optimizer_view": "needs_human_review",
            "reasons": reasons,
        }

    reasons.append("no optimizer-side stop condition triggered")
    return {
        "optimizer_view": "continue",
        "reasons": reasons,
    }


def build_stopping_signals_document(
    *,
    config: OptimizerConfig,
    iteration: int,
    history: pd.DataFrame,
    rec: pd.DataFrame | None,
    boundary: pd.DataFrame,
    surrogate: dict[str, Any] | None,
) -> dict[str, Any]:
    objective_config = config.objective_config()
    fit = _fit(history)

    best = _build_best_score_signals(
        fit,
        iteration=iteration,
        objective_config=objective_config,
    )

    signals: dict[str, Any] = {
        **best,
        "best_score_improvement_window": _build_improvement_window_signal(
            fit,
            window=3,
        ),
        "candidate_novelty": _build_candidate_novelty_signal(rec),
        "boundary_saturation": _build_boundary_signal(boundary),
        "surrogate_reliability": _build_surrogate_reliability_signal(
            surrogate,
            diagnostics={},
        ),
    }

    return _json_clean(
        {
            "schema_version": 1,
            "created_at": now_utc(),
            "iteration": int(iteration),
            "objective_config_id": objective_config.get("config_id"),
            "signals": signals,
            "recommendation": _optimizer_recommendation_from_signals(signals),
            "non_goals": [
                "does not submit jobs",
                "does not call sbatch/srun/mpiexec/mpirun",
                "does not launch WarpX",
                "does not run analysis adapters",
                "does not read HDF5/openPMD",
                "does not delete data",
                "does not mutate campaign workflow state",
            ],
        }
    )


def build_stopping_signals(config: OptimizerConfig, iteration: int) -> Path:
    iter_dir = config.iteration_dir(iteration)
    reports_dir = iter_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    obs = _optional_table(iter_dir / "inputs" / "observations.csv")
    obj = _optional_table(iter_dir / "inputs" / "objective_table.csv")
    rec = _optional_table(iter_dir / "outputs" / "recommended_candidates.tsv", sep="\t")
    surrogate = read_json_optional(iter_dir / "outputs" / "surrogate_summary.json")

    hist = _history(obs, obj)
    boundary = build_boundary_saturation_summary(rec, config.parameter_space())

    document = build_stopping_signals_document(
        config=config,
        iteration=iteration,
        history=hist,
        rec=rec,
        boundary=boundary,
        surrogate=surrogate,
    )

    return write_json(reports_dir / "stopping_signals.json", document)


def build_report(config: OptimizerConfig, iteration: int) -> dict[str, Path]:
    iter_dir = config.iteration_dir(iteration)
    plot_dir = iter_dir / "plots"
    reports_dir = iter_dir / "reports"
    plot_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    obs = _optional_table(iter_dir / "inputs" / "observations.csv")
    obj = _optional_table(iter_dir / "inputs" / "objective_table.csv")
    rec = _optional_table(iter_dir / "outputs" / "recommended_candidates.tsv", sep="\t")
    surrogate = read_json_optional(iter_dir / "outputs" / "surrogate_summary.json")
    hist = _history(obs, obj)

    conv = build_convergence_summary(hist)
    param_conv = build_parameter_convergence(hist)
    acq = build_acquisition_summary(rec)
    boundary = build_boundary_saturation_summary(rec, config.parameter_space())

    stopping_signals = build_stopping_signals_document(
        config=config,
        iteration=iteration,
        history=hist,
        rec=rec,
        boundary=boundary,
        surrogate=surrogate,
    )

    paths = {
        "convergence_summary_csv": write_csv(
            reports_dir / "convergence_summary.csv", conv
        ),
        "convergence_summary_json": write_json(
            reports_dir / "convergence_summary.json",
            {
                "schema_version": 1,
                "rows": json.loads(conv.to_json(orient="records"))
                if not conv.empty
                else [],
            },
        ),
        "parameter_convergence_csv": write_csv(
            reports_dir / "parameter_convergence.csv", param_conv
        ),
        "acquisition_summary_csv": write_csv(
            reports_dir / "acquisition_summary.csv", acq
        ),
        "boundary_saturation_summary_csv": write_csv(
            reports_dir / "boundary_saturation_summary.csv", boundary
        ),
        "surrogate_diagnostics_json": write_json(
            reports_dir / "surrogate_diagnostics.json",
            _surrogate_diagnostics(
                hist, obj, rec, surrogate, config.objective_config()
            ),
        ),
        "stopping_signals_json": write_json(
            reports_dir / "stopping_signals.json",
            stopping_signals,
        ),
    }

    manifest: list[PlotRecord] = []
    _plot_histograms(obj, plot_dir, manifest)
    _plot_candidates(rec, config.parameter_space(), plot_dir, manifest)
    _plot_tradeoffs(hist, plot_dir, manifest)
    _plot_traces(conv, plot_dir, manifest)

    paths["plot_manifest"] = write_json(
        reports_dir / "plot_manifest.json",
        {
            "schema_version": 1,
            "created_at": now_utc(),
            "iteration": int(iteration),
            "iteration_dir": str(iter_dir),
            "plot_dir": str(plot_dir),
            "reports_dir": str(reports_dir),
            "plots": [r.as_dict() for r in manifest],
            "counts": {
                "created": sum(r.status == "created" for r in manifest),
                "skipped": sum(r.status == "skipped" for r in manifest),
            },
        },
    )
    return paths
