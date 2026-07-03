from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

from .regions import ACTIVE, RegionRecord, region_bounds
from .search_space import SearchSpaceCodec


@dataclass(frozen=True)
class BotorchRegionalConfig:
    enabled: bool = False
    max_train_rows: int = 256
    candidate_pool_size_per_region: int = 96
    mc_samples: int = 32
    fit_maxiter: int = 50
    ref_point_quantile: float = 0.05
    ref_point_margin: float = 0.05
    prune_baseline: bool = True
    fallback_to_random: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if int(self.max_train_rows) < 2:
            raise ValueError("max_train_rows must be >= 2")
        if int(self.candidate_pool_size_per_region) < 1:
            raise ValueError("candidate_pool_size_per_region must be >= 1")
        if int(self.mc_samples) < 1:
            raise ValueError("mc_samples must be >= 1")
        if int(self.fit_maxiter) < 1:
            raise ValueError("fit_maxiter must be >= 1")
        if not (0.0 <= float(self.ref_point_quantile) <= 0.5):
            raise ValueError("ref_point_quantile must be in [0, 0.5]")
        if float(self.ref_point_margin) < 0.0:
            raise ValueError("ref_point_margin must be >= 0")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed must be >= 0 when provided")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "BotorchRegionalConfig":
        data = dict(payload or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_train_rows=int(data.get("max_train_rows", 256)),
            candidate_pool_size_per_region=int(
                data.get("candidate_pool_size_per_region", 96)
            ),
            mc_samples=int(data.get("mc_samples", 32)),
            fit_maxiter=int(data.get("fit_maxiter", 50)),
            ref_point_quantile=float(data.get("ref_point_quantile", 0.05)),
            ref_point_margin=float(data.get("ref_point_margin", 0.05)),
            prune_baseline=bool(data.get("prune_baseline", True)),
            fallback_to_random=bool(data.get("fallback_to_random", True)),
            seed=None if data.get("seed") is None else int(data["seed"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_train_rows": self.max_train_rows,
            "candidate_pool_size_per_region": self.candidate_pool_size_per_region,
            "mc_samples": self.mc_samples,
            "fit_maxiter": self.fit_maxiter,
            "ref_point_quantile": self.ref_point_quantile,
            "ref_point_margin": self.ref_point_margin,
            "prune_baseline": self.prune_baseline,
            "fallback_to_random": self.fallback_to_random,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class BotorchModelCandidate:
    params: dict[str, Any]
    candidate_signature: str
    region_id: str
    acquisition_value: float


@dataclass(frozen=True)
class BotorchRegionalResult:
    status: str
    candidates: tuple[BotorchModelCandidate, ...] = ()
    reason: str | None = None
    ref_point_raw: dict[str, float] | None = None
    ref_point_model_units: dict[str, float] | None = None
    y_transform: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None


def suggest_botorch_regional_candidates(
    *,
    encoded_X: Sequence[Sequence[float]],
    canonical_Y: Sequence[Sequence[float]],
    objective_names: Sequence[str],
    codec: SearchSpaceCodec,
    active_regions: Sequence[RegionRecord],
    rng: random.Random,
    n: int,
    blocked_signatures: set[str],
    config: BotorchRegionalConfig,
) -> BotorchRegionalResult:
    """Fit a BoTorch qNEHVI/qLogNEHVI model and rank regional candidate pools.

    This function is intentionally self-contained and uses lazy imports so that
    the repository can still run unit tests without torch/botorch installed.
    """

    if n <= 0:
        return BotorchRegionalResult(status="empty_request", candidates=())

    if not config.enabled:
        return BotorchRegionalResult(
            status="disabled",
            reason="BoTorch regional model mode is disabled",
            diagnostics={"config": config.as_dict()},
        )

    active = tuple(region for region in active_regions if region.status == ACTIVE)
    if not active:
        return BotorchRegionalResult(
            status="no_active_regions",
            reason="No active regions available for regional model suggestions",
            diagnostics={"config": config.as_dict()},
        )

    try:
        imports = _lazy_botorch_imports()
    except Exception as exc:
        return BotorchRegionalResult(
            status="unavailable",
            reason=f"BoTorch imports failed: {exc}",
            diagnostics={"config": config.as_dict()},
        )

    try:
        return _suggest_with_botorch(
            imports=imports,
            encoded_X=encoded_X,
            canonical_Y=canonical_Y,
            objective_names=tuple(str(item) for item in objective_names),
            codec=codec,
            active_regions=active,
            rng=rng,
            n=n,
            blocked_signatures=set(blocked_signatures),
            config=config,
        )
    except Exception as exc:
        return BotorchRegionalResult(
            status="failed",
            reason=f"BoTorch regional model failed: {exc}",
            diagnostics={"config": config.as_dict()},
        )


def _lazy_botorch_imports() -> dict[str, Any]:
    import torch

    try:
        from botorch import fit_gpytorch_mll
    except Exception:
        from botorch.fit import fit_gpytorch_mll

    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )

        acqf_class = qLogNoisyExpectedHypervolumeImprovement
        acqf_name = "qLogNoisyExpectedHypervolumeImprovement"
    except Exception:
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement,
        )

        acqf_class = qNoisyExpectedHypervolumeImprovement
        acqf_name = "qNoisyExpectedHypervolumeImprovement"

    try:
        from botorch.models.gp_regression import SingleTaskGP
    except Exception:
        from botorch.models import SingleTaskGP

    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize

    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except Exception:
        from botorch.sampling import SobolQMCNormalSampler

    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    return {
        "torch": torch,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "acqf_class": acqf_class,
        "acqf_name": acqf_name,
        "SingleTaskGP": SingleTaskGP,
        "ModelListGP": ModelListGP,
        "Standardize": Standardize,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "SumMarginalLogLikelihood": SumMarginalLogLikelihood,
    }


def _suggest_with_botorch(
    *,
    imports: Mapping[str, Any],
    encoded_X: Sequence[Sequence[float]],
    canonical_Y: Sequence[Sequence[float]],
    objective_names: tuple[str, ...],
    codec: SearchSpaceCodec,
    active_regions: Sequence[RegionRecord],
    rng: random.Random,
    n: int,
    blocked_signatures: set[str],
    config: BotorchRegionalConfig,
) -> BotorchRegionalResult:
    torch = imports["torch"]

    X_rows = _finite_matrix(encoded_X, "encoded_X")
    Y_rows = _finite_matrix(canonical_Y, "canonical_Y")

    if len(X_rows) != len(Y_rows):
        raise ValueError("encoded_X and canonical_Y must have the same number of rows")
    if len(X_rows) < 2:
        raise ValueError("at least two training rows are required for BoTorch")
    if not objective_names:
        raise ValueError("objective_names must be non-empty")

    y_dim = len(objective_names)
    for row in Y_rows:
        if len(row) != y_dim:
            raise ValueError("canonical_Y width must match objective_names")

    X_rows, Y_rows = _limit_training_rows(X_rows, Y_rows, config.max_train_rows)

    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    train_X = torch.tensor(X_rows, **tkwargs)
    train_Y = torch.tensor(Y_rows, **tkwargs)

    model, mll = _initialize_model(imports, train_X, train_Y)
    _fit_model(imports, mll, config.fit_maxiter)

    ref_raw = _compute_reference_point_raw(
        Y_rows,
        objective_names,
        quantile=config.ref_point_quantile,
        margin=config.ref_point_margin,
    )
    y_transform = _compute_y_transform(Y_rows, objective_names)
    ref_model = _to_model_units(ref_raw, y_transform)

    pool = _sample_regional_pool(
        codec=codec,
        regions=active_regions,
        rng=rng,
        blocked_signatures=blocked_signatures,
        pool_size_per_region=config.candidate_pool_size_per_region,
    )
    if not pool:
        return BotorchRegionalResult(
            status="no_candidate_pool",
            reason="No unique regional candidate pool could be sampled",
            diagnostics={"config": config.as_dict()},
            ref_point_raw=ref_raw,
            ref_point_model_units=ref_model,
            y_transform=y_transform,
        )

    X_pool = torch.tensor([item["encoded"] for item in pool], **tkwargs)

    sampler = imports["SobolQMCNormalSampler"](
        sample_shape=torch.Size([config.mc_samples])
    )
    acqf_class = imports["acqf_class"]
    acqf = acqf_class(
        model=model,
        ref_point=[ref_raw[name] for name in objective_names],
        X_baseline=train_X,
        prune_baseline=config.prune_baseline,
        sampler=sampler,
    )

    with torch.no_grad():
        values_tensor = acqf(X_pool.unsqueeze(1)).detach().cpu()

    values = [float(item) for item in values_tensor.reshape(-1).tolist()]
    ranked = sorted(
        range(len(pool)),
        key=lambda idx: values[idx] if math.isfinite(values[idx]) else float("-inf"),
        reverse=True,
    )

    selected: list[BotorchModelCandidate] = []
    selected_signatures: set[str] = set()

    for idx in ranked:
        value = values[idx]
        if not math.isfinite(value):
            continue

        item = pool[idx]
        signature = str(item["candidate_signature"])
        if signature in selected_signatures:
            continue

        selected.append(
            BotorchModelCandidate(
                params=dict(item["params"]),
                candidate_signature=signature,
                region_id=str(item["region_id"]),
                acquisition_value=float(value),
            )
        )
        selected_signatures.add(signature)

        if len(selected) >= n:
            break

    status = "ok" if selected else "no_finite_acquisition_values"
    reason = None if selected else "No finite acquisition values were produced"

    return BotorchRegionalResult(
        status=status,
        candidates=tuple(selected),
        reason=reason,
        ref_point_raw=ref_raw,
        ref_point_model_units=ref_model,
        y_transform=y_transform,
        diagnostics={
            "config": config.as_dict(),
            "acquisition_function": imports["acqf_name"],
            "train_rows": len(X_rows),
            "encoded_dim": len(X_rows[0]),
            "objective_names": list(objective_names),
            "candidate_pool_rows": len(pool),
            "selected_rows": len(selected),
        },
    )


def _initialize_model(
    imports: Mapping[str, Any], train_X: Any, train_Y: Any
) -> tuple[Any, Any]:
    SingleTaskGP = imports["SingleTaskGP"]
    ModelListGP = imports["ModelListGP"]
    Standardize = imports["Standardize"]
    SumMarginalLogLikelihood = imports["SumMarginalLogLikelihood"]

    models = []
    for idx in range(train_Y.shape[-1]):
        train_y = train_Y[..., idx : idx + 1]
        models.append(
            SingleTaskGP(train_X, train_y, outcome_transform=Standardize(m=1))
        )

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def _fit_model(imports: Mapping[str, Any], mll: Any, fit_maxiter: int) -> None:
    fit_gpytorch_mll = imports["fit_gpytorch_mll"]

    try:
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": int(fit_maxiter)}},
        )
    except TypeError:
        fit_gpytorch_mll(mll)


def _sample_regional_pool(
    *,
    codec: SearchSpaceCodec,
    regions: Sequence[RegionRecord],
    rng: random.Random,
    blocked_signatures: set[str],
    pool_size_per_region: int,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen = set(blocked_signatures)

    for region in sorted(regions, key=lambda item: item.region_id):
        bounds = region_bounds(region)
        accepted = 0
        attempts = 0
        max_attempts = max(50, int(pool_size_per_region) * 20)

        while accepted < int(pool_size_per_region) and attempts < max_attempts:
            attempts += 1
            params = codec.project_params(codec.sample_within_bounds(bounds, rng))
            signature = codec.signature(params)
            if signature in seen:
                continue

            seen.add(signature)
            accepted += 1
            pool.append(
                {
                    "params": params,
                    "encoded": codec.encode(params),
                    "candidate_signature": signature,
                    "region_id": region.region_id,
                }
            )

    return pool


def _finite_matrix(values: Sequence[Sequence[float]], label: str) -> list[list[float]]:
    rows: list[list[float]] = []

    for row_idx, row in enumerate(values):
        out_row = []
        for col_idx, value in enumerate(row):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{label}[{row_idx}][{col_idx}] is not finite")
            out_row.append(numeric)
        rows.append(out_row)

    if not rows:
        raise ValueError(f"{label} must be non-empty")

    width = len(rows[0])
    if width == 0:
        raise ValueError(f"{label} rows must be non-empty")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"{label} rows must have constant width")

    return rows


def _limit_training_rows(
    X_rows: list[list[float]],
    Y_rows: list[list[float]],
    max_rows: int,
) -> tuple[list[list[float]], list[list[float]]]:
    if len(X_rows) <= int(max_rows):
        return X_rows, Y_rows
    return X_rows[-int(max_rows) :], Y_rows[-int(max_rows) :]


def _compute_reference_point_raw(
    Y_rows: Sequence[Sequence[float]],
    objective_names: Sequence[str],
    *,
    quantile: float,
    margin: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    width = len(objective_names)

    for col in range(width):
        values = sorted(float(row[col]) for row in Y_rows)
        q_idx = min(
            max(int(round(float(quantile) * (len(values) - 1))), 0),
            len(values) - 1,
        )
        lo = values[q_idx]
        hi = values[-1]
        span = max(abs(hi - lo), abs(hi) * 1.0e-12, 1.0e-12)
        out[str(objective_names[col])] = float(lo - float(margin) * span)

    return out


def _compute_y_transform(
    Y_rows: Sequence[Sequence[float]],
    objective_names: Sequence[str],
) -> dict[str, Any]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for col, name in enumerate(objective_names):
        values = [float(row[col]) for row in Y_rows]
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        else:
            var = 0.0
        std = math.sqrt(max(var, 0.0))
        if std < 1.0e-12:
            std = 1.0
        means[str(name)] = float(mean)
        stds[str(name)] = float(std)

    return {
        "type": "botorch_outcome_standardize",
        "objective_names": [str(item) for item in objective_names],
        "mean": means,
        "std": stds,
        "note": (
            "qNEHVI/qLogNEHVI is evaluated with raw canonical objective units; "
            "model-unit reference points are stored for audit only."
        ),
    }


def _to_model_units(
    ref_point_raw: Mapping[str, float],
    y_transform: Mapping[str, Any],
) -> dict[str, float]:
    means = dict(y_transform.get("mean", {}))
    stds = dict(y_transform.get("std", {}))

    out: dict[str, float] = {}
    for name, value in ref_point_raw.items():
        mean = float(means[name])
        std = float(stds[name])
        out[str(name)] = float((float(value) - mean) / std)

    return out
