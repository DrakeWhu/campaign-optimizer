from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .backend import CandidateProposal
from .state import BACKEND_NAME


DEFAULT_RECOMMENDATION_STATUS = "recommended_placeholder_no_launch"
DEFAULT_RANKING_SOURCE = "morbo_like_random_rank"


BASE_RECOMMENDED_COLUMNS = [
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
]


def proposals_to_recommended_dataframe(
    proposals: Sequence[CandidateProposal],
    *,
    iteration: int,
    backend: str = BACKEND_NAME,
    candidate_id_prefix: str | None = None,
    recommendation_status: str = DEFAULT_RECOMMENDATION_STATUS,
    ranking_source: str = DEFAULT_RANKING_SOURCE,
    required_parameter_columns: Sequence[str] = (),
    require_unique_signatures: bool = True,
) -> pd.DataFrame:
    """Convert MORBO-like proposals into recommended_candidates rows.

    The output intentionally follows the existing campaign-optimization
    ``recommended_candidates.tsv`` contract, so downstream candidate-batch
    builders can consume the result without knowing about the MORBO internals.

    The random/no-BoTorch backend has no true acquisition value. For contract
    compatibility, ``acquisition_value`` is a deterministic rank-priority score
    where earlier proposals receive larger values.
    """

    if int(iteration) < 0:
        raise ValueError("iteration must be non-negative")

    prefix = candidate_id_prefix or f"morbo_{int(iteration):03d}"
    backend_name = str(backend).strip()
    if not backend_name:
        raise ValueError("backend must be non-empty")

    required = tuple(str(item) for item in required_parameter_columns)
    rows: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    total = len(proposals)

    for idx, proposal in enumerate(proposals):
        if not isinstance(proposal, CandidateProposal):
            raise TypeError("proposals must contain CandidateProposal instances")

        signature = str(proposal.candidate_signature).strip()
        if not signature:
            raise ValueError("candidate_signature must be non-empty")

        if require_unique_signatures:
            if signature in seen_signatures:
                raise ValueError(f"duplicate candidate_signature: {signature}")
            seen_signatures.add(signature)

        missing = [column for column in required if column not in proposal.params]
        if missing:
            raise ValueError(
                "proposal is missing required parameter column(s): "
                + ", ".join(missing)
            )

        rank = idx + 1
        row: dict[str, Any] = {
            "recommendation_id": f"iter_{int(iteration):03d}_rank_{rank:03d}",
            "optimizer_iteration": int(iteration),
            "candidate_id": f"{prefix}_{idx:03d}",
            "rank": rank,
            "recommendation_status": recommendation_status,
            "candidate_source": proposal.strategy,
            "recommendation_backend": backend_name,
            "surrogate_backend": backend_name,
            "ranking_source": ranking_source,
            "acquisition_value": (
                float(proposal.acquisition_value)
                if getattr(proposal, "acquisition_value", None) is not None
                else float(total - idx)
            ),
            "candidate_signature": signature,
            "region_id": proposal.region_id or "",
            "morbo_strategy": proposal.strategy,
        }
        row.update(dict(proposal.params))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[*BASE_RECOMMENDED_COLUMNS, *required])

    ordered = [column for column in BASE_RECOMMENDED_COLUMNS if column in df.columns]
    ordered += [column for column in df.columns if column not in ordered]
    return df[ordered]


def write_recommended_candidates_tsv(
    path: str | Path,
    proposals: Sequence[CandidateProposal],
    *,
    iteration: int,
    backend: str = BACKEND_NAME,
    candidate_id_prefix: str | None = None,
    recommendation_status: str = DEFAULT_RECOMMENDATION_STATUS,
    ranking_source: str = DEFAULT_RANKING_SOURCE,
    required_parameter_columns: Sequence[str] = (),
) -> Path:
    """Write MORBO-like proposals as ``recommended_candidates.tsv``."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = proposals_to_recommended_dataframe(
        proposals,
        iteration=iteration,
        backend=backend,
        candidate_id_prefix=candidate_id_prefix,
        recommendation_status=recommendation_status,
        ranking_source=ranking_source,
        required_parameter_columns=required_parameter_columns,
    )
    df.to_csv(out_path, sep="\t", index=False)
    return out_path
