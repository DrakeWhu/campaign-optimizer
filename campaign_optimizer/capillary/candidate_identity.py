from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from campaign_optimizer.morbo.search_space import SearchSpaceCodec

from .parameters import active_optimizer_parameter_columns, capillary_search_space


NITROGEN_PARAMETER = "nitrogen_fraction"


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_persisted_candidate_signature(
    candidate_signature: Any,
    *,
    parameter_space: dict[str, Any],
) -> str:
    """Normalize a persisted capillary signature into the active search space.

    New signatures that already contain the complete active parameter keyset are
    preserved byte-for-byte. This is required so that a float round-trip through
    TSV/CSV cannot break the candidate identity fixed by PR #2.

    When the active space adds ``nitrogen_fraction`` and the persisted signature
    has exactly the otherwise-complete legacy keyset, the historical H-only point
    is migrated deterministically by adding ``nitrogen_fraction=0`` and signing it
    with the current codec.

    Empty signatures remain empty so callers can use the historical recomputation
    fallback. Any other non-empty keyset fails closed in a nitrogen-enabled space:
    accepting it would allow an identity that cannot block an equivalent current
    candidate.
    """

    text = _optional_text(candidate_signature)
    if not text:
        return ""

    parameter_names = tuple(active_optimizer_parameter_columns(parameter_space))
    if NITROGEN_PARAMETER not in parameter_names:
        return text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "cannot normalize non-JSON persisted candidate signature in "
            "nitrogen-enabled capillary search space"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "persisted candidate signature must be a JSON object in "
            "nitrogen-enabled capillary search space"
        )

    expected_keys = set(parameter_names)
    observed_keys = set(payload)
    if observed_keys == expected_keys:
        return text

    legacy_keys = expected_keys - {NITROGEN_PARAMETER}
    if observed_keys != legacy_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            "persisted candidate signature keyset is incompatible with the "
            "nitrogen-enabled capillary search space: "
            f"missing={missing}, extra={extra}"
        )

    migrated = dict(payload)
    migrated[NITROGEN_PARAMETER] = 0.0
    codec = SearchSpaceCodec(capillary_search_space(parameter_space))
    return codec.signature({name: migrated[name] for name in parameter_names})


def normalize_observation_candidate_signatures(
    observations_path: str | Path,
    *,
    parameter_space: dict[str, Any],
) -> Path:
    """Rewrite only candidate identity fields to the active 6D signature.

    Source ``cases.tsv`` files remain untouched. The optimizer-facing observation
    table is the identity boundary: historical persisted 5D signatures become the
    canonical 6D H-only identity while current 6D persisted signatures remain
    byte-for-byte unchanged. CSV text for every other field is preserved rather
    than round-tripped through a numeric dataframe parser.
    """

    path = Path(observations_path)
    parameter_names = tuple(active_optimizer_parameter_columns(parameter_space))
    if NITROGEN_PARAMETER not in parameter_names:
        return path

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not rows or "candidate_signature" not in fieldnames:
        return path

    changed = False
    for row in rows:
        original = _optional_text(row.get("candidate_signature", ""))
        normalized = normalize_persisted_candidate_signature(
            original,
            parameter_space=parameter_space,
        )
        if normalized != original:
            row["candidate_signature"] = normalized
            changed = True

    if not changed:
        return path

    tmp = path.with_suffix(path.suffix + ".candidate_identity.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path
