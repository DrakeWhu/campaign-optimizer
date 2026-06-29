from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_FORBIDDEN_SUFFIXES = {".h5", ".hdf5"}
RAW_FORBIDDEN_PARTS = {"diags", "fields", "plasma_electrons", "ionized_electrons"}


def assert_not_raw_diagnostic_path(path: str | Path) -> None:
    p = Path(path)
    suffix = p.suffix.lower()
    parts = {part.lower() for part in p.parts}

    if suffix in RAW_FORBIDDEN_SUFFIXES:
        raise ValueError(f"Refusing to read raw diagnostic file: {p}")

    if "diags" in parts and parts.intersection(RAW_FORBIDDEN_PARTS - {"diags"}):
        raise ValueError(f"Refusing to read raw diagnostic directory path: {p}")


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json_optional(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None

    with p.open("r", encoding="utf-8-sig") as f_in:
        return json.load(f_in)


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, sort_keys=True)
        f_out.write("\n")

    tmp.replace(p)
    return p


def read_table(path: str | Path, *, sep: str | None = None) -> pd.DataFrame:
    p = Path(path)
    assert_not_raw_diagnostic_path(p)

    if not p.is_file():
        raise FileNotFoundError(p)

    if sep is not None:
        return pd.read_csv(p, sep=sep, encoding="utf-8-sig")

    if p.suffix.lower() == ".tsv":
        df = pd.read_csv(p, sep="\t", encoding="utf-8-sig")
        if len(df.columns) == 1:
            df = pd.read_csv(p, sep=r"\s+", encoding="utf-8-sig", engine="python")
        return df

    return pd.read_csv(p, encoding="utf-8-sig")


def write_csv(path: str | Path, df: pd.DataFrame) -> Path:
    p = ensure_parent(path)
    df.to_csv(p, index=False)
    return p


def write_tsv(path: str | Path, df: pd.DataFrame) -> Path:
    p = ensure_parent(path)
    df.to_csv(p, sep="\t", index=False)
    return p


def read_first_row_csv(path: str | Path) -> tuple[dict[str, Any], str, str]:
    p = Path(path)
    assert_not_raw_diagnostic_path(p)

    if not p.is_file():
        return {}, "missing_reduced_output", f"missing file: {p}"

    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception as exc:
        return {}, "analysis_failed", f"csv read failed: {exc}"

    if df.empty:
        return {}, "missing_metric", f"empty csv: {p}"

    return dict(df.iloc[0]), "ok", ""
