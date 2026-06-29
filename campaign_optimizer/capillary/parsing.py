from __future__ import annotations

import math
import re
from typing import Any


LASER_CASE_TO_F_NUMBER = {
    "f20": 20.0,
    "f32": 32.0,
    "f40": 40.0,
}


def parse_float(value: Any) -> float:
    if value is None:
        return float("nan")

    if isinstance(value, float) and math.isnan(value):
        return float("nan")

    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return float("nan")

    for suffix in ("cm^-3", "cm-3", "cm3", "mm", "um"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()

    return float(text)


def normalize_laser_case(value: Any, *, allowed: list[str] | None = None) -> str:
    text = str(value).strip().lower()

    if not text:
        raise ValueError("empty LASER_CASE")

    if not text.startswith("f"):
        text = f"f{text}"

    allowed_values = allowed or list(LASER_CASE_TO_F_NUMBER)
    if text not in allowed_values:
        raise ValueError(f"LASER_CASE {text!r} not in allowed set {allowed_values!r}")

    return text


def f_number_from_laser_case(laser_case: str) -> float:
    laser = normalize_laser_case(laser_case)

    if laser not in LASER_CASE_TO_F_NUMBER:
        raise ValueError(f"No f-number mapping for laser case {laser!r}")

    return LASER_CASE_TO_F_NUMBER[laser]


def parse_case_name_parameters(case_name: str) -> dict[str, Any]:
    """Fallback parser for legacy names.

    Explicit cases.tsv columns are the canonical source. This parser exists only
    for old products that do not carry parameter columns.
    """
    s = str(case_name)

    m_f = re.search(r"(?:^|_)f(?P<fnum>\d+)(?:_|$)", s)
    m_n = re.search(r"_n(?P<mant>\d+(?:p\d+)?)e(?P<exp>\d+)cm3_", s)
    m_L = re.search(r"_L(?P<L>\d+(?:p\d+)?)mm_", s)
    m_d = re.search(r"_d(?P<d>\d+(?:p\d+)?)um_", s)
    m_focus = re.search(r"_foc(?P<focus>m?\d+(?:p\d+)?|p\d+(?:p\d+)?|0)mm_", s)

    if not all([m_f, m_n, m_L, m_d, m_focus]):
        return {"fallback_parse_status": "failed"}

    def pnum(text: str) -> float:
        text = text.replace("p", ".")
        if text.startswith("m"):
            return -float(text[1:])
        return float(text)

    fnum = float(m_f.group("fnum"))
    laser_case = f"f{int(fnum)}"

    return {
        "fallback_parse_status": "ok",
        "laser_case": laser_case,
        "f_number": fnum,
        "n0_1e18cm3": pnum(m_n.group("mant")) * 10.0 ** (int(m_n.group("exp")) - 18),
        "plateau_mm_num": pnum(m_L.group("L")),
        "diameter_um_num": pnum(m_d.group("d")),
        "focus_mm_num": pnum(m_focus.group("focus")),
    }
