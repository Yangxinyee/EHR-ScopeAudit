"""Per-patient identifier hashing and time shifting for Tier-A (plan §7.2).

Every timestamp of a patient (evidence rows, index time, gold times, dates quoted in questions) moves by the same
whole-day offset, so windows, orderings and scope decisions are unchanged while no real date leaves the enclave.
The offset is derived deterministically from the secret enclave salt and the raw MRN; the salt is never committed,
so neither the patient hash nor the offset can be recomputed from the released manifests."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any

from ..schema import parse_time

_MIN_DAYS, _MAX_DAYS = 30, 400   # |offset| ∈ [30, 400] days, sign from the hash


def hash_patient(mrn: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|patient|{str(mrn).strip()}".encode()).hexdigest()[:16]


def offset_days(mrn: str, salt: str) -> int:
    h = hashlib.sha256(f"{salt}|shift|{str(mrn).strip()}".encode()).digest()
    magnitude = _MIN_DAYS + int.from_bytes(h[:4], "big") % (_MAX_DAYS - _MIN_DAYS + 1)
    return magnitude if h[4] % 2 == 0 else -magnitude


_DT_PATTERNS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%m/%d/%y %H:%M", "%m/%d/%y")
_EXCEL_EPOCH = datetime(1899, 12, 30)


def parse_any(value: Any) -> datetime | None:
    """CCDA exports mix ISO strings, US dates, Excel serial numbers and datetime objects; None/NULL/NaT -> None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    s = str(value).strip()
    if s == "" or s.upper() in ("NULL", "NONE", "NAN", "NAT", "N/A"):
        return None
    if re.fullmatch(r"\d{4,6}(\.\d+)?", s):
        try:
            return _EXCEL_EPOCH + timedelta(days=float(s))
        except ValueError:
            return None
    s2 = s.replace("T", " ").replace("Z", "")
    for fmt in _DT_PATTERNS:
        try:
            return datetime.strptime(s2, fmt)
        except ValueError:
            continue
    try:
        return parse_time(s.replace(" ", "T") + ("Z" if len(s) == 19 else ""))
    except Exception:  # noqa: BLE001
        return None


def shifted_iso(value: Any, days: int) -> str | None:
    """Parse, shift by `days`, return the canonical ISO form used everywhere in ScopeBench (`YYYY-MM-DDTHH:MM:SSZ`)."""
    dt = parse_any(value)
    if dt is None:
        return None
    return (dt + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def shift_dt(dt: datetime, days: int) -> datetime:
    return dt + timedelta(days=days)
