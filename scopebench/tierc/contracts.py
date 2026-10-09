# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Contract derivation for FHIR-AgentBench (plan §6.1), frozen in the Tier-C protocol.

Every FHIR-AgentBench question is anchored on a patient and, usually, on an explicit "current time"
(the `assumption` column). Its time filters are template placeholders whose natural-language form
(`val_dict.time_placeholder.*.nlq`) is one of a small closed set. Mapping:

  no global filter                          -> scope full, index = now
  "since <year>|<N> month(s)|... ago"       -> scope pre, index = now, window = now - start
  "this year|this month"                    -> scope pre, index = now, window = now - period start
  "last year|last month|yesterday|in MM/…"  -> scope pre, index = period END, window = period length
  "on the first|last|current hospital visit"-> scope post, index = visit start, window = visit length
  "on the first|last|current ICU visit"     -> same with the ICU encounter
Two global filters (change-between-two-values templates) take the union interval.
Basis: the benchmark's native basis is occurrence (effectiveDateTime / authoredOn / performed /
period.start / collectedDateTime); availability (`Observation.issued`) is the wrong-basis control.
Permitted sources: the FHIR resource types implied by the SQL main table. R_t = true_fhir_ids.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..schema import Contract, EvidenceObject

MAIN_TABLE_SOURCES = {
    "chartevents": ("Observation",), "labevents": ("Observation",), "outputevents": ("Observation",), "microbiologyevents": ("Observation", "Specimen"),
    "prescriptions": ("MedicationRequest", "Medication"), "inputevents": ("MedicationAdministration", "Medication"), "procedures_icd": ("Procedure",),
    "diagnoses_icd": ("Condition",), "admissions": ("Encounter",), "icustays": ("Encounter",), "patients": ("Patient",), "transfers": ("Encounter", "Location"),
}
REFERENCE_TYPES = {"Medication", "Patient", "Location", "Organization", "Practitioner"}
MONTHS = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6, "07": 7, "08": 8, "09": 9, "10": 10, "11": 11, "12": 12}


def parse_now(assumption: str | None) -> datetime | None:
    m = re.search(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", assumption or "")
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if m else None


def time_placeholders(val_dict: str) -> dict[str, str]:
    try:
        v = ast.literal_eval(val_dict) if val_dict else {}
    except (ValueError, SyntaxError):
        return {}
    out = {}
    for k, val in (v.get("time_placeholder") or {}).items():
        out[k] = (val.get("nlq") if isinstance(val, dict) else str(val)) or ""
    return out


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=timezone.utc)
    return start, end


def interval_for(nlq: str, now: datetime, visits: dict[str, list[tuple[datetime, datetime | None]]]) -> tuple[str, datetime, datetime | None] | None:
    """Return (kind, start, end) for a global time filter; end None = open until now. kind ∈ {calendar, visit}."""
    s = nlq.strip().lower()
    if not s:
        return None
    m = re.match(r"since (\d{4})$", s)
    if m:
        return ("calendar", datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc), None)
    m = re.match(r"since (\d+) (year|month|day)s? ago$", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        # the benchmark's SQL uses datetime(now, '-N month') / '-N year': calendar arithmetic, not 30/365-day spans
        if unit == "day":
            return ("calendar", now - timedelta(days=n), None)
        months = n * 12 if unit == "year" else n
        y, mth = now.year, now.month - months
        while mth <= 0:
            y -= 1; mth += 12
        import calendar as _cal
        day = min(now.day, _cal.monthrange(y, mth)[1])
        return ("calendar", now.replace(year=y, month=mth, day=day), None)
    if s == "this year":
        return ("calendar", datetime(now.year, 1, 1, tzinfo=timezone.utc), None)
    if s == "this month":
        return ("calendar", datetime(now.year, now.month, 1, tzinfo=timezone.utc), None)
    if s == "last year":
        return ("calendar", datetime(now.year - 1, 1, 1, tzinfo=timezone.utc), datetime(now.year, 1, 1, tzinfo=timezone.utc))
    if s == "last month":
        y, mth = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        return ("calendar", *_month_bounds(y, mth))
    if s == "yesterday":
        d = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        return ("calendar", d, d + timedelta(days=1))
    if s == "today":
        d = now.replace(hour=0, minute=0, second=0)
        return ("calendar", d, None)
    m = re.match(r"in (\d{2})/(this|last) year$", s)
    if m:
        year = now.year if m.group(2) == "this" else now.year - 1
        return ("calendar", *_month_bounds(year, MONTHS[m.group(1)]))
    m = re.match(r"in (\d{4})$", s)
    if m:
        y = int(m.group(1)); return ("calendar", datetime(y, 1, 1, tzinfo=timezone.utc), datetime(y + 1, 1, 1, tzinfo=timezone.utc))
    m = re.match(r"in (\d{2})/(\d{4})$", s)
    if m:
        return ("calendar", *_month_bounds(int(m.group(2)), MONTHS[m.group(1)]))
    m = re.match(r"since (\d{2})/(\d{4})$", s)
    if m:
        return ("calendar", datetime(int(m.group(2)), MONTHS[m.group(1)], 1, tzinfo=timezone.utc), None)
    m = re.match(r"since (\d{2})/(\d{2})/(\d{4})$", s)
    if m:
        return ("calendar", datetime(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)), tzinfo=timezone.utc), None)
    m = re.match(r"on (\d{2})/(\d{2})/(\d{4})$", s)
    if m:
        d = datetime(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)), tzinfo=timezone.utc); return ("calendar", d, d + timedelta(days=1))
    m = re.match(r"on (\d{2})/(\d{2})/(this|last) year$", s)
    if m:
        year = now.year if m.group(3) == "this" else now.year - 1
        d = datetime(year, MONTHS[m.group(1)], int(m.group(2)), tzinfo=timezone.utc); return ("calendar", d, d + timedelta(days=1))
    m = re.match(r"on (this|last) month/(\d{2})$", s)
    if m:
        y, mth = (now.year, now.month) if m.group(1) == "this" else ((now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1))
        d = datetime(y, mth, int(m.group(2)), tzinfo=timezone.utc); return ("calendar", d, d + timedelta(days=1))
    m = re.match(r"on the (first|last|current) (hospital|icu) visit$", s)
    if m:
        which, kind = m.group(1), m.group(2)
        vs = visits.get(kind, [])
        if not vs:
            return None
        if which == "first":
            start, end = vs[0]
        elif which == "last":
            start, end = vs[-1]
        else:
            # benchmark semantics (its SQL: dischtime / outtime IS NULL): the current visit is an OPEN one; a patient
            # with no open visit has no current visit, so the window is empty and the released answer is []
            open_ended = [v for v in vs if v[0] <= now and v[1] is None]
            if open_ended:
                start, end = open_ended[-1]
            else:
                return ("visit", now, now)
        return ("visit", start, end)
    return None


def derive_contract(row: dict[str, Any], visits: dict[str, list[tuple[datetime, datetime | None]]], latest_time: datetime | None) -> tuple[Contract | None, dict[str, Any]]:
    """Contract for a FHIR-AgentBench row plus derivation notes; None when the filter cannot be mapped."""
    now = parse_now(row.get("assumption")) or latest_time
    if now is None:
        return None, {"reason": "no current time and no patient timeline"}
    ph = time_placeholders(row.get("val_dict", ""))
    globals_ = [ph.get(k, "") for k in ("time_filter_global1", "time_filter_global2") if ph.get(k)]
    sources = MAIN_TABLE_SOURCES.get(str(row.get("main_table_name", "")).strip())
    notes: dict[str, Any] = {"now": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "filters": globals_, "main_table": row.get("main_table_name")}
    if not globals_:
        return Contract(index_time=notes["now"], scope="full", basis="occurred", permitted_sources=sources), {**notes, "rule": "no_filter_full"}
    intervals = []
    for g in globals_:
        iv = interval_for(g, now, visits)
        if iv is None:
            return None, {**notes, "reason": f"unmapped filter {g!r}"}
        intervals.append(iv)
    kinds = {iv[0] for iv in intervals}
    start = min(iv[1] for iv in intervals)
    ends = [iv[2] for iv in intervals]
    end = None if any(e is None for e in ends) else max(ends)
    if "visit" in kinds:
        # during a visit: post-index window from the visit start
        end_eff = end or now
        idx = start - timedelta(seconds=1)   # the visit's own encounter (period.start == start) must be legal under post
        c = Contract(index_time=idx.strftime("%Y-%m-%dT%H:%M:%SZ"), scope="post", basis="occurred", permitted_sources=sources,
                     window_hours=max(0.0, (end_eff - idx).total_seconds() / 3600.0))
        return c, {**notes, "rule": "visit_post_window", "window_end": end_eff.strftime("%Y-%m-%dT%H:%M:%SZ")}
    end_eff = end or now
    c = Contract(index_time=end_eff.strftime("%Y-%m-%dT%H:%M:%SZ"), scope="pre", basis="occurred", permitted_sources=sources,
                 window_hours=max(0.0, (end_eff - start).total_seconds() / 3600.0))
    return c, {**notes, "rule": "calendar_pre_window", "window_start": start.strftime("%Y-%m-%dT%H:%M:%SZ")}


# ------------------------------------------------------------------ FHIR resource -> evidence object
_OFFSET_RE = re.compile(r"([+-]\d\d:\d\d|Z)$")


def wallclock(s: str | None) -> str | None:
    """FHIR-AgentBench's time semantics are the MIMIC wall-clock times (its SQL gold and its 'current time' are naive
    local timestamps); the FHIR export carries them with a -05:00/-04:00 offset. The contract compares wall-clock to
    wall-clock: the offset is dropped and the naive time is marked UTC. (Pilot v0.1 compared UTC-converted FHIR
    times with naive windows, a 4–5 h shift at every window edge.)"""
    if not isinstance(s, str) or not s:
        return s
    return _OFFSET_RE.sub("", s) + "Z"


def _first(resource: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = resource.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and isinstance(v.get("start"), str):
            return v["start"]
    return None


def resource_to_evidence(resource: dict[str, Any], patient_hash: str) -> EvidenceObject:
    rt = str(resource.get("resourceType"))
    if rt == "MedicationRequest":
        # the benchmark's prescription time is prescriptions.starttime = dispenseRequest.validityPeriod.start;
        # authoredOn is the order-entry time
        vp = (resource.get("dispenseRequest") or {}).get("validityPeriod") or {}
        occurred = vp.get("start") if isinstance(vp.get("start"), str) else resource.get("authoredOn")
    else:
        occurred = _first(resource, "effectiveDateTime", "effectivePeriod", "authoredOn", "performedDateTime", "performedPeriod", "occurrenceDateTime",
                          "occurrencePeriod", "period", "recordedDate", "dateAsserted", "whenHandedOver")
    if occurred is None and isinstance(resource.get("collection"), dict):
        occurred = resource["collection"].get("collectedDateTime")
    available = resource.get("issued") if isinstance(resource.get("issued"), str) else occurred
    end = None
    for k in ("effectivePeriod", "performedPeriod", "occurrencePeriod", "period"):
        if isinstance(resource.get(k), dict) and isinstance(resource[k].get("end"), str):
            end = resource[k]["end"]
            break
    occurred, available, end = wallclock(occurred), wallclock(available), wallclock(end)
    fields = {"resourceType": rt}
    code = resource.get("code")
    if isinstance(code, dict):
        fields["code_text"] = code.get("text") or (code.get("coding") or [{}])[0].get("display")
    if "valueQuantity" in resource and isinstance(resource["valueQuantity"], dict):
        fields["value"] = resource["valueQuantity"].get("value"); fields["unit"] = resource["valueQuantity"].get("unit")
    elif "valueString" in resource:
        fields["value"] = resource["valueString"]
    if isinstance(resource.get("status"), str):
        fields["status"] = resource["status"]
    return EvidenceObject(id=str(resource["id"]), source=rt, patient=patient_hash, occurred=occurred, available=available, occurred_end=end, row_ref=str(resource["id"]), fields=fields)
