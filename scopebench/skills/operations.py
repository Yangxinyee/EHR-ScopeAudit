"""Inference-time ChartSkills, implementer B (plan §1.8).

Written strictly from scopebench/SKILL_SPEC.md version 1.2, operations §1–§7.
Every operation is a pure function

    fn(evidence: list[EvidenceObject], contract: Contract, params: dict) -> dict

with no I/O, no randomness and no module-level mutable state. Choices the
spec does not pin down are listed in NOTES.md next to this file.

Conventions used throughout (spec §0):
  * legality is decided by ``scopebench.predicate.legal`` only; each operation
    applies it once on entry, which is a no-op when the caller pre-filtered;
  * "chronological" is the total order ``(time on basis, id)``; "nearest" is
    ``(|t - index| in seconds, id)``; ids compare as strings ascending;
  * a numeric value is ``float(fields[key])``; anything that raises is
    non-numeric and is skipped by numeric operations.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable

from ..predicate import legal
from ..schema import BASES, Contract, EvidenceObject, parse_time

SPEC_VERSION = "1.2"

Result = dict[str, Any]
Params = dict[str, Any]

# Spec §0, total order T: untimed objects sort after every timed object,
# ties by id. The sentinel keeps tuple comparison type-safe.
_NO_TIME = datetime(1, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _params(params: Params | None) -> Params:
    return dict(params) if params else {}


def _legal_only(evidence: list[EvidenceObject], contract: Contract) -> list[EvidenceObject]:
    return [e for e in evidence if legal(e, contract)]


def _basis(params: Params, contract: Contract) -> str:
    basis = params.get("basis") or contract.basis
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}")
    return basis


def _when(e: EvidenceObject, basis: str) -> datetime | None:
    """Parsed time on ``basis``; missing or unparseable -> None (NOTES.md, N2)."""
    try:
        return parse_time(getattr(e, basis))
    except (TypeError, ValueError):
        return None


def _number(e: EvidenceObject, key: str = "value") -> float | None:
    """float(fields[key]) or None when the object is non-numeric for ``key``."""
    try:
        return float(e.fields[key])
    except (KeyError, TypeError, ValueError):
        return None


def _chrono_key(basis: str) -> Callable[[EvidenceObject], tuple[int, datetime, str]]:
    def key(e: EvidenceObject) -> tuple[int, datetime, str]:
        t = _when(e, basis)
        return (1, _NO_TIME, e.id) if t is None else (0, t, e.id)

    return key


def _chronological(objs: list[EvidenceObject], basis: str) -> list[EvidenceObject]:
    return sorted(objs, key=_chrono_key(basis))


def _timed(objs: list[EvidenceObject], basis: str) -> list[tuple[EvidenceObject, datetime]]:
    out = []
    for e in objs:
        t = _when(e, basis)
        if t is not None:
            out.append((e, t))
    return out


def _nearest(objs: list[EvidenceObject], contract: Contract, basis: str, k: int) -> list[EvidenceObject]:
    """The k objects closest to the index on ``basis``; key (|dt| seconds, id)."""
    index = contract.index
    ranked = sorted(
        _timed(objs, basis),
        key=lambda et: (abs((et[1] - index).total_seconds()), et[0].id),
    )
    return [e for e, _ in ranked[:k]]


def _k(params: Params, default: int | None = None, minimum: int = 0) -> int:
    """§0/§1/§2/§3: k < minimum raises (minimum 1 for the frozen ops, 0 for nearest_k)."""
    k = params.get("k", default)
    if k is None:
        raise ValueError("params['k'] is required")
    k = int(k)
    if k < minimum:
        raise ValueError(f"k must be >= {minimum}, got {k}")
    return k


def _same_text(a: Any, b: Any) -> bool:
    """§6 medication match: case-insensitive exact match after str.strip().lower(); None never matches."""
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()


# ---------------------------------------------------------------------------
# §1  lab_trend_k3  (frozen ScopePair-FHIR LabTrend semantics; basis = available)
# ---------------------------------------------------------------------------


def lab_trend_k3(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    p = _params(params)
    k = _k(p, default=3, minimum=1)
    pool = [e for e in _legal_only(evidence, contract) if _number(e, "value") is not None]
    chosen = _nearest(pool, contract, "available", k)
    if len(chosen) < k:
        # Shape of the frozen reference (tasktypes.REFERENCE_SKILLS), the arbiter for §1.
        return {"status": "insufficient", "provenance": []}
    ordered = _chronological(chosen, "available")
    first = _number(ordered[0], "value")
    last = _number(ordered[-1], "value")
    assert first is not None and last is not None
    delta = last - first
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        direction = "stable"
    elif delta > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return {
        "status": "ok",
        "ordered_result_ids": [e.id for e in ordered],
        "delta_last_minus_first": delta,
        "direction": direction,
        "provenance": [
            {
                "citation": e.id,
                "value": e.fields.get("value"),
                "unit": e.fields.get("unit"),
                "issued_time": e.available,
            }
            for e in ordered
        ],
    }


# ---------------------------------------------------------------------------
# §2  med_dose_k3  (frozen ScopePair-MedAdmin MedDose semantics; basis = occurred)
# ---------------------------------------------------------------------------


def med_dose_k3(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    p = _params(params)
    k = _k(p, default=3, minimum=1)
    pool = [e for e in _legal_only(evidence, contract) if _number(e, "dose") is not None]
    chosen = _nearest(pool, contract, "occurred", k)
    if len(chosen) < k:
        return {
            "status": "insufficient",
            "ordered_result_ids": [],
            "total_dose": None,
            "mean_dose": None,
            "provenance": [],
        }
    ordered = _chronological(chosen, "occurred")
    doses = [_number(e, "dose") for e in ordered]
    total = float(sum(d for d in doses if d is not None))
    return {
        "status": "ok",
        "ordered_result_ids": [e.id for e in ordered],
        "total_dose": total,
        "mean_dose": total / k,
        "provenance": [
            {
                "citation": e.id,
                "dose": e.fields.get("dose"),
                "unit": e.fields.get("unit"),
                "effective_time": e.occurred,
            }
            for e in ordered
        ],
    }


# ---------------------------------------------------------------------------
# §3  nearest_k
# ---------------------------------------------------------------------------


def nearest_k(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    p = _params(params)
    k = _k(p)
    basis = _basis(p, contract)
    chosen = _nearest(_legal_only(evidence, contract), contract, basis, k)
    if len(chosen) < k:
        return {"status": "insufficient", "ordered_result_ids": [], "provenance": []}
    ordered = _chronological(chosen, basis)
    return {
        "status": "ok",
        "ordered_result_ids": [e.id for e in ordered],
        "provenance": [
            {
                "citation": e.id,
                "value": e.fields.get("value"),
                "unit": e.fields.get("unit"),
                "time": getattr(e, basis),
            }
            for e in ordered
        ],
    }


# ---------------------------------------------------------------------------
# §4  first_after / last_before
# ---------------------------------------------------------------------------

_BOUNDARY_INSUFFICIENT: Result = {
    "status": "insufficient",
    "citation": None,
    "time": None,
    "fields": None,
    "provenance": [],
}


def _boundary_pick(
    objs: list[EvidenceObject], contract: Contract, basis: str, which: str
) -> EvidenceObject | None:
    index = contract.index
    timed = _timed(objs, basis)
    if which == "first_after":
        cands = [(e, t) for e, t in timed if t > index]
        if not cands:
            return None
        return min(cands, key=lambda et: (et[1], et[0].id))[0]
    if which == "last_before":
        cands = [(e, t) for e, t in timed if t < index]
        if not cands:
            return None
        return max(cands, key=lambda et: (et[1], et[0].id))[0]
    raise ValueError(f"unknown which {which!r}")


def _boundary(evidence: list[EvidenceObject], contract: Contract, params: Params, which: str) -> Result:
    p = _params(params)
    if p.get("which", which) != which:
        raise ValueError(f"params['which']={p['which']!r} conflicts with operation {which!r}")
    basis = _basis(p, contract)
    e = _boundary_pick(_legal_only(evidence, contract), contract, basis, which)
    if e is None:
        return dict(_BOUNDARY_INSUFFICIENT)
    return {
        "status": "ok",
        "citation": e.id,
        "time": getattr(e, basis),
        "fields": dict(e.fields),
        "provenance": [e.id],
    }


def first_after(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    return _boundary(evidence, contract, params, "first_after")


def last_before(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    return _boundary(evidence, contract, params, "last_before")


# ---------------------------------------------------------------------------
# §5  window_count / window_sum / window_mean
# ---------------------------------------------------------------------------


def _aggregate(evidence: list[EvidenceObject], contract: Contract, params: Params, agg: str) -> Result:
    p = _params(params)
    if p.get("agg", agg) != agg:
        raise ValueError(f"params['agg']={p['agg']!r} conflicts with operation {agg!r}")
    basis = _basis(p, contract)
    pool = _chronological(_legal_only(evidence, contract), basis)
    if agg == "count":
        return {
            "status": "ok",
            "agg": "count",
            "value": len(pool),
            "n": len(pool),
            "n_skipped_non_numeric": 0,
            "provenance": [e.id for e in pool],
        }
    numeric = [(e, v) for e in pool if (v := _number(e, "value")) is not None]
    skipped = len(pool) - len(numeric)
    if agg == "mean" and not numeric:
        # §0: the label key `agg` keeps its value on insufficient.
        return {
            "status": "insufficient",
            "agg": agg,
            "value": None,
            "n": None,
            "n_skipped_non_numeric": None,
            "provenance": [],
        }
    total = float(sum(v for _, v in numeric))
    if agg == "sum":
        value = total
    elif agg == "mean":
        value = total / len(numeric)
    else:
        raise ValueError(f"unknown agg {agg!r}")
    return {
        "status": "ok",
        "agg": agg,
        "value": value,
        "n": len(numeric),
        "n_skipped_non_numeric": skipped,
        "provenance": [e.id for e, _ in numeric],
    }


def window_count(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    return _aggregate(evidence, contract, params, "count")


def window_sum(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    return _aggregate(evidence, contract, params, "sum")


def window_mean(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    return _aggregate(evidence, contract, params, "mean")


# ---------------------------------------------------------------------------
# §6  outcome_lookup
# ---------------------------------------------------------------------------

def _outcome_insufficient(target: Any) -> Result:
    # §0: the label key `target` keeps its value on insufficient.
    return {
        "status": "insufficient",
        "target": target,
        "value": None,
        "citation": None,
        "time": None,
        "provenance": [],
    }


_DISPOSITION_SOURCE = {"discharge_disposition": "admissions", "ed_disposition": "edstays"}


def _disposition_pick(rows: list[EvidenceObject]) -> EvidenceObject:
    """§6/§0: max-type pick on `occurred` among timed rows; if none is timed, the smallest untimed id."""
    timed = _timed(rows, "occurred")
    if timed:
        return max(timed, key=lambda et: (et[1], et[0].id))[0]
    return min(rows, key=lambda r: r.id)


def outcome_lookup(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    p = _params(params)
    target = p.get("target")
    field_name = p.get("field")
    basis = _basis(p, contract)
    pool = _legal_only(evidence, contract)

    if target == "first_careunit":
        transfers = [e for e in pool if e.source == "transfers"]
        e = _boundary_pick(transfers, contract, basis, "first_after")
        if e is None:
            return _outcome_insufficient(target)
        return {
            "status": "ok",
            "target": target,
            "value": e.fields.get(field_name),
            "citation": e.id,
            "time": getattr(e, basis),
            "provenance": [e.id],
        }

    if target in _DISPOSITION_SOURCE:
        rows = [e for e in pool if e.source == _DISPOSITION_SOURCE[target]]
        if not rows:
            return _outcome_insufficient(target)
        e = _disposition_pick(rows)
        return {
            "status": "ok",
            "target": target,
            "value": e.fields.get(field_name),
            "citation": e.id,
            "time": getattr(e, basis),  # §6: time on params["basis"], None if absent
            "provenance": [e.id],
        }

    if target == "administration_count":
        drug = p.get("drug")
        hits = [e for e in pool if e.source == "emar" and _same_text(e.fields.get("medication"), drug)]
        hits = _chronological(hits, basis)
        return {
            "status": "ok",
            "target": target,
            "value": len(hits),
            "citation": None,
            "time": None,
            "provenance": [e.id for e in hits],
        }

    raise ValueError(f"unknown outcome target {target!r}")


# ---------------------------------------------------------------------------
# §7  cohort_criteria
# ---------------------------------------------------------------------------

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, thr: v >= thr,
    "<=": lambda v, thr: v <= thr,
    ">": lambda v, thr: v > thr,
    "<": lambda v, thr: v < thr,
}


def _domain_and_hits(
    crit: Params, pool: list[EvidenceObject]
) -> tuple[list[EvidenceObject], list[EvidenceObject]]:
    """§7 per kind: (domain = the legal objects examined, hits = those that satisfy the criterion)."""
    kind = crit.get("kind")

    if kind == "lab_threshold":
        concept = crit.get("concept")
        op = _OPS.get(crit.get("op"))
        if op is None:
            raise ValueError(f"unknown lab_threshold op {crit.get('op')!r}")
        threshold = float(crit["threshold"])
        domain = [e for e in pool if e.fields.get("concept") == concept]
        hits = [e for e in domain if (v := _number(e, "value")) is not None and op(v, threshold)]
        return domain, hits

    if kind == "code_present":
        prefix = str(crit.get("prefix", "")).upper()
        domain = [e for e in pool if e.source == "diagnoses_icd"]
        hits = [
            e
            for e in domain
            if e.fields.get("code") is not None and str(e.fields.get("code")).upper().startswith(prefix)
        ]
        return domain, hits

    if kind == "exposure":
        drug = crit.get("drug")
        domain = [e for e in pool if e.source == "emar"]
        hits = [e for e in domain if _same_text(e.fields.get("medication"), drug)]
        return domain, hits

    if kind == "acuity":
        max_acuity = int(crit["max_acuity"])
        domain = [e for e in pool if e.source == "triage"]
        hits = []
        for e in domain:
            try:
                acuity = int(e.fields["acuity"])
            except (KeyError, TypeError, ValueError):
                continue
            if acuity <= max_acuity:
                hits.append(e)
        return domain, hits

    raise ValueError(f"unknown criterion kind {kind!r}")


def _evaluate_criterion(
    crit: Params, pool: list[EvidenceObject], contract: Contract
) -> tuple[bool | None, list[EvidenceObject]]:
    """Returns (satisfied | None when the domain is empty, provenance objects chronological on the criterion basis).

    Spec v1.2 §7: satisfied -> all satisfying objects; unsatisfied -> the whole
    domain (examined and failed); undecidable -> [].
    """
    basis = _basis(crit, contract)
    domain, hits = _domain_and_hits(crit, pool)
    if not domain:
        return None, []
    if hits:
        return True, _chronological(hits, basis)
    return False, _chronological(domain, basis)


def cohort_criteria(evidence: list[EvidenceObject], contract: Contract, params: Params) -> Result:
    p = _params(params)
    criteria = list(p.get("criteria") or [])
    pool = _legal_only(evidence, contract)

    evaluated: list[Result] = []
    all_hits: list[list[EvidenceObject]] = []
    verdicts: list[bool | None] = []
    decisive: list[EvidenceObject] | None = None
    for crit in criteria:
        satisfied, hits = _evaluate_criterion(dict(crit), pool, contract)
        verdicts.append(satisfied)
        all_hits.append(hits)
        evaluated.append({**crit, "satisfied": satisfied, "provenance": [e.id for e in hits]})
        if satisfied is False and decisive is None:
            decisive = hits

    if decisive is not None:
        decision = "exclude"
        provenance = [e.id for e in decisive]
    else:
        decision = "uncertain" if any(v is None for v in verdicts) else "include"
        seen: dict[str, EvidenceObject] = {}
        for hits in all_hits:
            for e in hits:
                seen.setdefault(e.id, e)
        provenance = [e.id for e in _chronological(list(seen.values()), contract.basis)]

    return {
        "status": "ok",
        "decision": decision,
        "criteria": evaluated,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, Callable[[list[EvidenceObject], Contract, Params], Result]] = {
    "lab_trend_k3": lab_trend_k3,
    "med_dose_k3": med_dose_k3,
    "nearest_k": nearest_k,
    "first_after": first_after,
    "last_before": last_before,
    "window_count": window_count,
    "window_sum": window_sum,
    "window_mean": window_mean,
    "outcome_lookup": outcome_lookup,
    "cohort_criteria": cohort_criteria,
}
