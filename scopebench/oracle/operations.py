"""Oracle (gold-generation) implementation of scopebench/SKILL_SPEC.md v1.2, operations §1–§7.

Written strictly from the specification (implementer A, plan §1.8). Every function is pure:
no I/O, no randomness, no global state. Evidence is assumed already legal for the contract
(spec §0); the predicate is not re-applied here. Residual decisions are recorded in
scopebench/oracle/NOTES.md.
"""

from __future__ import annotations

import math
import operator
from datetime import datetime
from typing import Any, Callable

from ..schema import BASES, Contract, EvidenceObject

Result = dict[str, Any]


# ----------------------------------------------------------------------------------------------
# Shared helpers (spec §0)
# ----------------------------------------------------------------------------------------------


def _check_basis(basis: str) -> str:
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}")
    return basis


def _time(e: EvidenceObject, basis: str) -> datetime | None:
    """Time on the basis, or None when missing (`None`/``""``) or unparseable (spec §0)."""
    _check_basis(basis)
    try:
        return e.time(basis)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    """float(value) or None when the value cannot be parsed as float (spec §0, 'numeric')."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_numeric(e: EvidenceObject, key: str = "value") -> bool:
    return _as_float(e.fields.get(key)) is not None


def _t_key(basis: str) -> Callable[[EvidenceObject], tuple]:
    """The total order T (spec §0): (0, time, id) for timed, (1, "", id) for untimed objects."""

    def key(e: EvidenceObject) -> tuple:
        t = _time(e, basis)
        if t is None:
            return (1, "", e.id)
        return (0, t, e.id)

    return key


def _chronological(objs: list[EvidenceObject], basis: str) -> list[EvidenceObject]:
    return sorted(objs, key=_t_key(basis))


def _timed(objs: list[EvidenceObject], basis: str) -> list[EvidenceObject]:
    return [e for e in objs if _time(e, basis) is not None]


def _nearest(objs: list[EvidenceObject], basis: str, index: datetime, k: int) -> list[EvidenceObject]:
    """The k objects nearest the index on `basis`; key (|t - index|, id) over timed objects only."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    ranked = sorted(_timed(objs, basis), key=lambda e: (abs(_time(e, basis) - index), e.id))
    return ranked[:k]


def _min_pick(objs: list[EvidenceObject], basis: str) -> EvidenceObject | None:
    """Min-type pick (spec §0): first element of T among timed candidates."""
    timed = _timed(objs, basis)
    return min(timed, key=_t_key(basis)) if timed else None


def _max_pick(objs: list[EvidenceObject], basis: str) -> EvidenceObject | None:
    """Max-type pick (spec §0): last element of T among timed candidates (ties → largest id)."""
    timed = _timed(objs, basis)
    return max(timed, key=_t_key(basis)) if timed else None


def _basis(contract: Contract, params: dict) -> str:
    basis = params.get("basis")
    return _check_basis(contract.basis if basis is None else str(basis))


def _time_str(e: EvidenceObject, basis: str) -> str | None:
    """Verbatim time string on the basis; None when the object has no time on that basis."""
    if _time(e, basis) is None:
        return None
    return getattr(e, basis)


def _check_label(params: dict, key: str, expected: str) -> None:
    """spec §0: params["which"] / params["agg"], when present, must equal the registry key."""
    if key in params and params[key] != expected:
        raise ValueError(f"params[{key!r}] = {params[key]!r} does not match operation {expected!r}")


def _norm_drug(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _drug_matches(e: EvidenceObject, drug: Any) -> bool:
    """spec §6: case-insensitive exact match after strip().lower(); None never matches."""
    target = _norm_drug(drug)
    med = _norm_drug(e.fields.get("medication"))
    return target is not None and med is not None and med == target


# ----------------------------------------------------------------------------------------------
# §1 lab_trend_k3 (frozen ScopePair-FHIR semantics; arbiter = REFERENCE_SKILLS["lab_trend_k3"])
# ----------------------------------------------------------------------------------------------


def lab_trend_k3(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    k = int(params.get("k", 3))
    if k < 1:
        raise ValueError(f"k must be >= 1 for lab_trend_k3, got {k}")
    basis = "available"  # fixed by §1
    numeric = [e for e in evidence if _is_numeric(e, "value")]
    selected = _nearest(numeric, basis, contract.index, k)
    if len(selected) < k:
        return {"status": "insufficient", "provenance": []}  # reference shape (§0, §1)
    ordered = _chronological(selected, basis)
    delta = float(ordered[-1].fields["value"]) - float(ordered[0].fields["value"])
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
            {"citation": e.id, "value": e.fields.get("value"), "unit": e.fields.get("unit"), "issued_time": e.available}
            for e in ordered
        ],
    }


# ----------------------------------------------------------------------------------------------
# §2 med_dose_k3 (frozen ScopePair-MedAdmin semantics)
# ----------------------------------------------------------------------------------------------


def med_dose_k3(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    k = int(params.get("k", 3))
    if k < 1:
        raise ValueError(f"k must be >= 1 for med_dose_k3, got {k}")
    basis = "occurred"  # fixed by §2
    numeric = [e for e in evidence if _is_numeric(e, "dose")]
    selected = _nearest(numeric, basis, contract.index, k)
    if len(selected) < k:
        return {"status": "insufficient", "ordered_result_ids": [], "total_dose": None, "mean_dose": None, "provenance": []}
    ordered = _chronological(selected, basis)
    total_dose = sum((float(e.fields["dose"]) for e in ordered), 0.0)
    return {
        "status": "ok",
        "ordered_result_ids": [e.id for e in ordered],
        "total_dose": total_dose,
        "mean_dose": total_dose / k,
        "provenance": [
            {"citation": e.id, "dose": e.fields.get("dose"), "unit": e.fields.get("unit"), "effective_time": e.occurred}
            for e in ordered
        ],
    }


# ----------------------------------------------------------------------------------------------
# §3 nearest_k
# ----------------------------------------------------------------------------------------------


def nearest_k(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    k = int(params["k"])
    basis = _basis(contract, params)
    selected = _nearest(evidence, basis, contract.index, k)
    if len(selected) < k:
        return {"status": "insufficient", "ordered_result_ids": [], "provenance": []}
    ordered = _chronological(selected, basis)
    return {
        "status": "ok",
        "ordered_result_ids": [e.id for e in ordered],
        "provenance": [
            {"citation": e.id, "value": e.fields.get("value"), "unit": e.fields.get("unit"), "time": _time_str(e, basis)}
            for e in ordered
        ],
    }


# ----------------------------------------------------------------------------------------------
# §4 first_after / last_before
# ----------------------------------------------------------------------------------------------


def _pick_first_after(objs: list[EvidenceObject], basis: str, index: datetime) -> EvidenceObject | None:
    return _min_pick([e for e in _timed(objs, basis) if _time(e, basis) > index], basis)


def _pick_last_before(objs: list[EvidenceObject], basis: str, index: datetime) -> EvidenceObject | None:
    return _max_pick([e for e in _timed(objs, basis) if _time(e, basis) < index], basis)


def _boundary(evidence: list[EvidenceObject], contract: Contract, params: dict, which: str) -> Result:
    _check_label(params, "which", which)
    basis = _basis(contract, params)
    if which == "first_after":
        pick = _pick_first_after(evidence, basis, contract.index)
    elif which == "last_before":
        pick = _pick_last_before(evidence, basis, contract.index)
    else:
        raise ValueError(f"unknown boundary lookup {which!r}")
    if pick is None:
        return {"status": "insufficient", "citation": None, "time": None, "fields": None, "provenance": []}
    return {"status": "ok", "citation": pick.id, "time": _time_str(pick, basis), "fields": dict(pick.fields), "provenance": [pick.id]}


def first_after(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    return _boundary(evidence, contract, params, "first_after")


def last_before(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    return _boundary(evidence, contract, params, "last_before")


# ----------------------------------------------------------------------------------------------
# §5 window_count / window_sum / window_mean
# ----------------------------------------------------------------------------------------------


def _window(evidence: list[EvidenceObject], contract: Contract, params: dict, agg: str) -> Result:
    _check_label(params, "agg", agg)
    basis = _basis(contract, params)
    ordered = _chronological(list(evidence), basis)
    if agg == "count":
        return {"status": "ok", "agg": "count", "value": len(ordered), "n": len(ordered), "n_skipped_non_numeric": 0, "provenance": [e.id for e in ordered]}
    if agg not in ("sum", "mean"):
        raise ValueError(f"unknown aggregation {agg!r}")
    numeric = [e for e in ordered if _is_numeric(e, "value")]
    n_skipped = len(ordered) - len(numeric)
    values = [float(e.fields["value"]) for e in numeric]
    if agg == "mean" and not values:
        return {"status": "insufficient", "agg": "mean", "value": None, "n": None, "n_skipped_non_numeric": None, "provenance": []}
    total = sum(values, 0.0)
    value = total if agg == "sum" else total / len(values)
    return {"status": "ok", "agg": agg, "value": value, "n": len(values), "n_skipped_non_numeric": n_skipped, "provenance": [e.id for e in numeric]}


def window_count(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    return _window(evidence, contract, params, "count")


def window_sum(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    return _window(evidence, contract, params, "sum")


def window_mean(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    return _window(evidence, contract, params, "mean")


# ----------------------------------------------------------------------------------------------
# §6 outcome_lookup
# ----------------------------------------------------------------------------------------------

_DISPOSITION_SOURCES = {"discharge_disposition": "admissions", "ed_disposition": "edstays"}


def _outcome_insufficient(target: str) -> Result:
    return {"status": "insufficient", "target": target, "value": None, "citation": None, "time": None, "provenance": []}


def _disposition_pick(pool: list[EvidenceObject]) -> EvidenceObject | None:
    """Max-type pick on `occurred`; untimed fallback = smallest id when nothing is timed (§0)."""
    if not pool:
        return None
    pick = _max_pick(pool, "occurred")
    return pick if pick is not None else min(pool, key=lambda e: e.id)


def outcome_lookup(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    target = params["target"]
    field_key = params.get("field")
    basis = _basis(contract, params)
    if target == "first_careunit":
        pick = _pick_first_after([e for e in evidence if e.source == "transfers"], basis, contract.index)
    elif target in _DISPOSITION_SOURCES:
        pick = _disposition_pick([e for e in evidence if e.source == _DISPOSITION_SOURCES[target]])
    elif target == "administration_count":
        matches = _chronological([e for e in evidence if e.source == "emar" and _drug_matches(e, params.get("drug"))], basis)
        return {"status": "ok", "target": target, "value": len(matches), "citation": None, "time": None, "provenance": [e.id for e in matches]}
    else:
        raise ValueError(f"unknown outcome target {target!r}")
    if pick is None:
        return _outcome_insufficient(target)
    return {
        "status": "ok",
        "target": target,
        "value": pick.fields.get(field_key),
        "citation": pick.id,
        "time": _time_str(pick, basis),
        "provenance": [pick.id],
    }


# ----------------------------------------------------------------------------------------------
# §7 cohort_criteria
# ----------------------------------------------------------------------------------------------

_THRESHOLD_OPS: dict[str, Callable[[float, float], bool]] = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


def _evaluate_criterion(criterion: dict, evidence: list[EvidenceObject], contract: Contract) -> tuple[bool | None, list[str]]:
    """(satisfied, satisfying ids chronological on the criterion basis); satisfied is None iff the
    domain is empty (spec §7)."""
    kind = criterion.get("kind")
    basis = criterion.get("basis")
    basis = _check_basis(contract.basis if basis is None else str(basis))
    if kind == "lab_threshold":
        op = _THRESHOLD_OPS.get(criterion.get("op"))
        if op is None:
            raise ValueError(f"unknown threshold op {criterion.get('op')!r}")
        threshold = float(criterion["threshold"])
        concept = criterion.get("concept")
        domain = [e for e in evidence if e.fields.get("concept") == concept]
        matches = [e for e in domain if _is_numeric(e, "value") and op(float(e.fields["value"]), threshold)]
    elif kind == "code_present":
        prefix = str(criterion.get("prefix", "")).upper()
        domain = [e for e in evidence if e.source == "diagnoses_icd"]
        matches = [e for e in domain if e.fields.get("code") is not None and str(e.fields["code"]).upper().startswith(prefix)]
    elif kind == "exposure":
        domain = [e for e in evidence if e.source == "emar"]
        matches = [e for e in domain if _drug_matches(e, criterion.get("drug"))]
    elif kind == "acuity":
        max_acuity = int(criterion["max_acuity"])
        domain = [e for e in evidence if e.source == "triage"]
        matches = []
        for e in domain:
            try:
                acuity = int(e.fields.get("acuity"))
            except (TypeError, ValueError):
                continue
            if acuity <= max_acuity:
                matches.append(e)
    else:
        raise ValueError(f"unknown criterion kind {kind!r}")
    if not domain:
        return None, []  # undecidable (§7)
    if matches:
        return True, [e.id for e in _chronological(matches, basis)]
    # v1.2: an unsatisfied criterion cites its whole domain (examined-and-failed evidence).
    return False, [e.id for e in _chronological(domain, basis)]


def cohort_criteria(evidence: list[EvidenceObject], contract: Contract, params: dict) -> Result:
    by_id = {e.id: e for e in evidence}
    evaluated: list[Result] = []
    decisive: list[str] | None = None
    for criterion in params.get("criteria", []):
        satisfied, ids = _evaluate_criterion(dict(criterion), evidence, contract)
        evaluated.append({**dict(criterion), "satisfied": satisfied, "provenance": ids})
        if satisfied is False and decisive is None:
            decisive = ids
    if decisive is not None:
        return {"status": "ok", "decision": "exclude", "criteria": evaluated, "provenance": list(decisive)}
    decision = "uncertain" if any(c["satisfied"] is None for c in evaluated) else "include"
    union = _chronological([by_id[i] for i in {i for c in evaluated for i in c["provenance"]}], contract.basis)
    return {"status": "ok", "decision": decision, "criteria": evaluated, "provenance": [e.id for e in union]}


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------

OPERATIONS: dict[str, Callable[[list[EvidenceObject], Contract, dict], Result]] = {
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
