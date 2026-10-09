"""The scope predicate rho (plan §4.1). ONE implementation, used by retrieval, ScopeGuard and the scorer.

Semantics (t = evidence time on the contract's basis, i = index time):
  pre  : t <  i                       [and t >= i - window if a window is set]
  at   : t <= i + at_tolerance        [and t >= i - window if a window is set]
         or the evidence id is one of the contract's index_evidence_ids
  post : t >  i                       [and t <= i + window if a window is set]
  full : always (subject to the source filter)
An evidence object with no timestamp on the governing basis is illegal for
pre/at/post (conservative exclusion) and legal for full.
Source filter: if permitted_sources is set, evidence.source must be in it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from .schema import Contract, EvidenceObject


def legal(evidence: EvidenceObject, contract: Contract) -> bool:
    if contract.permitted_sources is not None and evidence.source not in contract.permitted_sources:
        return False
    if contract.scope == "full":
        return True
    if evidence.id in contract.index_evidence_ids and contract.scope == "at":
        return True
    t = evidence.time(contract.basis)
    if t is None:
        return False
    i = contract.index
    window = timedelta(hours=contract.window_hours) if contract.window_hours is not None else None
    if contract.strict_boundaries:
        # sensitivity variant (plan §8.5): every inclusive edge becomes exclusive
        if contract.scope == "pre":
            return t < i and (window is None or t > i - window)
        if contract.scope == "at":
            # "<=" -> "<" at the index: with tolerance 0 the index-time record itself is excluded (unless it is an index event)
            return t < i + timedelta(hours=contract.at_tolerance_hours) and (window is None or t > i - window)
        if contract.scope == "post":
            return t > i and (window is None or t < i + window)
        raise ValueError(contract.scope)
    if contract.scope == "pre":
        return t < i and (window is None or t >= i - window)
    if contract.scope == "at":
        return t <= i + timedelta(hours=contract.at_tolerance_hours) and (window is None or t >= i - window)
    if contract.scope == "post":
        return t > i and (window is None or t <= i + window)
    raise ValueError(contract.scope)


def legal_set(pool: Iterable[EvidenceObject], contract: Contract) -> set[str]:
    return {e.id for e in pool if legal(e, contract)}


def filter_legal(pool: Iterable[EvidenceObject], contract: Contract) -> list[EvidenceObject]:
    return [e for e in pool if legal(e, contract)]


def wrong_basis(contract: Contract) -> Contract:
    """The wrong-basis control: occurrence time where availability governs, and vice versa."""
    swap = {"available": "occurred", "occurred": "available", "documented": "occurred"}
    return contract.with_basis(swap[contract.basis])
