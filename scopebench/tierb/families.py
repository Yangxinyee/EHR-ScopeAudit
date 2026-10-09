"""Family builders (plan §5.3): anchor -> Case + two Tasks (one scope pair), gold via the oracle package.

Each builder returns (Case, [Task, Task]) or None when the anchor cannot yield a well-formed pair
(e.g. fewer than k legal objects in a scope after loading the full evidence). The pool is the
case's candidate evidence: what prompt-only sees in full and what scoped retrieval filters.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from .. import oracle
from ..predicate import filter_legal
from ..schema import Case, Contract, EvidenceObject, Task, parse_time, stable_hex
from .sampling import LAB_ITEMS, Anchor
from .sources import SourceLoader, hash_patient

Builder = Any


def _case_id(family: str, anchor: Anchor) -> str:
    return stable_hex("case", family, anchor.subject_id, *anchor.key, n=20)


def _task_id(case_id: str, scope: str) -> str:
    return f"{case_id}-{scope}"


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ F1 evidence retrieval (LabTrend on Tier-B)
def build_retrieval(loader: SourceLoader, anchor: Anchor, split: str) -> tuple[Case, list[Task]] | None:
    itemid = anchor.payload["itemid"]
    index_time = parse_time(anchor.payload["index_time"].replace(" ", "T") + ("Z" if len(anchor.payload["index_time"]) == 19 else ""))
    labs = loader.labevents(anchor.subject_id, [itemid])
    hadm = None
    # keep the admission's results plus a margin, so the pool is the patient's concept history around the index
    lo, hi = index_time - timedelta(days=30), index_time + timedelta(days=30)
    pool = [e for e in labs if e.time("available") is not None and lo <= e.time("available") <= hi and e.fields.get("value") is not None]
    if len(pool) < 6 or len(pool) > 60:
        return None
    case_id = _case_id("evidence_retrieval", anchor)
    case = Case(case_id=case_id, patient=hash_patient(anchor.subject_id), index_event={"type": "icu_transfer", "index_time": _iso(index_time)}, evidence=pool)
    concept = LAB_ITEMS[itemid]
    unit = next((e.fields.get("unit") for e in pool if e.fields.get("unit")), None)
    tasks = []
    for scope in ("pre", "post"):
        contract = Contract(index_time=_iso(index_time), scope=scope, basis="available", permitted_sources=("labevents",))
        legal_pool = filter_legal(pool, contract)
        gold = oracle.OPERATIONS["lab_trend_k3"](legal_pool, contract, {"k": 3})
        if gold["status"] != "ok":
            return None
        side = "strictly before" if scope == "pre" else "strictly after"
        question = (f"Return the three {concept} results whose results became available nearest to but {side} {_iso(index_time)}. "
                    f"Order them chronologically, then report the last-minus-first numeric change and whether it is increasing, decreasing, or stable.")
        gold_answer = {"status": "ok", "results": gold["provenance"], "delta": gold["delta_last_minus_first"], "direction": gold["direction"]}
        tasks.append(Task(task_id=_task_id(case_id, scope), case_id=case_id, patient=case.patient, tier="B", task_type="lab_trend_k3",
                          question=question, contract=contract, gold=gold_answer, required_ids=tuple(gold["ordered_result_ids"]), split=split,
                          metadata={"family": "evidence_retrieval", "concept": concept, "itemid": itemid, "unit": unit, "skill_params": {"k": 3}}))
    # leakage: a pre-index answer that changes when availability is replaced by occurrence time
    wb = Contract(index_time=_iso(index_time), scope="pre", basis="occurred", permitted_sources=("labevents",))
    wb_gold = oracle.OPERATIONS["lab_trend_k3"](filter_legal(pool, wb), wb, {"k": 3})
    leak = wb_gold["status"] == "ok" and tuple(wb_gold.get("ordered_result_ids") or ()) != tuple(tasks[0].required_ids)
    for t in tasks:
        t.leakage = bool(leak)
    return case, tasks


from .families_notes import build_claim, build_reconstruction, build_summarization  # noqa: E402
from .families_structured import build_cohort, build_outcome  # noqa: E402

BUILDERS: dict[str, Builder] = {
    "evidence_retrieval": build_retrieval,
    "outcome_abstraction": build_outcome,
    "cohort_abstraction": build_cohort,
    "claim_verification": build_claim,
    "report_reconstruction": build_reconstruction,
    "summarization": build_summarization,
}
