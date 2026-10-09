"""Per-task endpoints (plan §4.4 / paper §3.2):
answer_correct, completeness |C∩R|/|R|, access_legal [A⊆L], citation_legal [C≠∅ ∧ C⊆L],
auditable (every atomic field copied from a resolved citation or a declared operation),
abstained, workflow_compliant, plus tokens/calls/latency.
"""

from __future__ import annotations

from typing import Any

from .predicate import legal
from .schema import EvidenceObject, Task
from .tasktypes import TaskType

Payload = dict[str, Any]


def score(
    parsed: Payload | None,
    task: Task,
    task_type: TaskType,
    pool: dict[str, EvidenceObject],
    access_ids: set[str],
    *,
    tokens_in: int,
    tokens_out: int,
    calls: int,
    latency: float,
) -> dict[str, Any]:
    contract = task.contract
    citations = task_type.citations(parsed)
    cset = set(citations)
    resolved = {c for c in cset if c in pool}
    legal_c = {c for c in resolved if legal(pool[c], contract)}
    substantive = task_type.substantive(parsed)
    abstained = parsed is not None and not substantive
    parse_failed = parsed is None
    required = set(task.required_ids)
    completeness = (len(cset & required) / len(required)) if required else (1.0 if substantive else 0.0)
    access_legal = all(legal(pool[i], contract) for i in access_ids if i in pool)
    citation_legal = bool(substantive and cset and cset == legal_c)
    answer_correct = bool(task_type.answer_correct(parsed, task))
    auditable = bool(substantive and cset and cset == resolved and _fields_traceable(task_type.atomic_fields(parsed), resolved))
    return {
        "parse_failed": parse_failed,
        "abstained": abstained,
        "substantive": substantive,
        "answer_correct": answer_correct,
        "completeness": completeness,
        "access_legal": access_legal,
        "citation_legal": citation_legal,
        "auditable": auditable,
        "workflow_compliant": bool(substantive and access_legal and citation_legal),
        "n_citations": len(cset),
        "n_illegal_citations": len(resolved - legal_c),
        "n_unresolved_citations": len(cset - resolved),
        "n_illegal_access": sum(1 for i in access_ids if i in pool and not legal(pool[i], contract)),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "calls": calls,
        "latency_seconds": latency,
        **task_type.secondary(parsed, task),
    }


def _fields_traceable(fields: list[dict[str, Any]], resolved: set[str]) -> bool:
    for f in fields:
        if f.get("citation") is not None:
            if f["citation"] not in resolved:
                return False
        elif not f.get("operation"):
            return False
    return True


def late_filter(parsed: Payload | None, task: Task, task_type: TaskType, pool: dict[str, EvidenceObject]) -> Payload | None:
    """The late-filtering control: drop illegal citations from a prompt-only answer, no refill.
    Task types whose results are per-citation objects lose those objects; others keep the answer."""
    if parsed is None:
        return None
    illegal = {c for c in task_type.citations(parsed) if c not in pool or not legal(pool[c], task.contract)}
    if not illegal:
        return parsed

    def strip(x):
        # per-citation objects ({"citation": id, ...}) are dropped whole; bare id strings are dropped from lists;
        # everything else is kept unchanged
        if isinstance(x, dict):
            if isinstance(x.get("citation"), str) and x["citation"] in illegal:
                return None
            kept = {}
            for k, v in x.items():
                sv = strip(v)
                if sv is not None or v is None:
                    kept[k] = sv
            return kept
        if isinstance(x, list):
            return [v for v in (strip(i) for i in x if not (isinstance(i, str) and i in illegal)) if v is not None]
        return None if (isinstance(x, str) and x in illegal) else x

    out = strip(parsed)
    results = parsed.get("results")
    if isinstance(results, list) and len(out.get("results", [])) != len(results) and task_type.name.endswith("_k3"):
        # a k-of-3 answer with a removed member is no longer a valid substantive answer
        out["status"] = "insufficient"
        out["delta"] = None
        out["direction"] = None
    return out
