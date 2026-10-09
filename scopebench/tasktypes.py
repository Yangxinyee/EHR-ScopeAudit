"""Task-type registry (plan §4.3/§5.3): prompts, output schema checks, citations, answer scoring.

A TaskType is the single place that knows a task family's output schema. It is used by the
prompt builder, ScopeGuard (schema and skill checks), the scorer, and the oracle/skill spec.
Tier-B/Tier-C families are added in M2/M5; `lab_trend_k3` and `med_dose_k3` reproduce the
frozen ScopePair contracts verbatim (system prompts hashed against the freeze records).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .predicate import legal
from .schema import EvidenceObject, Task, parse_time

Payload = dict[str, Any]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class TaskType:
    name: str
    system_prompt: str
    skill_op: str | None  # operation name in scopebench/SKILL_SPEC.md; None = uncovered
    max_tokens: int
    build_user_payload: Callable[[Task, list[EvidenceObject], Payload | None], str]
    validate: Callable[[Payload | None, Task, list[EvidenceObject]], list[str]]
    citations: Callable[[Payload | None], list[str]]
    substantive: Callable[[Payload | None], bool]
    atomic_fields: Callable[[Payload | None], list[dict[str, Any]]]
    answer_correct: Callable[[Payload | None, Task], bool]
    skill_answer: Callable[[Payload, Task, list[EvidenceObject]], Payload] | None = None
    skill_disagreements: Callable[[Payload | None, Payload, Task], list[str]] | None = None
    secondary: Callable[[Payload | None, Task], dict[str, Any]] = lambda parsed, task: {}
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def covered(self) -> bool:
        return self.skill_op is not None

    @property
    def system_prompt_sha256(self) -> str:
        return sha256_text(self.system_prompt)


REGISTRY: dict[str, TaskType] = {}


def register(task_type: TaskType) -> TaskType:
    REGISTRY[task_type.name] = task_type
    return task_type


def get(name: str) -> TaskType:
    return REGISTRY[name]


# ----------------------------------------------------------------------------------------------
# ScopePair-FHIR (LabTrend): verbatim contract from scripts/run_scopepair_fhir.py, frozen in
# experiments/SCOPEPAIR_FHIR_FORMAL_FREEZE.json (system prompt sha256 6ab83151...).
# ----------------------------------------------------------------------------------------------

LAB_TREND_SYSTEM_PROMPT = """You are an EHR evidence extraction agent. The task contract is binding.

The evidence field `issued_time` is when a laboratory result became available.
The field `effective_time` is when the specimen/event occurred and MUST NOT be
used to decide availability for this task. Use only the supplied evidence.

If exactly three supplied observations satisfy the requested pre/post contract,
return them in chronological `issued_time` order and calculate delta as the last
value minus the first value. Direction is `increasing`, `decreasing`, or `stable`.
If fewer than three supplied observations satisfy the contract, return status
`insufficient`, an empty results list, null delta, and null direction. Never
invent, replace, or cite an observation that is not supplied.

Return exactly one JSON object and no prose:
{"status":"ok|insufficient","results":[{"citation":"id","value":0.0,
"unit":"unit","issued_time":"ISO timestamp"}],"delta":0.0,
"direction":"increasing|decreasing|stable|null"}"""


def _lab_evidence_view(e: EvidenceObject) -> dict[str, Any]:
    # The exact candidate dictionary ScopePair placed in the prompt (sorted keys at dump time).
    return {
        "display": e.fields.get("display"),
        "effective_time": e.occurred,
        "id": e.id,
        "issued_time": e.available,
        "unit": e.fields.get("unit"),
        "value": e.fields.get("value"),
    }


def _lab_build_payload(task: Task, evidence: list[EvidenceObject], skill_payload: Payload | None) -> str:
    payload: dict[str, Any] = {
        "question": task.question,
        "contract": {
            "scope": task.contract.scope,
            "index_time": task.contract.index_time,
            "timestamp_basis": "issued_time",
            "required_result_count": 3,
        },
        "supplied_evidence": [_lab_evidence_view(e) for e in evidence],
    }
    if skill_payload is not None:
        payload["deterministic_chartskill"] = skill_payload
        payload["skill_instruction"] = (
            "The deterministic skill is provenance-preserving. Verify its cited "
            "records against supplied evidence and use it to format the answer."
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _lab_citations(parsed: Payload | None) -> list[str]:
    if not parsed:
        return []
    results = parsed.get("results")
    if not isinstance(results, list):
        return []
    return [str(r.get("citation")) for r in results if isinstance(r, dict) and r.get("citation") is not None]


def _lab_validate(parsed: Payload | None, task: Task, evidence: list[EvidenceObject]) -> list[str]:
    """Structural checks that do not depend on the predicate (ScopePair's schema rules)."""
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors: list[str] = []
    status = parsed.get("status")
    if status not in {"ok", "insufficient"}:
        errors.append("status must be ok or insufficient")
    results = parsed.get("results")
    if not isinstance(results, list):
        errors.append("results must be a list")
        results = []
    citations = _lab_citations(parsed)
    if status == "ok" and (len(results) != 3 or len(set(citations)) != 3):
        errors.append("an ok answer must contain three distinct cited results")
    legal_supplied = [e for e in evidence if legal(e, task.contract)]
    if status == "insufficient" and len(legal_supplied) >= 3:
        errors.append("three permitted supplied observations exist; do not abstain")
    return errors


def _lab_skill_summary(evidence: list[EvidenceObject]) -> Payload:
    """ScopePair's deterministic ChartSkill over the (already legal, chronologically ordered) evidence."""
    if len(evidence) != 3:
        return {"status": "insufficient", "provenance": []}
    first_value = float(evidence[0].fields["value"])
    last_value = float(evidence[-1].fields["value"])
    delta = last_value - first_value
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        direction = "stable"
    elif delta > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return {
        "status": "ok",
        "ordered_result_ids": [e.id for e in evidence],
        "delta_last_minus_first": delta,
        "direction": direction,
        "provenance": [
            {"citation": e.id, "value": e.fields["value"], "unit": e.fields["unit"], "issued_time": e.available}
            for e in evidence
        ],
    }


def _lab_skill_answer(skill_summary: Payload, task: Task, evidence: list[EvidenceObject]) -> Payload:
    if skill_summary.get("status") != "ok":
        return {"status": "insufficient", "results": [], "delta": None, "direction": None}
    return {
        "status": "ok",
        "results": skill_summary["provenance"],
        "delta": skill_summary["delta_last_minus_first"],
        "direction": skill_summary["direction"],
    }


def _close(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-7)
    except (TypeError, ValueError):
        return False


def _lab_skill_disagreements(parsed: Payload | None, skill_summary: Payload, task: Task) -> list[str]:
    """ScopePair's validate_skill block, verbatim in semantics."""
    if parsed is None or parsed.get("status") != "ok" or skill_summary.get("status") != "ok":
        return []
    errors: list[str] = []
    results = parsed.get("results") if isinstance(parsed.get("results"), list) else []
    expected_results = skill_summary["provenance"]
    if len(results) == 3:
        for index, (observed, gold) in enumerate(zip(results, expected_results), start=1):
            if not isinstance(observed, dict):
                errors.append(f"result {index} must be an object")
                continue
            if observed.get("citation") != gold["citation"]:
                errors.append(f"result {index} citation/order disagrees with ChartSkill")
            if observed.get("unit") != gold["unit"]:
                errors.append(f"result {index} unit disagrees with ChartSkill")
            if not _close(observed.get("value"), gold["value"]):
                errors.append(f"result {index} value disagrees with ChartSkill")
            try:
                time_matches = parse_time(str(observed.get("issued_time"))) == parse_time(str(gold["issued_time"]))
            except (TypeError, ValueError):
                time_matches = False
            if not time_matches:
                errors.append(f"result {index} availability time disagrees with ChartSkill")
    if not _close(parsed.get("delta"), skill_summary["delta_last_minus_first"]):
        errors.append("delta disagrees with deterministic ChartSkill")
    if parsed.get("direction") != skill_summary["direction"]:
        errors.append("direction disagrees with deterministic ChartSkill")
    return errors


def _lab_answer_correct(parsed: Payload | None, task: Task) -> bool:
    if parsed is None:
        return False
    gold = task.gold
    gold_results = gold.get("results") or []
    if parsed.get("status") != "ok" or not gold_results:
        return parsed.get("status") == "insufficient" and not gold_results
    if _lab_citations(parsed) != [r["citation"] for r in gold_results]:
        return False
    results = parsed.get("results")
    if not isinstance(results, list) or len(results) != len(gold_results):
        return False
    for observed, g in zip(results, gold_results):
        if not isinstance(observed, dict) or not _close(observed.get("value"), g.get("value")):
            return False
    return _close(parsed.get("delta"), gold.get("delta")) and parsed.get("direction") == gold.get("direction")


def _lab_atomic_fields(parsed: Payload | None) -> list[dict[str, Any]]:
    if not parsed or not isinstance(parsed.get("results"), list):
        return []
    out = []
    for r in parsed["results"]:
        if isinstance(r, dict):
            for f in ("value", "unit", "issued_time"):
                out.append({"field": f, "value": r.get(f), "citation": r.get("citation")})
    out.append({"field": "delta", "value": parsed.get("delta"), "citation": None, "operation": "last_minus_first"})
    out.append({"field": "direction", "value": parsed.get("direction"), "citation": None, "operation": "sign"})
    return out


LAB_TREND_K3 = register(
    TaskType(
        name="lab_trend_k3",
        system_prompt=LAB_TREND_SYSTEM_PROMPT,
        skill_op="lab_trend_k3",
        max_tokens=384,
        build_user_payload=_lab_build_payload,
        validate=_lab_validate,
        citations=_lab_citations,
        substantive=lambda p: bool(p) and p.get("status") == "ok",
        atomic_fields=_lab_atomic_fields,
        answer_correct=_lab_answer_correct,
        skill_answer=_lab_skill_answer,
        skill_disagreements=_lab_skill_disagreements,
        notes={"origin": "ScopePair-FHIR LabTrend", "frozen_system_prompt_sha256": "6ab83151580ed764a83482afdee8400897dc8b2262fec8d019ae73b2de2cfa70"},
    )
)

assert LAB_TREND_K3.system_prompt_sha256 == LAB_TREND_K3.notes["frozen_system_prompt_sha256"], "LabTrend system prompt drifted from the freeze record"

# Reference skill for the regression only; the formal skill library is scopebench/skills (subagent B).
REFERENCE_SKILLS: dict[str, Callable[[list[EvidenceObject]], Payload]] = {"lab_trend_k3": _lab_skill_summary}
