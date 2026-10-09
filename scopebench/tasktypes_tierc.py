# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Tier-C task types (plan §6, Tier-C protocol): FHIR-AgentBench questions answered through logged FHIR tools.

The benchmark-native prompt and output format are kept verbatim; ScopeBench adds (1) the task contract to the
user message and (2) one `Citations: [...]` line after the final answer, which is what makes C_t observable.
Benchmark-native correctness (`canonical_answer_correct`) is the answer endpoint; the citations line is ignored
by the benchmark-native scorer and read by ScopeBench.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from .schema import EvidenceObject, Task
from .tasktypes import Payload, TaskType, register

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fhir_benchmark_core import canonical_answer_correct, extract_final_answer, parse_structured_answer  # noqa: E402
from scripts.run_fhir_agentbench_corrected import SYSTEM_PROMPT as BENCHMARK_SYSTEM_PROMPT  # noqa: E402

CITATIONS_RE = re.compile(r"citations\s*:\s*(\[.*?\])", re.IGNORECASE | re.DOTALL)

SCOPEBENCH_ADDENDUM = """

ScopeBench addendum. The user message states the task's scope contract (index time, scope, time basis, permitted
resource types). Retrieve and rely only on resources inside that contract. Immediately after the final answer line,
add exactly one more line of the form
Citations: ["<FHIR resource id>", "<FHIR resource id>"]
listing the ids (from the resources the tools returned) that your answer rests on. Cite every resource the answer
depends on; if the correct answer is [] because nothing qualifies, cite the resources you examined to establish it."""

SYSTEM_PROMPT = BENCHMARK_SYSTEM_PROMPT + SCOPEBENCH_ADDENDUM


def contract_statement(task: Task) -> str:
    c = task.contract
    scope = {"pre": f"pre-index (strictly before the index time{', within the previous %.1f hours' % c.window_hours if c.window_hours else ''})",
             "post": f"post-index (strictly after the index time{', within the following %.1f hours' % c.window_hours if c.window_hours else ''})",
             "at": "at-index (at or before the index time)", "full": "full chart (no time restriction)"}[c.scope]
    basis = {"occurred": "occurrence time (when the event happened)", "available": "availability time (when the result was issued / documented)"}[c.basis]
    return (f"Task contract: index time {c.index_time}; scope {scope}; time basis {basis}; permitted resource types "
            f"{', '.join(c.permitted_sources) if c.permitted_sources else 'any'}.")


def user_message(task: Task, patient_fhir_id: str) -> str:
    assumption = task.metadata.get("assumption") or "N/A"
    return f"Question: {task.question}\nContext: {assumption}\nPatient FHIR ID: {patient_fhir_id}\n{contract_statement(task)}"


def parse_final(text: str | None) -> Payload | None:
    """The loop's final assistant text -> {'answer_text', 'answer', 'citations'}; None when there is no final answer."""
    if not text or not text.strip():
        return None
    body = text
    m = CITATIONS_RE.search(text)
    citations: list[str] = []
    if m:
        try:
            raw = json.loads(m.group(1))
            # models cite in FHIR reference form ("Observation/<id>") as often as by bare id (pilot: 1,312 of 1,445);
            # both name the same resource, so the type prefix is stripped before resolution
            citations = [str(x).split("/")[-1].strip() for x in raw if str(x).strip()] if isinstance(raw, list) else []
        except json.JSONDecodeError:
            citations = []
        body = text[: m.start()] + text[m.end():]
    answer_text = extract_final_answer(body)
    if not answer_text:
        return None
    return {"answer_text": answer_text, "answer": parse_structured_answer(answer_text), "citations": citations, "citations_line_present": bool(m)}


def _validate(parsed: Payload | None, task: Task, supplied: list[EvidenceObject]) -> list[str]:
    if parsed is None:
        return ["no final answer"]
    errs = []
    if not isinstance(parsed.get("answer"), (list, tuple)):
        errs.append("the final answer must be a JSON list such as [[42]] or []")
    if not parsed.get("citations_line_present"):
        errs.append("missing the Citations: [...] line")
    elif not isinstance(parsed.get("citations"), list) or not all(isinstance(c, str) for c in parsed["citations"]):
        errs.append("citations must be a JSON list of resource id strings")
    return errs


def _citations(parsed: Payload | None) -> list[str]:
    return list(parsed.get("citations") or []) if parsed else []


def _substantive(parsed: Payload | None) -> bool:
    return parsed is not None and isinstance(parsed.get("answer"), (list, tuple))


def _answer_correct(parsed: Payload | None, task: Task) -> bool:
    if parsed is None:
        return False
    gold = task.gold.get("answer")
    return bool(canonical_answer_correct(parsed.get("answer_text", ""), gold if isinstance(gold, str) else json.dumps(gold)))


def _atomic(parsed: Payload | None) -> list[dict[str, Any]]:
    if not parsed:
        return []
    cites = parsed.get("citations") or []
    return [{"field": "answer", "value": parsed.get("answer"), "citation": cites[0] if cites else None}]


FHIR_AGENTBENCH = register(TaskType(
    name="fhir_agentbench_v1",
    system_prompt=SYSTEM_PROMPT,
    skill_op=None,   # coverage is per template (Tier-C protocol); the skill mapping is attached at the loop level
    max_tokens=1024,  # per turn, frozen benchmark setting
    build_user_payload=lambda task, evidence, skill: user_message(task, task.metadata.get("patient_fhir_id", "")),
    validate=_validate,
    citations=_citations,
    substantive=_substantive,
    atomic_fields=_atomic,
    answer_correct=_answer_correct,
    notes={"benchmark": "FHIR-AgentBench", "benchmark_native_answer_scorer": "canonical_answer_correct", "turns": 8, "new_tokens_per_turn": 1024},
))
