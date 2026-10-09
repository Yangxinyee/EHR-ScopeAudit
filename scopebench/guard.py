"""ScopeGuard (plan §4.3): citation resolution, access scope, citation scope, completeness, schema,
skill agreement; one repair; deterministic replacement. Every outcome is a logged stage.

The guard never edits an answer. It reports; the runner decides (repair, replace) per configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .predicate import legal
from .schema import EvidenceObject, Task
from .tasktypes import TaskType

Payload = dict[str, Any]


@dataclass
class GuardReport:
    parseable: bool
    schema_errors: list[str] = field(default_factory=list)
    unresolved_citations: list[str] = field(default_factory=list)   # not in the case pool
    unsupplied_citations: list[str] = field(default_factory=list)   # in pool, not shown to the model
    illegal_citations: list[str] = field(default_factory=list)      # shown/known but outside L_t
    illegal_access: list[str] = field(default_factory=list)         # A_t \ L_t
    missing_required: list[str] = field(default_factory=list)       # R_t \ C_t (only when checked)
    skill_disagreements: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    abstained_with_evidence: bool = False

    @property
    def scope_errors(self) -> list[str]:
        out = []
        if self.unresolved_citations:
            out.append(f"citations that resolve to no evidence object: {self.unresolved_citations}")
        if self.unsupplied_citations:
            out.append(f"citations not present in supplied evidence: {self.unsupplied_citations}")
        if self.illegal_citations:
            out.append(f"out-of-contract citations: {self.illegal_citations}")
        return out

    @property
    def completeness_errors(self) -> list[str]:
        if not self.missing_required:
            return []
        if self.abstained_with_evidence:
            return [f"required evidence is supplied; do not abstain: {self.missing_required}"]
        return [f"required evidence not cited: {self.missing_required}"]

    def errors(self, *, schema: bool = True, scope: bool = True, completeness: bool = True, skill: bool = True) -> list[str]:
        out: list[str] = []
        if not self.parseable:
            return ["submission is not one parseable JSON object"]
        if schema:
            out += self.schema_errors
        if scope:
            out += self.scope_errors
        if completeness:
            out += self.completeness_errors
        if skill:
            out += self.skill_disagreements
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "parseable": self.parseable,
            "schema_errors": self.schema_errors,
            "unresolved_citations": self.unresolved_citations,
            "unsupplied_citations": self.unsupplied_citations,
            "illegal_citations": self.illegal_citations,
            "illegal_access": self.illegal_access,
            "missing_required": self.missing_required,
            "skill_disagreements": self.skill_disagreements,
            "citations": self.citations,
        }


def check(
    parsed: Payload | None,
    task: Task,
    task_type: TaskType,
    pool: dict[str, EvidenceObject],
    supplied: list[EvidenceObject],
    access_ids: set[str],
    *,
    check_completeness: bool,
    skill_summary: Payload | None = None,
) -> GuardReport:
    if parsed is None:
        return GuardReport(parseable=False)
    report = GuardReport(parseable=True)
    report.schema_errors = task_type.validate(parsed, task, supplied)
    citations = task_type.citations(parsed)
    report.citations = citations
    supplied_ids = {e.id for e in supplied}
    contract = task.contract
    for cid in sorted(set(citations)):
        if cid not in pool:
            report.unresolved_citations.append(cid)
        elif cid not in supplied_ids:
            report.unsupplied_citations.append(cid)
            if not legal(pool[cid], contract):
                report.illegal_citations.append(cid)
        elif not legal(pool[cid], contract):
            report.illegal_citations.append(cid)
    report.illegal_access = sorted(i for i in access_ids if i in pool and not legal(pool[i], contract))
    if check_completeness:
        if task_type.substantive(parsed):
            report.missing_required = sorted(set(task.required_ids) - set(citations))
        elif task.required_ids and set(task.required_ids) <= supplied_ids:
            # abstention although every required record was supplied: the answer is available (ScopePair rule generalized)
            report.missing_required = sorted(task.required_ids)
            report.abstained_with_evidence = True
    if skill_summary is not None and task_type.skill_disagreements is not None:
        report.skill_disagreements = task_type.skill_disagreements(parsed, skill_summary, task)
    return report


REPAIR_WORDINGS: dict[str, dict[str, str]] = {
    # wording "a" is the frozen reference; "b" is the alternative repair wording of the sensitivity suite (plan §8.5)
    "a": {
        "generic": "Re-check your submission and return exactly the required JSON object.",
        "schema_only": "Repair the submission's format and return exactly the required JSON object.",
        "feedback_no_evidence": "Repair the submission using only the evidence already supplied; cite only contract-permitted evidence and return exactly the required JSON object.",
        "feedback": "Repair the submission. Use only the following contract-permitted evidence and return exactly the required JSON object.",
        # refill (partial re-supply): the earlier evidence stays available; the listed records are the ones the guard
        # named. Saying "use only the following" for a partial list made small models answer "insufficient".
        "feedback_refill": ("Repair the submission. The evidence supplied earlier remains available and permitted; the records listed "
                            "below are also contract-permitted and were required but not cited. Use them together with the earlier "
                            "evidence and return exactly the required JSON object."),
    },
    "b": {
        "generic": "Please review your previous answer once more and resubmit it as the single JSON object the task asks for.",
        "schema_only": "Your previous answer did not match the required format. Resubmit it as the single JSON object the task asks for.",
        "feedback_no_evidence": "Your previous answer failed the checks listed below. Correct it with the evidence you were already given, citing only records the contract permits, and resubmit the single JSON object.",
        "feedback": "Your previous answer failed the checks listed below. Correct it using the permitted records listed here, cite them by id, and resubmit the single JSON object.",
        "feedback_refill": ("Your previous answer failed the checks listed below. The evidence you were given earlier still applies; the records "
                            "listed here are permitted too and were required. Combine them, cite records by id, and resubmit the single JSON object."),
    },
}


def repair_message(
    mode: str,
    errors: list[str],
    permitted_evidence_view: list[dict[str, Any]],
    skill_summary: Payload | None,
    wording: str = "a",
) -> str:
    """mode: 'feedback' (scope/completeness/schema errors + permitted evidence), 'schema_only'
    (schema errors only, no evidence), 'generic' (equal-call placebo: no information)."""
    text = REPAIR_WORDINGS[wording]
    if mode == "generic":
        payload: dict[str, Any] = {"instruction": text["generic"]}
    elif mode == "schema_only":
        payload = {"scopeguard_errors": errors, "instruction": text["schema_only"]}
    elif mode == "feedback_no_evidence":
        payload = {"scopeguard_errors": errors, "instruction": text["feedback_no_evidence"]}
    elif mode == "feedback":
        payload = {"scopeguard_errors": errors, "instruction": text["feedback"], "permitted_evidence": permitted_evidence_view}
    elif mode == "feedback_refill":
        payload = {"scopeguard_errors": errors, "instruction": text["feedback_refill"], "required_permitted_evidence": permitted_evidence_view}
    else:
        raise ValueError(mode)
    if skill_summary is not None and mode == "feedback":
        payload["deterministic_chartskill"] = skill_summary
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
