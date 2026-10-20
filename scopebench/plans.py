"""Evidence plans (DEV-34, experiments/SCOPEBENCH_PLAN_PROTOCOL.md): the auditor without reference inputs.

The reference auditor reads three things derived from each task's reference answer: the required evidence R_t,
the skill parameters, and the skill result computed from them. An evidence plan replaces all three by one object the
model declares itself — the typed chart operation its answer rests on. The auditor executes the plan on the
contract-permitted records:

    R_t                 ->  R̂_t = provenance of the executed plan
    skill parameters    ->  the plan's operation and arguments
    reference result    ->  the executed plan's result

Executing the reference plan (`reference_plan`) reproduces R_t and the reference skill result exactly on every covered
task (scripts/scopebench/plans_offline.py identity), so every difference between the two auditors is the cost of not
knowing the plan.

This module holds the tier-independent pieces and the Tier-B catalog; Tier-C lives in `scopebench.tierc.plans`.
Nothing here reads `task.gold`, `task.required_ids` or `task.metadata["skill_params"]` except `reference_plan`,
which exists for evaluation and the equivalence tests and is never called at inference time. The cohort operation
takes its criteria from the request (`metadata["criteria"]`, shown in every arm's prompt), as the contract is taken
from the request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import EvidenceObject, Task

PLAN_KEY = "evidence_plan"


@dataclass(frozen=True)
class Plan:
    """A normalized evidence plan. `operation is None` means the model declared that no catalog operation applies."""

    operation: str | None
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any] | None:
        return None if self.operation is None else {"operation": self.operation, **self.args}


@dataclass
class PlanReading:
    """What a submission declares. `plan` is None when the declaration is missing or invalid (see `errors`)."""

    present: bool
    plan: Plan | None
    errors: list[str]
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"present": self.present, "plan": self.plan.to_dict() if self.plan else None, "declared_none": bool(self.plan and self.plan.operation is None),
                "errors": self.errors, "raw": self.raw}


@dataclass
class PlanResult:
    """The executed plan. `required` is R̂_t: the provenance of an `ok` result, empty otherwise."""

    plan: Plan
    summary: dict[str, Any]
    required: list[str]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.summary.get("status") == "ok"

    def to_dict(self) -> dict[str, Any]:
        public = {k: v for k, v in self.summary.items() if not k.startswith("_")}
        return {"plan": self.plan.to_dict(), "summary": public, "required": self.required, "error": self.error}


class PlanError(ValueError):
    """An invalid plan; the message is shown to the model in the repair turn."""


# ------------------------------------------------------------------------------------------- Tier-B catalog
OUTCOME_TARGETS: dict[str, dict[str, str]] = {
    # target -> (record field, time basis of the pick): schema knowledge, the same for every task of a source
    "first_careunit": {"field": "careunit", "basis": "occurred"},
    "discharge_disposition": {"field": "discharge_location", "basis": "occurred"},
}
# DEV-34 addendum 2: the CCDA tier stores the disposition in another field (scopebench/tiera/families.py); the operation is
# the same, only the schema differs. Tiers not listed use OUTCOME_TARGETS.
OUTCOME_TARGETS_BY_TIER: dict[str, dict[str, dict[str, str]]] = {
    "A": {"first_careunit": {"field": "careunit", "basis": "occurred"},
          "discharge_disposition": {"field": "admission_type", "basis": "occurred"}},
}


@dataclass(frozen=True)
class Operation:
    """One catalog entry. A Tier-B plan is just an operation name (v0.3): everything the skill needs besides the name is
    fixed by the operation or stated in the request."""

    name: str
    question: str                                             # which questions it answers (shown to the model)
    skill: str                                                # the SKILL_SPEC operation it runs
    params: Callable[[Task], dict[str, Any]]                  # the request -> SKILL_SPEC params


def _outcome(target: str) -> Callable[[Task], dict[str, Any]]:
    return lambda t: {"target": target, **OUTCOME_TARGETS_BY_TIER.get(t.tier, OUTCOME_TARGETS)[target], "drug": None}


TIER_B_OPERATIONS: dict[str, Operation] = {op.name: op for op in (
    Operation("lab_trend_k3", "the three results nearest the index time on the side the contract allows, in time order, and their trend",
              "lab_trend_k3", lambda t: {"k": 3}),
    Operation("first_careunit", "the first care unit the patient was transferred to after the index admission",
              "outcome_lookup", _outcome("first_careunit")),
    Operation("discharge_disposition", "the discharge disposition (discharge location) recorded for the index admission",
              "outcome_lookup", _outcome("discharge_disposition")),
    # the criteria are part of the request: every arm's prompt shows them (`criteria`), so the plan only names the operation
    Operation("cohort_criteria", "a decision on whether the patient meets all of the listed criteria (include / exclude / uncertain)",
              "cohort_criteria", lambda t: {"criteria": [dict(c) for c in (t.metadata.get("criteria") or [])]}),
)}


def normalize(raw: Any, catalog: dict[str, Operation] = TIER_B_OPERATIONS) -> Plan:
    """A submitted plan -> Plan, or PlanError with the reason. Read leniently by fixed rules: `null`, an operation name
    as a bare string, `{"operation": name}`, and the earlier form `{"operation": "outcome_lookup", "target": ...}`."""
    if isinstance(raw, str):
        raw = None if raw.strip().lower() in ("", "null", "none") else {"operation": raw.strip()}
    if raw is None:
        return Plan(None)
    if not isinstance(raw, dict):
        raise PlanError("evidence_plan must be an operation name, an object with `operation`, or null")
    op = raw.get("operation")
    if op == "outcome_lookup" and raw.get("target") in catalog:
        op = raw["target"]
    if op is None or (isinstance(op, str) and op.strip().lower() in ("null", "none")):
        return Plan(None)
    if op not in catalog:
        raise PlanError(f"unknown operation {op!r}; use one of {', '.join(catalog)} or null")
    return Plan(op, {})


def read(parsed: dict[str, Any] | None) -> PlanReading:
    """The plan a Tier-B submission declares (key `evidence_plan` of the answer object)."""
    if not isinstance(parsed, dict) or PLAN_KEY not in parsed:
        return PlanReading(False, None, [f"the submission must include `{PLAN_KEY}` (an operation from the catalog, or null)"])
    raw = parsed.get(PLAN_KEY)
    try:
        return PlanReading(True, normalize(raw), [], raw)
    except PlanError as exc:
        return PlanReading(True, None, [f"invalid {PLAN_KEY}: {exc}"], raw)


def execute(plan: Plan, evidence: list[EvidenceObject], task: Task, operations: dict[str, Callable[..., dict[str, Any]]]) -> PlanResult | None:
    """Run a Tier-B plan on the contract-permitted evidence (chronological). None for a plan that declares no
    operation. `operations` is the inference-time skill package (`scopebench.skills.OPERATIONS`)."""
    if plan.operation is None:
        return None
    op = TIER_B_OPERATIONS[plan.operation]
    try:
        summary = operations[op.skill](evidence, task.contract, op.params(task))
    except (ValueError, KeyError, TypeError) as exc:
        return PlanResult(plan, {"status": "error"}, [], error=f"{type(exc).__name__}: {exc}")
    required = provenance_ids(summary) if summary.get("status") == "ok" else []
    return PlanResult(plan, summary, required)


def provenance_ids(summary: dict[str, Any]) -> list[str]:
    """Evidence ids of a skill result's provenance (entries are ids, or dicts with a `citation`, as in lab_trend_k3)."""
    return [str(p["citation"]) if isinstance(p, dict) else str(p) for p in (summary.get("provenance") or [])]


def reference_plan(task: Task) -> Plan:
    """The plan the reference system implicitly used (evaluation and equivalence tests only; never at inference)."""
    sp = task.metadata.get("skill_params") or {}
    if task.task_type == "lab_trend_k3":
        return Plan("lab_trend_k3", {})
    if task.task_type == "outcome_lookup_v1":
        return Plan(sp["target"], {})
    if task.task_type == "cohort_v1":
        return Plan("cohort_criteria", {})
    return Plan(None)


def plan_accuracy(pred: Plan | None, ref: Plan) -> dict[str, Any]:
    """Field-level agreement of a declared plan with the reference plan."""
    if pred is None:
        return {"valid": False, "operation": False, "args": False, "exact": False}
    op = pred.operation == ref.operation
    args = op and json.dumps(pred.args, sort_keys=True) == json.dumps(ref.args, sort_keys=True)
    return {"valid": True, "operation": op, "args": args, "exact": op and args}


def required_overlap(pred: list[str], ref: list[str]) -> dict[str, Any]:
    """R̂_t against R_t (precision / recall; both None when the respective set is empty)."""
    p, r = set(pred), set(ref)
    inter = len(p & r)
    return {"n_pred": len(p), "n_ref": len(r), "precision": inter / len(p) if p else None, "recall": inter / len(r) if r else None, "exact": p == r}


# ------------------------------------------------------------------------------------------- prompt text
def tier_b_addendum() -> str:
    lines = [
        "",
        "Evidence plan. Add one more key to your JSON object: \"evidence_plan\": {\"operation\": \"<name>\"} naming the",
        "deterministic chart operation that computes what the question asks, or \"evidence_plan\": null when no operation",
        "does (for example free-text report tasks). The auditor runs the operation on the contract-permitted records; it",
        "checks that you cite the records the operation uses and that your answer agrees with its result. Operations:",
    ]
    for op in TIER_B_OPERATIONS.values():
        lines.append(f"- \"{op.name}\": questions asking for {op.question}")
    return "\n".join(lines)


TIER_B_ADDENDUM = tier_b_addendum()
