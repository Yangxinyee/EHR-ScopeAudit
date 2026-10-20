# SPDX-License-Identifier: MIT
# MedAgentBench-related adaptation; Copyright (c) 2025 Stanford Machine Learning Group.
# Upstream license: licenses/MedAgentBench-MIT.txt; changes: THIRD_PARTY_NOTICES.md
"""MedAgentBench evidence plans (DEV-34, experiments/SCOPEBENCH_PLAN_PROTOCOL.md): the reference skill's query
computations as operations a model declares.

The model names the operation its FINISH answer rests on, on one line after the Citations line:

    Plan: {"operation": "latest_lab_in_window", "code": "<code>", "window_hours": <hours>}

or `Plan: null` when no operation computes the answer (lookups, orders). The executor runs the operation on the legal
resources the agent retrieved, with the code paths of the reference skill (`medagent.skill_answer` calls `compute`
with its category's parameters, so there is one implementation). Only `reference_plan` reads the task's category; it
is used for evaluation and the equivalence tests, never at inference time.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from typing import Any

from ..plans import Plan, PlanError, PlanReading, PlanResult
from ..schema import Task, parse_time
from .plans import PLAN_LINE

# operation -> (computation, answer carries the time, arguments)
OPERATIONS: dict[str, tuple[str, bool, tuple[str, ...]]] = {
    "patient_age": ("age", False, ()),
    "latest_lab": ("latest", False, ("code",)),
    "latest_lab_with_time": ("latest", True, ("code",)),
    "latest_lab_in_window": ("latest_in_window", False, ("code", "window_hours")),
    "mean_lab_in_window": ("all_in_window", False, ("code", "window_hours")),
}
_NULL = ("", "null", "none")


# ------------------------------------------------------------------------------------------- the executor
def compute(kind: str, store, index: datetime, *, code: str | None = None, window_hours: float | None = None, with_time: bool = False) -> dict[str, Any] | None:
    """One query computation over the store's retrieved resources (`kind`: age | latest | latest_in_window |
    all_in_window, the R_t rules of CATEGORY_RULES); None for age without a Patient birth date. `index` is naive UTC."""
    if kind in ("latest", "latest_in_window", "all_in_window"):
        obs = [r for r in store.retrieved.values() if r.get("resourceType") == "Observation" and store.is_legal(r)]

        def has_code(r):
            c = r.get("code") or {}
            return (c.get("text") == code) or any(x.get("code") == code for x in c.get("coding", []))
        obs = [r for r in obs if has_code(r) and r.get("effectiveDateTime")]
        if not obs:
            return {"status": "insufficient", "answer": [-1], "provenance": []}
        timed = sorted(((datetime.fromisoformat(r["effectiveDateTime"]).replace(tzinfo=None), str(r["id"]), (r.get("valueQuantity") or {}).get("value"), r["effectiveDateTime"]) for r in obs))
        if kind == "all_in_window":
            lo = index - timedelta(hours=window_hours)
            sel = [t for t in timed if lo < t[0] <= index and t[2] is not None]
            return {"status": "ok", "answer": [sum(t[2] for t in sel) / len(sel) if sel else -1], "provenance": [t[1] for t in sel]}
        if kind == "latest_in_window":
            lo = index - timedelta(hours=window_hours)
            sel = [t for t in timed if lo < t[0] <= index]
        else:
            sel = [t for t in timed if t[0] <= index]
        if not sel:
            return {"status": "ok", "answer": [-1], "provenance": []}
        best = sel[-1]
        if with_time:   # the recorded effectiveDateTime string, as the released grader compares it (task10)
            return {"status": "ok", "answer": [best[2], best[3]], "provenance": [best[1]]}
        return {"status": "ok", "answer": [best[2]], "provenance": [best[1]]}
    if kind == "age":
        pats = [r for r in store.retrieved.values() if r.get("resourceType") == "Patient" and r.get("birthDate")]
        if not pats:
            return None
        p = pats[0]
        dob = datetime.fromisoformat(p["birthDate"])
        age = index.year - dob.year - ((index.month, index.day) < (dob.month, dob.day))
        return {"status": "ok", "answer": [age], "provenance": [str(p["id"])]}
    return None


def execute(plan: Plan | None, store, index_time: str) -> PlanResult | None:
    """Run a declared plan at the context's time; None for no plan or `Plan: null`. R̂_t = the provenance of an ok result."""
    if plan is None or plan.operation is None:
        return None
    kind, with_time, _ = OPERATIONS[plan.operation]
    try:
        summary = compute(kind, store, parse_time(index_time).replace(tzinfo=None), code=plan.args.get("code"), window_hours=plan.args.get("window_hours"),
                          with_time=with_time)
    except (ValueError, KeyError, TypeError) as exc:
        return PlanResult(plan, {"status": "error"}, [], error=f"{type(exc).__name__}: {exc}")
    if summary is None:
        return PlanResult(plan, {"status": "insufficient", "answer": None, "provenance": []}, [])
    return PlanResult(plan, summary, list(summary["provenance"]) if summary["status"] == "ok" else [])


def answers_agree(answer: Any, expected: list[Any]) -> bool:
    """The submitted FINISH list against the executed result: same length, numbers within 1e-6, times by wall clock."""
    from .medagent import wallclock_iso
    if not isinstance(answer, list) or len(answer) != len(expected):
        return False
    for a, e in zip(answer, expected):
        num = [isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, e)]
        if all(num):
            if not abs(a - e) <= 1e-6:
                return False
        elif any(num):
            return False
        elif isinstance(a, str) and isinstance(e, str) and wallclock_iso(a) and wallclock_iso(e):
            if wallclock_iso(a) != wallclock_iso(e):
                return False
        elif a != e:
            return False
    return True


# ------------------------------------------------------------------------------------------- reading plans
def normalize(raw: Any) -> Plan:
    """A declared plan -> Plan, or PlanError with the reason. Read leniently by fixed rules: null / "none", a bare
    operation name, operation names and codes in any case, window_hours as a number or a numeric string."""
    if isinstance(raw, str):
        raw = None if raw.strip().lower() in _NULL else {"operation": raw}
    if raw is None:
        return Plan(None)
    if not isinstance(raw, dict):
        raise PlanError("the plan must be a JSON object with \"operation\", an operation name, or null")
    op = raw.get("operation")
    if op is None or (isinstance(op, str) and op.strip().lower() in _NULL):
        return Plan(None)
    name = op.strip().lower() if isinstance(op, str) else None
    if name not in OPERATIONS:
        raise PlanError(f"unknown operation {op!r}; use one of {', '.join(OPERATIONS)} or null")
    args: dict[str, Any] = {}
    for arg in OPERATIONS[name][2]:
        value = raw.get(arg)
        if arg == "code":
            if not isinstance(value, str) or not value.strip():
                raise PlanError(f"{name} needs \"code\": the observation code")
            args["code"] = value.strip().upper()
        else:
            try:
                hours = float(value) if not isinstance(value, bool) else math.nan
            except (TypeError, ValueError):
                hours = math.nan
            if not (math.isfinite(hours) and hours > 0):
                raise PlanError(f"{name} needs \"window_hours\": a positive number of hours")
            args["window_hours"] = hours
    return Plan(name, args)


def read(text: str | None) -> tuple[str, PlanReading]:
    """Split a final response into (the text without the Plan line, the declared plan). The last Plan line counts."""
    if not text:
        return text or "", PlanReading(False, None, ["add one line Plan: {...} (an operation from the list) or Plan: null after the Citations line"])
    matches = list(PLAN_LINE.finditer(text))
    if not matches:
        return text, PlanReading(False, None, ["add one line Plan: {...} (an operation from the list) or Plan: null after the Citations line"])
    m = matches[-1]
    rest = (text[: m.start()] + text[m.end():]).strip()
    body = m.group(1).strip().strip("`").strip()
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        if not re.fullmatch(r"[A-Za-z_]+", body):
            return rest, PlanReading(True, None, ["the Plan line must be one JSON object, an operation name, or null"], body)
        raw = body
    try:
        return rest, PlanReading(True, normalize(raw), [], raw)
    except PlanError as exc:
        return rest, PlanReading(True, None, [f"invalid plan: {exc}"], raw)


def category_computation(category: str) -> dict[str, Any] | None:
    """The reference skill's computation for a category (CATEGORY_RULES; task10 also returns the time, the answer shapes
    of the released grader, DEV-37); None when uncovered."""
    from .medagent import CATEGORY_RULES
    rule = CATEGORY_RULES[category]
    kind = "age" if category == "task2" else rule["rt"]
    if kind not in ("age", "latest", "latest_in_window", "all_in_window"):
        return None
    return {"kind": kind, "code": rule["code"], "window_hours": rule["window_hours"], "with_time": category == "task10"}


def reference_plan(task: Task) -> Plan:
    """The plan behind the reference skill call (evaluation and equivalence tests only; never at inference). Plan(None)
    for the categories the skill does not cover (task1, task3, task5, task8)."""
    from .medagent import SKILL_COVERED
    cat = task.metadata["category"]
    comp = category_computation(cat)
    if cat not in SKILL_COVERED or comp is None:
        return Plan(None)
    name = next(n for n, (kind, with_time, _) in OPERATIONS.items() if kind == comp["kind"] and with_time == comp["with_time"])
    return Plan(name, {a: comp[a] for a in OPERATIONS[name][2]})


# ------------------------------------------------------------------------------------------- prompt text
PLAN_ADDENDUM = """

Evidence plan. Your final response ends with one more line after the Citations line, naming the computation your
FINISH answer rests on:
Plan: {"operation": "<name>", <arguments>}
or Plan: null when none of the operations below computes your answer (for example when the task asks you to look up an
identifier or only to place an order). The auditor runs the operation on the contract-permitted resources you
retrieved; it checks that you cite the resources the operation uses and that your FINISH answer agrees with its result.
"Now" is the time the context states; <code> is the observation code; <hours> is a number of hours.
- "patient_age" (no arguments): the patient's age in whole years now, from the Patient's birth date; result [age]
- "latest_lab" {"code": "<code>"}: the most recent value of that observation at or before now; result [value]
- "latest_lab_with_time" {"code": "<code>"}: the most recent value at or before now and the time it was recorded;
  result [value, "<time>"]
- "latest_lab_in_window" {"code": "<code>", "window_hours": <hours>}: the most recent value within the last <hours>
  hours; result [value]
- "mean_lab_in_window" {"code": "<code>", "window_hours": <hours>}: the mean of the values within the last <hours>
  hours; result [mean]
When no value qualifies, the result is [-1]."""
