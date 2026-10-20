# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Tier-C evidence plans (DEV-34, experiments/SCOPEBENCH_PLAN_PROTOCOL.md §2): the covered FHIR-AgentBench templates
as operations a model declares.

A Tier-C plan names one catalog template (`experiments/scopebench_tierC/plan_catalog.json`, built from the dev split
by `scripts/scopebench/build_tierc_plan_catalog.py`), fills its value placeholders and states the operators the
question uses. The executor turns the plan into the task specification the template skill expects
(`scopebench.tierc.skills.solve`) and runs it on the contract-permitted resources, exactly as the reference skill
does with the benchmark's own metadata. The model writes the plan on one line after the Citations line:

    Plan: {"template": "C07", "values": {"patient_id": "10016810", "lab_name": "phosphate"}, "extreme": "min", "order": "last"}

or `Plan: null` when no catalog template fits. Only `reference_plan` reads the task's generator metadata; it is used
for evaluation and the equivalence tests, never at inference time.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..plans import Plan, PlanError, PlanReading, PlanResult, provenance_ids
from ..schema import Task

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "experiments" / "scopebench_tierC" / "plan_catalog.json"

# benchmark operator slot -> plan argument, and the values each argument takes
OPERATOR_SLOTS = {"comparison": "comparison", "agg_function": "aggregate", "sort": "extreme", "unit_count": "unit", "time_filter_exact1": "order"}
OPERATOR_VALUES = {"comparison": ("<", ">"), "aggregate": ("max", "min", "avg"), "extreme": ("max", "min"), "unit": ("hours", "days"), "order": ("first", "last")}
_WORD = {"max": "maximum", "min": "minimum", "avg": "average"}
PLAN_LINE = re.compile(r"^[ \t>*`]*plan[ \t*`]*:[ \t]*(.*?)[ \t`]*$", re.IGNORECASE | re.MULTILINE)


@lru_cache(maxsize=4)
def load_catalog(path: str | Path = CATALOG_PATH) -> dict[str, dict[str, Any]]:
    record = json.loads(Path(path).read_text())
    return {e["id"]: e for e in record["templates"]}


# words a model uses for each operator choice (DEV-34 v0.2: the plan is read leniently, by fixed rules, from what the
# model wrote — the benchmark's own slot names and everyday words are accepted; nothing here depends on the task)
OPERATOR_WORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "comparison": {"<": ("<", "less", "lower", "below", "under", "smaller"), ">": (">", "greater", "higher", "above", "over", "more", "larger", "exceed", "exceeds")},
    "aggregate": {"max": ("max", "maximum", "highest", "largest", "peak"), "min": ("min", "minimum", "lowest", "smallest"), "avg": ("avg", "average", "mean")},
    "extreme": {"max": ("max", "maximum", "highest", "largest", "peak"), "min": ("min", "minimum", "lowest", "smallest")},
    "unit": {"hours": ("hours", "hour", "hrs", "hr", "h"), "days": ("days", "day", "d")},
    "order": {"first": ("first", "earliest", "initial", "oldest"), "last": ("last", "latest", "most recent", "recent", "final", "newest")},
}
SLOT_OF = {arg: slot for slot, arg in OPERATOR_SLOTS.items()}   # aggregate -> agg_function, ...
FIXED_VALUES = ("patient_id",)   # the task fixes the patient; the executor never reads it


def _template_id(ref: Any, catalog: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(ref, str):
        return None
    ref = ref.strip()
    if ref in catalog:
        return catalog[ref]
    m = re.fullmatch(r"[Cc]\s*0*(\d{1,3})", ref)   # "C023", "c7"
    if m and f"C{int(m.group(1)):02d}" in catalog:
        return catalog[f"C{int(m.group(1)):02d}"]
    return next((e for e in catalog.values() if e["template"] == ref), None)   # the template text itself


def _choice(arg: str, text: Any) -> str | None:
    """The operator choice a free-form value names, or None when it names none or more than one."""
    if not isinstance(text, (str, int, float)) or isinstance(text, bool):
        return None
    t = str(text).strip().lower()
    words = OPERATOR_WORDS[arg]
    if t in words:
        return t
    hits = {choice for choice, ws in words.items() for w in ws if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t)}
    return hits.pop() if len(hits) == 1 else None


def _operator(arg: str, raw: dict[str, Any], values: dict[str, Any]) -> str | None:
    """Look for an operator under its plan name or the benchmark's slot name, at the top level, in `values` or in an
    `operators` object; the first place that names exactly one choice wins."""
    ops = raw.get("operators") if isinstance(raw.get("operators"), dict) else {}
    for where in (raw, ops, values):
        for key in (arg, SLOT_OF[arg]):
            if key in where:
                c = _choice(arg, where[key])
                if c is not None:
                    return c
    return None


def normalize(raw: Any, catalog: dict[str, dict[str, Any]]) -> Plan:
    if raw is None:
        return Plan(None)
    if not isinstance(raw, dict):
        raise PlanError("the plan must be a JSON object or null")
    entry = _template_id(raw.get("template"), catalog)
    if entry is None:
        raise PlanError(f"unknown template {raw.get('template')!r}; use a catalog id such as C01, or null")
    values = raw.get("values") if isinstance(raw.get("values"), dict) else {}
    needed = [v for v in entry["values"] if v not in FIXED_VALUES]
    missing = [v for v in needed if values.get(v) is None or str(values.get(v)).strip() == ""]
    if missing:
        raise PlanError(f"template {entry['id']} needs values for {', '.join(missing)}")
    args: dict[str, Any] = {"values": {v: str(values[v]).strip() for v in needed}}
    for arg in entry["operators"]:
        c = _operator(arg, raw, values)
        if c is None:
            raise PlanError(f"template {entry['id']} needs {arg} as one of {', '.join(OPERATOR_VALUES[arg])}")
        args[arg] = c
    return Plan(entry["id"], args)


def read(text: str | None, catalog: dict[str, dict[str, Any]]) -> tuple[str, PlanReading]:
    """Split a final answer into (text without the Plan line, the declared plan). The last Plan line counts."""
    if not text:
        return text or "", PlanReading(False, None, ["add one line Plan: {...} (a catalog template) or Plan: null"])
    matches = list(PLAN_LINE.finditer(text))
    if not matches:
        return text, PlanReading(False, None, ["add one line Plan: {...} (a catalog template) or Plan: null after the Citations line"])
    m = matches[-1]
    rest = (text[: m.start()] + text[m.end():]).strip()
    body = m.group(1).strip()
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        return rest, PlanReading(True, None, ["the Plan line must be one JSON object or null"], body)
    try:
        return rest, PlanReading(True, normalize(raw, catalog), [], raw)
    except PlanError as exc:
        return rest, PlanReading(True, None, [f"invalid plan: {exc}"], raw)


def task_spec(plan: Plan, catalog: dict[str, dict[str, Any]], context: str | None) -> dict[str, Any]:
    """The template skill's input for a plan (`solve(task_spec, legal resources)`): the shape of the benchmark's
    metadata, with each operator spelled the way the skill reads it."""
    entry = catalog[plan.operation]
    a = plan.args
    ops: dict[str, Any] = {}
    if "comparison" in a:
        ops["comparison"] = {"sql": a["comparison"]}
    if "aggregate" in a:
        ops["agg_function"] = {"nlq": _WORD[a["aggregate"]]}
    if "extreme" in a:
        ops["sort"] = {"nlq": _WORD[a["extreme"]]}
    if "unit" in a:
        ops["unit_count"] = {"nlq": a["unit"]}
    time = {"time_filter_exact1": {"nlq": a["order"]}} if "order" in a else {}
    return {"template": entry["template"], "main_table": entry["main_table"], "assumption": context,
            "skill_params": {"values": dict(a["values"]), "ops": ops, "time": time}}


def reference_plan(task: Task, catalog: dict[str, dict[str, Any]]) -> Plan:
    """The plan behind the reference skill call (evaluation and equivalence tests only; never at inference)."""
    m = task.metadata
    entry = next((e for e in catalog.values() if e["template"] == m.get("template")), None)
    if entry is None:
        return Plan(None)
    sp = m.get("skill_params") or {}
    ops, time = sp.get("ops") or {}, sp.get("time") or {}

    def word(slot: dict | None, table: dict[str, str]) -> str | None:
        text = " ".join(str(x) for x in ((slot or {}).get("nlq"), (slot or {}).get("sql"), (slot or {}).get("type")) if x).lower()
        return next((v for k, v in table.items() if k in text), None)

    raw: dict[str, Any] = {"template": entry["id"], "values": sp.get("values") or {}}
    for arg in entry["operators"]:
        if arg == "comparison":
            raw[arg] = (ops.get("comparison") or {}).get("sql")
        elif arg == "aggregate":
            raw[arg] = word(ops.get("agg_function"), {"max": "max", "min": "min", "avg": "avg", "average": "avg"})
        elif arg == "extreme":
            raw[arg] = word(ops.get("sort"), {"max": "max", "desc": "max", "min": "min", "asc": "min"})
        elif arg == "unit":
            raw[arg] = word(ops.get("unit_count"), {"hour": "hours", "day": "days"})
        elif arg == "order":
            raw[arg] = word(time.get("time_filter_exact1"), {"first": "first", "last": "last"})
    return normalize(raw, catalog)


class Executor:
    """Runs Tier-C plans on a task's contract-permitted resources (and the untimed references they need), the legal
    set the reference template skills use. Patient resource pairs are cached per patient."""

    def __init__(self, patient_map: dict[str, str], catalog_path: str | Path = CATALOG_PATH):
        self.catalog = load_catalog(str(catalog_path))
        self.patient_map = patient_map
        self._pairs: dict[str, list] = {}

    def __call__(self, plan: Plan | None, task_store, task: Task, context: str | None) -> PlanResult | None:
        from .skills import legal_set_with_references, patient_pairs, solve

        if plan is None or plan.operation is None:
            return None
        pid = self.patient_map[task.patient]
        if pid not in self._pairs:
            self._pairs[pid] = patient_pairs(task_store.store, pid, task.patient)
        pairs = self._pairs[pid]
        legal_rs, extra = legal_set_with_references(task_store.store, pairs, task.contract, pid)
        try:
            summary = solve(task_spec(plan, self.catalog, context), legal_rs + extra)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            return PlanResult(plan, {"status": "error"}, [], error=f"{type(exc).__name__}: {exc}")
        summary["template"] = self.catalog[plan.operation]["template"]
        if summary.get("status") != "ok":
            return PlanResult(plan, summary, [])
        ev = {e.id: e for _, e in pairs}
        prov = set(provenance_ids(summary))
        summary["_provenance_pairs"] = [(r, ev.get(str(r.get("id")))) for r in legal_rs + extra if str(r.get("id")) in prov]
        return PlanResult(plan, summary, provenance_ids(summary))


def skeleton(entry: dict[str, Any]) -> str:
    """The plan a template takes, as the model should write it."""
    plan: dict[str, Any] = {"template": entry["id"], "values": {v: f"<{v}>" for v in entry["values"] if v not in FIXED_VALUES}}
    for arg in entry["operators"]:
        plan[arg] = "|".join(OPERATOR_VALUES[arg])
    return json.dumps(plan)


def addendum(catalog: dict[str, dict[str, Any]]) -> str:
    lines = [
        "",
        "",
        "Evidence plan. Your final answer ends with one more line after the Citations line:",
        "Plan: <the plan of the catalog template your answer follows>   or   Plan: null   (when no template fits).",
        "Copy the plan of the template from the catalog, fill each <placeholder> with the value as the question states it,",
        "and pick one choice for each operator (for example \"order\": \"first\"). The patient is fixed by the task; time phrases",
        "such as \"since 05/2137\" are already enforced by the contract and do not go into the plan. The auditor runs your plan",
        "on the contract-permitted resources and checks that you cite the resources it uses and that your answer agrees with",
        "its result. Use a template only when the question asks exactly what it asks; otherwise (for example a change between",
        "two measurements, a comparison with an earlier value, a care unit or an admission type) write Plan: null.",
        "Catalog (template, then its plan):",
    ]
    for e in catalog.values():
        lines.append(f"- {e['id']}: {e['template']}")
        lines.append(f"  Plan: {skeleton(e)}")
    # v0.3: stated last — without it the JSON above pulled Gemma-3 into fenced ```tool_call blocks (dev probe: 39/40 → 1/40)
    lines += ["", "Your first responses are tool calls: write each one exactly as <tool_call>\n{...}\n</tool_call>, never inside a ``` code",
              "block. Only the final answer carries the Citations and Plan lines."]
    return "\n".join(lines)
