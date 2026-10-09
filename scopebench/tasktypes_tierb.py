"""Tier-B task types beyond LabTrend (plan §5.3): outcome lookup, cohort criteria, claim verification,
report reconstruction, exam summarization. Importing this module registers them.

Every type shares one contract preamble (scope, index time, basis, permitted sources) and returns
exactly one JSON object. Structured types are skill-covered (SKILL_SPEC §6/§7); the three free-text
types are uncovered by construction (paper Table 4).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .predicate import legal
from .schema import EvidenceObject, Task
from .tasktypes import TaskType, register

Payload = dict[str, Any]

PREAMBLE = """You are an EHR evidence extraction agent. The task contract is binding.

Evidence objects carry two timestamps: `occurred` (when the event happened) and `available`
(when the information became available in the record). The contract names which one governs
legality and the scope relative to the index time: `pre` = strictly before, `at` = up to the
index (the index document itself is legal), `post` = strictly after, `full` = the whole chart.
Use only the supplied evidence. Cite evidence by its `id`. Never invent, alter, or cite an
object that is not supplied. Return exactly one JSON object and no prose."""


def norm(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower()) if v is not None else ""


def struct_view(e: EvidenceObject) -> dict[str, Any]:
    return {"id": e.id, "source": e.source, "occurred": e.occurred, "available": e.available, **{k: v for k, v in e.fields.items() if k not in ("hadm", "stay", "note")}}


def line_view(e: EvidenceObject) -> dict[str, Any]:
    f = e.fields
    return {"id": e.id, "source": e.source, "occurred": e.occurred, "available": e.available, "note": f.get("note"), "line": f.get("line"),
            "section": f.get("section"), "exam_name": f.get("exam_name"), "text": f.get("text")}


def view(e: EvidenceObject) -> dict[str, Any]:
    return line_view(e) if e.source.endswith("_line") else struct_view(e)


def contract_view(task: Task, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    c = task.contract
    d = {"scope": c.scope, "index_time": c.index_time, "timestamp_basis": c.basis, "permitted_sources": list(c.permitted_sources) if c.permitted_sources else "any"}
    if c.window_hours is not None:
        d["window_hours"] = c.window_hours
    d.update(extra or {})
    return d


def grouped_evidence(evidence: list[EvidenceObject]) -> dict[str, Any]:
    """Compact prompt view: structured objects as before; report lines grouped by note with the note's
    metadata stated once and each line reduced to id / line / section / text (≈ 3× fewer tokens)."""
    structured = [struct_view(e) for e in evidence if not e.source.endswith("_line")]
    notes: dict[str, dict[str, Any]] = {}
    for e in evidence:
        if not e.source.endswith("_line"):
            continue
        f = e.fields
        n = notes.setdefault(str(f.get("note")), {"note": f.get("note"), "source": e.source, "exam_name": f.get("exam_name"), "occurred": e.occurred, "available": e.available, "lines": []})
        n["lines"].append({"id": e.id, "line": f.get("line"), "section": f.get("section"), "text": f.get("text")})
    for n in notes.values():
        n["lines"].sort(key=lambda l: int(l["line"] or 0))
    out: dict[str, Any] = {}
    if structured:
        out["supplied_evidence"] = structured
    if notes:
        out["supplied_report_lines"] = list(notes.values())
        out["citation_rule"] = "cite report evidence by the line `id`; a note's timestamps apply to every line in it"
    return out


def payload_builder(schema_line: str, extra_key: str | None = None):
    def build(task: Task, evidence: list[EvidenceObject], skill_payload: Payload | None) -> str:
        payload: dict[str, Any] = {
            "question": task.question,
            "contract": contract_view(task, task.metadata.get("contract_extra")),
            "output_schema": schema_line,
            **grouped_evidence(evidence),
        }
        if extra_key and task.metadata.get(extra_key) is not None:
            payload[extra_key] = task.metadata[extra_key]
        if skill_payload is not None:
            payload["deterministic_chartskill"] = skill_payload
            payload["skill_instruction"] = ("The deterministic skill is provenance-preserving. Verify its cited records against "
                                            "supplied evidence and use it to format the answer.")
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return build


def _str_list(x: Any) -> list[str]:
    return [str(i) for i in x if isinstance(i, (str, int))] if isinstance(x, list) else []


# ------------------------------------------------------------------ outcome lookup (covered)
OUTCOME_SCHEMA = '{"status":"ok|insufficient","value":"<string or null>","time":"<ISO timestamp or null>","citation":"<evidence id or null>"}'


def _outcome_validate(parsed, task, evidence):
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors = []
    if parsed.get("status") not in {"ok", "insufficient"}:
        errors.append("status must be ok or insufficient")
    if parsed.get("status") == "ok":
        if not isinstance(parsed.get("citation"), str) or not parsed["citation"]:
            errors.append("an ok answer must cite exactly one evidence id")
        if parsed.get("value") is None:
            errors.append("an ok answer must contain a value")
    return errors


def _outcome_citations(parsed):
    return [parsed["citation"]] if parsed and isinstance(parsed.get("citation"), str) and parsed["citation"] else []


def _outcome_correct(parsed, task):
    if parsed is None:
        return False
    g = task.gold
    if parsed.get("status") != g.get("status"):
        return False
    if g.get("status") != "ok":
        return True
    return parsed.get("citation") == g.get("citation") and norm(parsed.get("value")) == norm(g.get("value"))


def _outcome_skill_answer(summary, task, evidence):
    if summary.get("status") != "ok":
        return {"status": "insufficient", "value": None, "time": None, "citation": None}
    return {"status": "ok", "value": summary.get("value"), "time": summary.get("time"), "citation": summary.get("citation")}


def _outcome_skill_disagreements(parsed, summary, task):
    if parsed is None or summary.get("status") != "ok":
        return []
    if parsed.get("status") != "ok":
        return ["abstained although the deterministic ChartSkill produced a result"]
    errs = []
    if parsed.get("citation") != summary.get("citation"):
        errs.append("citation disagrees with deterministic ChartSkill")
    if norm(parsed.get("value")) != norm(summary.get("value")):
        errs.append("value disagrees with deterministic ChartSkill")
    return errs


register(TaskType(
    name="outcome_lookup_v1", system_prompt=PREAMBLE, skill_op="outcome_lookup", max_tokens=256,
    build_user_payload=payload_builder(OUTCOME_SCHEMA),
    validate=_outcome_validate, citations=_outcome_citations,
    substantive=lambda p: bool(p) and p.get("status") == "ok",
    atomic_fields=lambda p: ([{"field": "value", "value": p.get("value"), "citation": p.get("citation")}, {"field": "time", "value": p.get("time"), "citation": p.get("citation")}] if p and p.get("status") == "ok" else []),
    answer_correct=_outcome_correct, skill_answer=_outcome_skill_answer, skill_disagreements=_outcome_skill_disagreements,
    notes={"evidence_view": view},
))

# ------------------------------------------------------------------ cohort criteria (covered)
COHORT_SCHEMA = '{"decision":"include|exclude|uncertain","citations":["<evidence ids establishing each satisfied criterion, or the decisive failed one>"]}'


def _cohort_validate(parsed, task, evidence):
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors = []
    if parsed.get("decision") not in {"include", "exclude", "uncertain"}:
        errors.append("decision must be include, exclude, or uncertain")
    if not isinstance(parsed.get("citations"), list):
        errors.append("citations must be a list")
    elif parsed.get("decision") == "include" and not parsed["citations"]:
        errors.append("an include decision must cite the evidence establishing the criteria")
    return errors


def _cohort_skill_answer(summary, task, evidence):
    return {"decision": summary.get("decision"), "citations": list(summary.get("provenance") or [])}


def _cohort_skill_disagreements(parsed, summary, task):
    if parsed is None:
        return []
    if parsed.get("decision") == "uncertain" and summary.get("decision") in ("include", "exclude"):
        return ["abstained (uncertain) although the deterministic ChartSkill decided"]
    return ["decision disagrees with deterministic ChartSkill"] if parsed.get("decision") != summary.get("decision") else []


register(TaskType(
    name="cohort_v1", system_prompt=PREAMBLE, skill_op="cohort_criteria", max_tokens=384,
    build_user_payload=payload_builder(COHORT_SCHEMA, extra_key="criteria"),
    validate=_cohort_validate, citations=lambda p: _str_list(p.get("citations")) if p else [],
    substantive=lambda p: bool(p) and p.get("decision") in {"include", "exclude"},
    atomic_fields=lambda p: ([{"field": "decision", "value": p.get("decision"), "citation": None, "operation": "criteria_conjunction"}] if p else []),
    answer_correct=lambda p, t: bool(p) and p.get("decision") == t.gold.get("decision"),
    skill_answer=_cohort_skill_answer, skill_disagreements=_cohort_skill_disagreements,
    notes={"evidence_view": view},
))

# ------------------------------------------------------------------ claim verification (uncovered)
CLAIM_SCHEMA = '{"label":"supported|contradicted|insufficient","citations":["<report line ids that decide the label>"]}'

register(TaskType(
    name="claim_v1", system_prompt=PREAMBLE, skill_op=None, max_tokens=256,
    build_user_payload=payload_builder(CLAIM_SCHEMA, extra_key="claim"),
    validate=lambda p, t, e: (["submission is not one parseable JSON object"] if p is None else
                              ([] if p.get("label") in {"supported", "contradicted", "insufficient"} else ["label must be supported, contradicted, or insufficient"]) +
                              ([] if isinstance(p.get("citations"), list) else ["citations must be a list"]) +
                              ([] if p.get("label") == "insufficient" or (isinstance(p.get("citations"), list) and p["citations"]) else ["a decided label must cite the deciding lines"])),
    citations=lambda p: _str_list(p.get("citations")) if p else [],
    substantive=lambda p: bool(p) and p.get("label") in {"supported", "contradicted"},
    atomic_fields=lambda p: ([{"field": "label", "value": p.get("label"), "citation": (p.get("citations") or [None])[0]}] if p and p.get("label") in {"supported", "contradicted"} else []),
    answer_correct=lambda p, t: bool(p) and p.get("label") == t.gold.get("label"),
    notes={"evidence_view": view},
))

# ------------------------------------------------------------------ report reconstruction (uncovered)
RECON_SCHEMA = '{"status":"ok|insufficient","lines":[{"citation":"<line id>","text":"<line text copied verbatim>"}]}'


def _recon_validate(parsed, task, evidence):
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors = []
    if parsed.get("status") not in {"ok", "insufficient"}:
        errors.append("status must be ok or insufficient")
    lines = parsed.get("lines")
    if not isinstance(lines, list):
        errors.append("lines must be a list")
    elif parsed.get("status") == "ok" and (not lines or not all(isinstance(l, dict) and isinstance(l.get("citation"), str) for l in lines)):
        errors.append("an ok answer must list cited lines in order")
    return errors


def _recon_citations(parsed):
    return [l["citation"] for l in parsed.get("lines", []) if isinstance(l, dict) and isinstance(l.get("citation"), str)] if parsed and isinstance(parsed.get("lines"), list) else []


def _recon_correct(parsed, task):
    if parsed is None or parsed.get("status") != task.gold.get("status"):
        return False
    if task.gold.get("status") != "ok":
        return True
    return _recon_citations(parsed) == list(task.gold.get("line_ids", []))


def _token_f1(a: str, b: str) -> float:
    ta, tb = norm(a).split(), norm(b).split()
    if not ta or not tb:
        return 0.0
    from collections import Counter

    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def _recon_secondary(parsed, task):
    if not parsed or parsed.get("status") != "ok" or task.gold.get("status") != "ok":
        return {"token_f1": 0.0}
    text = " ".join(str(l.get("text", "")) for l in parsed.get("lines", []) if isinstance(l, dict))
    return {"token_f1": round(_token_f1(text, task.gold.get("text", "")), 4)}


register(TaskType(
    name="reconstruct_v1", system_prompt=PREAMBLE, skill_op=None, max_tokens=768,
    build_user_payload=payload_builder(RECON_SCHEMA),
    validate=_recon_validate, citations=_recon_citations,
    substantive=lambda p: bool(p) and p.get("status") == "ok" and bool(p.get("lines")),
    atomic_fields=lambda p: ([{"field": "text", "value": l.get("text"), "citation": l.get("citation")} for l in p.get("lines", []) if isinstance(l, dict)] if p and p.get("status") == "ok" else []),
    answer_correct=_recon_correct, secondary=_recon_secondary,
    notes={"evidence_view": view},
))

# ------------------------------------------------------------------ exam summarization (uncovered)
SUMM_SCHEMA = '{"status":"ok|insufficient","exams":[{"exam_name":"<name>","time":"<availability time>","first_impression_line":"<the line copied verbatim>","citation":"<id of that line>"}]}'


def _summ_validate(parsed, task, evidence):
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors = []
    if parsed.get("status") not in {"ok", "insufficient"}:
        errors.append("status must be ok or insufficient")
    exams = parsed.get("exams")
    if not isinstance(exams, list):
        errors.append("exams must be a list")
    elif parsed.get("status") == "ok" and (not exams or not all(isinstance(x, dict) and isinstance(x.get("citation"), str) for x in exams)):
        errors.append("an ok answer must list exams each citing its impression line")
    return errors


def _summ_citations(parsed):
    return [x["citation"] for x in parsed.get("exams", []) if isinstance(x, dict) and isinstance(x.get("citation"), str)] if parsed and isinstance(parsed.get("exams"), list) else []


def _summ_correct(parsed, task):
    if parsed is None or parsed.get("status") != task.gold.get("status"):
        return False
    if task.gold.get("status") != "ok":
        return True
    gold = {g["citation"]: g for g in task.gold.get("exams", [])}
    got = {x.get("citation"): x for x in parsed.get("exams", []) if isinstance(x, dict)}
    if set(gold) != set(got):
        return False
    return all(norm(got[c].get("exam_name")) == norm(g.get("exam_name")) and norm(got[c].get("first_impression_line")) == norm(g.get("first_impression_line")) for c, g in gold.items())


register(TaskType(
    name="summarize_exams_v1", system_prompt=PREAMBLE, skill_op=None, max_tokens=768,
    build_user_payload=payload_builder(SUMM_SCHEMA),
    validate=_summ_validate, citations=_summ_citations,
    substantive=lambda p: bool(p) and p.get("status") == "ok" and bool(p.get("exams")),
    atomic_fields=lambda p: ([f for x in p.get("exams", []) if isinstance(x, dict) for f in ({"field": "exam_name", "value": x.get("exam_name"), "citation": x.get("citation")}, {"field": "first_impression_line", "value": x.get("first_impression_line"), "citation": x.get("citation")})] if p and p.get("status") == "ok" else []),
    answer_correct=_summ_correct,
    notes={"evidence_view": view},
))
