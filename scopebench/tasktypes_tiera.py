"""Tier-A-only task types (registered alongside the Tier-B types; freeze_tier.py TIER_ONLY_MODULES["A"]).

outcome_count_v1 — the outcome-abstraction count target ("how many administrations of <drug> in <window>"). The Tier-B
outcome type (`outcome_lookup_v1`) cites exactly one evidence object, which fits its single-row targets (first care unit,
discharge disposition) but cannot express a count whose support is every counted administration. The Tier-A development
pilot (2026-09-29) showed the mismatch: the frozen ChartSkill's count answer fails the single-citation validator, which
aborted the `full` configuration. This type keeps the Tier-B preamble, contract view, evidence view and payload builder,
and changes only the output schema, the validator, the citation accessor, correctness and the skill mapping:
  {"status": "ok|insufficient", "value": <integer count or null>, "citations": ["<evidence id>", ...]}
Correct = status and count equal to the gold; completeness (scorer) = share of the counted administrations cited.
Skill: the frozen `outcome_lookup` operation (target `administration_count`), whose provenance is the counted rows.
"""

from __future__ import annotations

from typing import Any

from .tasktypes import TaskType, register
from .tasktypes_tierb import PREAMBLE, payload_builder

COUNT_SCHEMA = '{"status":"ok|insufficient","value":"<integer count or null>","citations":["<evidence id of every counted administration>"]}'


def _as_int(v: Any) -> int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def _validate(parsed, task, evidence):
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors = []
    if parsed.get("status") not in {"ok", "insufficient"}:
        errors.append("status must be ok or insufficient")
    if parsed.get("status") == "ok":
        if _as_int(parsed.get("value")) is None:
            errors.append("an ok answer must contain an integer count")
        cites = parsed.get("citations")
        if not isinstance(cites, list) or not cites or not all(isinstance(c, str) and c for c in cites):
            errors.append("an ok answer must cite the counted administrations as a non-empty list of evidence ids")
    return errors


def _citations(parsed):
    if not parsed or not isinstance(parsed.get("citations"), list):
        return []
    return [c for c in parsed["citations"] if isinstance(c, str) and c]


def _correct(parsed, task):
    if parsed is None:
        return False
    g = task.gold
    if parsed.get("status") != g.get("status"):
        return False
    if g.get("status") != "ok":
        return True
    return _as_int(parsed.get("value")) is not None and _as_int(parsed.get("value")) == _as_int(g.get("value"))


def _skill_answer(summary, task, evidence):
    if summary.get("status") != "ok":
        return {"status": "insufficient", "value": None, "citations": []}
    return {"status": "ok", "value": summary.get("value"), "citations": list(summary.get("provenance") or [])}


def _skill_disagreements(parsed, summary, task):
    if parsed is None or summary.get("status") != "ok":
        return []
    if parsed.get("status") != "ok":
        return ["abstained although the deterministic ChartSkill produced a result"]
    errs = []
    if _as_int(parsed.get("value")) != _as_int(summary.get("value")):
        errs.append("count disagrees with deterministic ChartSkill")
    if set(_citations(parsed)) != set(summary.get("provenance") or []):
        errs.append("cited administrations disagree with the deterministic ChartSkill's counted rows")
    return errs


OUTCOME_COUNT = register(TaskType(
    name="outcome_count_v1", system_prompt=PREAMBLE, skill_op="outcome_lookup", max_tokens=1024,
    build_user_payload=payload_builder(COUNT_SCHEMA),
    validate=_validate, citations=_citations,
    substantive=lambda p: bool(p) and p.get("status") == "ok",
    atomic_fields=lambda p: ([{"field": "value", "value": p.get("value"), "citation": (p.get("citations") or [None])[0]}] if p and p.get("status") == "ok" else []),
    answer_correct=_correct, skill_answer=_skill_answer, skill_disagreements=_skill_disagreements,
    notes={"tier": "A", "reason": "count targets cite every counted row (Tier-A pilot finding 11)"},
))
