"""Late filtering (plan §4.2): illegal citations are dropped from every task type's answer, never refilled."""

from __future__ import annotations

from scopebench import tasktypes_tierb  # noqa: F401
from scopebench.scoring import late_filter, score
from scopebench.schema import Contract, EvidenceObject, Task
from scopebench.tasktypes import get as get_task_type


def _ev(eid: str, t: str) -> EvidenceObject:
    return EvidenceObject(id=eid, source="labevents", patient="p", occurred=t, available=t, row_ref=eid, fields={"value": 1.0, "unit": "u", "concept": "c"})


CONTRACT = Contract(index_time="2020-01-10T00:00:00Z", scope="pre", basis="available")
POOL = {"in1": _ev("in1", "2020-01-09T00:00:00Z"), "in2": _ev("in2", "2020-01-08T00:00:00Z"), "out1": _ev("out1", "2020-01-11T00:00:00Z")}


def _task(tt: str) -> Task:
    return Task(task_id="t", case_id="c", patient="p", tier="B", task_type=tt, question="q", contract=CONTRACT, gold={}, required_ids=("in1",))


def test_cohort_style_citation_list_loses_illegal_ids():
    tt = get_task_type("cohort_v1")
    parsed = {"decision": "include", "citations": ["in1", "out1", "unknown"]}
    out = late_filter(parsed, _task("cohort_v1"), tt, POOL)
    assert out == {"decision": "include", "citations": ["in1"]}
    assert parsed["citations"] == ["in1", "out1", "unknown"]   # input untouched
    m = score(out, _task("cohort_v1"), tt, POOL, {"in1", "out1"}, tokens_in=1, tokens_out=1, calls=1, latency=0.0)
    assert m["citation_legal"] and not m["access_legal"]


def test_lab_trend_results_lose_objects_and_become_insufficient():
    tt = get_task_type("lab_trend_k3")
    parsed = {"status": "ok", "results": [{"citation": "in1", "value": 1.0}, {"citation": "in2", "value": 1.0}, {"citation": "out1", "value": 1.0}], "delta": 0.0, "direction": "stable"}
    out = late_filter(parsed, _task("lab_trend_k3"), tt, POOL)
    assert [r["citation"] for r in out["results"]] == ["in1", "in2"] and out["status"] == "insufficient" and out["delta"] is None


def test_legal_answer_is_returned_unchanged():
    tt = get_task_type("cohort_v1")
    parsed = {"decision": "exclude", "citations": ["in1", "in2"]}
    assert late_filter(parsed, _task("cohort_v1"), tt, POOL) is parsed
    assert late_filter(None, _task("cohort_v1"), tt, POOL) is None
