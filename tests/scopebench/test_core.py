"""Invariant tests for the ScopeBench core (plan §4.6). Run: .venv-analysis/bin/python -m pytest -q tests/scopebench"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scopebench import parser, scoring  # noqa: E402
from scopebench.guard import check  # noqa: E402
from scopebench.predicate import filter_legal, legal, legal_set, wrong_basis  # noqa: E402
from scopebench.retrieval import AccessLog, TemporalNearestRetriever, rrf  # noqa: E402
from scopebench.schema import Case, Contract, EvidenceObject, Task, parse_time  # noqa: E402
from scopebench.tasktypes import LAB_TREND_K3, REFERENCE_SKILLS  # noqa: E402


def ev(i: str, occ: str | None, avail: str | None, value: float = 1.0, source: str = "lab") -> EvidenceObject:
    return EvidenceObject(id=i, source=source, patient="p", occurred=occ, available=avail, fields={"display": "PT", "value": value, "unit": "sec"})


INDEX = "2187-02-17T11:52:30Z"
POOL = [
    ev("a", "2187-02-14T08:30:00Z", "2187-02-14T09:08:00Z", 13.9),
    ev("b", "2187-02-15T09:00:00Z", "2187-02-15T09:29:00Z", 13.5),
    ev("c", "2187-02-16T08:40:00Z", "2187-02-16T09:42:00Z", 13.0),
    ev("d", "2187-02-17T11:00:00Z", "2187-02-17T12:30:00Z", 12.8),   # drawn before, available after the index
    ev("e", "2187-02-18T11:25:00Z", "2187-02-18T14:03:00Z", 12.5),
    ev("f", "2187-02-19T08:00:00Z", "2187-02-19T09:10:00Z", 12.2),
    ev("g", None, None, 99.0),                                        # no timestamps
]


def test_predicate_scopes_and_bases():
    pre = Contract(INDEX, "pre", "available")
    post = Contract(INDEX, "post", "available")
    assert legal_set(POOL, pre) == {"a", "b", "c"}
    assert legal_set(POOL, post) == {"d", "e", "f"}
    # occurrence basis moves d to the pre side: the wrong-basis leak
    assert legal_set(POOL, wrong_basis(pre)) == {"a", "b", "c", "d"}
    assert legal_set(POOL, Contract(INDEX, "full", "available")) == {e.id for e in POOL}
    assert "g" not in legal_set(POOL, pre) and "g" in legal_set(POOL, Contract(INDEX, "full", "available"))
    at = Contract(INDEX, "at", "available", at_tolerance_hours=1.0, index_evidence_ids=("f",))
    assert legal_set(POOL, at) == {"a", "b", "c", "d", "f"}
    windowed = Contract(INDEX, "post", "available", window_hours=24)
    assert legal_set(POOL, windowed) == {"d"}
    assert legal_set(POOL, Contract(INDEX, "pre", "available", permitted_sources=("micro",))) == set()


def test_predicate_rejects_bad_contract():
    with pytest.raises(ValueError):
        Contract(INDEX, "sideways", "available")
    with pytest.raises(ValueError):
        Contract(INDEX, "pre", "issued")


def test_temporal_nearest_matches_scopepair_semantics():
    task = Task("t", "c", "p", "x", "lab_trend_k3", "q", Contract(INDEX, "post", "available"), gold={}, required_ids=())
    r = TemporalNearestRetriever()
    top = r.rank(task, POOL, 3, lambda e: legal(e, task.contract))
    assert [e.id for e in top] == ["d", "e", "f"]            # chronological output
    unscoped = r.rank(task, POOL, 3, None)
    assert [e.id for e in unscoped] == ["c", "d", "e"]       # nearest by |t-index| regardless of side
    assert [e.id for e in r.refill(task, POOL, 3, lambda e: legal(e, task.contract))] == ["d", "e", "f"]


def test_rrf_and_access_log():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["a", "b"]
    log = AccessLog()
    log.record("prompt", ["a", "b"])
    log.record("repair", ["b", "c"])
    assert log.ids() == {"a", "b", "c"}


def test_parser_variants():
    assert parser.extract_json('```json\n{"status":"ok"}\n```')["status"] == "ok"
    assert parser.extract_json('Sure: {"a": 1} trailing')["a"] == 1
    assert parser.extract_json("no json here") is None
    assert parser.extract_json('[{"a":1}]') is None   # frozen ScopePair semantics: a top-level list is not an object
    assert parser.extract_json('{"a": {"b": "}"}}')["a"]["b"] == "}"
    # frozen rule: a fenced object followed by prose that contains braces is unparseable
    assert parser.extract_json('```json\n{"status":"ok"}\n```\nExplanation: {"id": "x"} was used') is None
    assert parser.extract_json(None) is None


def _post_task():
    gold = {"status": "ok", "results": [{"citation": "d", "value": 12.8, "unit": "sec", "issued_time": "2187-02-17T12:30:00Z"},
                                        {"citation": "e", "value": 12.5, "unit": "sec", "issued_time": "2187-02-18T14:03:00Z"},
                                        {"citation": "f", "value": 12.2, "unit": "sec", "issued_time": "2187-02-19T09:10:00Z"}],
            "delta": -0.6000000000000005, "direction": "decreasing"}
    return Task("t", "c", "p", "x", "lab_trend_k3", "q", Contract(INDEX, "post", "available"), gold=gold, required_ids=("d", "e", "f"))


def test_guard_and_scoring_on_labtrend_shapes():
    task = _post_task()
    pool = {e.id: e for e in POOL}
    supplied = [pool["d"], pool["e"], pool["f"]]
    good = {"status": "ok", "results": [{"citation": "d", "value": 12.8, "unit": "sec", "issued_time": "2187-02-17T12:30:00Z"},
                                        {"citation": "e", "value": 12.5, "unit": "sec", "issued_time": "2187-02-18T14:03:00Z"},
                                        {"citation": "f", "value": 12.2, "unit": "sec", "issued_time": "2187-02-19T09:10:00Z"}],
            "delta": -0.6, "direction": "decreasing"}
    rep = check(good, task, LAB_TREND_K3, pool, supplied, {"d", "e", "f"}, check_completeness=True, skill_summary=REFERENCE_SKILLS["lab_trend_k3"](supplied))
    assert rep.errors() == []
    m = scoring.score(good, task, LAB_TREND_K3, pool, {"d", "e", "f"}, tokens_in=1, tokens_out=1, calls=1, latency=0.1)
    assert m["answer_correct"] and m["completeness"] == 1.0 and m["access_legal"] and m["citation_legal"] and m["auditable"] and m["workflow_compliant"]

    leaky = dict(good); leaky["results"] = [dict(good["results"][0], citation="c")] + good["results"][1:]
    rep = check(leaky, task, LAB_TREND_K3, pool, supplied, {"c", "d", "e", "f"}, check_completeness=True)
    assert rep.illegal_citations == ["c"] and rep.unsupplied_citations == ["c"] and rep.illegal_access == ["c"] and rep.missing_required == ["d"]
    m = scoring.score(leaky, task, LAB_TREND_K3, pool, {"c", "d", "e", "f"}, tokens_in=1, tokens_out=1, calls=1, latency=0.1)
    assert not m["access_legal"] and not m["citation_legal"] and m["completeness"] == pytest.approx(2 / 3) and not m["answer_correct"]

    abstain = {"status": "insufficient", "results": [], "delta": None, "direction": None}
    m = scoring.score(abstain, task, LAB_TREND_K3, pool, {"d", "e", "f"}, tokens_in=1, tokens_out=1, calls=1, latency=0.1)
    assert m["abstained"] and not m["citation_legal"] and not m["auditable"] and not m["answer_correct"] and not m["workflow_compliant"]

    m = scoring.score(None, task, LAB_TREND_K3, pool, {"d"}, tokens_in=1, tokens_out=1, calls=1, latency=0.1)
    assert m["parse_failed"] and not m["answer_correct"]
    assert check(None, task, LAB_TREND_K3, pool, supplied, set(), check_completeness=True).errors() == ["submission is not one parseable JSON object"]


def test_late_filter_drops_illegal_and_invalidates_k3():
    task = _post_task()
    pool = {e.id: e for e in POOL}
    ans = {"status": "ok", "results": [{"citation": "c"}, {"citation": "d"}, {"citation": "e"}], "delta": 1, "direction": "increasing"}
    out = scoring.late_filter(ans, task, LAB_TREND_K3, pool)
    assert [r["citation"] for r in out["results"]] == ["d", "e"] and out["status"] == "insufficient"


def test_reference_skill_matches_frozen_semantics():
    s = REFERENCE_SKILLS["lab_trend_k3"]([e for e in POOL if e.id in ("d", "e", "f")])
    assert s["direction"] == "decreasing" and s["ordered_result_ids"] == ["d", "e", "f"] and abs(s["delta_last_minus_first"] + 0.6) < 1e-9
    assert REFERENCE_SKILLS["lab_trend_k3"]([POOL[0]])["status"] == "insufficient"


def test_parse_time_forms():
    assert parse_time("2187-02-17T11:52:30Z") == parse_time("2187-02-17 11:52:30+00:00")
    assert parse_time(None) is None
