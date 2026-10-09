"""Hand-built unit cases for scopebench.oracle (SKILL_SPEC.md v1.1, §1–§7), implementer A."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from scopebench.oracle import OPERATIONS, SPEC_VERSION
from scopebench.predicate import filter_legal
from scopebench.schema import Contract, EvidenceObject, parse_time
from scopebench.tasktypes import REFERENCE_SKILLS

INDEX = "2024-01-10T12:00:00Z"
I0 = parse_time(INDEX)


def ts(hours: float) -> str:
    return (I0 + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def ev(id_, hours=None, source="labevents", basis="available", **fields) -> EvidenceObject:
    kw = {"id": id_, "source": source, "patient": "p", "fields": fields}
    if hours is not None:
        kw[basis] = ts(hours)
    return EvidenceObject(**kw)


def contract(scope="full", basis="available", **kw) -> Contract:
    return Contract(index_time=INDEX, scope=scope, basis=basis, **kw)


def test_surface():
    assert SPEC_VERSION == "1.2"
    assert list(OPERATIONS) == [
        "lab_trend_k3", "med_dose_k3", "nearest_k", "first_after", "last_before",
        "window_count", "window_sum", "window_mean", "outcome_lookup", "cohort_criteria",
    ]


# ------------------------------------------------------------------ §1 lab_trend_k3


def test_lab_trend_insufficient_shape_matches_reference():
    pool = [ev("a", -1, value=1.0, unit="u"), ev("b", -2, value=2.0, unit="u")]
    out = OPERATIONS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out == {"status": "insufficient", "provenance": []}
    assert out == REFERENCE_SKILLS["lab_trend_k3"](pool)


def test_lab_trend_selects_nearest_then_orders_chronologically():
    pool = [
        ev("far", -30, value=9.0, unit="u"),
        ev("c", -1, value=3.0, unit="u"),
        ev("a", -5, value=1.0, unit="u"),
        ev("b", -3, value=5.0, unit="u"),
    ]
    out = OPERATIONS["lab_trend_k3"](pool, contract("pre"), {"k": 3})
    assert out["status"] == "ok"
    assert out["ordered_result_ids"] == ["a", "b", "c"]
    assert out["delta_last_minus_first"] == 2.0 and out["direction"] == "increasing"
    assert out["provenance"][0] == {"citation": "a", "value": 1.0, "unit": "u", "issued_time": ts(-5)}


def test_lab_trend_ties_by_id_for_selection_and_order():
    # Four objects at the same distance; z is dropped (id order), and order is by id among ties.
    pool = [ev("z", -2, value=0.0), ev("b", -2, value=2.0), ev("a", -2, value=1.0), ev("c", 2, value=3.0)]
    out = OPERATIONS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out["ordered_result_ids"] == ["a", "b", "c"]
    assert out["direction"] == "increasing"


def test_lab_trend_skips_non_numeric_and_untimed():
    pool = [
        ev("n", -1, value="abc"), ev("m", -1, value=None), ev("u", None, value=1.0),
        ev("a", -4, value=2.0), ev("b", -3, value=2.0), ev("c", -2, value=2.0),
    ]
    out = OPERATIONS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out["ordered_result_ids"] == ["a", "b", "c"] and out["direction"] == "stable"


def _random_lab_pool(rng: random.Random, n: int) -> list[EvidenceObject]:
    ids = rng.sample([f"obs-{i:03d}" for i in range(50)], n)
    offsets = [-1.0, -2.0, -2.0, -3.5, -7.0, -24.0, -48.0]  # duplicates → ties
    values = [rng.choice([1.0, 2.5, 2.5, 7.25, 100.0, -3.0]) for _ in range(n)]
    return [ev(i, rng.choice(offsets), value=v, unit=rng.choice(["mg/dL", "mmol/L"])) for i, v in zip(ids, values)]


def test_lab_trend_equals_reference_on_random_legal_pools():
    rng = random.Random(20260927)
    ctr = contract("pre")
    n_ok = 0
    for _ in range(200):
        pool = filter_legal(_random_lab_pool(rng, rng.randint(0, 3)), ctr)  # ScopePair regime: ≤ 3 legal
        ordered = sorted(pool, key=lambda e: (parse_time(e.available), e.id))
        expected = REFERENCE_SKILLS["lab_trend_k3"](ordered)
        assert OPERATIONS["lab_trend_k3"](pool, ctr, {"k": 3}) == expected
        n_ok += expected["status"] == "ok"
    assert n_ok > 30
    # More than 3 legal: the reference is undefined (spec §1 selects the nearest 3); check the
    # oracle against the reference applied to an independently computed nearest-3.
    for _ in range(50):
        pool = filter_legal(_random_lab_pool(rng, rng.randint(4, 8)), ctr)
        nearest = sorted(pool, key=lambda e: (abs(parse_time(e.available) - I0), e.id))[:3]
        ordered = sorted(nearest, key=lambda e: (parse_time(e.available), e.id))
        assert OPERATIONS["lab_trend_k3"](pool, ctr, {"k": 3}) == REFERENCE_SKILLS["lab_trend_k3"](ordered)


# ------------------------------------------------------------------ §2 med_dose_k3


def med(id_, hours, dose, unit="mg"):
    return ev(id_, hours, source="emar", basis="occurred", dose=dose, unit=unit)


def test_med_dose_basic_and_shape():
    pool = [med("b", -2, 10), med("a", -3, 20), med("c", -1, 30), med("far", -40, 1000)]
    out = OPERATIONS["med_dose_k3"](pool, contract(basis="occurred"), {"k": 3})
    assert out["ordered_result_ids"] == ["a", "b", "c"]
    assert out["total_dose"] == 60.0 and out["mean_dose"] == 20.0
    assert out["provenance"][2] == {"citation": "c", "dose": 30, "unit": "mg", "effective_time": ts(-1)}


def test_med_dose_insufficient_ties_and_non_numeric():
    ctr = contract(basis="occurred")
    assert OPERATIONS["med_dose_k3"]([med("a", -1, 1), med("b", -1, "x"), med("c", -1, None)], ctr, {"k": 3}) == {
        "status": "insufficient", "ordered_result_ids": [], "total_dose": None, "mean_dose": None, "provenance": [],
    }
    pool = [med("d", -1, 4), med("c", -1, 3), med("b", -1, 2), med("a", -1, 1)]
    out = OPERATIONS["med_dose_k3"](pool, ctr, {"k": 3})
    assert out["ordered_result_ids"] == ["a", "b", "c"] and out["total_dose"] == 6.0


# ------------------------------------------------------------------ §3 nearest_k


def test_nearest_k_ties_untimed_and_verbatim_values():
    pool = [ev("b", 1, value="n/a", unit=None), ev("a", -1, value=None), ev("u", None, value=5), ev("c", 3, value=2)]
    out = OPERATIONS["nearest_k"](pool, contract(), {"k": 2, "basis": "available"})
    assert out["ordered_result_ids"] == ["a", "b"]
    assert out["provenance"] == [
        {"citation": "a", "value": None, "unit": None, "time": ts(-1)},
        {"citation": "b", "value": "n/a", "unit": None, "time": ts(1)},
    ]
    assert OPERATIONS["nearest_k"](pool, contract(), {"k": 4, "basis": "available"}) == {
        "status": "insufficient", "ordered_result_ids": [], "provenance": [],
    }


def test_nearest_k_uses_params_basis():
    pool = [
        EvidenceObject(id="x", source="s", patient="p", occurred=ts(-1), available=ts(-10), fields={}),
        EvidenceObject(id="y", source="s", patient="p", occurred=ts(-10), available=ts(-1), fields={}),
    ]
    assert OPERATIONS["nearest_k"](pool, contract(), {"k": 1, "basis": "occurred"})["ordered_result_ids"] == ["x"]
    assert OPERATIONS["nearest_k"](pool, contract(), {"k": 1, "basis": "available"})["ordered_result_ids"] == ["y"]


# ------------------------------------------------------------------ §4 first_after / last_before


def test_boundaries_are_strict_and_tie_by_id():
    pool = [ev("at", 0, v=0), ev("b", 2, v=1), ev("a", 2, v=2), ev("q", -3, v=3), ev("p", -3, v=4), ev("u", None)]
    fa = OPERATIONS["first_after"](pool, contract(), {"basis": "available", "which": "first_after"})
    assert fa == {"status": "ok", "citation": "a", "time": ts(2), "fields": {"v": 2}, "provenance": ["a"]}
    lb = OPERATIONS["last_before"](pool, contract(), {"basis": "available", "which": "last_before"})
    assert lb["citation"] == "q" and lb["time"] == ts(-3)  # max-type pick: equal times → largest id (§0)
    only_index = [ev("at", 0), ev("u", None)]
    assert OPERATIONS["first_after"](only_index, contract(), {"basis": "available"}) == {
        "status": "insufficient", "citation": None, "time": None, "fields": None, "provenance": [],
    }
    assert OPERATIONS["last_before"](only_index, contract(), {"basis": "available"})["status"] == "insufficient"


def test_which_and_agg_must_match_registry_key():
    pool = [ev("a", -1, value=1)]
    with pytest.raises(ValueError):
        OPERATIONS["first_after"](pool, contract(), {"basis": "available", "which": "last_before"})
    with pytest.raises(ValueError):
        OPERATIONS["window_sum"](pool, contract(), {"basis": "available", "agg": "count"})
    assert OPERATIONS["window_sum"](pool, contract(), {"basis": "available", "agg": "sum"})["value"] == 1.0
    assert OPERATIONS["last_before"](pool, contract(), {"basis": None})["citation"] == "a"  # basis None → contract.basis


def test_unparseable_timestamp_is_untimed():
    bad = EvidenceObject(id="bad", source="labevents", patient="p", available="not-a-time", fields={"value": 1})
    empty = EvidenceObject(id="empty", source="labevents", patient="p", available="", fields={"value": 2})
    pool = [bad, empty, ev("ok", -1, value=3)]
    assert OPERATIONS["nearest_k"](pool, contract(), {"k": 1, "basis": "available"})["ordered_result_ids"] == ["ok"]
    assert OPERATIONS["nearest_k"](pool, contract(), {"k": 2, "basis": "available"})["status"] == "insufficient"
    assert OPERATIONS["last_before"](pool, contract(), {"basis": "available"})["citation"] == "ok"
    out = OPERATIONS["window_sum"](pool, contract(), {"basis": "available"})
    assert out["value"] == 6.0 and out["provenance"] == ["ok", "bad", "empty"]  # untimed after timed, by id


# ------------------------------------------------------------------ §5 window_*


def test_window_count_sum_mean():
    pool = [ev("c", 1, value="x"), ev("a", -2, value=1.5), ev("u", None, value=2.5), ev("b", -1, value="4")]
    p = {"basis": "available"}
    assert OPERATIONS["window_count"](pool, contract(), p) == {
        "status": "ok", "agg": "count", "value": 4, "n": 4, "n_skipped_non_numeric": 0, "provenance": ["a", "b", "c", "u"],
    }
    s = OPERATIONS["window_sum"](pool, contract(), p)
    assert s == {"status": "ok", "agg": "sum", "value": 8.0, "n": 3, "n_skipped_non_numeric": 1, "provenance": ["a", "b", "u"]}
    m = OPERATIONS["window_mean"](pool, contract(), p)
    assert m["value"] == pytest.approx(8.0 / 3) and m["n"] == 3 and m["n_skipped_non_numeric"] == 1


def test_window_zero_cases():
    p = {"basis": "available"}
    assert OPERATIONS["window_count"]([], contract(), p)["value"] == 0
    assert OPERATIONS["window_sum"]([ev("x", 0, value="bad")], contract(), p)["value"] == 0.0
    assert OPERATIONS["window_mean"]([ev("x", 0, value="bad")], contract(), p) == {
        "status": "insufficient", "agg": "mean", "value": None, "n": None, "n_skipped_non_numeric": None, "provenance": [],
    }


def test_window_boundaries_via_predicate():
    # pre with a 24 h window: t >= i - 24h inclusive, t < i strict (predicate); the oracle counts what it is given.
    pool = [ev("edge", -24, value=1), ev("out", -24.0001, value=1), ev("in", -1, value=1), ev("idx", 0, value=1), ev("post", 1, value=1)]
    ctr = contract("pre", window_hours=24)
    out = OPERATIONS["window_count"](filter_legal(pool, ctr), ctr, {"basis": "available"})
    assert out["provenance"] == ["edge", "in"] and out["value"] == 2
    ctr_post = contract("post", window_hours=1)
    assert OPERATIONS["window_count"](filter_legal(pool, ctr_post), ctr_post, {})["provenance"] == ["post"]


# ------------------------------------------------------------------ §6 outcome_lookup


def test_outcome_first_careunit_and_dispositions():
    pool = [
        ev("t2", 5, source="transfers", careunit="MICU"), ev("t1", 2, source="transfers", careunit="ED"),
        ev("t0", -1, source="transfers", careunit="prior"), ev("lab", 1, value=3),
        ev("adm-b", -3, source="admissions", basis="occurred", disp="HOME"),
        ev("adm-a", -3, source="admissions", basis="occurred", disp="SNF"),
        ev("adm-old", -9, source="admissions", basis="occurred", disp="DIED"),
    ]
    p = {"target": "first_careunit", "field": "careunit", "basis": "available", "drug": None}
    assert OPERATIONS["outcome_lookup"](pool, contract(), p) == {
        "status": "ok", "target": "first_careunit", "value": "ED", "citation": "t1", "time": ts(2), "provenance": ["t1"],
    }
    p = {"target": "discharge_disposition", "field": "disp", "basis": "occurred", "drug": None}
    out = OPERATIONS["outcome_lookup"](pool, contract(basis="occurred"), p)
    assert out["value"] == "HOME" and out["citation"] == "adm-b" and out["time"] == ts(-3)  # max-type: largest id
    p = {"target": "ed_disposition", "field": "disp", "basis": "occurred", "drug": None}
    assert OPERATIONS["outcome_lookup"](pool, contract(), p) == {
        "status": "insufficient", "target": "ed_disposition", "value": None, "citation": None, "time": None, "provenance": [],
    }
    # time is on params["basis"]: admissions have no `available`, so time is None while the pick is on occurred.
    p = {"target": "discharge_disposition", "field": "disp", "basis": "available", "drug": None}
    out = OPERATIONS["outcome_lookup"](pool, contract(), p)
    assert out["citation"] == "adm-b" and out["time"] is None and out["status"] == "ok"
    # Untimed fallback: no timed admissions → smallest id; missing field → value None with status ok.
    untimed = [
        EvidenceObject(id="u-b", source="admissions", patient="p", fields={"disp": "X"}),
        EvidenceObject(id="u-a", source="admissions", patient="p", occurred="garbage", fields={}),
    ]
    out = OPERATIONS["outcome_lookup"](untimed, contract(), {"target": "discharge_disposition", "field": "disp", "basis": "occurred"})
    assert out == {"status": "ok", "target": "discharge_disposition", "value": None, "citation": "u-a", "time": None, "provenance": ["u-a"]}


def test_outcome_administration_count():
    pool = [
        EvidenceObject(id="e2", source="emar", patient="p", occurred=ts(2), fields={"medication": " Heparin "}),
        EvidenceObject(id="e1", source="emar", patient="p", occurred=ts(-2), fields={"medication": "heparin"}),
        EvidenceObject(id="e0", source="emar", patient="p", occurred=ts(-5), fields={"medication": "insulin"}),
        EvidenceObject(id="not-emar", source="labevents", patient="p", occurred=ts(-5), fields={"medication": "heparin"}),
    ]
    p = {"target": "administration_count", "field": None, "basis": "occurred", "drug": "HEPARIN"}
    assert OPERATIONS["outcome_lookup"](pool, contract(basis="occurred"), p) == {
        "status": "ok", "target": "administration_count", "value": 2, "citation": None, "time": None, "provenance": ["e1", "e2"],
    }
    p["drug"] = "warfarin"
    out = OPERATIONS["outcome_lookup"](pool, contract(basis="occurred"), p)
    assert out["status"] == "ok" and out["value"] == 0 and out["provenance"] == []


# ------------------------------------------------------------------ §7 cohort_criteria


def _cohort_pool():
    return [
        ev("cr2", -2, concept="creatinine", value=2.1), ev("cr1", -6, concept="creatinine", value=0.9),
        ev("crx", -1, concept="creatinine", value="cancelled"),
        ev("dx", None, source="diagnoses_icd", code="n17.0"),
        EvidenceObject(id="em", source="emar", patient="p", available=ts(-3), fields={"medication": "Vancomycin"}),
        ev("tr", -8, source="triage", acuity="2"),
    ]


def test_cohort_include_and_provenance_union_order():
    criteria = [
        {"kind": "lab_threshold", "concept": "creatinine", "op": ">=", "threshold": 1.5, "basis": "available"},
        {"kind": "code_present", "prefix": "N17"},
        {"kind": "exposure", "drug": "vancomycin"},
        {"kind": "acuity", "max_acuity": 2},
    ]
    out = OPERATIONS["cohort_criteria"](_cohort_pool(), contract(), {"criteria": criteria})
    assert out["status"] == "ok" and out["decision"] == "include"
    assert [c["satisfied"] for c in out["criteria"]] == [True, True, True, True]
    assert out["criteria"][0] == {**criteria[0], "satisfied": True, "provenance": ["cr2"]}
    assert out["criteria"][1]["provenance"] == ["dx"]
    assert out["provenance"] == ["tr", "em", "cr2", "dx"]  # chronological on contract basis, untimed last


def test_cohort_exclude_uses_first_unsatisfied_and_uncertain():
    criteria = [
        {"kind": "acuity", "max_acuity": 1},  # decidable, unsatisfied → decisive; cites its domain (v1.2)
        {"kind": "lab_threshold", "concept": "creatinine", "op": "<", "threshold": 1.0},  # satisfied by cr1
        {"kind": "code_present", "prefix": "I50"},  # unsatisfied → cites its domain
    ]
    out = OPERATIONS["cohort_criteria"](_cohort_pool(), contract(), {"criteria": criteria})
    assert out["decision"] == "exclude" and out["provenance"] == ["tr"]
    assert [c["satisfied"] for c in out["criteria"]] == [False, True, False]
    assert [c["provenance"] for c in out["criteria"]] == [["tr"], ["cr1"], ["dx"]]
    # Decisive first criterion: the whole examined domain (incl. the non-numeric crx), chronological on its basis.
    criteria2 = [{"kind": "lab_threshold", "concept": "creatinine", "op": ">", "threshold": 100.0}, criteria[0]]
    assert OPERATIONS["cohort_criteria"](_cohort_pool(), contract(), {"criteria": criteria2})["provenance"] == ["cr1", "cr2", "crx"]
    # Uncertain: no triage object at all; other criteria satisfied; undecidable provenance is [].
    pool = [e for e in _cohort_pool() if e.source != "triage"]
    out = OPERATIONS["cohort_criteria"](pool, contract(), {"criteria": [{"kind": "acuity", "max_acuity": 3}, {"kind": "exposure", "drug": "VANCOMYCIN"}]})
    assert out["decision"] == "uncertain" and out["criteria"][0]["satisfied"] is None and out["criteria"][0]["provenance"] == []
    assert out["provenance"] == ["em"]
    # Exclusion beats uncertainty; the decisive exposure criterion cites every emar object examined.
    out = OPERATIONS["cohort_criteria"](pool, contract(), {"criteria": [{"kind": "acuity", "max_acuity": 3}, {"kind": "exposure", "drug": "nothing"}]})
    assert out["decision"] == "exclude" and out["provenance"] == ["em"]
    out = OPERATIONS["cohort_criteria"](pool, contract(), {"criteria": [{"kind": "lab_threshold", "concept": "sodium", "op": "<", "threshold": 1}]})
    assert out["decision"] == "uncertain"
    assert OPERATIONS["cohort_criteria"]([], contract(), {"criteria": []})["decision"] == "include"


def test_cohort_ties_by_id_and_non_numeric_only_is_unsatisfied():
    pool = [ev("b", -1, concept="k", value=6), ev("a", -1, concept="k", value=7), ev("z", -1, concept="k", value="?")]
    out = OPERATIONS["cohort_criteria"](pool, contract(), {"criteria": [{"kind": "lab_threshold", "concept": "k", "op": ">", "threshold": 5}]})
    assert out["provenance"] == ["a", "b"]
    out = OPERATIONS["cohort_criteria"]([pool[2]], contract(), {"criteria": [{"kind": "lab_threshold", "concept": "k", "op": ">", "threshold": 5}]})
    assert out["decision"] == "exclude" and out["criteria"][0]["satisfied"] is False
    assert out["provenance"] == ["z"]  # v1.2: the non-numeric object was examined and failed, so it is cited


# ------------------------------------------------------------------ purity


def test_operations_do_not_mutate_inputs():
    pool = _cohort_pool()
    snapshot = [e.to_dict() for e in pool]
    ctr = contract()
    for name, fn in OPERATIONS.items():
        params = {
            "lab_trend_k3": {"k": 3}, "med_dose_k3": {"k": 3}, "nearest_k": {"k": 1, "basis": "available"},
            "first_after": {"basis": "available"}, "last_before": {"basis": "available"},
            "window_count": {}, "window_sum": {}, "window_mean": {},
            "outcome_lookup": {"target": "administration_count", "field": None, "basis": "available", "drug": "x"},
            "cohort_criteria": {"criteria": [{"kind": "code_present", "prefix": "N"}]},
        }[name]
        a, b = fn(pool, ctr, params), fn(pool, ctr, params)
        assert a == b
    assert [e.to_dict() for e in pool] == snapshot
