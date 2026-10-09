"""Hand-built unit cases for scopebench/skills (implementer B), spec v1.2 §1–§7.

Run: .venv-analysis/bin/python -m pytest -q tests/scopebench/test_skills_unit.py -p no:cacheprovider
"""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from scopebench import skills
from scopebench.predicate import legal
from scopebench.schema import Contract, EvidenceObject, parse_time
from scopebench.tasktypes import REFERENCE_SKILLS

INDEX = "2024-03-10T12:00:00Z"
OPS = skills.OPERATIONS


def ev(id: str, *, occurred=None, available=None, source="labevents", **fields) -> EvidenceObject:
    return EvidenceObject(id=id, source=source, patient="p1", occurred=occurred, available=available, fields=fields)


def at(hours: float) -> str:
    """ISO string `hours` after the index (negative = before)."""
    return (parse_time(INDEX) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def contract(scope="full", basis="available", **kw) -> Contract:
    return Contract(index_time=INDEX, scope=scope, basis=basis, **kw)


# --------------------------------------------------------------------------- registry


def test_registry_surface():
    assert skills.SPEC_VERSION == "1.2"
    assert set(OPS) == {
        "lab_trend_k3",
        "med_dose_k3",
        "nearest_k",
        "first_after",
        "last_before",
        "window_count",
        "window_sum",
        "window_mean",
        "outcome_lookup",
        "cohort_criteria",
    }
    assert all(callable(fn) for fn in OPS.values())


# --------------------------------------------------------------------------- §1 lab_trend_k3


def test_lab_trend_insufficient_matches_reference_shape():
    pool = [ev("a", available=at(-1), value=1.0, unit="u"), ev("b", available=at(-2), value=2.0, unit="u")]
    out = OPS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out == {"status": "insufficient", "provenance": []}
    assert out == REFERENCE_SKILLS["lab_trend_k3"](pool)


def test_lab_trend_ties_by_id_and_chronological_output():
    # b and c are equidistant (+-2h); d is nearer; "a" and "z" tie at exactly -3h -> "a" wins over "z".
    pool = [
        ev("z", available=at(-3), value=9.0, unit="u"),
        ev("a", available=at(-3), value=5.0, unit="u"),
        ev("c", available=at(+2), value=7.0, unit="u"),
        ev("b", available=at(-2), value=6.0, unit="u"),
        ev("d", available=at(-1), value=8.0, unit="u"),
    ]
    out = OPS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out["status"] == "ok"
    assert out["ordered_result_ids"] == ["b", "d", "c"]
    assert out["delta_last_minus_first"] == pytest.approx(1.0)
    assert out["direction"] == "increasing"
    assert [p["citation"] for p in out["provenance"]] == ["b", "d", "c"]
    assert out["provenance"][0] == {"citation": "b", "value": 6.0, "unit": "u", "issued_time": at(-2)}


def test_lab_trend_same_time_tie_uses_id_not_insertion_order():
    pool = [
        ev("y", available=at(-1), value=1.0, unit="u"),
        ev("x", available=at(-1), value=3.0, unit="u"),
        ev("w", available=at(-5), value=2.0, unit="u"),
    ]
    out = OPS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out["ordered_result_ids"] == ["w", "x", "y"]
    assert out["direction"] == "decreasing"


def test_lab_trend_skips_non_numeric_and_reports_stable():
    pool = [
        ev("n1", available=at(-0.5), value="not a number", unit="u"),
        ev("n2", available=at(-0.25), value=None, unit="u"),
        ev("a", available=at(-1), value=4.0, unit="u"),
        ev("b", available=at(-2), value=4.5, unit="u"),
        ev("c", available=at(-3), value="4.0", unit="u"),
    ]
    out = OPS["lab_trend_k3"](pool, contract(), {"k": 3})
    assert out["ordered_result_ids"] == ["c", "b", "a"]
    assert out["direction"] == "stable"
    assert out["provenance"][0]["value"] == "4.0"  # verbatim copy


def test_lab_trend_respects_contract_legality():
    pool = [
        ev("post", available=at(+1), value=1.0, unit="u"),
        ev("a", available=at(-1), value=1.0, unit="u"),
        ev("b", available=at(-2), value=2.0, unit="u"),
        ev("c", available=at(-3), value=3.0, unit="u"),
        ev("d", available=at(-4), value=4.0, unit="u"),
    ]
    out = OPS["lab_trend_k3"](pool, contract(scope="pre"), {"k": 3})
    assert out["ordered_result_ids"] == ["c", "b", "a"]
    # §1/§2: k defaults to 3, k < 1 raises; a missing `available` is dropped before selection
    assert OPS["lab_trend_k3"](pool, contract(scope="pre"), {})["ordered_result_ids"] == ["c", "b", "a"]
    with pytest.raises(ValueError):
        OPS["lab_trend_k3"](pool, contract(scope="pre"), {"k": 0})
    with pytest.raises(ValueError):
        OPS["med_dose_k3"](pool, contract(basis="occurred"), {"k": -1})
    untimed = pool + [ev("no_time", value=0.0, unit="u")]
    assert OPS["lab_trend_k3"](untimed, contract(), {"k": 3})["ordered_result_ids"] == ["b", "a", "post"]


def _expected_from_reference(pool: list[EvidenceObject], c: Contract) -> dict:
    """Independent tiny selection + the frozen reference over the 3 selected objects."""
    idx = c.index
    legal_pool = [e for e in pool if legal(e, c)]
    ranked = sorted(legal_pool, key=lambda e: (abs((e.time("available") - idx).total_seconds()), e.id))
    chosen = ranked[:3]
    if len(chosen) < 3:
        return REFERENCE_SKILLS["lab_trend_k3"]([])
    chosen.sort(key=lambda e: (e.time("available"), e.id))
    return REFERENCE_SKILLS["lab_trend_k3"](chosen)


def test_lab_trend_equals_reference_on_200_random_legal_pools():
    rng = random.Random(20260927)
    offsets = [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6]  # hours; small set forces duplicates
    values = [1.0, 2.5, 2.5, 3.75, 10.0]
    n_ok = 0
    for trial in range(200):
        scope = rng.choice(["full", "pre", "post"])
        c = contract(scope=scope, basis="available")
        n = rng.randint(0, 6)
        ids = rng.sample([f"o{j:02d}" for j in range(20)], n)
        pool = []
        for id_ in ids:
            hours = rng.choice(offsets)
            if scope == "pre":
                hours = -abs(hours)
            elif scope == "post":
                hours = abs(hours)
            pool.append(ev(id_, available=at(hours), value=rng.choice(values), unit=rng.choice(["mg/dL", "mmol/L"])))
        rng.shuffle(pool)
        got = OPS["lab_trend_k3"](pool, c, {"k": 3})
        assert got == _expected_from_reference(pool, c), (trial, pool)
        n_ok += got["status"] == "ok"
    assert n_ok > 50  # the sample exercised the ok branch


# --------------------------------------------------------------------------- §2 med_dose_k3


def test_med_dose_basic_ties_and_non_numeric():
    c = contract(basis="occurred")
    pool = [
        ev("m3", occurred=at(-3), dose=10.0, unit="mg", source="emar"),
        ev("m1", occurred=at(+1), dose=20.0, unit="mg", source="emar"),
        ev("m2", occurred=at(-1), dose="5", unit="mg", source="emar"),
        ev("bad", occurred=at(-0.1), dose="n/a", unit="mg", source="emar"),
        ev("m4", occurred=at(-3), dose=1.0, unit="mg", source="emar"),  # ties m3 at 3h; m3 < m4 by id
    ]
    out = OPS["med_dose_k3"](pool, c, {"k": 3})
    assert out["status"] == "ok"
    assert out["ordered_result_ids"] == ["m3", "m2", "m1"]
    assert out["total_dose"] == pytest.approx(35.0)
    assert out["mean_dose"] == pytest.approx(35.0 / 3)
    assert out["provenance"][1] == {"citation": "m2", "dose": "5", "unit": "mg", "effective_time": at(-1)}


def test_med_dose_insufficient_shape():
    c = contract(basis="occurred")
    pool = [ev("m1", occurred=at(-1), dose=1.0, unit="mg", source="emar")]
    assert OPS["med_dose_k3"](pool, c, {"k": 3}) == {
        "status": "insufficient",
        "ordered_result_ids": [],
        "total_dose": None,
        "mean_dose": None,
        "provenance": [],
    }


# --------------------------------------------------------------------------- §3 nearest_k


def test_nearest_k_ties_by_id_verbatim_values_and_insufficient():
    c = contract(basis="occurred")
    pool = [
        ev("q", occurred=at(-2), available=at(+9), value="abc", unit=None),
        ev("p", occurred=at(+2), value=3),
        ev("r", occurred=at(-5), value=1.0, unit="u"),
        ev("untimed", value=0.0),
    ]
    out = OPS["nearest_k"](pool, c, {"k": 2, "basis": "occurred"})
    assert out["ordered_result_ids"] == ["q", "p"]  # ordered chronologically on occurred
    assert out["provenance"] == [
        {"citation": "q", "value": "abc", "unit": None, "time": at(-2)},
        {"citation": "p", "value": 3, "unit": None, "time": at(+2)},
    ]
    # ties at |dt| = 2h: p vs q -> "p" < "q" by id, so k=1 selects p (not the earlier-inserted q)
    assert OPS["nearest_k"](pool, c, {"k": 1, "basis": "occurred"})["ordered_result_ids"] == ["p"]
    assert OPS["nearest_k"](pool, c, {"k": 4, "basis": "occurred"}) == {
        "status": "insufficient",
        "ordered_result_ids": [],
        "provenance": [],
    }
    assert OPS["nearest_k"](pool, c, {"k": 0, "basis": "occurred"})["status"] == "ok"


def test_nearest_k_uses_param_basis_for_distance_and_order():
    c = contract(basis="occurred")
    pool = [
        ev("a", occurred=at(-1), available=at(+5), value=1),
        ev("b", occurred=at(-4), available=at(+1), value=2),
    ]
    assert OPS["nearest_k"](pool, c, {"k": 1, "basis": "available"})["ordered_result_ids"] == ["b"]
    assert OPS["nearest_k"](pool, c, {"k": 1, "basis": "occurred"})["ordered_result_ids"] == ["a"]


# --------------------------------------------------------------------------- §4 first_after / last_before


def test_boundaries_are_strict_and_tie_by_id():
    c = contract(basis="available")
    pool = [
        ev("at_index", available=at(0), value=0),
        ev("b_after", available=at(+1), value=1),
        ev("a_after", available=at(+1), value=2),
        ev("c_after", available=at(+3), value=3),
        ev("b_before", available=at(-1), value=4),
        ev("a_before", available=at(-1), value=5),
        ev("untimed", value=6),
    ]
    fa = OPS["first_after"](pool, c, {"basis": "available", "which": "first_after"})
    assert fa == {"status": "ok", "citation": "a_after", "time": at(+1), "fields": {"value": 2}, "provenance": ["a_after"]}
    lb = OPS["last_before"](pool, c, {"basis": "available", "which": "last_before"})
    assert lb["citation"] == "b_before"  # max of the (time, id) order -> largest id among equal times
    assert lb["provenance"] == ["b_before"]


def test_boundaries_insufficient_shape():
    c = contract(basis="available")
    pool = [ev("at_index", available=at(0), value=0)]
    expected = {"status": "insufficient", "citation": None, "time": None, "fields": None, "provenance": []}
    assert OPS["first_after"](pool, c, {"basis": "available"}) == expected
    assert OPS["last_before"](pool, c, {"basis": "available"}) == expected
    with pytest.raises(ValueError):
        OPS["first_after"](pool, c, {"basis": "available", "which": "last_before"})


# --------------------------------------------------------------------------- §5 window aggregations


@pytest.fixture
def window_pool():
    return [
        ev("w1", available=at(-30), value=100.0),  # outside a 24h pre window
        ev("w2", available=at(-24), value=10.0),  # exactly on the window edge -> legal (t >= i - window)
        ev("w3", available=at(-2), value="x"),  # non-numeric
        ev("w4", available=at(-1), value="2.5"),
        ev("w5", available=at(0), value=1.0),  # t == index is NOT pre
        ev("w6", available=at(+1), value=1000.0),
    ]


def test_window_count_sum_mean_within_pre_window(window_pool):
    c = contract(scope="pre", basis="available", window_hours=24)
    p = {"basis": "available"}
    assert OPS["window_count"](window_pool, c, {**p, "agg": "count"}) == {
        "status": "ok",
        "agg": "count",
        "value": 3,
        "n": 3,
        "n_skipped_non_numeric": 0,
        "provenance": ["w2", "w3", "w4"],
    }
    assert OPS["window_sum"](window_pool, c, {**p, "agg": "sum"}) == {
        "status": "ok",
        "agg": "sum",
        "value": pytest.approx(12.5),
        "n": 2,
        "n_skipped_non_numeric": 1,
        "provenance": ["w2", "w4"],
    }
    mean = OPS["window_mean"](window_pool, c, {**p, "agg": "mean"})
    assert mean["value"] == pytest.approx(6.25) and mean["n"] == 2 and mean["provenance"] == ["w2", "w4"]


def test_window_zero_cases():
    c = contract(scope="post", basis="available", window_hours=1)
    only_non_numeric = [ev("x", available=at(+0.5), value="?"), ev("far", available=at(+5), value=1.0)]
    assert OPS["window_count"]([], c, {"basis": "available"})["value"] == 0
    assert OPS["window_count"]([], c, {"basis": "available"})["status"] == "ok"
    assert OPS["window_sum"](only_non_numeric, c, {"basis": "available"}) == {
        "status": "ok",
        "agg": "sum",
        "value": 0.0,
        "n": 0,
        "n_skipped_non_numeric": 1,
        "provenance": [],
    }
    assert OPS["window_mean"](only_non_numeric, c, {"basis": "available"}) == {
        "status": "insufficient",
        "agg": "mean",  # label key keeps its value (spec §0)
        "value": None,
        "n": None,
        "n_skipped_non_numeric": None,
        "provenance": [],
    }


def test_window_count_full_scope_counts_untimed_last():
    c = contract(scope="full", basis="available")
    pool = [ev("u2", value=1), ev("t1", available=at(+2), value=1), ev("u1", value="z"), ev("t0", available=at(-2), value=1)]
    out = OPS["window_count"](pool, c, {"basis": "available"})
    assert out["value"] == 4
    assert out["provenance"] == ["t0", "t1", "u1", "u2"]


# --------------------------------------------------------------------------- §6 outcome_lookup


def test_outcome_first_careunit_and_insufficient():
    c = contract(basis="occurred")
    pool = [
        ev("tr_prev", occurred=at(-1), source="transfers", careunit="ED"),
        ev("tr_b", occurred=at(+2), source="transfers", careunit="MICU"),
        ev("tr_a", occurred=at(+2), source="transfers", careunit="SICU"),
        ev("not_tr", occurred=at(+1), source="admissions", careunit="X"),
    ]
    params = {"target": "first_careunit", "field": "careunit", "basis": "occurred", "drug": None}
    out = OPS["outcome_lookup"](pool, c, params)
    assert out == {"status": "ok", "target": "first_careunit", "value": "SICU", "citation": "tr_a", "time": at(+2), "provenance": ["tr_a"]}
    assert OPS["outcome_lookup"](pool[:1], c, params) == {
        "status": "insufficient",
        "target": "first_careunit",  # label key keeps its value (spec §0)
        "value": None,
        "citation": None,
        "time": None,
        "provenance": [],
    }


def test_outcome_disposition_takes_largest_occurred_ties_by_id():
    c = contract(basis="occurred")
    pool = [
        ev("adm_b", occurred=at(-10), source="admissions", disposition="HOME"),
        ev("adm_c", occurred=at(-5), source="admissions", disposition="SNF"),
        ev("adm_a", occurred=at(-5), source="admissions", disposition="DIED"),
        ev("ed_1", occurred=at(-20), source="edstays", disposition="ADMITTED"),
    ]
    out = OPS["outcome_lookup"](pool, c, {"target": "discharge_disposition", "field": "disposition", "basis": "occurred", "drug": None})
    assert out["citation"] == "adm_c" and out["value"] == "SNF" and out["time"] == at(-5)
    ed = OPS["outcome_lookup"](pool, c, {"target": "ed_disposition", "field": "disposition", "basis": "occurred", "drug": None})
    assert ed["value"] == "ADMITTED" and ed["provenance"] == ["ed_1"]
    assert OPS["outcome_lookup"](pool[:3], c, {"target": "ed_disposition", "field": "disposition", "basis": "occurred", "drug": None})["status"] == "insufficient"


def test_outcome_disposition_time_on_param_basis_and_untimed_fallback():
    c = contract(basis="occurred")
    p = {"target": "discharge_disposition", "field": "disposition", "basis": "available", "drug": None}
    # the pick is on `occurred`, but `time` is reported on params["basis"] (None when absent on that basis)
    pool = [
        ev("adm_old", occurred=at(-9), available=at(-8), source="admissions", disposition="HOME"),
        ev("adm_new", occurred=at(-1), source="admissions", disposition="SNF"),
    ]
    out = OPS["outcome_lookup"](pool, c, p)
    assert out["citation"] == "adm_new" and out["value"] == "SNF" and out["time"] is None
    pool[0] = ev("adm_old", occurred=at(-1), available=at(-8), source="admissions", disposition="HOME")
    out = OPS["outcome_lookup"](pool, c, p)
    assert out["citation"] == "adm_old" and out["time"] == at(-8)  # tie at -1h -> max-type pick -> largest id ("adm_old" > "adm_new")
    # no timed admission at all -> the SMALLEST untimed id
    untimed_only = [
        ev("u_b", source="admissions", disposition="B"),
        ev("u_a", source="admissions", disposition="A"),
    ]
    out = OPS["outcome_lookup"](untimed_only, c, p)
    assert out == {"status": "ok", "target": "discharge_disposition", "value": "A", "citation": "u_a", "time": None, "provenance": ["u_a"]}
    # a timed row beats every untimed row even when its id is larger
    out = OPS["outcome_lookup"](untimed_only + [ev("z_timed", occurred=at(-3), source="admissions", disposition="Z")], c, p)
    assert out["citation"] == "z_timed"


def test_outcome_administration_count_case_insensitive_and_zero():
    c = contract(scope="pre", basis="occurred")
    pool = [
        ev("e2", occurred=at(-1), source="emar", medication=" Heparin "),
        ev("e1", occurred=at(-3), source="emar", medication="heparin"),
        ev("e3", occurred=at(-2), source="emar", medication="insulin"),
        ev("e4", occurred=at(+1), source="emar", medication="heparin"),  # post -> illegal for pre
        ev("l1", occurred=at(-1), source="labevents", medication="heparin"),
    ]
    p = {"target": "administration_count", "field": None, "basis": "occurred", "drug": "HEPARIN"}
    assert OPS["outcome_lookup"](pool, c, p) == {
        "status": "ok",
        "target": "administration_count",
        "value": 2,
        "citation": None,
        "time": None,
        "provenance": ["e1", "e2"],
    }
    zero = OPS["outcome_lookup"](pool, c, {**p, "drug": "warfarin"})
    assert zero["status"] == "ok" and zero["value"] == 0 and zero["provenance"] == []


# --------------------------------------------------------------------------- §7 cohort_criteria


@pytest.fixture
def cohort_pool():
    return [
        ev("lab_hi", occurred=at(-2), value=2.5, concept="lactate"),
        ev("lab_lo", occurred=at(-4), value=1.0, concept="lactate"),
        ev("lab_bad", occurred=at(-1), value="pending", concept="lactate"),
        ev("dx_a", occurred=at(-3), source="diagnoses_icd", code="a41.9"),
        ev("dx_b", occurred=at(-6), source="diagnoses_icd", code="I50"),
        ev("rx_1", occurred=at(-5), source="emar", medication="Vancomycin"),
        ev("tri_1", occurred=at(-7), source="triage", acuity="2"),
        ev("tri_x", occurred=at(-8), source="triage", acuity=None),
    ]


def test_cohort_include_union_chronological(cohort_pool):
    c = contract(scope="pre", basis="occurred")
    crit = [
        {"kind": "lab_threshold", "concept": "lactate", "op": ">=", "threshold": 2.0, "basis": "occurred"},
        {"kind": "code_present", "prefix": "A41"},
        {"kind": "exposure", "drug": "vancomycin"},
        {"kind": "acuity", "max_acuity": 2},
    ]
    out = OPS["cohort_criteria"](cohort_pool, c, {"criteria": crit})
    assert out["status"] == "ok" and out["decision"] == "include"
    assert [x["satisfied"] for x in out["criteria"]] == [True, True, True, True]
    assert out["criteria"][0]["provenance"] == ["lab_hi"]
    assert out["criteria"][0]["kind"] == "lab_threshold" and out["criteria"][0]["threshold"] == 2.0
    assert out["provenance"] == ["tri_1", "rx_1", "dx_a", "lab_hi"]


def test_cohort_exclude_cites_domain_of_first_unsatisfied_criterion(cohort_pool):
    c = contract(scope="pre", basis="occurred")
    crit = [
        {"kind": "acuity", "max_acuity": 1},  # triage present, acuity 2 > 1 -> unsatisfied; domain = both triage rows
        {"kind": "lab_threshold", "concept": "lactate", "op": "<", "threshold": 2.0, "basis": "occurred"},
        {"kind": "code_present", "prefix": "Z99"},
    ]
    out = OPS["cohort_criteria"](cohort_pool, c, {"criteria": crit})
    assert out["decision"] == "exclude"
    assert [x["satisfied"] for x in out["criteria"]] == [False, True, False]
    # v1.2: an unsatisfied criterion cites its whole domain (incl. the unparseable-acuity row), chronological
    assert out["criteria"][0]["provenance"] == ["tri_x", "tri_1"]
    assert out["criteria"][1]["provenance"] == ["lab_lo"]  # satisfied -> satisfying ids only
    assert out["criteria"][2]["provenance"] == ["dx_b", "dx_a"]  # unsatisfied -> all diagnoses examined
    assert out["provenance"] == ["tri_x", "tri_1"]  # top level = FIRST unsatisfied criterion's domain
    # the decisive criterion is the first unsatisfied one in list order, not the first in the list
    out2 = OPS["cohort_criteria"](cohort_pool, c, {"criteria": crit[1:]})
    assert out2["decision"] == "exclude" and out2["provenance"] == ["dx_b", "dx_a"]


def test_cohort_uncertain_when_source_absent_and_exclude_wins(cohort_pool):
    c = contract(scope="pre", basis="occurred")
    no_triage = [e for e in cohort_pool if e.source != "triage"]
    crit = [{"kind": "acuity", "max_acuity": 3}, {"kind": "exposure", "drug": "vancomycin"}]
    out = OPS["cohort_criteria"](no_triage, c, {"criteria": crit})
    assert out["decision"] == "uncertain"
    assert out["criteria"][0]["satisfied"] is None and out["criteria"][0]["provenance"] == []
    assert out["provenance"] == ["rx_1"]
    # unknown concept -> uncertain; but an unsatisfied criterion anywhere -> exclude
    crit2 = [
        {"kind": "lab_threshold", "concept": "troponin", "op": ">", "threshold": 0.0, "basis": "occurred"},
        {"kind": "exposure", "drug": "warfarin"},
    ]
    out2 = OPS["cohort_criteria"](cohort_pool, c, {"criteria": crit2})
    assert out2["decision"] == "exclude" and out2["criteria"][0]["satisfied"] is None
    assert out2["criteria"][0]["provenance"] == []  # undecidable keeps []
    assert out2["provenance"] == ["rx_1"]  # the emar domain that failed the warfarin exposure
    assert OPS["cohort_criteria"](cohort_pool, c, {"criteria": []})["decision"] == "include"


def test_cohort_respects_legality_window(cohort_pool):
    # 3h window: lab_hi (-2h) is legal, lab_lo (-4h) is not
    c = contract(scope="pre", basis="occurred", window_hours=3)
    crit = [{"kind": "lab_threshold", "concept": "lactate", "op": "<", "threshold": 2.0, "basis": "occurred"}]
    out = OPS["cohort_criteria"](cohort_pool, c, {"criteria": crit})
    assert out["decision"] == "exclude" and out["criteria"][0]["satisfied"] is False
    assert out["provenance"] == ["lab_hi", "lab_bad"]  # only the legal (in-window) domain is cited


# --------------------------------------------------------------------------- purity


def test_operations_do_not_mutate_inputs():
    c = contract(basis="available")
    e = ev("a", available=at(+1), value=1.0, unit="u")
    pool = [e, ev("b", available=at(-1), value=2.0, unit="u"), ev("c", available=at(-2), value=3.0, unit="u")]
    snapshot = [x.to_dict() for x in pool]
    params = {"k": 3, "basis": "available", "which": "first_after"}
    out = OPS["first_after"](pool, c, params)
    out["fields"]["value"] = "mutated"
    OPS["lab_trend_k3"](pool, c, params)
    assert [x.to_dict() for x in snapshot and pool] == snapshot
    assert params == {"k": 3, "basis": "available", "which": "first_after"}
