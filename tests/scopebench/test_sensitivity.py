"""Sensitivity-suite variants (plan §8.5): strict boundaries, ±6 h at-index window, alternative repair wording,
and the contract's serialization staying byte-identical for frozen manifests."""

from __future__ import annotations

from scopebench.guard import REPAIR_WORDINGS, repair_message
from scopebench.predicate import legal
from scopebench.schema import Contract, EvidenceObject, Task
from scopebench.sensitivity import apply_variant, variant_contract


def _ev(eid: str, t: str) -> EvidenceObject:
    return EvidenceObject(id=eid, source="lab", patient="p", occurred=t, available=t, row_ref=eid, fields={})


def test_strict_boundaries_flip_inclusive_edges():
    c = Contract(index_time="2020-01-10T00:00:00Z", scope="at", basis="available", window_hours=24.0)
    s = variant_contract(c, "strict_boundaries")
    at_index = _ev("a", "2020-01-10T00:00:00Z")
    at_edge = _ev("b", "2020-01-09T00:00:00Z")
    inside = _ev("c", "2020-01-09T12:00:00Z")
    assert legal(at_index, c) and legal(at_edge, c) and legal(inside, c)
    assert not legal(at_index, s) and not legal(at_edge, s) and legal(inside, s)
    post = Contract(index_time="2020-01-10T00:00:00Z", scope="post", basis="available", window_hours=24.0)
    edge = _ev("d", "2020-01-11T00:00:00Z")
    assert legal(edge, post) and not legal(edge, variant_contract(post, "strict_boundaries"))
    # index events named by the contract stay legal under at even when strict
    named = Contract(index_time="2020-01-10T00:00:00Z", scope="at", basis="available", index_evidence_ids=("a",), strict_boundaries=True)
    assert legal(at_index, named)


def test_at_window_6h_only_touches_at_scope():
    at = Contract(index_time="2020-01-10T00:00:00Z", scope="at", basis="available")
    pre = Contract(index_time="2020-01-10T00:00:00Z", scope="pre", basis="available")
    later = _ev("e", "2020-01-10T05:00:00Z")
    assert not legal(later, at) and legal(later, variant_contract(at, "at_window_6h"))
    assert variant_contract(pre, "at_window_6h") == pre
    assert variant_contract(at, "at_window_6h").at_tolerance_hours == 6.0 and at.at_tolerance_hours == 0.0


def test_contract_serialization_unchanged_for_frozen_manifests():
    c = Contract(index_time="2020-01-10T00:00:00Z", scope="pre", basis="available", window_hours=12.0)
    d = c.to_dict()
    assert "strict_boundaries" not in d
    assert Contract.from_dict(d) == c
    s = variant_contract(c, "strict_boundaries")
    assert Contract.from_dict(s.to_dict()) == s and s.to_dict()["strict_boundaries"] is True


def test_extended_rt_widens_lab_trend_only():
    import os
    from pathlib import Path

    from scopebench.schema import read_jsonl
    from scopebench.sensitivity import EXTENDED_K
    d = Path(os.environ.get("SCOPEBENCH_TIERB_DIR", "experiments/scopebench_tierB_dev"))
    if not (d / "tasks.jsonl").exists():
        import pytest
        pytest.skip("development manifest not present")
    from scripts.scopebench.run_config import load  # type: ignore
    tasks, cases = load("scopebench", d, None)
    lab = [t for t in tasks if t.task_type == "lab_trend_k3"][:20]
    other = [t for t in tasks if t.task_type != "lab_trend_k3"][:20]
    assert apply_variant(lab + other, "extended_rt", cases) == lab + other   # scoring-only: the runner sees the reference tasks
    from scopebench.sensitivity import extended_scoring_tasks
    out = extended_scoring_tasks(lab + other, cases)
    widened = [t for t in out if t.task_type == "lab_trend_k3" and len(t.required_ids) == EXTENDED_K]
    assert widened, "no lab-trend task had five legal results"
    for t_new, t_old in zip(out, lab + other):
        if t_new.task_type == "lab_trend_k3" and len(t_new.required_ids) == EXTENDED_K:
            assert set(t_old.required_ids) <= set(t_new.required_ids)   # the nearest three lie inside the nearest five
            assert t_new.metadata["reference_required_ids"] == list(t_old.required_ids)
        elif t_new.task_type != "lab_trend_k3":
            assert t_new.required_ids == t_old.required_ids
        assert t_new.gold == t_old.gold and t_new.contract == t_old.contract


def test_alternative_repair_wording_differs_but_keeps_structure():
    a = repair_message("feedback", ["e1"], [{"id": "x"}], None, wording="a")
    b = repair_message("feedback", ["e1"], [{"id": "x"}], None, wording="b")
    assert a != b
    import json
    ja, jb = json.loads(a), json.loads(b)
    assert set(ja) == set(jb) == {"scopeguard_errors", "instruction", "permitted_evidence"}
    assert ja["instruction"] == REPAIR_WORDINGS["a"]["feedback"] and jb["instruction"] == REPAIR_WORDINGS["b"]["feedback"]
    assert repair_message("generic", [], [], None) == repair_message("generic", [], [], None, wording="a")


def test_contract_variants_redefine_gold_and_rt_consistently():
    """DEV-14: under strict_boundaries / at_window_6h the oracle-backed task types get their gold and R_t recomputed on
    the variant legal set, every R_t lies inside it, and the frozen ChartSkill's answer cites all of R_t (the check whose
    failure aborted the first strict_boundaries/full run)."""
    import os
    from pathlib import Path

    import pytest
    from scopebench import skills as pkg
    from scopebench.runner import _call_skill, _provenance_ids, legal_all
    from scopebench.tasktypes import get as get_tt
    d = Path(os.environ.get("SCOPEBENCH_TIERB_DIR", "experiments/scopebench_tierB_dev"))
    if not (d / "tasks.jsonl").exists():
        pytest.skip("development manifest not present")
    from scripts.scopebench.run_config import load  # type: ignore
    tasks, cases = load("scopebench", d, None)
    for name in ("strict_boundaries", "at_window_6h"):
        out = apply_variant(tasks, name, cases)
        assert len(out) == len(tasks)
        for t in out:
            case = cases[t.case_id]
            pool = case.by_id()
            assert all(legal(pool[i], t.contract) for i in t.required_ids if i in pool), (name, t.task_id)
            tt = get_tt(t.task_type)
            fn = pkg.OPERATIONS.get(tt.skill_op) if tt.skill_op else None
            if fn is not None:
                summ = _call_skill(fn, legal_all(t, case), t)
                if summ and summ.get("status") == "ok":
                    assert set(t.required_ids) <= set(_provenance_ids(summ)), (name, t.task_id)


def test_uncertain_variant_gold_drops_the_task():
    """DEV-14 addendum: a cohort task whose recomputed gold is "uncertain" under the variant window is dropped from the
    variant (the Tier-B builder treats an uncertain gold as unscorable), and rescore_rows drops its saved row."""
    from scopebench.sensitivity import rescore_rows
    import os
    from pathlib import Path
    import pytest
    d = Path(os.environ.get("SCOPEBENCH_TIERB_FORMAL_DIR", "experiments/scopebench_tierB"))
    if not (d / "tasks.jsonl").exists():
        pytest.skip("formal manifest not present")
    from scripts.scopebench.run_config import load  # type: ignore
    tasks, cases = load("scopebench", d, None)
    dropped: list[str] = []
    out = apply_variant(tasks, "strict_boundaries", cases, dropped)
    assert len(out) + len(dropped) == len(tasks)
    assert all(t.gold.get("decision") != "uncertain" for t in out if t.task_type == "cohort_v1")
    kept = {t.task_id for t in out}
    assert not (kept & set(dropped))
    rows = [{"task_id": tid, "parsed_answer": None, "access_ids": [], "metrics": {"tokens_in": 0, "tokens_out": 0, "calls": 1, "latency_seconds": 0.0}}
            for tid in list(kept)[:3] + dropped[:2]]
    new = rescore_rows(rows, {t.task_id: t for t in out}, cases)
    assert [r["task_id"] for r in new] == list(kept)[:3]
