"""Tier-A manifest invariants (plan §7.2 / §12 gate G6). Runs only when the enclave manifest exists; never prints a row.

Invariants: four tasks per case (one per scope), tier "A", one case per patient, dev/test patients disjoint, R_t non-empty
and inside the legal set of every task, evidence times canonical ISO, no raw identifier keys in any evidence object,
patient ids are 16-hex salted hashes, the text-free release copy carries no free-text field, and the recorded modality /
family cell counts match the construction statistics."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scopebench.predicate import legal_set
from scopebench.schema import Case, EvidenceObject, Task, read_jsonl
from scopebench.tiera.config import ENCLAVE
import scopebench.tasktypes_tiera  # noqa: F401,E402  (registers outcome_count_v1)

import os

TEST_DIR = Path(os.environ["SCOPEBENCH_TIERA_DIR"]) if os.environ.get("SCOPEBENCH_TIERA_DIR") else ENCLAVE / "tierA"   # SCOPEBENCH_TIERA_DIR=<enclave>/tierA_dev checks the dev pool
DEV_DIR = ENCLAVE / "tierA_dev"
pytestmark = pytest.mark.skipif(not (TEST_DIR / "tasks.jsonl").exists(), reason="Tier-A enclave manifest not built")
FORBIDDEN_KEYS = {"mrn", "csn", "accession", "accession_number", "order_proc_id", "pat_name", "birth_date", "dob", "provider", "ordering_provider"}
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _load(d: Path):
    tasks = [Task.from_dict(r) for r in read_jsonl(d / "tasks.jsonl")]
    cases = {}
    for t in tasks:
        if t.case_id not in cases:
            ev = [EvidenceObject.from_dict(r) for r in read_jsonl(d / "evidence" / f"{t.case_id}.jsonl")]
            cases[t.case_id] = Case(case_id=t.case_id, patient=t.patient, index_event={"index_time": t.contract.index_time}, evidence=ev)
    return tasks, cases


@pytest.fixture(scope="module")
def test_manifest():
    return _load(TEST_DIR)


def test_shape_and_scopes(test_manifest):
    tasks, cases = test_manifest
    stats = json.load(open(TEST_DIR / "construction_stats.json"))
    assert len(tasks) == stats["n_tasks"] == 4 * len(cases)
    by_case = {}
    for t in tasks:
        assert t.tier == "A"
        by_case.setdefault(t.case_id, set()).add(t.contract.scope)
    assert all(s == {"pre", "at", "post", "full"} for s in by_case.values())
    patients = [c.patient for c in cases.values()]
    assert len(set(patients)) == len(patients), "one case per patient"
    assert all(re.fullmatch(r"[0-9a-f]{16}", p) for p in patients)


def test_required_inside_legal_and_nonempty(test_manifest):
    tasks, cases = test_manifest
    for t in tasks:
        assert t.required_ids, t.task_id
        legal = legal_set(cases[t.case_id].evidence, t.contract)
        assert set(t.required_ids) <= legal, t.task_id


def test_no_raw_identifiers_or_bad_times(test_manifest):
    _, cases = test_manifest
    for c in cases.values():
        ids = [e.id for e in c.evidence]
        assert len(set(ids)) == len(ids), c.case_id      # one object per id (the exports repeat reports under several date formats)
        for e in c.evidence:
            assert re.fullmatch(r"[0-9a-f]{24}", e.id)
            assert not (set(k.lower() for k in e.fields) & FORBIDDEN_KEYS), sorted(e.fields)
            for t in (e.occurred, e.available, e.occurred_end):
                assert t is None or ISO.match(t), t


def test_dev_test_disjoint():
    if not (DEV_DIR / "patients.json").exists() or TEST_DIR.resolve() == DEV_DIR.resolve():
        pytest.skip("dev manifest not built, or checking the dev pool itself")
    dev = set(json.load(open(DEV_DIR / "patients.json"))["patient_hashes"])
    test = set(json.load(open(TEST_DIR / "patients.json"))["patient_hashes"])
    assert not (dev & test)


def test_release_copy_is_text_free():
    rel = ENCLAVE / "release" / "tierA"
    if not (rel / "tasks.jsonl").exists():
        pytest.skip("release copy not built")
    bad = 0
    for p in [rel / "tasks.jsonl", *sorted((rel / "evidence").glob("*.jsonl"))[:50]]:
        for line in open(p):
            if re.search(r'"(text|first_impression_line|claim)":', line):
                bad += 1
    assert bad == 0


def test_cells_match_stats(test_manifest):
    tasks, _ = test_manifest
    stats = json.load(open(TEST_DIR / "construction_stats.json"))
    from collections import Counter
    cells = Counter((t.metadata["modality"], t.metadata["family"]) for t in tasks if t.contract.scope == "pre")
    for key, v in stats["cells"].items():
        m, f = key.split("|")
        assert cells[(m, f)] == v["kept"], key


def test_gold_recomputes_from_written_evidence(test_manifest):
    """Oracle differential (plan §4.3 / §7.2): for the structured families the gold written into the manifest equals the
    oracle re-applied to the legal subset of the written evidence file with the task's own parameters."""
    from scopebench import oracle
    from scopebench.predicate import filter_legal
    tasks, cases = test_manifest
    checked = 0
    for t in tasks:
        legal = filter_legal(cases[t.case_id].evidence, t.contract)
        if t.task_type == "lab_trend_k3":
            g = oracle.OPERATIONS["lab_trend_k3"](legal, t.contract, {"k": 3})
            assert g["status"] == "ok" and tuple(g["ordered_result_ids"]) == tuple(t.required_ids) and g["direction"] == t.gold["direction"], t.task_id
        elif t.task_type in ("outcome_lookup_v1", "outcome_count_v1"):
            g = oracle.OPERATIONS["outcome_lookup"](legal, t.contract, t.metadata["skill_params"])
            assert g["status"] == "ok" and g["value"] == t.gold["value"] and tuple(g["provenance"]) == tuple(t.required_ids), t.task_id
        elif t.task_type == "cohort_v1":
            g = oracle.OPERATIONS["cohort_criteria"](legal, t.contract, {"criteria": t.metadata["criteria"]})
            assert g["decision"] == t.gold["decision"] and tuple(g["provenance"]) == tuple(t.required_ids), t.task_id
        else:
            continue
        checked += 1
    assert checked > 0
