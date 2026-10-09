"""Tier-B manifest invariants (plan §5.5). Point SCOPEBENCH_TIERB_DIR at a built manifest; skipped otherwise.

Checks: one case per patient; two tasks per case with distinct scopes; unique evidence ids per case;
every task's R_t is a subset of its legal set L_t (and non-empty unless the gold is an abstention);
gold is reproducible from the oracle for the covered families; scope balance and family counts when
--expect-per-family is given; no raw MIMIC identifiers in manifests.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scopebench import oracle  # noqa: E402
from scopebench.manifest import load_manifest  # noqa: E402
from scopebench.predicate import filter_legal, legal_set  # noqa: E402

DIR = os.environ.get("SCOPEBENCH_TIERB_DIR")
pytestmark = pytest.mark.skipif(not DIR or not (Path(DIR) / "tasks.jsonl").exists(), reason="set SCOPEBENCH_TIERB_DIR to a built manifest")
EXPECT = int(os.environ.get("SCOPEBENCH_EXPECT_PER_FAMILY", "0"))


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(Path(DIR))


def test_structure(manifest):
    tasks, cases = manifest
    per_case = Counter(t.case_id for t in tasks)
    assert set(per_case.values()) == {2}, "every case has exactly two tasks"
    assert all(len({t.scope for t in tasks if t.case_id == c}) == 2 for c in list(per_case)[:200]), "the two tasks of a case have distinct scopes"
    patients = Counter(cases[c].patient for c in per_case)
    assert max(patients.values()) == 1, "one case per patient"
    for c in cases.values():
        ids = [e.id for e in c.evidence]
        assert len(ids) == len(set(ids)), f"duplicate evidence ids in {c.case_id}"
        assert all(e.patient == c.patient for e in c.evidence)


def test_required_within_legal(manifest):
    tasks, cases = manifest
    for t in tasks:
        L = legal_set(cases[t.case_id].evidence, t.contract)
        assert set(t.required_ids) <= L, f"{t.task_id}: required evidence outside L_t"
        abstain = (t.task_type == "claim_v1" and t.gold.get("label") == "insufficient") or (t.gold.get("status") == "insufficient")
        assert t.required_ids or abstain, f"{t.task_id}: empty R_t on a substantive task"


def test_gold_reproducible_for_covered_families(manifest):
    tasks, cases = manifest
    for t in tasks:
        if t.task_type == "lab_trend_k3":
            g = oracle.OPERATIONS["lab_trend_k3"](filter_legal(cases[t.case_id].evidence, t.contract), t.contract, {"k": 3})
            assert tuple(g["ordered_result_ids"]) == t.required_ids and g["direction"] == t.gold["direction"]
        elif t.task_type == "cohort_v1":
            g = oracle.OPERATIONS["cohort_criteria"](filter_legal(cases[t.case_id].evidence, t.contract), t.contract, {"criteria": t.metadata["criteria"]})
            assert g["decision"] == t.gold["decision"] and tuple(g["provenance"]) == t.required_ids


def test_balance_and_counts(manifest):
    tasks, _ = manifest
    fam = Counter(t.metadata["family"] for t in tasks)
    scopes = Counter(t.scope for t in tasks)
    if EXPECT:
        assert all(v == 2 * EXPECT for v in fam.values()), fam
        assert all(v == 3 * EXPECT for v in scopes.values()), scopes
    assert set(scopes) == {"pre", "at", "post", "full"}


def test_no_raw_identifiers():
    text = Path(DIR, "tasks.jsonl").read_text()[:5_000_000]
    assert not re.search(r"\"subject_id\"|\"hadm_id\"|\"note_id\"|\"stay_id\"", text), "raw MIMIC identifiers in the manifest"
    stats = json.loads(Path(DIR, "construction_stats.json").read_text())
    assert "subject_ids" not in json.dumps(stats)
