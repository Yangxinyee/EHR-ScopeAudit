from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_scopepair_fhir import (  # noqa: E402
    extract_json,
    guard_errors,
    select_evidence,
    skill_answer,
)


TASKS = ROOT / "experiments" / "scopepair_fhir" / "tasks.jsonl"


def load_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_frozen_manifest_shape_and_pairing() -> None:
    tasks = load_tasks()
    assert len(tasks) == 160
    assert {task["scope"] for task in tasks} == {"pre", "post"}
    assert sum(task["split"] == "dev" for task in tasks) == 32
    assert sum(task["split"] == "test" for task in tasks) == 128
    assert len({task["patient_cluster_id"] for task in tasks}) == 80
    assert len({task["case_id"] for task in tasks}) == 80
    assert sum(task["timestamp_stress"] for task in tasks) == 40
    by_case: dict[str, list[dict]] = {}
    for task in tasks:
        by_case.setdefault(task["case_id"], []).append(task)
        assert "patient_id" not in task
        assert len(task["patient_cluster_id"]) == 16
        assert len(task["candidates"]) == 6
        assert len(task["gold_answer"]["results"]) == 3
    for pair in by_case.values():
        assert {task["scope"] for task in pair} == {"pre", "post"}
        assert pair[0]["candidates"] == pair[1]["candidates"]
        assert pair[0]["index_time"] == pair[1]["index_time"]


def test_oracle_scope_first_and_guard_refill_invariants() -> None:
    for task in load_tasks():
        scope_first = select_evidence(task, "scope_first")
        guard_refill = select_evidence(task, "guard_refill")
        expected_ids = [
            result["citation"] for result in task["gold_answer"]["results"]
        ]
        assert [item["id"] for item in scope_first] == expected_ids
        assert [item["id"] for item in guard_refill] == expected_ids


def test_timestamp_stress_flag_matches_wrong_basis_disagreement() -> None:
    by_case: dict[str, list[dict]] = {}
    for task in load_tasks():
        by_case.setdefault(task["case_id"], []).append(task)
    for pair in by_case.values():
        disagreement = any(
            {
                item["id"]
                for item in select_evidence(task, "wrong_time_scope_first")
            }
            != {item["id"] for item in select_evidence(task, "scope_first")}
            for task in pair
        )
        assert disagreement == bool(pair[0]["timestamp_stress"])


def test_parser_and_guard_reject_illegal_citation() -> None:
    task = load_tasks()[0]
    evidence = select_evidence(task, "scope_first")
    good = task["gold_answer"]
    parsed = extract_json(json.dumps(good))
    assert parsed == good
    assert guard_errors(parsed, task, evidence) == []
    bad = json.loads(json.dumps(good))
    illegal = next(
        candidate["id"]
        for candidate in task["candidates"]
        if candidate["id"] not in {item["id"] for item in evidence}
    )
    bad["results"][0]["citation"] = illegal
    errors = guard_errors(bad, task, evidence)
    assert any("out-of-contract" in error for error in errors)


def test_full_guard_checks_deterministic_aggregation() -> None:
    task = load_tasks()[0]
    evidence = select_evidence(task, "scope_first")
    bad_math = json.loads(json.dumps(task["gold_answer"]))
    bad_math["delta"] = float(bad_math["delta"]) + 1.0
    errors = guard_errors(
        bad_math,
        task,
        evidence,
        validate_skill=True,
    )
    assert any("delta disagrees" in error for error in errors)
    fallback = skill_answer(evidence)
    assert guard_errors(
        fallback,
        task,
        evidence,
        validate_skill=True,
    ) == []
