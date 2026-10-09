from __future__ import annotations

import json
from pathlib import Path

from scripts.run_scopepair_medadmin import (
    guard_errors,
    permitted,
    select_evidence,
    skill_answer,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "experiments" / "scopepair_medadmin" / "tasks.jsonl"


def load_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_manifest_has_patient_disjoint_dev_and_test_pairs() -> None:
    tasks = load_tasks()
    assert len(tasks) == 160
    assert len({task["case_id"] for task in tasks}) == 80
    dev_cases = {task["case_id"] for task in tasks if task["split"] == "dev"}
    test_cases = {task["case_id"] for task in tasks if task["split"] == "test"}
    assert len(dev_cases) == 16
    assert len(test_cases) == 64
    dev_patients = {
        task["patient_cluster_id"] for task in tasks if task["split"] == "dev"
    }
    test_patients = {
        task["patient_cluster_id"] for task in tasks if task["split"] == "test"
    }
    assert dev_patients.isdisjoint(test_patients)
    for case_id in {task["case_id"] for task in tasks}:
        pair = [task for task in tasks if task["case_id"] == case_id]
        assert {task["scope"] for task in pair} == {"pre", "post"}
        assert len({json.dumps(task["candidates"], sort_keys=True) for task in pair}) == 1


def test_every_case_has_six_rows_and_varying_doses() -> None:
    for task in load_tasks():
        assert len(task["candidates"]) == 6
        assert task["dose_variation"]
        assert len({item["dose"] for item in task["candidates"]}) > 1
        assert sum(permitted(item, task) for item in task["candidates"]) == 3


def test_oracle_equals_scope_first_deterministic_skill() -> None:
    for task in load_tasks():
        selected = select_evidence(task, "scope_first")
        assert len(selected) == 3
        assert skill_answer(selected) == task["gold_answer"]


def test_information_complete_and_refill_selectors() -> None:
    for task in load_tasks():
        assert len(select_evidence(task, "prompt_all6")) == 6
        assert select_evidence(task, "guard_refill") == select_evidence(
            task, "scope_first"
        )


def test_full_guard_detects_aggregation_error() -> None:
    task = load_tasks()[0]
    evidence = select_evidence(task, "scope_first")
    answer = skill_answer(evidence)
    answer["total_dose"] += 1
    errors = guard_errors(answer, task, evidence, validate_skill=True)
    assert any("total dose" in error for error in errors)
