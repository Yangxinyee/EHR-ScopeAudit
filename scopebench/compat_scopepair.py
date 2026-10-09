"""Load the frozen ScopePair manifests as ScopeBench cases/tasks (regression harness, plan §12 M1).

LabTrend candidates -> EvidenceObject(source="lab", occurred=effective_time, available=issued_time).
The contract is availability-based, scope pre/post relative to index_time, no source filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .schema import Case, Contract, EvidenceObject, Task, read_jsonl


def labtrend_tasks(path: Path, split: str | None = None) -> tuple[list[Task], dict[str, Case]]:
    tasks: list[Task] = []
    cases: dict[str, Case] = {}
    for row in read_jsonl(path):
        if split is not None and row.get("split") != split:
            continue
        evidence = [
            EvidenceObject(
                id=str(c["id"]),
                source="lab",
                patient=str(row.get("patient_cluster_id", "")),
                occurred=c.get("effective_time"),
                available=c.get("issued_time"),
                fields={"display": c.get("display"), "value": c.get("value"), "unit": c.get("unit")},
            )
            for c in row["candidates"]
        ]
        gold = row["gold_answer"]
        task = Task(
            task_id=str(row["task_id"]),
            case_id=str(row["case_id"]),
            patient=str(row.get("patient_cluster_id", "")),
            tier="scopepair_labtrend",
            task_type="lab_trend_k3",
            question=str(row["question"]),
            contract=Contract(index_time=str(row["index_time"]), scope=str(row["scope"]), basis="available"),
            gold=gold,
            required_ids=tuple(r["citation"] for r in gold.get("results", [])),
            split=str(row.get("split", "test")),
            metadata={"lab_code": row.get("lab_code"), "lab_display": row.get("lab_display"), "unit": row.get("unit"),
                      "timestamp_stress": row.get("timestamp_stress", False), "oracle_version": row.get("oracle_version")},
        )
        tasks.append(task)
        cases.setdefault(task.case_id, Case(case_id=task.case_id, patient=task.patient, index_event={"index_time": task.contract.index_time}, evidence=evidence))
    return tasks, cases
