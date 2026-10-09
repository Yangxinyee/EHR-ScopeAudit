"""Load a ScopeBench manifest directory (tasks.jsonl + evidence/<case_id>.jsonl) into tasks and cases."""

from __future__ import annotations

from pathlib import Path

from . import tasktypes_tierb  # noqa: F401  (registers the Tier-B task types)
from .schema import Case, EvidenceObject, Task, read_jsonl


def load_manifest(directory: Path, split: str | None = None, families: list[str] | None = None) -> tuple[list[Task], dict[str, Case]]:
    directory = Path(directory)
    tasks = [Task.from_dict(r) for r in read_jsonl(directory / "tasks.jsonl")]
    if split is not None:
        tasks = [t for t in tasks if t.split == split]
    if families:
        tasks = [t for t in tasks if t.metadata.get("family") in families]
    cases: dict[str, Case] = {}
    for t in tasks:
        if t.case_id in cases:
            continue
        evidence = [EvidenceObject.from_dict(r) for r in read_jsonl(directory / "evidence" / f"{t.case_id}.jsonl")]
        cases[t.case_id] = Case(case_id=t.case_id, patient=t.patient, index_event={"index_time": t.contract.index_time}, evidence=evidence)
    return tasks, cases
