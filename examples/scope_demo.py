"""A CPU-only synthetic demonstration of access/citation divergence.

The backend returns a fixed oracle answer. These outputs demonstrate the
pipeline and are not model evaluation results or clinical observations.
"""
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.backends.oracle_backend import OracleBackend
from scopebench.configurations import CONFIGS
from scopebench.retrieval import TemporalNearestRetriever
from scopebench.runner import RunSpec, run
from scopebench.schema import Case, Contract, EvidenceObject, Task, read_jsonl
from scopebench.tasktypes import REFERENCE_SKILLS


def demonstrate():
    patient = "synthetic-example"
    index = "2027-01-04T12:00:00Z"
    evidence = [EvidenceObject(
        id=f"example-{day}", source="lab", patient=patient,
        occurred=f"2027-01-{day:02}T12:00:00Z",
        available=f"2027-01-{day:02}T12:00:00Z",
        fields={"display": "Synthetic measure", "value": value, "unit": "a.u."},
    ) for day, value in ((1, 10), (2, 11), (3, 12), (5, 99), (6, 100), (7, 101))]
    earlier = evidence[:3]
    gold = {"status": "ok", "results": [
        {"citation": e.id, "value": e.fields["value"], "unit": "a.u.", "issued_time": e.available}
        for e in earlier], "delta": 2.0, "direction": "increasing"}
    task = Task(task_id="synthetic-task", case_id="synthetic-case", patient=patient,
                tier="synthetic", task_type="lab_trend_k3",
                question="Return the last three available pre-index measurements and their trend.",
                contract=Contract(index, "pre", "available", permitted_sources=("lab",)),
                gold=gold, required_ids=tuple(e.id for e in earlier))
    case = Case(case_id=task.case_id, patient=patient, index_event={"index_time": index}, evidence=evidence)
    results = {}
    with tempfile.TemporaryDirectory(prefix="ehr-scopeaudit-synthetic-") as directory:
        for name in ("prompt_only", "scope_aware", "full"):
            destination = Path(directory) / name
            spec = RunSpec(CONFIGS[name], TemporalNearestRetriever(), k=3, skills=REFERENCE_SKILLS)
            run(spec, [task], {case.case_id: case}, OracleBackend([task], {case.case_id: case}), destination)
            row = next(iter(read_jsonl(destination / "predictions.jsonl")))
            metrics = row["metrics"]
            results[name] = {"answer_correct": metrics["answer_correct"],
                             "citation_legal": metrics["citation_legal"],
                             "access_legal": metrics["access_legal"],
                             "completeness": metrics["completeness"],
                             "accessed_objects": len(row["access_ids"])}
    assert results["prompt_only"]["answer_correct"] and results["prompt_only"]["citation_legal"]
    assert not results["prompt_only"]["access_legal"]
    assert results["full"]["access_legal"] and results["full"]["answer_correct"]
    return results


if __name__ == "__main__":
    print("Synthetic oracle demonstration; no patient data or model inference.")
    print(json.dumps(demonstrate(), indent=2))
