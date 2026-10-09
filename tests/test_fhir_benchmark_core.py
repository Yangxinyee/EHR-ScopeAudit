from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fhir_benchmark_core import (
    LocalFhirStore,
    canonical_answer_correct,
    parse_tool_call,
    parse_tool_calls,
    retrieval_precision,
    retrieval_recall,
    run_safe_python,
)
from run_fhir_agentbench_corrected import paired_comparison


def test_parse_qwen_tool_call() -> None:
    text = (
        '<tool_call>\n{"name":"get_resources_by_resource_id",'
        '"arguments":{"resource_type":"Observation","resource_id":"abc"}}\n'
        "</tool_call>"
    )
    assert parse_tool_call(text) == {
        "name": "get_resources_by_resource_id",
        "arguments": {"resource_type": "Observation", "resource_id": "abc"},
    }
    assert parse_tool_call('''execute_python_code("""
print(2 + 2)
""")''') == {
        "name": "execute_python_code",
        "arguments": {"code": "print(2 + 2)"},
    }
    assert parse_tool_call(
        'execute_python_code("len([r for r in rows if r[\\"id\\"]])")'
    ) == {
        "name": "execute_python_code",
        "arguments": {"code": 'len([r for r in rows if r["id"]])'},
    }
    assert [call["arguments"]["resource_type"] for call in parse_tool_calls(
        '<tool_call>{"name":"get_resources_by_patient_fhir_id",'
        '"arguments":{"resource_type":"Encounter","patient_fhir_id":"p"}}</tool_call>'
        '<tool_call>{"name":"get_resources_by_patient_fhir_id",'
        '"arguments":{"resource_type":"Procedure","patient_fhir_id":"p"}}</tool_call>'
    )] == ["Encounter", "Procedure"]


def test_canonical_answers() -> None:
    assert canonical_answer_correct("[[1]]", "[[1]]")
    assert canonical_answer_correct('[["female"]]', "[['F']]")
    assert canonical_answer_correct("[[3.49]]", "[[3]]")
    assert canonical_answer_correct('[["2133-12-31T02:00:30"]]', "[['2133-12-31 02:00:00']]")
    assert canonical_answer_correct(
        '[["2134-09-07T18:30:00-04:00"]]',
        "[['2134-09-07 18:30:00']]",
    )
    assert canonical_answer_correct("[[67], [42]]", "[[67.]\\n [42.]]")
    assert canonical_answer_correct("[]", "[]")
    assert not canonical_answer_correct("[[1]]", "[[0]]")
    assert not canonical_answer_correct("[[1], [2]]", "[[1]]")


def test_native_retrieval_conventions() -> None:
    assert retrieval_precision([], []) == 1.0
    assert retrieval_recall([], []) == 1.0
    assert retrieval_precision(["a"], []) == 0.0
    assert math.isnan(retrieval_recall(["a"], []))
    assert math.isnan(retrieval_precision([], ["a"]))
    assert retrieval_recall([], ["a"]) == 0.0
    assert retrieval_precision(["a", "b"], ["a"]) == 0.5
    assert retrieval_recall(["a"], ["a", "b"]) == 0.5


def test_reference_aware_mode_keeps_broad_candidate_set() -> None:
    store = LocalFhirStore.__new__(LocalFhirStore)
    store.by_type_id = {}
    resources = [
        {"resourceType": "Observation", "id": str(index), "subject": {"reference": "Patient/p"}}
        for index in range(200)
    ]
    store.by_patient_type = {("p", "Observation"): resources}
    result = store.get_by_patient(
        "Observation",
        "p",
        question="latest glucose",
        mode="reference_aware",
        model_limit=8,
        retrieval_limit=128,
    )
    assert len(result.resources) == 200
    assert len(result.returned_to_model) == 8


def test_reference_aware_paired_comparison() -> None:
    scores = pd.DataFrame(
        [
            {
                "question_id": "a",
                "system": "broad",
                "answer_correct": 0,
                "retrieval_precision": 0.5,
                "retrieval_recall": 1.0,
            },
            {
                "question_id": "a",
                "system": "reference_aware",
                "answer_correct": 1,
                "retrieval_precision": 0.5,
                "retrieval_recall": 1.0,
            },
        ]
    )
    rows = paired_comparison(scores, 2027)
    assert rows[0]["comparison_system"] == "reference_aware"
    assert rows[0]["comparison_minus_baseline"] == 1.0


def test_safe_python_reads_resources_and_blocks_files() -> None:
    resources = {"Observation": [{"id": "x", "valueQuantity": {"value": 7}}]}
    assert run_safe_python(
        "print(retrieved_resources['Observation'][0]['valueQuantity']['value'])",
        resources,
    ) == "7"
    assert run_safe_python(
        "retrieved_resources['Observation'][0]['valueQuantity']['value']",
        resources,
    ) == "7"
    assert run_safe_python(
        "values = [r['valueQuantity']['value'] for r in retrieved_resources['Observation']]\n"
        "sum(values)",
        resources,
    ) == "7"
    try:
        run_safe_python("open('/tmp/forbidden', 'w')", resources)
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe file access was not blocked")
