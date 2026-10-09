#!/usr/bin/env python3
"""Post-hoc, descriptive error analysis for frozen ScopePair-FHIR runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_scopepair_fhir import exact_answer_correct
from run_scopepair_fhir import cited_ids, extract_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def result_payload_matches(actual: Any, gold: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    observed = actual.get("results")
    expected = gold["results"]
    if not isinstance(observed, list) or len(observed) != len(expected):
        return False
    for left, right in zip(observed, expected):
        if not isinstance(left, dict):
            return False
        if left.get("citation") != right["citation"]:
            return False
        if left.get("unit") != right["unit"]:
            return False
        try:
            if abs(float(left.get("value")) - float(right["value"])) > 1e-7:
                return False
        except (TypeError, ValueError):
            return False
        if left.get("issued_time") != right["issued_time"]:
            return False
    return True


def classify_failure(actual: Any, gold: dict[str, Any]) -> str:
    if not isinstance(actual, dict):
        return "parse_failure"
    if actual.get("status") == "insufficient":
        return "overrestriction"
    if result_payload_matches(actual, gold):
        return "aggregation_only"
    gold_ids = [item["citation"] for item in gold["results"]]
    if cited_ids(actual) != gold_ids:
        return "citation_or_order"
    return "value_unit_or_time"


def summarize_run(
    tasks: dict[str, dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    model = Path(metadata["model_path"]).name
    records = load_jsonl(run_dir / "predictions.jsonl")
    systems: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        systems.setdefault(record["system"], []).append(record)
    output: dict[str, Any] = {"model": model, "systems": {}}
    for system, predictions in systems.items():
        failures: Counter[str] = Counter()
        correct = 0
        for prediction in predictions:
            gold = tasks[prediction["task_id"]]["gold_answer"]
            actual = prediction.get("parsed_answer")
            if exact_answer_correct(actual, gold):
                correct += 1
            else:
                failures[classify_failure(actual, gold)] += 1
        output["systems"][system] = {
            "n": len(predictions),
            "final_exact_accuracy": correct / len(predictions),
            "failure_types": dict(sorted(failures.items())),
        }

    full = systems.get("clinehr_os", [])
    if full:
        initial_correct = 0
        post_repair_correct = 0
        error_types: Counter[str] = Counter()
        for prediction in full:
            gold = tasks[prediction["task_id"]]["gold_answer"]
            initial = extract_json(prediction["response"])
            if exact_answer_correct(initial, gold):
                initial_correct += 1
            candidate = (
                extract_json(prediction["repair_response"])
                if prediction.get("repair_response")
                else initial
            )
            if exact_answer_correct(candidate, gold):
                post_repair_correct += 1
            for error in prediction.get("initial_guard_errors", []):
                if "delta disagrees" in error:
                    error_types["delta"] += 1
                elif "direction disagrees" in error:
                    error_types["direction"] += 1
                elif "out-of-contract" in error:
                    error_types["out_of_contract"] += 1
                elif "do not abstain" in error:
                    error_types["overrestriction"] += 1
                elif "parseable JSON" in error:
                    error_types["parse"] += 1
                else:
                    error_types["other_schema_or_field"] += 1
        output["clinehr_os_stages"] = {
            "n": len(full),
            "initial_llm_exact_accuracy": initial_correct / len(full),
            "after_one_repair_exact_accuracy": post_repair_correct / len(full),
            "final_system_exact_accuracy": output["systems"]["clinehr_os"][
                "final_exact_accuracy"
            ],
            "repair_rate": sum(bool(item["guard_repaired"]) for item in full)
            / len(full),
            "deterministic_fallback_rate": sum(
                bool(item.get("guard_deterministic_fallback")) for item in full
            )
            / len(full),
            "initial_guard_error_instances": dict(sorted(error_types.items())),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {
        task["task_id"]: task
        for task in load_jsonl(args.tasks)
    }
    summaries = [summarize_run(tasks, run_dir) for run_dir in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
