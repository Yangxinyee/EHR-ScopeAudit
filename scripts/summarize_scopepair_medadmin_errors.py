#!/usr/bin/env python3
"""Deterministic descriptive error coding for ScopePair-MedAdmin."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from analyze_scopepair_medadmin import exact_answer_correct
    from run_scopepair_medadmin import cited_ids
except ModuleNotFoundError:
    from scripts.analyze_scopepair_medadmin import exact_answer_correct
    from scripts.run_scopepair_medadmin import cited_ids


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
            if abs(float(left.get("dose")) - float(right["dose"])) > 1e-7:
                return False
        except (TypeError, ValueError):
            return False
        if left.get("event_time") != right["event_time"]:
            return False
    return True


def classify(actual: Any, gold: dict[str, Any]) -> str:
    if not isinstance(actual, dict):
        return "parse_failure"
    if actual.get("status") == "insufficient":
        return "overrestriction"
    if result_payload_matches(actual, gold):
        return "aggregation_only"
    if cited_ids(actual) != [item["citation"] for item in gold["results"]]:
        return "citation_or_order"
    return "dose_unit_or_time"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {task["task_id"]: task for task in load_jsonl(args.tasks)}
    output = []
    for run_dir in args.run_dirs:
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        model = Path(metadata["model_path"]).name
        by_system: dict[str, list[dict[str, Any]]] = {}
        for record in load_jsonl(run_dir / "predictions.jsonl"):
            by_system.setdefault(record["system"], []).append(record)
        systems = {}
        for system, records in sorted(by_system.items()):
            failures: Counter[str] = Counter()
            correct = 0
            for record in records:
                gold = tasks[record["task_id"]]["gold_answer"]
                actual = record.get("parsed_answer")
                if exact_answer_correct(actual, gold):
                    correct += 1
                else:
                    failures[classify(actual, gold)] += 1
            systems[system] = {
                "n": len(records),
                "exact_accuracy": correct / len(records),
                "failure_types": dict(sorted(failures.items())),
            }
        output.append({"model": model, "systems": systems})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
