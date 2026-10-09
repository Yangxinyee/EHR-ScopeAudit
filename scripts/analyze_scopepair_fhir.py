#!/usr/bin/env python3
"""Score ScopePair-FHIR runs with patient-clustered uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from run_scopepair_fhir import cited_ids, parse_time, permitted


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "experiments" / "scopepair_fhir" / "tasks.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_fhir_analysis"
METRICS = (
    "joint_valid",
    "answer_correct",
    "scope_violation",
    "evidence_precision",
    "evidence_recall",
    "overrestriction",
    "parse_failure",
    "guard_repaired",
    "guard_deterministic_fallback",
)
COMPARISONS = (
    ("clinehr_os", "unscoped"),
    ("clinehr_os", "late_no_refill"),
    ("clinehr_os", "scope_first"),
    ("scope_first", "wrong_time_scope_first"),
)


def close_number(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        return False
    return math.isclose(
        float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-7
    )


def exact_answer_correct(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    if actual.get("status") != expected["status"]:
        return False
    results = actual.get("results")
    if not isinstance(results, list) or len(results) != len(expected["results"]):
        return False
    for observed, gold in zip(results, expected["results"]):
        if not isinstance(observed, dict):
            return False
        if observed.get("citation") != gold["citation"]:
            return False
        if observed.get("unit") != gold["unit"]:
            return False
        if not close_number(observed.get("value"), gold["value"]):
            return False
        try:
            if parse_time(str(observed.get("issued_time"))) != parse_time(
                str(gold["issued_time"])
            ):
                return False
        except (TypeError, ValueError):
            return False
    if not close_number(actual.get("delta"), expected["delta"]):
        return False
    return actual.get("direction") == expected["direction"]


def score_record(
    task: dict[str, Any],
    prediction: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    parsed = prediction.get("parsed_answer")
    predicted_ids = cited_ids(parsed)
    gold_ids = [item["citation"] for item in task["gold_answer"]["results"]]
    allowed_ids = {
        str(candidate["id"])
        for candidate in task["candidates"]
        if permitted(candidate, task, basis="issued")
    }
    predicted_set = set(predicted_ids)
    gold_set = set(gold_ids)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(gold_set)
    violation = int(bool(predicted_set - allowed_ids))
    correct = int(exact_answer_correct(parsed, task["gold_answer"]))
    return {
        "task_id": task["task_id"],
        "case_id": task["case_id"],
        "patient_cluster_id": task["patient_cluster_id"],
        "split": task["split"],
        "scope": task["scope"],
        "timestamp_stress": bool(task["timestamp_stress"]),
        "model": model,
        "system": prediction["system"],
        "joint_valid": int(correct and not violation),
        "answer_correct": correct,
        "scope_violation": violation,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "overrestriction": int(
            isinstance(parsed, dict) and parsed.get("status") == "insufficient"
        ),
        "parse_failure": int(not isinstance(parsed, dict)),
        "guard_repaired": int(bool(prediction.get("guard_repaired"))),
        "guard_deterministic_fallback": int(
            bool(prediction.get("guard_deterministic_fallback"))
        ),
        "final_guard_failure": int(bool(prediction.get("final_guard_errors"))),
        "generated_tokens": int(prediction.get("generated_tokens", 0)),
        "latency_seconds": float(prediction.get("latency_seconds", 0.0)),
        "n_citations": len(predicted_set),
        "selected_evidence_count": len(prediction.get("selected_evidence_ids", [])),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    per_cluster = (
        frame.groupby("patient_cluster_id", sort=True)[metric].mean().astype(float)
    )
    values = per_cluster.tolist()
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(values) for _ in range(len(values)))
        for _ in range(replicates)
    ]
    estimates.sort()
    return (
        estimates[int(0.025 * replicates)],
        estimates[int(0.975 * replicates)],
    )


def summaries(
    scores: pd.DataFrame, seed: int, replicates: int
) -> pd.DataFrame:
    rows = []
    strata = [
        ("all", scores),
        ("pre", scores[scores["scope"] == "pre"]),
        ("post", scores[scores["scope"] == "post"]),
        ("timestamp_stress", scores[scores["timestamp_stress"]]),
        ("nonstress", scores[~scores["timestamp_stress"]]),
    ]
    for stratum, subset in strata:
        for (model, system), frame in subset.groupby(["model", "system"], sort=True):
            row: dict[str, Any] = {
                "stratum": stratum,
                "model": model,
                "system": system,
                "n_tasks": len(frame),
                "n_patients": frame["patient_cluster_id"].nunique(),
                "mean_generated_tokens": frame["generated_tokens"].mean(),
                "mean_latency_seconds": frame["latency_seconds"].mean(),
            }
            for offset, metric in enumerate(METRICS):
                low, high = cluster_bootstrap(
                    frame, metric, seed + offset, replicates
                )
                row[metric] = frame[metric].mean()
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_cluster_bootstrap(
    merged: pd.DataFrame,
    difference_column: str,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    per_cluster = (
        merged.groupby("patient_cluster_id", sort=True)[difference_column]
        .mean()
        .astype(float)
        .tolist()
    )
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(per_cluster) for _ in range(len(per_cluster)))
        for _ in range(replicates)
    ]
    estimates.sort()
    return (
        estimates[int(0.025 * replicates)],
        estimates[int(0.975 * replicates)],
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_comparisons(
    scores: pd.DataFrame, seed: int, replicates: int
) -> pd.DataFrame:
    rows = []
    for model, model_frame in scores.groupby("model", sort=True):
        model_rows = []
        for comparison_index, (first, second) in enumerate(COMPARISONS):
            left = model_frame[model_frame["system"] == first][
                ["task_id", "patient_cluster_id", "joint_valid"]
            ].rename(columns={"joint_valid": "first"})
            right = model_frame[model_frame["system"] == second][
                ["task_id", "joint_valid"]
            ].rename(columns={"joint_valid": "second"})
            merged = left.merge(right, on="task_id", validate="one_to_one")
            if merged.empty:
                continue
            merged["difference"] = merged["first"] - merged["second"]
            first_only = int(
                ((merged["first"] == 1) & (merged["second"] == 0)).sum()
            )
            second_only = int(
                ((merged["first"] == 0) & (merged["second"] == 1)).sum()
            )
            low, high = paired_cluster_bootstrap(
                merged,
                "difference",
                seed + comparison_index,
                replicates,
            )
            model_rows.append(
                {
                    "model": model,
                    "comparison_system": first,
                    "baseline_system": second,
                    "metric": "joint_valid",
                    "n_paired_tasks": len(merged),
                    "n_patient_clusters": merged["patient_cluster_id"].nunique(),
                    "comparison_minus_baseline": merged["difference"].mean(),
                    "cluster_ci_low": low,
                    "cluster_ci_high": high,
                    "comparison_only_success": first_only,
                    "baseline_only_success": second_only,
                    "mcnemar_exact_p": exact_mcnemar_p(first_only, second_only),
                }
            )
        adjusted = holm_adjust([row["mcnemar_exact_p"] for row in model_rows])
        for row, p_adjusted in zip(model_rows, adjusted):
            row["holm_adjusted_p"] = p_adjusted
            rows.append(row)
    return pd.DataFrame(rows)


def load_predictions(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    model = Path(metadata["model_path"]).name
    records = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return model, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    tasks = {
        task["task_id"]: task
        for task in (
            json.loads(line)
            for line in args.tasks.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if task["split"] == args.split
    }
    score_rows = []
    seen = set()
    for run_dir in args.run_dirs:
        model, predictions = load_predictions(run_dir)
        for prediction in predictions:
            task_id = prediction["task_id"]
            if task_id not in tasks:
                continue
            key = (model, prediction["system"], task_id)
            if key in seen:
                raise ValueError(f"Duplicate prediction: {key}")
            seen.add(key)
            score_rows.append(score_record(tasks[task_id], prediction, model))
    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise ValueError("No predictions matched the selected split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    summary = summaries(scores, args.seed, args.bootstrap_replicates)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired = paired_comparisons(scores, args.seed, args.bootstrap_replicates)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    print(
        summary[summary["stratum"] == "all"][
            [
                "model",
                "system",
                "n_tasks",
                "joint_valid",
                "answer_correct",
                "scope_violation",
                "evidence_recall",
                "overrestriction",
            ]
        ].to_string(index=False)
    )
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
