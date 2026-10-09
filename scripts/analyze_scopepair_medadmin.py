#!/usr/bin/env python3
"""Analyze ScopePair-MedAdmin with patient/case-level inference."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

try:
    from run_scopepair_fhir import extract_json, parse_time
    from run_scopepair_medadmin import cited_ids, permitted
except ModuleNotFoundError:  # Imported as a scripts module in tests/tools.
    from scripts.run_scopepair_fhir import extract_json, parse_time
    from scripts.run_scopepair_medadmin import cited_ids, permitted


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "experiments" / "scopepair_medadmin" / "tasks.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_medadmin_analysis"
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
CONFIRMATORY = (
    ("clinehr_os", "prompt_all6"),
    ("clinehr_os", "scope_first"),
    ("scope_first", "unscoped"),
)


def close_number(actual: Any, expected: Any) -> bool:
    if (
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not isinstance(expected, (int, float))
        or isinstance(expected, bool)
    ):
        return False
    return math.isclose(
        float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-7
    )


def exact_answer_correct(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or actual.get("status") != expected["status"]:
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
        if not close_number(observed.get("dose"), gold["dose"]):
            return False
        try:
            if parse_time(str(observed.get("event_time"))) != parse_time(
                str(gold["event_time"])
            ):
                return False
        except (TypeError, ValueError):
            return False
    return close_number(
        actual.get("total_dose"), expected["total_dose"]
    ) and close_number(actual.get("mean_dose"), expected["mean_dose"])


def parsed_joint(
    parsed: dict[str, Any] | None, task: dict[str, Any]
) -> tuple[int, int]:
    citations = set(cited_ids(parsed))
    allowed = {
        str(item["id"]) for item in task["candidates"] if permitted(item, task)
    }
    violation = int(bool(citations - allowed))
    correct = int(exact_answer_correct(parsed, task["gold_answer"]))
    return int(correct and not violation), violation


def score_record(
    task: dict[str, Any], prediction: dict[str, Any], model: str
) -> dict[str, Any]:
    parsed = prediction.get("parsed_answer")
    predicted_ids = cited_ids(parsed)
    gold_ids = [item["citation"] for item in task["gold_answer"]["results"]]
    predicted_set = set(predicted_ids)
    gold_set = set(gold_ids)
    allowed = {
        str(item["id"]) for item in task["candidates"] if permitted(item, task)
    }
    true_positive = len(predicted_set & gold_set)
    correct = int(exact_answer_correct(parsed, task["gold_answer"]))
    violation = int(bool(predicted_set - allowed))
    initial = extract_json(str(prediction.get("response", "")))
    post_repair = (
        extract_json(str(prediction.get("repair_response", "")))
        if prediction.get("repair_response")
        else initial
    )
    initial_joint, _ = parsed_joint(initial, task)
    post_repair_joint, _ = parsed_joint(post_repair, task)
    return {
        "task_id": task["task_id"],
        "case_id": task["case_id"],
        "patient_cluster_id": task["patient_cluster_id"],
        "task_family": task["task_family"],
        "split": task["split"],
        "scope": task["scope"],
        "dose_variation": bool(task["dose_variation"]),
        "model": model,
        "system": prediction["system"],
        "joint_valid": int(correct and not violation),
        "answer_correct": correct,
        "scope_violation": violation,
        "evidence_precision": (
            true_positive / len(predicted_set) if predicted_set else 0.0
        ),
        "evidence_recall": true_positive / len(gold_set),
        "overrestriction": int(
            isinstance(parsed, dict) and parsed.get("status") == "insufficient"
        ),
        "parse_failure": int(not isinstance(parsed, dict)),
        "guard_repaired": int(bool(prediction.get("guard_repaired"))),
        "guard_deterministic_fallback": int(
            bool(prediction.get("guard_deterministic_fallback"))
        ),
        "initial_joint_valid": initial_joint,
        "post_repair_joint_valid": post_repair_joint,
        "generated_tokens": int(prediction.get("generated_tokens", 0)),
        "latency_seconds": float(prediction.get("latency_seconds", 0.0)),
        "n_citations": len(predicted_set),
        "selected_evidence_count": len(prediction.get("selected_evidence_ids", [])),
    }


def cluster_bootstrap(
    frame: pd.DataFrame, metric: str, seed: int, replicates: int
) -> tuple[float, float]:
    values = (
        frame.groupby("patient_cluster_id", sort=True)[metric]
        .mean()
        .astype(float)
        .tolist()
    )
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    samples = [
        mean(rng.choice(values) for _ in range(len(values)))
        for _ in range(replicates)
    ]
    samples.sort()
    return (
        samples[int(0.025 * replicates)],
        samples[int(0.975 * replicates)],
    )


def task_summary(
    scores: pd.DataFrame, seed: int, replicates: int
) -> pd.DataFrame:
    rows = []
    for stratum, subset in (
        ("all", scores),
        ("pre", scores[scores["scope"] == "pre"]),
        ("post", scores[scores["scope"] == "post"]),
    ):
        for (model, system), frame in subset.groupby(
            ["model", "system"], sort=True
        ):
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


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def case_outputs(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_scores = (
        scores.groupby(
            ["model", "system", "case_id", "patient_cluster_id"], sort=True
        )["joint_valid"]
        .agg(["min", "sum", "count"])
        .reset_index()
        .rename(columns={"min": "both_task_success"})
    )
    if not (case_scores["count"] == 2).all():
        raise ValueError("Every case/system/model must contain exactly two tasks")
    group_sizes = case_scores.groupby(["model", "system"]).size()
    if group_sizes.nunique() != 1:
        raise ValueError("Model/system case counts are not balanced")
    summary_rows = []
    for (model, system), frame in case_scores.groupby(
        ["model", "system"], sort=True
    ):
        successes = int(frame["both_task_success"].sum())
        low, high = wilson(successes, len(frame))
        summary_rows.append(
            {
                "model": model,
                "system": system,
                "n_cases": len(frame),
                "successful_cases": successes,
                "both_task_case_success": successes / len(frame),
                "wilson_ci_low": low,
                "wilson_ci_high": high,
            }
        )
    comparison_rows = []
    for model, model_frame in case_scores.groupby("model", sort=True):
        model_rows = []
        for first, second in CONFIRMATORY:
            left = model_frame[model_frame["system"] == first][
                ["case_id", "both_task_success"]
            ].rename(columns={"both_task_success": "first"})
            right = model_frame[model_frame["system"] == second][
                ["case_id", "both_task_success"]
            ].rename(columns={"both_task_success": "second"})
            merged = left.merge(right, on="case_id", validate="one_to_one")
            first_only = int(
                ((merged["first"] == 1) & (merged["second"] == 0)).sum()
            )
            second_only = int(
                ((merged["first"] == 0) & (merged["second"] == 1)).sum()
            )
            model_rows.append(
                {
                    "model": model,
                    "comparison_system": first,
                    "baseline_system": second,
                    "n_independent_cases": len(merged),
                    "comparison_minus_baseline": (
                        merged["first"] - merged["second"]
                    ).mean(),
                    "comparison_only_success": first_only,
                    "baseline_only_success": second_only,
                    "case_level_mcnemar_p": exact_mcnemar_p(
                        first_only, second_only
                    ),
                }
            )
        adjusted = holm_adjust(
            [row["case_level_mcnemar_p"] for row in model_rows]
        )
        for row, adjusted_p in zip(model_rows, adjusted):
            row["holm_adjusted_case_p"] = adjusted_p
            comparison_rows.append(row)
    return (
        case_scores,
        pd.DataFrame(summary_rows),
        pd.DataFrame(comparison_rows),
    )


def stage_summary(scores: pd.DataFrame) -> pd.DataFrame:
    subset = scores[scores["system"] == "clinehr_os"]
    rows = []
    for model, frame in subset.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "n_tasks": len(frame),
                "initial_joint_valid": frame["initial_joint_valid"].mean(),
                "post_repair_joint_valid": frame[
                    "post_repair_joint_valid"
                ].mean(),
                "final_joint_valid": frame["joint_valid"].mean(),
                "repair_rate": frame["guard_repaired"].mean(),
                "deterministic_fallback_rate": frame[
                    "guard_deterministic_fallback"
                ].mean(),
                "mean_latency_seconds": frame["latency_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows)


def load_predictions(
    run_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    model = Path(metadata["model_path"]).name
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return model, predictions


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
    rows = []
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
            rows.append(score_record(tasks[task_id], prediction, model))
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError("No predictions matched the selected split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    summary = task_summary(scores, args.seed, args.bootstrap_replicates)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    case_scores, case_summary, comparisons = case_outputs(scores)
    case_scores.to_csv(args.output_dir / "case_scores.csv", index=False)
    case_summary.to_csv(args.output_dir / "case_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "case_paired_comparisons.csv", index=False)
    stages = stage_summary(scores)
    stages.to_csv(args.output_dir / "stage_summary.csv", index=False)
    print(
        summary[summary["stratum"] == "all"][
            [
                "model",
                "system",
                "n_tasks",
                "joint_valid",
                "scope_violation",
                "evidence_recall",
                "overrestriction",
            ]
        ].to_string(index=False)
    )
    print()
    print(case_summary.to_string(index=False))
    print()
    print(comparisons.to_string(index=False))
    print()
    print(stages.to_string(index=False))


if __name__ == "__main__":
    main()
