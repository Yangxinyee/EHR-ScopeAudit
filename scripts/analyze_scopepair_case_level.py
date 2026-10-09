#!/usr/bin/env python3
"""Independent-case sensitivity analysis for frozen ScopePair-FHIR scores.

This post-formal correction does not rescore model outputs. It aggregates the
two paired tasks of each case before exact McNemar testing so that the
inferential unit is the independent patient/case.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


COMPARISONS = (
    ("clinehr_os", "unscoped"),
    ("clinehr_os", "late_no_refill"),
    ("clinehr_os", "scope_first"),
    ("scope_first", "wrong_time_scope_first"),
)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
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
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scores = pd.read_csv(args.scores)

    case_scores = (
        scores.groupby(
            [
                "model",
                "system",
                "case_id",
                "patient_cluster_id",
                "timestamp_stress",
            ],
            as_index=False,
        )
        .agg(
            both_tasks_joint_valid=("joint_valid", "min"),
            mean_task_joint_valid=("joint_valid", "mean"),
            any_scope_violation=("scope_violation", "max"),
            mean_evidence_recall=("evidence_recall", "mean"),
            both_tasks_answer_correct=("answer_correct", "min"),
        )
    )
    if not (case_scores.groupby(["model", "system"]).size() == 64).all():
        raise ValueError("Expected 64 independent cases per model/system")

    summary_rows = []
    for (model, system), frame in case_scores.groupby(["model", "system"], sort=True):
        successes = int(frame["both_tasks_joint_valid"].sum())
        low, high = wilson(successes, len(frame))
        summary_rows.append(
            {
                "model": model,
                "system": system,
                "n_independent_cases": len(frame),
                "both_tasks_joint_valid": successes / len(frame),
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "any_scope_violation": frame["any_scope_violation"].mean(),
                "mean_task_joint_valid": frame["mean_task_joint_valid"].mean(),
                "mean_evidence_recall": frame["mean_evidence_recall"].mean(),
            }
        )

    paired_rows = []
    for model, model_frame in case_scores.groupby("model", sort=True):
        current = []
        for comparison, baseline in COMPARISONS:
            first = model_frame[model_frame["system"] == comparison][
                ["case_id", "both_tasks_joint_valid"]
            ].rename(columns={"both_tasks_joint_valid": "comparison"})
            second = model_frame[model_frame["system"] == baseline][
                ["case_id", "both_tasks_joint_valid"]
            ].rename(columns={"both_tasks_joint_valid": "baseline"})
            merged = first.merge(second, on="case_id", validate="one_to_one")
            comparison_only = int(
                ((merged["comparison"] == 1) & (merged["baseline"] == 0)).sum()
            )
            baseline_only = int(
                ((merged["comparison"] == 0) & (merged["baseline"] == 1)).sum()
            )
            current.append(
                {
                    "model": model,
                    "comparison_system": comparison,
                    "baseline_system": baseline,
                    "n_independent_cases": len(merged),
                    "comparison_minus_baseline": (
                        merged["comparison"] - merged["baseline"]
                    ).mean(),
                    "comparison_only_success": comparison_only,
                    "baseline_only_success": baseline_only,
                    "case_level_mcnemar_p": exact_mcnemar_p(
                        comparison_only, baseline_only
                    ),
                }
            )
        adjusted = holm_adjust([row["case_level_mcnemar_p"] for row in current])
        for row, corrected in zip(current, adjusted):
            row["holm_adjusted_case_p"] = corrected
            paired_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_scores.to_csv(args.output_dir / "case_scores.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "case_summary.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "case_paired_comparisons.csv", index=False
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print(pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
