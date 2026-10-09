#!/usr/bin/env python3
"""Check every universal claim the manuscript makes against the saved artifacts.

The paper asserts things like "positive for every checkpoint" and "zero for
every checkpoint". Those are cheap to write and easy to get wrong once the
panel grows, so each one is restated here as an executable predicate over the
frozen analysis CSVs. Exit code 1 means the manuscript says something the
artifacts do not support.

Loads no model. Reads only saved CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from generate_submission_tables import (
        LAB_ALL_SIX,
        MED_ALL_SIX,
        MODEL_LABELS,
        PANEL_ORDER,
    )
except ModuleNotFoundError:
    from scripts.generate_submission_tables import (
        LAB_ALL_SIX,
        MED_ALL_SIX,
        MODEL_LABELS,
        PANEL_ORDER,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary_map(path: Path, stratum: str = "all") -> dict[tuple[str, str], dict[str, str]]:
    return {
        (MODEL_LABELS[row["model"]], row["system"]): row
        for row in read_rows(path)
        if row["stratum"] == stratum and row["model"] in MODEL_LABELS
    }


def keyed_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (MODEL_LABELS[row["model"]], row["system"]): row
        for row in read_rows(path)
        if row["model"] in MODEL_LABELS
    }


def comparison_map(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (MODEL_LABELS[row["model"]], row["comparison_system"], row["baseline_system"]): row
        for row in read_rows(path)
        if row["model"] in MODEL_LABELS
    }


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, ok: bool, claim: str, detail: str = "") -> None:
        self.checks += 1
        if ok:
            print(f"  ok    {claim}")
        else:
            print(f"  FAIL  {claim}{(' -- ' + detail) if detail else ''}")
            self.failures.append(claim)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lab-analysis", type=Path,
        default=Path("experiments/scopepair_fhir_analysis_with_all6"),
    )
    parser.add_argument(
        "--lab-cases", type=Path,
        default=Path("experiments/scopepair_fhir_case_with_all6"),
    )
    parser.add_argument(
        "--med-analysis", type=Path,
        default=Path("experiments/scopepair_medadmin_test_analysis_frozen"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    lab = summary_map(args.lab_analysis / "summary.csv")
    med = summary_map(args.med_analysis / "summary.csv")
    lab_cases = keyed_map(args.lab_cases / "case_summary.csv")
    med_cases = keyed_map(args.med_analysis / "case_summary.csv")
    med_comparisons = comparison_map(args.med_analysis / "case_paired_comparisons.csv")

    models = [label for label in PANEL_ORDER if (label, "scope_first") in lab]
    print(f"Checkpoints found: {', '.join(models)}\n")

    check = Checker()
    families = (
        ("LabTrend", lab, lab_cases, LAB_ALL_SIX),
        ("MedDose", med, med_cases, MED_ALL_SIX),
    )

    print("Claim: guard-refill exceeds all-six for every checkpoint and family.")
    for family, rows, _, all_six in families:
        for model in models:
            gain = float(rows[(model, "guard_refill")]["joint_valid"]) - float(
                rows[(model, all_six)]["joint_valid"]
            )
            check.require(
                gain > 0, f"{family}/{model}: guard-refill > all-six", f"gain={gain:+.4f}"
            )

    print("\nClaim: guard-refill is never worse than scope-first.")
    for family, rows, _, _ in families:
        for model in models:
            delta = float(rows[(model, "guard_refill")]["joint_valid"]) - float(
                rows[(model, "scope_first")]["joint_valid"]
            )
            check.require(
                delta >= 0, f"{family}/{model}: guard-refill >= scope-first", f"delta={delta:+.4f}"
            )

    print("\nClaim: scope-first exceeds all-six in every run but the named exceptions.")
    # The manuscript states this is not universal and names the exception, so the
    # check enforces the exact exception set rather than a blanket inequality.
    expected_exceptions = {("MedDose", "MedGemma-27B")}
    observed_exceptions = set()
    for family, rows, _, all_six in families:
        for model in models:
            gain = float(rows[(model, "scope_first")]["joint_valid"]) - float(
                rows[(model, all_six)]["joint_valid"]
            )
            if gain <= 0:
                observed_exceptions.add((family, model))
    check.require(
        observed_exceptions == expected_exceptions,
        "scope-first exception set matches the manuscript",
        f"observed={sorted(observed_exceptions)}, stated={sorted(expected_exceptions)}",
    )

    print("\nClaim: all-six < guard-refill < ClinEHR-OS in every run.")
    for family, rows, _, all_six in families:
        for model in models:
            values = [
                float(rows[(model, system)]["joint_valid"])
                for system in (all_six, "guard_refill", "clinehr_os")
            ]
            ordered = values[0] < values[1] < values[2]
            check.require(ordered, f"{family}/{model}: ordering holds", f"values={values}")

    print("\nClaim: ClinEHR-OS reaches 100% task and case validity everywhere.")
    for family, rows, cases, _ in families:
        field = (
            "both_tasks_joint_valid" if family == "LabTrend" else "both_task_case_success"
        )
        for model in models:
            task = float(rows[(model, "clinehr_os")]["joint_valid"])
            case = float(cases[(model, "clinehr_os")][field])
            check.require(
                task == 1.0 and case == 1.0,
                f"{family}/{model}: full system 100%",
                f"task={task}, case={case}",
            )

    print("\nClaim: unscoped top-3 has zero joint validity for every checkpoint.")
    for family, rows, _, _ in families:
        for model in models:
            value = float(rows[(model, "unscoped")]["joint_valid"])
            check.require(value == 0.0, f"{family}/{model}: unscoped == 0", f"value={value}")

    print("\nClaim: late-no-refill never succeeds, and abstains on nearly every task.")
    for family, rows, _, _ in families:
        for model in models:
            joint = float(rows[(model, "late_no_refill")]["joint_valid"])
            check.require(
                joint == 0.0,
                f"{family}/{model}: late-no-refill joint validity == 0",
                f"value={joint}",
            )
            # The load-bearing claim above is exact. This one is a description of
            # how the failure presents, and is deliberately not asserted as
            # universal: a model may emit a short in-scope answer instead of
            # abstaining, which still fails but is not over-restriction.
            over = float(rows[(model, "late_no_refill")]["overrestriction"])
            check.require(
                over >= 0.99,
                f"{family}/{model}: late-no-refill abstains on >=99% of tasks",
                f"value={over}",
            )

    print("\nClaim: guard-refill removes all task-level scope violations.")
    for family, rows, _, _ in families:
        for model in models:
            value = float(rows[(model, "guard_refill")]["scope_violation"])
            check.require(
                value == 0.0, f"{family}/{model}: guard-refill scope violations == 0", f"value={value}"
            )

    print("\nClaim: MedDose confirmatory comparisons survive Holm correction everywhere.")
    for model in models:
        row = med_comparisons.get((model, "scope_first", "unscoped"))
        if row is not None:
            p_value = float(row["holm_adjusted_case_p"])
            check.require(
                p_value < args.alpha,
                f"MedDose/{model}: scope-first vs unscoped significant",
                f"p={p_value:.3g}",
            )
        for baseline in (MED_ALL_SIX, "scope_first"):
            row = med_comparisons.get((model, "clinehr_os", baseline))
            if row is None:
                check.require(False, f"MedDose/{model}: comparison vs {baseline} present")
                continue
            p_value = float(row["holm_adjusted_case_p"])
            check.require(
                p_value < args.alpha,
                f"MedDose/{model}: ClinEHR-OS vs {baseline} significant",
                f"p={p_value:.3g}",
            )

    print("\nClaim: availability-time scope beats occurrence-time scope on the "
          "LabTrend stress stratum for every checkpoint.")
    stress_path = args.lab_analysis / "summary.csv"
    stress = summary_map(stress_path, stratum="timestamp_stress")
    if not stress:
        print("  skip  no timestamp_stress stratum in the summary file")
    else:
        for model in models:
            right = float(stress[(model, "scope_first")]["joint_valid"])
            wrong = float(stress[(model, "wrong_time_scope_first")]["joint_valid"])
            check.require(
                right > wrong,
                f"LabTrend/{model}: availability-time > occurrence-time on stress",
                f"{right} vs {wrong}",
            )

    print("\nClaim: after guard-refill, aggregation-only is the largest residual "
          "category in nine of ten runs.")
    import json as _json
    dominant_aggregation = 0
    total_runs = 0
    for family, analysis in (
        ("LabTrend", args.lab_analysis / "error_analysis.json"),
        ("MedDose", args.med_analysis / "error_analysis.json"),
    ):
        if not analysis.exists():
            continue
        for entry in _json.loads(analysis.read_text(encoding="utf-8")):
            systems = entry.get("systems", {})
            failures = systems.get("guard_refill", {}).get("failure_types", {})
            if not failures:
                continue
            total_runs += 1
            if max(failures, key=failures.get) == "aggregation_only":
                dominant_aggregation += 1
    if total_runs:
        check.require(
            dominant_aggregation == total_runs - 1,
            "guard-refill residue is aggregation-dominated in all but one run",
            f"{dominant_aggregation}/{total_runs}",
        )

    print("\nClaim: parameter count does not order the checkpoints on scope-first.")
    # The panel is listed in ascending parameter order, so a strictly increasing
    # scope-first score would mean scale does order them, and the manuscript's
    # non-monotonicity sentence would be wrong.
    for family, rows, _, _ in families:
        scores = [float(rows[(model, "scope_first")]["joint_valid"]) for model in models]
        monotone = all(a <= b for a, b in zip(scores, scores[1:]))
        check.require(
            not monotone,
            f"{family}: scope-first is not monotone in parameter count",
            f"scores in panel order={scores}",
        )

    print("\nClaim: medical adaptation does not beat its general base.")
    if "MedGemma-27B" in models and "Gemma-3-27B" in models:
        for family, rows, _, all_six in families:
            for system, label in ((all_six, "all-six"), ("scope_first", "scope-first")):
                medical = float(rows[("MedGemma-27B", system)]["joint_valid"])
                general = float(rows[("Gemma-3-27B", system)]["joint_valid"])
                check.require(
                    medical <= general,
                    f"{family}/{label}: MedGemma <= Gemma-3",
                    f"{medical} vs {general}",
                )
    else:
        print("  skip  matched pair not present")

    print(f"\n{check.checks - len(check.failures)}/{check.checks} claims verified.")
    if check.failures:
        print("\nThe manuscript asserts the following, and the artifacts do not support it:")
        for failure in check.failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
