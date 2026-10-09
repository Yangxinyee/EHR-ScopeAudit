#!/usr/bin/env python3
"""Residual-composition table: what each intervention leaves behind.

The paper's mechanistic claim is that contract execution and test-time reasoning
repair disjoint failure modes. This script reads the deterministic error coding
for both and emits the table and macros supporting that claim, plus a strict
check of the claim itself. It loads no model and recodes nothing; it only reads
saved error_analysis.json files.

Failure categories come from the frozen coders: a citation/order failure means
the wrong rows or the wrong order; an aggregation-only failure means the cited
payload is exactly right and the derived value is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from generate_submission_tables import HEADER, MODEL_LABELS, PANEL_ORDER
except ModuleNotFoundError:
    from scripts.generate_submission_tables import HEADER, MODEL_LABELS, PANEL_ORDER


SELECTION = "citation_or_order"
ARITHMETIC = "aggregation_only"
COT_SYSTEMS = ("prompt_only_all6_cot", "prompt_all6_cot")
ALL_SIX = ("prompt_only_all6", "prompt_all6")


def load(path: Path) -> dict[str, dict[str, dict]]:
    """Map checkpoint label to its per-system entries.

    A checkpoint can appear more than once in one error-analysis file, because
    its six-system run and its all-six run are separate directories. Merge the
    system dictionaries rather than letting the later entry replace the earlier.
    """
    if not path.exists():
        return {}
    merged: dict[str, dict[str, dict]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        if entry["model"] not in MODEL_LABELS:
            continue
        merged.setdefault(MODEL_LABELS[entry["model"]], {}).update(entry["systems"])
    return merged


def pick(systems: dict, names: tuple[str, ...]) -> dict | None:
    for name in names:
        if name in systems:
            return systems[name]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lab-frozen", type=Path,
        default=Path("experiments/scopepair_fhir_analysis_with_all6/error_analysis.json"),
    )
    parser.add_argument(
        "--med-frozen", type=Path,
        default=Path("experiments/scopepair_medadmin_test_analysis_frozen/error_analysis.json"),
    )
    parser.add_argument(
        "--lab-cot", type=Path,
        default=Path("experiments/scopepair_lab_all6cot_analysis/error_analysis.json"),
    )
    parser.add_argument(
        "--med-cot", type=Path,
        default=Path("experiments/scopepair_med_all6cot_analysis/error_analysis.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/generated/complementarity_table.tex"),
    )
    parser.add_argument(
        "--macros", type=Path,
        default=Path("paper/generated/complementarity_macros.tex"),
    )
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if the complementarity claim fails")
    args = parser.parse_args()

    families = (
        ("LabTrend", load(args.lab_frozen), load(args.lab_cot)),
        ("MedDose", load(args.med_frozen), load(args.med_cot)),
    )

    body: list[str] = []
    cot_selection_dominant = 0
    guard_arithmetic_dominant = 0
    runs = 0
    failures: list[str] = []
    for index, (family, frozen, cot) in enumerate(families):
        models = [
            label for label in PANEL_ORDER
            if label in cot and label in frozen
        ]
        for position, model in enumerate(models):
            cot_entry = pick(cot.get(model, {}), COT_SYSTEMS)
            all_six_entry = pick(frozen.get(model, {}), ALL_SIX)
            guard_entry = frozen.get(model, {}).get("guard_refill")
            if not (cot_entry and all_six_entry and guard_entry):
                continue
            runs += 1
            cot_fail = cot_entry["failure_types"]
            all_fail = all_six_entry["failure_types"]
            guard_fail = guard_entry["failure_types"]

            if cot_fail and max(cot_fail, key=cot_fail.get) == SELECTION:
                cot_selection_dominant += 1
            else:
                failures.append(f"{family}/{model}: reasoning residual not selection-dominated ({cot_fail})")
            if guard_fail and max(guard_fail, key=guard_fail.get) == ARITHMETIC:
                guard_arithmetic_dominant += 1

            prefix = (
                f"\\multirow{{{len(models)}}}{{*}}{{{family}}}" if position == 0 else ""
            )
            body.append(
                f"{prefix} & {model} & "
                f"{all_fail.get(SELECTION, 0)} & {all_fail.get(ARITHMETIC, 0)} & "
                f"{cot_fail.get(SELECTION, 0)} & {cot_fail.get(ARITHMETIC, 0)} & "
                f"{guard_fail.get(SELECTION, 0)} & {guard_fail.get(ARITHMETIC, 0)} \\\\"
            )
        if index == 0 and body:
            body.append("\\midrule")

    lines = [
        HEADER,
        "\\begin{tabular}{ll rr rr rr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{All-six} & \\multicolumn{2}{c}{All-six + reasoning}"
        " & \\multicolumn{2}{c}{Guard-refill} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        "Family & Checkpoint & Sel. & Arith. & Sel. & Arith. & Sel. & Arith. \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    macros = [
        HEADER,
        "% Residual-composition macros for the complementarity claim.",
        f"\\newcommand{{\\CompRuns}}{{{runs}\\xspace}}",
        f"\\newcommand{{\\CompCotSelection}}{{{cot_selection_dominant}\\xspace}}",
        f"\\newcommand{{\\CompGuardArithmetic}}{{{guard_arithmetic_dominant}\\xspace}}",
        "",
    ]
    args.macros.write_text("\n".join(macros), encoding="utf-8")
    print(args.output)
    print(args.macros)
    print(
        f"reasoning residual selection-dominated in {cot_selection_dominant}/{runs}; "
        f"guard residual arithmetic-dominated in {guard_arithmetic_dominant}/{runs}"
    )
    for failure in failures:
        print(f"  NOTE {failure}")
    if args.strict and cot_selection_dominant != runs:
        print("\nComplementarity claim NOT supported in every run.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
