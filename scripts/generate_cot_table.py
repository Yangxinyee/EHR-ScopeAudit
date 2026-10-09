#!/usr/bin/env python3
"""Generate the reasoning-enabled all-six robustness table from saved analyses.

Reads the same summary CSVs the frozen scorer produced, so the robustness table
cannot drift from the frozen numbers it is compared against. Loads no model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from generate_submission_tables import (
        HEADER,
        LAB_ALL_SIX,
        MED_ALL_SIX,
        MODEL_LABELS,
        PANEL_ORDER,
        pct,
        signed_pct,
    )
except ModuleNotFoundError:
    from scripts.generate_submission_tables import (
        HEADER,
        LAB_ALL_SIX,
        MED_ALL_SIX,
        MODEL_LABELS,
        PANEL_ORDER,
        pct,
        signed_pct,
    )


LAB_COT = "prompt_only_all6_cot"
MED_COT = "prompt_all6_cot"


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stratum"] == "all"]
    return {
        (MODEL_LABELS[row["model"]], row["system"]): row
        for row in rows
        if row["model"] in MODEL_LABELS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lab-analysis", type=Path,
        default=Path("experiments/scopepair_lab_all6cot_analysis/summary.csv"),
    )
    parser.add_argument(
        "--med-analysis", type=Path,
        default=Path("experiments/scopepair_med_all6cot_analysis/summary.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/generated/cot_results_table.tex"),
    )
    parser.add_argument(
        "--macros", type=Path,
        default=Path("paper/generated/cot_macros.tex"),
    )
    args = parser.parse_args()

    families = (
        ("LabTrend", read_summary(args.lab_analysis), LAB_ALL_SIX, LAB_COT),
        ("MedDose", read_summary(args.med_analysis), MED_ALL_SIX, MED_COT),
    )

    body: list[str] = []
    deltas: list[float] = []
    residuals: list[float] = []
    guard_residuals: list[float] = []
    token_ratios: list[float] = []
    for index, (family, rows, frozen, cot) in enumerate(families):
        models = [
            label for label in PANEL_ORDER if (label, cot) in rows
        ]
        for position, model in enumerate(models):
            frozen_value = float(rows[(model, frozen)]["joint_valid"])
            cot_value = float(rows[(model, cot)]["joint_valid"])
            scope = float(rows[(model, "scope_first")]["joint_valid"])
            guard = float(rows[(model, "guard_refill")]["joint_valid"])
            tokens = float(rows[(model, cot)]["mean_generated_tokens"])
            frozen_tokens = float(rows[(model, frozen)]["mean_generated_tokens"])
            deltas.append(cot_value - frozen_value)
            residuals.append(scope - cot_value)
            guard_residuals.append(guard - cot_value)
            token_ratios.append(tokens / frozen_tokens if frozen_tokens else float("nan"))
            prefix = (
                f"\\multirow{{{len(models)}}}{{*}}{{{family}}}" if position == 0 else ""
            )
            body.append(
                f"{prefix} & {model} & {pct(frozen_value)} & {pct(cot_value)} & "
                f"{signed_pct(cot_value - frozen_value)} & {pct(scope)} & "
                f"{signed_pct(scope - cot_value)} & {pct(guard)} & "
                f"{signed_pct(guard - cot_value)} & {tokens:.0f} \\\\"
            )
        if index == 0:
            body.append("\\midrule")

    lines = [
        HEADER,
        "\\begin{tabular}{ll rrr rr rr r}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{All-six prompt} & "
        "\\multicolumn{2}{c}{vs. scope-first} & "
        "\\multicolumn{2}{c}{vs. guard-refill} & \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
        "Family & Checkpoint & Frozen & +Reason & $\\Delta$ & "
        "Score & Gap & Score & Gap & Tokens \\\\",
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
        "% Reasoning-enabled all-six robustness macros.",
        f"\\newcommand{{\\CotDeltaMin}}{{{signed_pct(min(deltas))}\\xspace}}",
        f"\\newcommand{{\\CotDeltaMax}}{{{signed_pct(max(deltas))}\\xspace}}",
        f"\\newcommand{{\\CotGapMin}}{{{signed_pct(min(residuals))}\\xspace}}",
        f"\\newcommand{{\\CotGapMax}}{{{signed_pct(max(residuals))}\\xspace}}",
        f"\\newcommand{{\\CotRuns}}{{{len(deltas)}\\xspace}}",
        f"\\newcommand{{\\CotClosed}}{{{sum(1 for value in residuals if value <= 0)}\\xspace}}",
        f"\\newcommand{{\\CotGuardGapMin}}{{{signed_pct(min(guard_residuals))}\\xspace}}",
        f"\\newcommand{{\\CotGuardGapMax}}{{{signed_pct(max(guard_residuals))}\\xspace}}",
        f"\\newcommand{{\\CotBeatsGuard}}{{{sum(1 for v in guard_residuals if v < 0)}\\xspace}}",
        f"\\newcommand{{\\CotTokenRatioMin}}{{{min(token_ratios):.1f}\\xspace}}",
        f"\\newcommand{{\\CotTokenRatioMax}}{{{max(token_ratios):.1f}\\xspace}}",
        "",
    ]
    args.macros.write_text("\n".join(macros), encoding="utf-8")
    print(args.output)
    print(args.macros)


if __name__ == "__main__":
    main()
