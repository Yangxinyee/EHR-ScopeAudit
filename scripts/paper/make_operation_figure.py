#!/usr/bin/env python3
"""Plot the frozen LabTrend/MedDose diagnostic panel in manuscript colours.

Reads task-level joint validity and patient-cluster confidence intervals from
saved ScopePair analysis CSVs. Exports a vector PDF/SVG, a PNG preview, and the
plotted values with source hashes. No models or experimental scorers are run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_submission_tables import LAB_ALL_SIX, MED_ALL_SIX, MODEL_LABELS, PANEL_ORDER

PALETTE_PATH = ROOT / "docs/figures/palette.json"
TICK_LABELS = {
    "Qwen3-4B": "Qwen3\n4B",
    "Phi-4-14B": "Phi-4\n14B",
    "Gemma-3-27B": "Gemma-3\n27B",
    "MedGemma-27B": "MedGemma\n27B",
    "Qwen3-30B": "Qwen3\n30B-A3B",
}
SYSTEMS = (
    ("all_six", "All-six prompt", "baseline_fill", "secondary", ""),
    ("scope_first", "Scope-first", "scope_fill", "scope", ""),
    ("guard_refill", "Guard-refill", "audit", "audit", ""),
    ("clinehr_os", "EHR-AgentGuard (verified)", "valid_fill", "valid", "///"),
)


def read_summary(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stratum"] != "all":
                continue
            if row["model"] not in MODEL_LABELS:
                raise ValueError(f"Unlabelled checkpoint in {path}: {row['model']}")
            key = (MODEL_LABELS[row["model"]], row["system"])
            if key in rows:
                raise ValueError(f"Duplicate overall result in {path}: {key}")
            mean, low, high = (
                100 * float(row[k])
                for k in ("joint_valid", "joint_valid_ci_low", "joint_valid_ci_high")
            )
            if not 0 <= low <= mean <= high <= 100:
                raise ValueError(f"Invalid estimate or confidence interval: {key}")
            rows[key] = {
                "mean_pct": mean,
                "ci_low_pct": low,
                "ci_high_pct": high,
                "n_tasks": int(row["n_tasks"]),
                "n_patients": int(row["n_patients"]),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-analysis", type=Path, default=ROOT / "results/scopepair/labtrend/summary.csv")
    parser.add_argument("--med-analysis", type=Path, default=ROOT / "results/scopepair/meddose/summary.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/operation_results.pdf")
    args = parser.parse_args()
    palette = json.loads(PALETTE_PATH.read_text())
    lab, med = read_summary(args.lab_analysis), read_summary(args.med_analysis)
    models = [model for model in PANEL_ORDER if (model, "scope_first") in lab]
    med_models = [model for model in PANEL_ORDER if (model, "scope_first") in med]
    if models != med_models or not models:
        raise ValueError("LabTrend and MedDose must cover the same nonempty model panel")

    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "text.color": palette["ink"],
        "axes.labelcolor": palette["ink"],
        "axes.edgecolor": palette["secondary"],
        "axes.linewidth": 0.55,
        "xtick.color": palette["secondary"],
        "ytick.color": palette["secondary"],
        "hatch.linewidth": 0.6,
        "svg.hashsalt": "clinehr-operation-results",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), sharey=True)
    fig.patch.set_facecolor("white")
    width = 0.18
    panel_specs = (
        ("LabTrend", "(a) LabTrend", "Availability time · change and direction", lab, LAB_ALL_SIX),
        ("MedDose", "(b) MedDose", "Occurrence time · total and mean dose", med, MED_ALL_SIX),
    )
    plotted = []
    for ax, (family, title, subtitle, rows, all_six) in zip(axes, panel_specs):
        for offset, (key, label, fill, edge, hatch) in enumerate(SYSTEMS):
            system = all_six if key == "all_six" else key
            selected = [rows[(model, system)] for model in models]
            xs = [i + (offset - 1.5) * width for i in range(len(models))]
            means = [r["mean_pct"] for r in selected]
            ax.bar(xs, means, width=width, color=palette[fill],
                   edgecolor=palette[edge], linewidth=0.55, hatch=hatch, zorder=3)
            ax.errorbar(xs, means,
                        yerr=[[r["mean_pct"] - r["ci_low_pct"] for r in selected],
                              [r["ci_high_pct"] - r["mean_pct"] for r in selected]],
                        fmt="none", ecolor=palette["ink"], elinewidth=0.45,
                        capsize=1.2, capthick=0.45, zorder=4)
            for model, row in zip(models, selected):
                plotted.append({"family": family, "model": model, "system": system,
                                "label": label, **row})
        ax.text(0, 1.19, title, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="left")
        ax.text(0, 1.065, subtitle, transform=ax.transAxes, fontsize=7.8,
                color=palette["secondary"], va="bottom", ha="left")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([TICK_LABELS[model] for model in models], fontsize=7.0,
                           linespacing=1.3)
        ax.set_xlim(-0.58, len(models) - 0.42)
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=2.5, width=0.55, pad=4, labelsize=8)
        ax.grid(axis="y", color=palette["rule"], linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Joint validity (%)", fontsize=8.5, labelpad=7)
    axes[1].tick_params(axis="y", length=0)
    handles = [Patch(facecolor=palette[fill], edgecolor=palette[edge],
                     hatch=hatch, linewidth=0.55, label=label)
               for _, label, fill, edge, hatch in SYSTEMS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=False, fontsize=8, handlelength=1.8,
               handletextpad=0.55, columnspacing=1.55, borderaxespad=0)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.285, wspace=0.11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"CreationDate": None, "ModDate": None, "Producer": None,
                "Creator": "make_operation_figure.py"}
    fig.savefig(args.out, format="pdf", metadata=metadata)
    fig.savefig(args.out.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(args.out.with_suffix(".png"), dpi=300, metadata={"Software": "make_operation_figure.py"})
    manifest = {
        "metric": "Exact structured answer correctness and no out-of-contract citations",
        "uncertainty": "95% patient-cluster bootstrap intervals from frozen summary CSVs",
        "scope": "Separate ScopePair diagnostic tests; not pooled into ScopeBench tier totals",
        "full_result": "Final verified protocol output, including deterministic fallback",
        "palette": palette,
        "sources": [{"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (args.lab_analysis, args.med_analysis)],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "values": plotted,
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt.close(fig)
    print(f"Wrote {args.out}, SVG/PNG previews, and source manifest ({len(models)} models; {len(plotted)} bars)")


if __name__ == "__main__":
    main()
