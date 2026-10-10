#!/usr/bin/env python3
"""Endpoint figure (single column): evidence completeness (top) and answer correctness (bottom) per Tier-B checkpoint as paired
dumbbells, prompt-only (open mark) to the full system (filled mark), with a panel-mean row and patient-cluster 95 %
intervals (plan §9.5). Reads panel.json; loads no model; Type 42 fonts; no creation date so rebuilds are byte-identical.

Usage: make_gap_figure.py --panel experiments/scopebench_analysis/tierB/panel/panel.json --out docs/figures/gap.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ORDER = ["qwen3_4b", "phi4_14b", "gemma3_27b", "medgemma_27b", "qwen3_32b", "qwen3_30b", "llama33_70b", "claude_opus"]
LABEL = {"qwen3_4b": "Qwen3-4B", "phi4_14b": "Phi-4", "gemma3_27b": "Gemma-3-27B", "medgemma_27b": "MedGemma-27B", "qwen3_32b": "Qwen3-32B",
         "qwen3_30b": "Qwen3-30B-A3B", "llama33_70b": "Llama-3.3-70B", "claude_opus": "Claude Opus"}
# Match the manuscript's architecture and operation figures: neutral baseline,
# navy intervention. Open/filled markers also distinguish the configurations.
PALETTE_PATH = Path(__file__).resolve().parents[2] / "docs/figures/palette.json"
PALETTE = json.loads(PALETTE_PATH.read_text())
LIGHT, DARK = PALETTE["secondary"], PALETTE["valid"]
INK, INK2 = PALETTE["ink"], PALETTE["secondary"]
MUTED, LINK, GRID = PALETTE["secondary"], PALETTE["baseline_fill"], PALETTE["rule"]
PANELS = (("completeness", "Evidence completeness (%)"), ("answer_correct", "Answer correctness (%)"))
COLUMN_IN = 3.3   # AAAI/ICML single column ≈ 3.3 in


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    models = json.loads(args.panel.read_text()).get("models", {})
    tags = [t for t in ORDER if t in models] + sorted(t for t in models if t not in ORDER)
    n = len(tags)
    rows = {t: n - i for i, t in enumerate(tags)}          # first checkpoint at the top
    rows["__mean__"] = -0.35                                # panel-mean row, set off below a separator

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 2.9), dpi=200, sharex=True, gridspec_kw={"hspace": 0.32})
    fig.patch.set_facecolor("white")
    lo = min(models[t][c][m]["ci_low"] for t in tags for c in ("prompt_only", "full") for m, _ in PANELS)
    x0, x1 = max(0, 10 * int(lo // 10)), 100

    for ax, (metric, title) in zip(axes, PANELS):
        ax.set_facecolor("white")
        ax.set_xlim(x0, x1); ax.set_ylim(-1.0, n + 0.75)
        ax.axhline(0.35, color=MUTED, lw=0.4, zorder=1)                    # separator above the mean row
        means = {}
        for cfg in ("prompt_only", "full"):
            means[cfg] = sum(models[t][cfg][metric]["mean"] for t in tags) / n
        for key in tags + ["__mean__"]:
            y = rows[key]
            if key == "__mean__":
                a, b = means["prompt_only"], means["full"]
                ax.plot([a, b], [y, y], color=LINK, lw=2.2, solid_capstyle="round", zorder=2)
                ax.plot(a, y, "o", ms=5.2, mfc="white", mec=LIGHT, mew=1.3, zorder=4)
                ax.plot(b, y, "o", ms=5.2, mfc=DARK, mec=DARK, zorder=4)
                ax.annotate(f"{a:.0f}", (a, y), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=6, color=INK, fontweight="bold")
                ax.annotate(f"{b:.0f}", (b, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6, color=INK, fontweight="bold")
                continue
            a, b = models[key]["prompt_only"][metric], models[key]["full"][metric]
            ax.plot([a["mean"], b["mean"]], [y, y], color=LINK, lw=1.6, solid_capstyle="round", zorder=2)
            for v, color, filled in ((a, LIGHT, False), (b, DARK, True)):
                ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=color, lw=0.7, zorder=3)        # 95 % interval
                ax.plot(v["mean"], y, "o", ms=4.6, mfc=color if filled else "white", mec=color, mew=1.1, zorder=4)
        ax.set_yticks([rows[t] for t in tags] + [rows["__mean__"]])
        ax.set_yticklabels([LABEL.get(t, t) for t in tags] + ["Panel mean"], fontsize=6.5, color=INK2)
        ax.get_yticklabels()[-1].set_fontweight("bold"); ax.get_yticklabels()[-1].set_color(INK)
        ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=7, fontweight="bold", color=INK, ha="left", va="bottom")
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=6.5, colors=INK2, length=2, width=0.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED); ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(True, axis="x", lw=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("Percent of tasks", fontsize=7, color=INK2, labelpad=2)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=LIGHT, mew=1.1, ms=4.6, label="prompt-only"),
               Line2D([], [], marker="o", ls="", mfc=DARK, mec=DARK, ms=4.6, label="full EHR-ScopeAudit")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.5, frameon=False, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.255, right=0.975, top=0.895, bottom=0.13, hspace=0.42)   # left margin holds the checkpoint names
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": "make_gap_figure.py"})
    print(f"wrote {args.out} with {n} checkpoints")


if __name__ == "__main__":
    main()
