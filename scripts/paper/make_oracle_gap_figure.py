#!/usr/bin/env python3
"""ICML revision plan §8: where the deployable auditor falls short of the reference-input system. Answer correctness of
scope-aware retrieval, the plan-audited system (no reference input) and the reference-input system, on plan-covered and
uncovered tasks, per tier (mean of the three protocol checkpoints; every checkpoint runs the same tasks) and for the
frontier model. Every number is read from the manuscript's frozen macros (DEP.*, scripts/paper/build_paper.py).
Usage: make_oracle_gap_figure.py [--out docs/figures/oracle_gap.pdf]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PALETTE = json.loads((ROOT / "docs/figures/palette.json").read_text())
INK, INK2, GRID, LINK = PALETTE["ink"], PALETTE["secondary"], PALETTE["rule"], PALETTE["baseline_fill"]
SCOPE, PLAN, REF = PALETTE["secondary"], PALETTE["valid"], PALETTE["audit"]
PRIMARY = ("qwen3_4b", "qwen3_30b", "gemma3_27b")
ROWS = (("Tier-B · open", "B", PRIMARY), ("Tier-A · open", "A", PRIMARY), ("Tier-C · open", "C", PRIMARY),
        ("Tier-C · frontier", "F", ("frontier",)), ("MedAgentBench · frontier", "M", ("frontier",)))
CONFIGS = ("scope_aware", "plan_full", "full")
PANELS = (("cov", "Plan-covered tasks"), ("unc", "Uncovered tasks"))
COLUMN_IN = 3.3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/oracle_gap.pdf")
    args = ap.parse_args()
    V = json.loads((ROOT / "results/paper/macros.json").read_text())

    def val(X, tags, cfg, part):
        xs = [float(V[f"DEP.{X}.{t}.{cfg}.{part}.ans"]) for t in tags]
        return sum(xs) / len(xs)

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                                "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK})
    n = len(ROWS)
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 2.75), dpi=200, sharex=True)
    fig.patch.set_facecolor("white")
    for ax, (part, title) in zip(axes, PANELS):
        ax.set_facecolor("white")
        ax.set_xlim(0, 102); ax.set_ylim(-0.6, n - 0.4)
        for i, (label, X, tags) in enumerate(ROWS):
            y = n - 1 - i
            s, p, r = (val(X, tags, c, part) for c in CONFIGS)
            ax.plot([s, p], [y, y], color=LINK, lw=1.8, solid_capstyle="round", zorder=2)              # gain of the plan audit
            ax.plot([p, r], [y, y], color=REF, lw=0.9, ls=(0, (1.5, 1.2)), zorder=2)                   # remaining gap to the reference
            ax.plot(s, y, "o", ms=4.4, mfc="white", mec=SCOPE, mew=1.1, zorder=4, clip_on=False)
            ax.plot(r, y, "D", ms=3.6, mfc="white", mec=REF, mew=1.0, zorder=4, clip_on=False)
            ax.plot(p, y, "o", ms=4.6, mfc=PLAN, mec=PLAN, zorder=5, clip_on=False)
        ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_yticks([n - 1 - i for i in range(n)])
        ax.set_yticklabels([lab for lab, _, _ in ROWS], fontsize=6.3, color=INK2)
        ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=7, fontweight="bold", color=INK, ha="left", va="bottom")
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=6.5, colors=INK2, length=2, width=0.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(True, axis="x", lw=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("Answer correctness (%)", fontsize=7, color=INK2, labelpad=2)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=SCOPE, mew=1.1, ms=4.4, label="scope-aware"),
               Line2D([], [], marker="o", ls="", mfc=PLAN, mec=PLAN, ms=4.6, label="plan-audited"),
               Line2D([], [], marker="D", ls="", mfc="white", mec=REF, mew=1.0, ms=3.6, label="reference-input")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=3, fontsize=5.9, frameon=False,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)
    fig.subplots_adjust(left=0.37, right=0.955, top=0.88, bottom=0.14, hspace=0.42)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": "make_oracle_gap_figure.py"})
    if os.environ.get("ORACLE_GAP_PNG"):
        fig.savefig(os.environ["ORACLE_GAP_PNG"], dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
