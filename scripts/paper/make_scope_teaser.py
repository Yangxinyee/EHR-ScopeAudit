#!/usr/bin/env python3
"""Draw an introductory scope counterexample and the hidden-violation rates of agents that issue their own requests.

The left panel is schematic, not a patient observation. The right panel reads
the manuscript's frozen macros (AH.*, scripts/scopebench/analyze_agent_hidden.py);
it does not rerun analyses or infer new rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/scope_teaser.pdf")
    args = parser.parse_args()
    values_path = ROOT / "results/paper/macros.json"
    palette_path = ROOT / "docs/figures/palette.json"
    values = json.loads(values_path.read_text())
    palette = json.loads(palette_path.read_text())
    # ICML revision §3.1: agents that issue their own requests (no scope enforcement), all-task denominators
    rows = [(f"Frontier · FHIR-AgentBench", "AH.F.fhir"), (f"Open-weight mean · FHIR-AgentBench", "AH.open.fhir"),
            (f"Open-weight ReAct · MIMIC-IV", "AH.B.react")]   # DEV-38: MedAgentBench agents stay in scope (AH.F.med, AH.open.med ≈ 0)
    rates = [(label, float(values[f"{k}.exposed"]), float(values[f"{k}.hidden"])) for label, k in rows]
    for _, exposed, hidden in rates:
        if not 0 <= hidden <= exposed <= 100:
            raise ValueError("The hidden violation rate must be a subset of the exposure rate")

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": palette["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "ehr-agentguard-scope-teaser",
        "hatch.linewidth": 0.6,
    })
    fig = plt.figure(figsize=(7.1, 2.15), facecolor="white")
    left = fig.add_axes((0.015, 0.06, 0.555, 0.90))
    right = fig.add_axes((0.62, 0.06, 0.365, 0.90))
    for ax in (left, right):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left.text(0, 1, "(a) Same answer, different evidence paths", va="top", weight="bold")
    # A decision boundary separates earlier available evidence from a later result.
    left.add_patch(Rectangle((0, 0.61), 0.55, 0.24, color=palette["scope_panel"], lw=0))
    left.add_patch(Rectangle((0.55, 0.61), 0.45, 0.24, color=palette["invalid"], alpha=0.09, lw=0))
    left.plot((0.04, 0.96), (0.71, 0.71), color=palette["secondary"], lw=0.8)
    left.plot((0.55, 0.55), (0.60, 0.89), color=palette["ink"], lw=0.85, ls=(0, (2, 2)))
    left.scatter((0.18, 0.82), (0.71, 0.71), s=25,
                 color=(palette["scope"], palette["invalid"]), zorder=3)
    left.text(0.18, 0.82, "Earlier record", ha="center", va="center", fontsize=8.5)
    left.text(0.82, 0.82, "Later result", ha="center", va="center", fontsize=8.5)
    left.text(0.18, 0.645, r"$e_1$", ha="center", va="center", color=palette["scope"])
    left.text(0.82, 0.645, r"$e_2$", ha="center", va="center", color=palette["invalid"])
    left.text(0.55, 0.905, "Decision time", ha="center", va="center", fontsize=8.2)
    left.text(0.015, 0.515, "Agent", fontsize=8.5, color=palette["secondary"])
    left.text(0.47, 0.515, "Accesses", ha="center", fontsize=8.5, color=palette["secondary"])
    left.text(0.71, 0.515, "Cites", ha="center", fontsize=8.5, color=palette["secondary"])
    left.text(0.915, 0.515, "Answer", ha="center", fontsize=8.5, color=palette["secondary"])
    left.plot((0, 1), (0.475, 0.475), color=palette["rule"], lw=0.7)
    # Keep the shared chart once; lightly distinguish the two execution paths.
    left.add_patch(Rectangle((0, 0.265), 1, 0.17,
                             facecolor=palette["invalid"], alpha=0.075, lw=0))
    left.add_patch(Rectangle((0, 0.065), 1, 0.17,
                             facecolor=palette["scope_panel"], lw=0))
    left.add_patch(Rectangle((0.495, 0.293), 0.062, 0.114,
                             facecolor="white", edgecolor=palette["invalid"],
                             lw=0.75, zorder=2))
    left.text(0.015, 0.35, "Unscoped", va="center")
    left.text(0.405, 0.35, r"$e_1$", va="center", ha="center", color=palette["scope"])
    left.text(0.465, 0.35, "+", va="center", ha="center")
    left.text(0.525, 0.35, r"$e_2$", va="center", ha="center", color=palette["invalid"])
    left.text(0.015, 0.15, "EHR-ScopeAudit", va="center", color=palette["scope"])
    left.text(0.47, 0.15, r"$e_1$", va="center", ha="center", color=palette["scope"])
    for y in (0.35, 0.15):
        left.text(0.71, y, r"$e_1$", va="center", ha="center", color=palette["scope"])
        left.text(0.915, y, "Correct", va="center", ha="center")

    right.text(0, 1, "(b) What citations miss", va="top", weight="bold")
    right.text(0, 0.885, "Agents issuing own requests (% of tasks)", fontsize=8.2, color=palette["secondary"])
    # One bar per setting on a shared 0–100% all-task scale: the solid part is any out-of-scope access, the hatched
    # part the joint event with fully admissible citations (a subset, not a conditional fraction).
    bar_start, bar_width = 0.01, 0.64
    for i, (label, exposed, hidden) in enumerate(rates):
        y = 0.64 - i * 0.205
        right.text(bar_start, y + 0.085, label, fontsize=8.0, va="bottom")
        right.add_patch(Rectangle((bar_start, y), bar_width, 0.075, facecolor=palette["neutral"], lw=0))
        right.add_patch(Rectangle((bar_start, y), bar_width * exposed / 100, 0.075, facecolor=palette["invalid"], alpha=0.35, lw=0))
        right.add_patch(Rectangle((bar_start, y), bar_width * hidden / 100, 0.075, facecolor=palette["invalid"], edgecolor="white", hatch="///", lw=0))
        right.text(bar_start + bar_width * exposed / 100 + 0.02, y + 0.0375, f"{hidden:.1f}% / {exposed:.1f}%",
                   va="center", fontsize=8.2, weight="bold", color=palette["invalid"])
    right.add_patch(Rectangle((bar_start, 0.105), 0.035, 0.035, facecolor=palette["invalid"], alpha=0.35, lw=0))
    right.text(bar_start + 0.05, 0.1225, "any out-of-scope access", va="center", fontsize=7.6)
    right.add_patch(Rectangle((bar_start, 0.035), 0.035, 0.035, facecolor=palette["invalid"], edgecolor="white", hatch="///", lw=0))
    right.text(bar_start + 0.05, 0.0525, "… with only admissible citations", va="center", fontsize=7.6)
    fig.add_artist(plt.Line2D((0.593, 0.593), (0.11, 0.88), transform=fig.transFigure,
                             color=palette["rule"], lw=0.7))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"CreationDate": None})
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    fig.savefig(args.out.with_suffix(".svg"))
    plt.close(fig)
    trace = {
        "artifact_id": "fig:scope_teaser",
        "source_data": str(values_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
        "palette_sha256": hashlib.sha256(palette_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "values": {f"{k}.{f}": float(values[f"{k}.{f}"]) for _, k in rows for f in ("exposed", "hidden")},
        "transformation": "Read rounded checkpoint means from frozen manuscript macros; no reanalysis.",
        "caption_claim": "Admissible citations can conceal inadmissible access.",
        "supported_manuscript_claims": [{"claim": "An agent can read a future result, return the correct answer, and cite earlier records that also support it.", "locator": "Introduction, paragraph 2"}],
        "limitations": ["Panel (a) is illustrative; both records can support the same answer.", "Panel (b) summarizes unscoped agent loops (frontier and open-weight); it does not measure clinician outcomes or condition on correct answers.", "These rounded descriptive means are shown without uncertainty intervals; detailed results are reported later in the paper."],
    }
    args.out.with_suffix(".json").write_text(json.dumps(trace, indent=2) + "\n")
    print(f"Wrote {args.out}: " + "; ".join(f"{l}: {h:.1f}/{e:.1f}%" for l, e, h in rates))


if __name__ == "__main__":
    main()
