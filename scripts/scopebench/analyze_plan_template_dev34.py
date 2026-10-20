#!/usr/bin/env python3
"""DEV-34: where the Tier-C oracle gap comes from. On covered FHIR-AgentBench tasks (train + test), correctness of
`plan_full` and of the reference-input `full` split by whether the model's final declared template is the reference
template, per checkpoint (protocol and extension panels) and for the frontier sample. Loads no model.
Usage: analyze_plan_template_dev34.py [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.plans import load_catalog, reference_plan  # noqa: E402

E = ROOT / "experiments"
TAGS = ("qwen3_4b", "qwen3_30b", "gemma3_27b", "phi4_14b", "medgemma_27b", "qwen3_32b", "frontier")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/plans_dev34/tierC_template_choice.json")
    args = ap.parse_args()
    keep = set(json.loads((E / "scopebench_tierC/plan_dev34/traintest_task_ids.json").read_text()))
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(E / "scopebench_tierC/fhir_agentbench/tasks.jsonl") if r["task_id"] in keep}
    cat = load_catalog()
    ref = {i: reference_plan(t, cat).operation for i, t in tasks.items()}
    load = lambda p: {r["task_id"]: r for r in read_jsonl(p) if r["task_id"] in keep} if p.exists() else {}  # noqa: E731
    out = {}
    for tag in TAGS:
        if tag == "frontier":
            pf = load(E / "scopebench_tierC_runs/frontier/fhir_agentbench/plan_full/predictions.jsonl")
            fu = load(E / "scopebench_tierC_runs_dev32/frontier/fhir_agentbench/full/predictions.jsonl")
        else:
            pf = load(E / "scopebench_tierC_runs_dev34" / tag / "plan_full/predictions.jsonl")
            fu = load(E / "scopebench_tierC_runs_dev32" / tag / "full/predictions.jsonl")
        if not pf:
            continue
        groups = {"right": [], "wrong": []}
        for i, r in pf.items():
            if ref[i] is None:
                continue
            declared = ((r.get("evidence_plan_final") or {}).get("plan") or {}).get("operation")
            groups["right" if declared == ref[i] else "wrong"].append(i)
        cell = {}
        for g, ids in groups.items():
            for name, rows in (("plan_full", pf), ("full", fu)):
                ii = [i for i in ids if i in rows]
                cell[f"{g}.{name}"] = {"n": len(ii), "answer_correct": round(100 * sum(bool(rows[i]["metrics"]["answer_correct"]) for i in ii) / len(ii), 1) if ii else None}
        cell["template_right_rate"] = round(100 * len(groups["right"]) / max(1, len(groups["right"]) + len(groups["wrong"])), 1)
        out[tag] = cell
        print(f"{tag:13s} right {cell['template_right_rate']:5}%  plan_full {cell['right.plan_full']['answer_correct']} vs full {cell['right.full']['answer_correct']}"
              f"  | wrong: plan_full {cell['wrong.plan_full']['answer_correct']} vs full {cell['wrong.full']['answer_correct']}")
    # rows of every DEV-34 Tier-C plan run (plan_full and plan_audit, all checkpoints, frontier) that fail access compliance
    files = sorted((E / "scopebench_tierC_runs_dev34").glob("*/plan_*/predictions.jsonl")) + [E / "scopebench_tierC_runs/frontier/fhir_agentbench/plan_full/predictions.jsonl"]
    total = leak = via_replacement = 0
    for f in files:
        for r in read_jsonl(f):
            total += 1
            if r["metrics"].get("access_legal") is False:
                leak += 1; via_replacement += bool(r.get("guard_deterministic_fallback"))
    out["_access_leak"] = {"rows": total, "access_illegal": leak, "via_replacement": via_replacement}
    print(f"access-illegal plan rows: {leak} of {total} ({via_replacement} through replacement)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
