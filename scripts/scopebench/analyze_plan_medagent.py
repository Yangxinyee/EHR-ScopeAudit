#!/usr/bin/env python3
"""Plan protocol addendum 3 (experiments/SCOPEBENCH_PLAN_PROTOCOL.md §10): the frontier `plan_full` on MedAgentBench
against the frontier's scope-aware, audit and reference-input (`full`) cells on the same 300 tasks, with the endpoint
cells, paired contrasts and plan diagnostics of analyze_plan_dev34.py (same functions; plan-covered = task2/4/6/7/9/10).
Also counts the covered failures whose executed plan agrees with a wrong answer because the agent retrieved too little
(the auditor checks the answer against the records the agent fetched). Loads no model.
Usage: analyze_plan_medagent.py [--out FILE] [--n-boot N] [--runs DIR]   (DEV-38 rows: scopebench_tierC_runs_dev38/frontier/medagentbench)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/scopebench"))

from analyze_plan_dev34 import contrast, endpoint_cells, plan_diagnostics  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.medagent_plans import reference_plan  # noqa: E402

E = ROOT / "experiments"
RUNS = E / "scopebench_tierC_runs/frontier/medagentbench"
CONFIGS = ("prompt_only", "scope_aware", "audit", "full", "plan_full")
CONTRASTS = (("scope_aware", "plan_full"), ("full", "plan_full"), ("audit", "plan_full"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/plans_dev34/medagent_frontier.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--runs", type=Path, default=RUNS)
    args = ap.parse_args()
    runs = args.runs
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(E / "scopebench_tierC/medagentbench/tasks.jsonl")}
    refs = {i: reference_plan(t) for i, t in tasks.items()}
    covered = {i: refs[i].operation is not None for i in tasks}
    rows = {c: {r["task_id"]: r for r in read_jsonl(runs / c / "predictions.jsonl")} for c in CONFIGS if (runs / c / "predictions.jsonl").exists()}
    pf = rows["plan_full"]
    entry = {"endpoints": {c: endpoint_cells(r, tasks, covered, args.n_boot) for c, r in rows.items()},
             "contrasts": {f"{b} - {a}": contrast(rows[a], rows[b], tasks, covered, args.n_boot) for a, b in CONTRASTS if a in rows and b in rows},
             "plans": {"plan_full": plan_diagnostics("C", pf, refs, tasks)}}
    wrong_cov = [i for i, r in pf.items() if covered[i] and not r["metrics"]["answer_correct"]]
    agreed = [i for i in wrong_cov if (r := pf[i]).get("plan_result_final") and not any(e.startswith("agreement") for e in r["final_guard_errors"])
              and (r["plan_result_final"].get("summary") or {}).get("answer") == (r["parsed_answer"] or {}).get("answer")]
    entry["covered_failures"] = {"n": len(wrong_cov), "plan_agrees_with_wrong_answer": len(agreed),
                                 "of_which_incomplete_evidence": sum(pf[i]["metrics"]["completeness"] < 1 for i in agreed),
                                 "full_correct_on_these": sum(bool(rows["full"][i]["metrics"]["answer_correct"]) for i in wrong_cov)}
    out = {"bench": "medagentbench", "n_tasks": len(tasks), "n_covered": sum(covered.values()), "checkpoints": {"frontier": entry}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for c, cell in entry["endpoints"].items():
        a = cell["all"]
        print(f"{c:12s} n={a['n']:<4} ans={a['answer_correct']['mean']:>5} evc={a['completeness']['mean']:>5} al={a['access_legal']['mean']:>5} cl={a['citation_legal']['mean']:>5}")
    for k, v in entry["contrasts"].items():
        print(k, v["all"].get("answer_correct"))
    print("plan diag:", {k: entry["plans"]["plan_full"][k]["rate"] for k in ("final_covered_operation", "final_covered_exact", "final_declared", "repaired", "replaced") if k in entry["plans"]["plan_full"]})
    print("covered failures:", entry["covered_failures"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
