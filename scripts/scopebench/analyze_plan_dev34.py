#!/usr/bin/env python3
"""DEV-34 analysis (experiments/SCOPEBENCH_PLAN_PROTOCOL.md §5): the plan-audited configurations against the existing
formal arms on the same tasks, and diagnostics of the declared plans. Loads no model.

Endpoints per (checkpoint, configuration), all tasks and covered / uncovered: answer correctness, completeness Ev.C,
access compliance A-L, citation compliance C-L — the shared scorer's per-row metrics, unchanged. Paired contrasts
(patient-cluster bootstrap, 10,000 resamples): plan_full − scope_aware, plan_full − full (the gap to the reference
system), plan_full − audit, plan_audit − audit; per checkpoint and pooled (rows of all checkpoints, clusters = patient).
Plan diagnostics: declared / valid / null rates, agreement with the reference plan (operation, arguments, exact),
R̂_t against R_t (precision, recall), repair and replacement rates. Output is aggregate only (Tier-B is MIMIC-derived;
Tier-A, addendum 2, reads the enclave run trees on the workstation and adds answer correctness per task type, so the
skill-covered but plan-uncovered `outcome_count_v1` is reported apart).

Usage: analyze_plan_dev34.py --tier A|B|C --phase dev|formal [--tags ...] [--plan-root DIR] [--out DIR] [--n-boot N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench import plans  # noqa: E402
from scopebench import tasktypes_tierb  # noqa: E402,F401
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.stats import cluster_bootstrap_mean, paired_cluster_contrast  # noqa: E402
from scopebench.tasktypes import get as task_type  # noqa: E402

E = ROOT / "experiments"
ENDPOINTS = ("answer_correct", "completeness", "access_legal", "citation_legal")
PLAN_CONFIGS = ("plan_full", "plan_audit")
REFERENCE_CONFIGS = ("prompt_only", "scope_aware", "audit", "full")
CONTRASTS = (("scope_aware", "plan_full"), ("full", "plan_full"), ("audit", "plan_full"), ("audit", "plan_audit"))
TAGS = ("qwen3_4b", "qwen3_30b", "gemma3_27b")


def layout(tier: str, phase: str) -> dict[str, Any]:
    suf = "_dev" if phase == "dev" else ""
    if tier == "A":
        assert phase == "formal", "Tier-A has no development split"
        from scopebench import tasktypes_tiera  # noqa: F401  (registers the Tier-A task types)
        from scopebench.tiera.config import ENCLAVE
        return {"plan_root": Path(ENCLAVE) / "runs_dev34", "ref_root": Path(ENCLAVE) / "runs", "manifest": Path(ENCLAVE) / "tierA", "ids": None}
    if tier == "B":
        return {"plan_root": E / f"scopebench_tierB_runs_dev34{suf}", "ref_root": E / "scopebench_tierB_runs" if phase == "formal" else None,
                "manifest": E / ("scopebench_tierB_dev" if phase == "dev" else "scopebench_tierB"),
                "ids": None if phase == "dev" else E / "scopebench_tierB/plan_dev34_task_ids.json"}
    return {"plan_root": E / f"scopebench_tierC_runs_dev34{suf}", "ref_root": E / "scopebench_tierC_runs_dev32",
            "manifest": E / "scopebench_tierC/fhir_agentbench",
            "ids": E / "scopebench_tierC/plan_dev34" / ("valid_task_ids.json" if phase == "dev" else "traintest_task_ids.json")}


def load_rows(path: Path, keep: set[str]) -> dict[str, dict]:
    p = path / "predictions.jsonl"
    return {r["task_id"]: r for r in read_jsonl(p) if r["task_id"] in keep} if p.exists() else {}


def reference_plans(tier: str, tasks: dict[str, Task]) -> dict[str, plans.Plan]:
    if tier in ("A", "B"):
        return {i: plans.reference_plan(t) for i, t in tasks.items()}
    from scopebench.tierc.plans import load_catalog, reference_plan

    cat = load_catalog()
    return {i: reference_plan(t, cat) for i, t in tasks.items()}


def endpoint_cells(rows: dict[str, dict], tasks: dict[str, Task], covered: dict[str, bool], n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part, sel in (("all", lambda i: True), ("covered", lambda i: covered[i]), ("uncovered", lambda i: not covered[i])):
        ids = [i for i in sorted(rows) if sel(i)]
        cell: dict[str, Any] = {"n": len(ids)}
        for ep in ENDPOINTS:
            vals = [(float(rows[i]["metrics"][ep]), tasks[i].patient) for i in ids if rows[i]["metrics"].get(ep) is not None]
            if vals:
                m, lo, hi = cluster_bootstrap_mean([v for v, _ in vals], [c for _, c in vals], n_boot=n_boot)
                cell[ep] = {"mean": round(100 * m, 1), "ci": [round(100 * lo, 1), round(100 * hi, 1)], "n": len(vals)}
        out[part] = cell
    return out


def contrast(a: dict[str, dict], b: dict[str, dict], tasks: dict[str, Task], covered: dict[str, bool], n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part, sel in (("all", lambda i: True), ("covered", lambda i: covered[i]), ("uncovered", lambda i: not covered[i])):
        res: dict[str, Any] = {}
        for ep in ("answer_correct", "completeness"):
            va = {i: float(r["metrics"][ep]) for i, r in a.items() if sel(i) and r["metrics"].get(ep) is not None}
            vb = {i: float(r["metrics"][ep]) for i, r in b.items() if sel(i) and r["metrics"].get(ep) is not None}
            if set(va) & set(vb):
                c = paired_cluster_contrast(va, vb, {i: tasks[i].patient for i in set(va) | set(vb)}, n_boot=n_boot)
                res[ep] = {"delta": round(100 * c["delta"], 1), "ci": [round(100 * c["ci_low"], 1), round(100 * c["ci_high"], 1)], "p": c["p_bootstrap"], "n": c["n"]}
        out[part] = res
    return out


def plan_diagnostics(tier: str, rows: dict[str, dict], refs: dict[str, plans.Plan], tasks: dict[str, Task]) -> dict[str, Any]:
    from scopebench.plans import Plan, plan_accuracy, required_overlap

    def as_plan(d: dict | None) -> Plan | None:
        if not d or d.get("plan") is None and not d.get("declared_none"):
            return None
        p = d.get("plan")
        if p is None:
            return Plan(None)
        return Plan(p["operation"], {k: v for k, v in p.items() if k != "operation"})

    acc: dict[str, list[float]] = defaultdict(list)
    for i, r in rows.items():
        ref = refs[i]
        for stage in ("initial", "final"):
            d = r.get(f"evidence_plan_{stage}")
            pred = as_plan(d)
            acc[f"{stage}_declared"].append(float(bool(d and d.get("present"))))
            acc[f"{stage}_valid"].append(float(pred is not None))
            if ref.operation is not None:
                a = plan_accuracy(pred if pred is not None and pred.operation is not None else None, ref)
                for k in ("operation", "args", "exact"):
                    acc[f"{stage}_covered_{k}"].append(float(a[k]))
                acc[f"{stage}_covered_declared_null"].append(float(pred is not None and pred.operation is None))
            else:
                acc[f"{stage}_uncovered_declared_null"].append(float(pred is not None and pred.operation is None))
            res = r.get(f"plan_result_{stage}")
            if res and res.get("error") is None and ref.operation is not None:
                ov = required_overlap(res.get("required") or [], list(tasks[i].required_ids))
                for k in ("precision", "recall"):
                    if ov[k] is not None:
                        acc[f"{stage}_covered_rhat_{k}"].append(ov[k])
                acc[f"{stage}_covered_rhat_exact"].append(float(ov["exact"]))
        acc["repaired"].append(float(bool(r.get("guard_repaired"))))
        acc["replaced"].append(float(bool(r.get("guard_deterministic_fallback"))))
    return {k: {"rate": round(100 * mean(v), 1), "n": len(v)} for k, v in sorted(acc.items()) if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", required=True, choices=["A", "B", "C"])
    ap.add_argument("--phase", required=True, choices=["dev", "formal"])
    ap.add_argument("--tags", nargs="*", default=list(TAGS))
    ap.add_argument("--plan-root", type=Path, default=None, help="override the plan run tree (e.g. an offline smoke run)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    lay = layout(args.tier, args.phase)
    plan_root = args.plan_root or lay["plan_root"]
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(lay["manifest"] / "tasks.jsonl")}
    if lay["ids"] is not None:
        keep = set(json.loads(lay["ids"].read_text()))
        tasks = {i: t for i, t in tasks.items() if i in keep}
    keep = set(tasks)
    refs = reference_plans(args.tier, tasks)
    covered = {i: refs[i].operation is not None for i in tasks}
    if args.tier == "B":
        assert all(covered[i] == task_type(t.task_type).covered for i, t in tasks.items())

    root_name = str(plan_root.relative_to(ROOT)) if plan_root.is_relative_to(ROOT) else ("<enclave>/" + plan_root.name if args.tier == "A" else str(plan_root))
    result: dict[str, Any] = {"tier": args.tier, "phase": args.phase, "plan_root": root_name,
                              "n_tasks": len(tasks), "n_covered": sum(covered.values()), "checkpoints": {}}
    pooled: dict[str, dict[str, dict]] = defaultdict(dict)
    for tag in args.tags:
        rows = {c: load_rows(plan_root / tag / c, keep) for c in PLAN_CONFIGS}
        if lay["ref_root"] is not None:
            rows.update({c: load_rows(lay["ref_root"] / tag / c, keep) for c in REFERENCE_CONFIGS})
        rows = {c: r for c, r in rows.items() if r}
        if not any(c in rows for c in PLAN_CONFIGS):
            continue
        entry: dict[str, Any] = {"endpoints": {c: endpoint_cells(r, tasks, covered, args.n_boot) for c, r in rows.items()},
                                 "contrasts": {f"{b} - {a}": contrast(rows[a], rows[b], tasks, covered, args.n_boot) for a, b in CONTRASTS if a in rows and b in rows},
                                 "plans": {c: plan_diagnostics(args.tier, rows[c], refs, tasks) for c in PLAN_CONFIGS if c in rows}}
        if args.tier == "A":
            entry["answer_correct_by_task_type"] = {c: {tt: {"mean": round(100 * mean(float(r[i]["metrics"]["answer_correct"]) for i in ids), 1), "n": len(ids)}
                                                         for tt in sorted({tasks[i].task_type for i in r})
                                                         for ids in [[i for i in r if tasks[i].task_type == tt]]}
                                                     for c, r in rows.items()}
        result["checkpoints"][tag] = entry
        for c, r in rows.items():
            pooled[c].update({f"{i}@{tag}": v for i, v in r.items()})
    if len(result["checkpoints"]) > 1:
        ptasks = {k: tasks[k.split("@")[0]] for rows in pooled.values() for k in rows}
        pcov = {k: covered[k.split("@")[0]] for k in ptasks}
        result["pooled"] = {"checkpoints": sorted(result["checkpoints"]),
                            "endpoints": {c: endpoint_cells(r, ptasks, pcov, args.n_boot) for c, r in pooled.items()},
                            "contrasts": {f"{b} - {a}": contrast(pooled[a], pooled[b], ptasks, pcov, args.n_boot) for a, b in CONTRASTS if a in pooled and b in pooled}}
    out = args.out or E / "scopebench_analysis" / "plans_dev34" / f"tier{args.tier}_{args.phase}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for tag, entry in result["checkpoints"].items():
        for c, cell in entry["endpoints"].items():
            a = cell["all"]
            print(f"{tag:12s} {c:12s} n={a['n']:<5} " + "  ".join(f"{ep[:6]}={a[ep]['mean']:>5}" for ep in ENDPOINTS if ep in a))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
