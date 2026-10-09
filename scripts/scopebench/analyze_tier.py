#!/usr/bin/env python3
"""Tier analysis (plan §9.1–9.4): endpoints per configuration overall / by family / by scope / leakage
stratum with patient-cluster bootstrap intervals; the prespecified paired contrasts (H1–H4) with Holm
correction; the covered-operation stage table for `full`; abstention and cost tables relative to
prompt_only. Loads no model. One checkpoint per invocation.

Usage: analyze_tier.py --runs-dir experiments/scopebench_tierB_runs/<tag> --out experiments/scopebench_analysis/tierB/<tag>
       (expects <runs-dir>/<configuration>/predictions.jsonl for the configurations present)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.stats import cluster_bootstrap_mean, holm, paired_cluster_contrast  # noqa: E402

ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "workflow_compliant", "abstained", "parse_failed"]
HYPOTHESES = {  # name: (endpoint, from_config, to_config)
    "H1_scope_vs_prompt": ("completeness", "prompt_only", "scope_aware"),
    "H2_audit_vs_scope": ("completeness", "scope_aware", "audit"),
    "H3_full_vs_audit": ("completeness", "audit", "full"),
    "H4_scope_vs_wrongbasis_citation": ("citation_legal", "wrong_basis", "scope_aware"),
}


def load_runs(runs_dir: Path) -> dict[str, dict[str, dict]]:
    runs = {}
    for d in sorted(runs_dir.iterdir()):
        f = d / "predictions.jsonl"
        if d.is_dir() and f.exists():
            runs[d.name] = {r["task_id"]: r for r in read_jsonl(f)}
    return runs


def cells(rows: dict[str, dict], n_boot: int, seed: int) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows.values():
        groups["all"].append(r)
        groups[f"family={r.get('task_type')}"].append(r)
        groups[f"scope={r['scope']}"].append(r)
        groups[f"leakage={bool(r.get('leakage'))}"].append(r)
    out = {}
    for name, rs in groups.items():
        cell = {"n": len(rs), "n_patients": len({r["patient"] for r in rs})}
        for ep in ENDPOINTS:
            point, lo, hi = cluster_bootstrap_mean([float(r["metrics"][ep]) for r in rs], [r["patient"] for r in rs], n_boot=n_boot, seed=seed)
            cell[ep] = {"mean": round(100 * point, 2), "ci_low": round(100 * lo, 2), "ci_high": round(100 * hi, 2)}
        cell["mean_tokens_in"] = round(sum(r["metrics"]["tokens_in"] for r in rs) / len(rs), 1)
        cell["mean_tokens_out"] = round(sum(r["metrics"]["tokens_out"] for r in rs) / len(rs), 1)
        cell["mean_calls"] = round(sum(r["metrics"]["calls"] for r in rs) / len(rs), 3)
        cell["median_latency_s"] = round(sorted(r["metrics"]["latency_seconds"] for r in rs)[len(rs) // 2], 3)
        cell["repaired_pct"] = round(100 * sum(bool(r.get("guard_repaired")) for r in rs) / len(rs), 2)
        cell["replaced_pct"] = round(100 * sum(bool(r.get("guard_deterministic_fallback")) for r in rs) / len(rs), 2)
        # infrastructure failures (HTTP errors, timeouts) are reported, never silently folded into parse failures
        cell["generation_error_pct"] = round(100 * sum(bool(r.get("generation_errors")) for r in rs) / len(rs), 2)
        out[name] = cell
    return out


def stage_table(rows: dict[str, dict], covered_types: set[str]) -> dict:
    cov = [r for r in rows.values() if r.get("task_type") in covered_types]
    if not cov:
        return {}
    n = len(cov)
    initial_valid = sum(1 for r in cov if r.get("initial_guard_errors") == [])
    repaired = sum(1 for r in cov if r.get("guard_repaired"))
    replaced = sum(1 for r in cov if r.get("guard_deterministic_fallback"))
    final_valid = sum(1 for r in cov if r.get("final_guard_errors") == [])
    initial_correct = sum(1 for r in cov if r["initial_metrics"]["answer_correct"])
    final_correct = sum(1 for r in cov if r["metrics"]["answer_correct"])
    wrong_fixed_by_replacement = sum(1 for r in cov if r.get("guard_deterministic_fallback") and not r["initial_metrics"]["answer_correct"] and r["metrics"]["answer_correct"])
    return {"n_covered": n, "initial_valid_pct": round(100 * initial_valid / n, 2), "repaired_pct": round(100 * repaired / n, 2),
            "replaced_pct": round(100 * replaced / n, 2), "final_valid_pct": round(100 * final_valid / n, 2),
            "initial_correct_pct": round(100 * initial_correct / n, 2), "final_correct_pct": round(100 * final_correct / n, 2),
            "wrong_answers_corrected_only_by_replacement": wrong_fixed_by_replacement}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs under {args.runs_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    covered = {"lab_trend_k3", "outcome_lookup_v1", "cohort_v1"}
    result = {"runs_dir": str(args.runs_dir), "configurations": sorted(runs), "n_boot": args.n_boot, "seed": args.seed, "cells": {}, "hypotheses": {}, "stages": {}, "cost": {}}
    for cfg, rows in runs.items():
        result["cells"][cfg] = cells(rows, args.n_boot, args.seed)
        if cfg in ("full", "sp_clinehr_os", "skill_verify_no_replace", "skill_replace_no_payload", "skill_payload_only"):
            result["stages"][cfg] = stage_table(rows, covered)
    # hypotheses
    raw_p = {}
    for name, (ep, a, b) in HYPOTHESES.items():
        if a in runs and b in runs:
            ta = {i: float(r["metrics"][ep]) for i, r in runs[a].items()}
            tb = {i: float(r["metrics"][ep]) for i, r in runs[b].items()}
            clusters = {i: r["patient"] for i, r in runs[a].items()}
            res = paired_cluster_contrast(ta, tb, clusters, n_boot=args.n_boot, seed=args.seed)
            res.update({"endpoint": ep, "from": a, "to": b, "delta_pct": round(100 * res["delta"], 2), "ci_low_pct": round(100 * res["ci_low"], 2), "ci_high_pct": round(100 * res["ci_high"], 2)})
            result["hypotheses"][name] = res
            raw_p[name] = res["p_bootstrap"]
    for name, adj in holm(raw_p).items():
        result["hypotheses"][name]["p_holm"] = adj
    # cost relative to prompt_only
    if "prompt_only" in runs:
        base = result["cells"]["prompt_only"]["all"]
        for cfg in runs:
            c = result["cells"][cfg]["all"]
            tok_base = base["mean_tokens_in"] + base["mean_tokens_out"]
            result["cost"][cfg] = {"tokens_ratio": round((c["mean_tokens_in"] + c["mean_tokens_out"]) / tok_base, 3) if tok_base else None,
                                  "calls": c["mean_calls"], "latency_ratio": round(c["median_latency_s"] / base["median_latency_s"], 3) if base["median_latency_s"] else None}
    (args.out / "analysis.json").write_text(json.dumps(result, indent=1))
    with open(args.out / "cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "cell", "n", "n_patients"] + [f"{ep}_{s}" for ep in ENDPOINTS for s in ("mean", "ci_low", "ci_high")] + ["mean_tokens_in", "mean_tokens_out", "mean_calls", "median_latency_s", "repaired_pct", "replaced_pct", "generation_error_pct"])
        for cfg, cs in result["cells"].items():
            for cell, v in cs.items():
                w.writerow([cfg, cell, v["n"], v["n_patients"]] + [v[ep][s] for ep in ENDPOINTS for s in ("mean", "ci_low", "ci_high")] + [v["mean_tokens_in"], v["mean_tokens_out"], v["mean_calls"], v["median_latency_s"], v["repaired_pct"], v["replaced_pct"], v["generation_error_pct"]])
    with open(args.out / "hypotheses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hypothesis", "endpoint", "from", "to", "n", "n_clusters", "delta_pct", "ci_low_pct", "ci_high_pct", "p_bootstrap", "p_holm"])
        for name, h in result["hypotheses"].items():
            w.writerow([name, h["endpoint"], h["from"], h["to"], h["n"], h["n_clusters"], h["delta_pct"], h["ci_low_pct"], h["ci_high_pct"], h["p_bootstrap"], h.get("p_holm")])
    for cfg in sorted(runs):
        a = result["cells"][cfg]["all"]
        print(f"{cfg:26s} n={a['n']:<5} correct={a['answer_correct']['mean']:>6}  compl={a['completeness']['mean']:>6}  A-L={a['access_legal']['mean']:>6}  C-L={a['citation_legal']['mean']:>6}  aud={a['auditable']['mean']:>6}  abst={a['abstained']['mean']:>5}")
    for name, h in result["hypotheses"].items():
        print(f"{name:32s} Δ={h['delta_pct']:>6} [{h['ci_low_pct']}, {h['ci_high_pct']}] p_holm={h.get('p_holm')}")
    print("written", args.out)


if __name__ == "__main__":
    main()
