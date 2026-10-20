#!/usr/bin/env python3
"""Frontier vs open-weight comparison on the 300-task FHIR-AgentBench frontier subsample (Tier-C protocol, frontier
addendum): per configuration and endpoint, the frontier mean and the open checkpoint's mean on the same 300 tasks with
patient-cluster intervals, the open checkpoint's all-task mean, and the paired frontier − open contrast; plus the
within-model contrasts (H1–H4, filtering on access and on answers) for both models on the 300 tasks.

Writes <out-dir>/frontier_vs_<tag>_fhir300.json (the files build_paper.py reads). Loads no model.
Usage: compare_frontier_open.py --runs-root experiments/scopebench_tierC_runs --out-dir experiments/scopebench_analysis/tierC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.stats import cluster_bootstrap_mean, paired_cluster_contrast  # noqa: E402

TAGS = ["gemma3_27b", "medgemma_27b", "phi4_14b", "qwen3_30b", "qwen3_32b", "qwen3_4b"]
CONFIGS = ["prompt_only", "late_filter", "wrong_basis", "scope_aware", "audit", "full"]
ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable"]
CONTRASTS = {"H1_scope_vs_prompt": ("completeness", "prompt_only", "scope_aware"), "H2_audit_vs_scope": ("completeness", "scope_aware", "audit"),
             "H3_full_vs_audit": ("completeness", "audit", "full"), "H4_scope_vs_wrongbasis_citation": ("citation_legal", "wrong_basis", "scope_aware"),
             "scope_vs_prompt_access": ("access_legal", "prompt_only", "scope_aware"), "scope_vs_prompt_answer": ("answer_correct", "prompt_only", "scope_aware")}


def load(d: Path) -> dict[str, dict]:
    f = d / "predictions.jsonl"
    return {r["task_id"]: {"patient": r["patient"], **{ep: float(r["metrics"][ep]) for ep in ENDPOINTS}} for r in read_jsonl(f)} if f.exists() else {}


def contrast(a: dict, b: dict, ep: str, n_boot: int, seed: int) -> dict:
    ids = sorted(set(a) & set(b))
    r = paired_cluster_contrast({i: a[i][ep] for i in ids}, {i: b[i][ep] for i in ids}, {i: a[i]["patient"] for i in ids}, n_boot=n_boot, seed=seed)
    return {"delta": r["delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_bootstrap": r["p_bootstrap"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tags", nargs="*", default=TAGS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    fr = {c: load(args.runs_root / "frontier" / "fhir_agentbench" / c) for c in CONFIGS}
    sub = set(fr["prompt_only"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        op_all = {c: load(args.runs_root / tag / c) for c in CONFIGS}
        op = {c: {i: r for i, r in rows.items() if i in sub} for c, rows in op_all.items()}
        out = {"tag": tag, "n_tasks": len(sub), "configs": {}, "within_model_contrasts": {}}
        for c in CONFIGS:
            if not fr[c] or not op[c]:
                continue
            cell = {}
            for ep in ENDPOINTS:
                f_ids = list(fr[c]); o_ids = list(op[c])   # file order: the interval resamples clusters in first-seen order
                cell[ep] = {"frontier": list(cluster_bootstrap_mean([fr[c][i][ep] for i in f_ids], [fr[c][i]["patient"] for i in f_ids], n_boot=args.n_boot, seed=args.seed)),
                            f"{tag}_300": list(cluster_bootstrap_mean([op[c][i][ep] for i in o_ids], [op[c][i]["patient"] for i in o_ids], n_boot=args.n_boot, seed=args.seed)),
                            f"{tag}_all": sum(r[ep] for r in op_all[c].values()) / len(op_all[c]),
                            "delta_frontier_minus_open": contrast(op[c], fr[c], ep, args.n_boot, args.seed)}
            out["configs"][c] = cell
        for name, rows in (("frontier", fr), (f"{tag}@300", op)):
            out["within_model_contrasts"][name] = {h: contrast(rows[a], rows[b], ep, args.n_boot, args.seed)
                                                   for h, (ep, a, b) in CONTRASTS.items() if rows.get(a) and rows.get(b)}
        (args.out_dir / f"frontier_vs_{tag}_fhir300.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"{tag}: {len(out['configs'])} configurations", flush=True)


if __name__ == "__main__":
    main()
