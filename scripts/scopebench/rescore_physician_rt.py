#!/usr/bin/env python3
"""ICML revision plan §4 (W1, the non-circular check): re-score every system on the direct-set tasks against the
evidence the physicians selected from the chart without seeing the programmatic schema. For each direct-set card the
raters mark the records a correct answer may use (allowed) and must use (needed); this script recomputes, on the same
task ids and for every configuration (the reference-input arms and the plan-audited one), completeness |C ∩ R| / |R| and
citation compliance [C ≠ ∅ ∧ C ⊆ allowed] with the physicians' sets in place of the programmatic R_t and L_t, using each
task type's own citation reader (the scorer's). Reported per rater (then averaged) and for the strict-majority set of
the raters who rated the card, beside the programmatic values on the same rows, with paired patient-cluster contrasts
(plan_full − scope_aware, full − scope_aware). Uses only the cards already sent; output is aggregate only (Tier-A rows
and ratings stay in the enclave). Loads no model.

Usage: rescore_physician_rt.py <packets root> [<packets root> ...] [--tags ...] [--out FILE] [--n-boot N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/scopebench"))

from adjudication_analysis import load_roots  # noqa: E402
from scopebench import tasktypes_tierb  # noqa: E402,F401
from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.stats import paired_cluster_contrast  # noqa: E402
from scopebench.tasktypes import get as task_type  # noqa: E402

E = ROOT / "experiments"
CONFIGS = ("prompt_only", "wrong_basis", "scope_aware", "audit", "full", "plan_full")
CONTRASTS = (("scope_aware", "plan_full"), ("scope_aware", "full"))
PRIMARY = ("qwen3_4b", "qwen3_30b", "gemma3_27b")


def run_dirs(tier: str) -> dict[str, Path]:
    """configuration -> run root (<root>/<tag>/<configuration>/predictions.jsonl)."""
    if tier == "A":
        from scopebench import tasktypes_tiera  # noqa: F401
        from scopebench.tiera.config import ENCLAVE
        base, plan = Path(ENCLAVE) / "runs", Path(ENCLAVE) / "runs_dev34"
    else:
        base, plan = E / "scopebench_tierB_runs", E / "scopebench_tierB_runs_dev34"
    return {c: (plan if c == "plan_full" else base) for c in CONFIGS}


def physician_sets(key_direct: dict, ratings: dict) -> dict[str, dict]:
    """task_id -> {tier, per_rater: {rater: (allowed, needed)}, majority: (allowed, needed)}."""
    out = {}
    for pid, k in key_direct.items():
        rs = {rt: (set(r.get("allowed_ids") or []), set(r.get("needed_ids") or [])) for rt, rows in ratings.items()
              for p, r in rows.items() if p == pid and r.get("packet_type") == "direct_set"}
        if not rs:
            continue
        n = len(rs)
        items_a = set().union(*(a for a, _ in rs.values())); items_n = set().union(*(nd for _, nd in rs.values()))
        maj = ({i for i in items_a if sum(i in a for a, _ in rs.values()) * 2 > n},
               {i for i in items_n if sum(i in nd for _, nd in rs.values()) * 2 > n})
        out[k["task_id"]] = {"tier": k["tier"], "per_rater": rs, "majority": maj, "programmatic_required": set(k.get("required_ids") or [])}
    return out


def rescore(row: dict, allowed: set[str], needed: set[str]) -> tuple[float, bool]:
    tt = task_type(row["task_type"])
    parsed = row.get("parsed_answer")
    cset = set(tt.citations(parsed))
    substantive = tt.substantive(parsed)
    completeness = (len(cset & needed) / len(needed)) if needed else (1.0 if substantive else 0.0)
    return completeness, bool(substantive and cset and cset <= allowed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--tags", nargs="+", default=list(PRIMARY))
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/clinician/physician_rt_rescore.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    _, key_direct, _, ratings = load_roots(args.roots)
    sets = physician_sets(key_direct, ratings)
    result: dict = {"n_direct_cards_rated": len(sets), "tiers": {}}
    for tier in sorted({v["tier"] for v in sets.values()}):
        ids = {t for t, v in sets.items() if v["tier"] == tier}
        dirs = run_dirs(tier)
        cells: dict = {}
        per_task: dict[str, dict[str, float]] = defaultdict(dict)      # config -> {task@tag: physician completeness (majority)}
        clusters: dict[str, str] = {}
        for cfg, base in dirs.items():
            acc = defaultdict(list)
            for tag in args.tags:
                f = base / tag / cfg / "predictions.jsonl"
                if not f.exists():
                    continue
                for r in read_jsonl(f):
                    if r["task_id"] not in ids:
                        continue
                    s = sets[r["task_id"]]
                    acc["programmatic_completeness"].append(r["metrics"]["completeness"])
                    acc["programmatic_citation_legal"].append(float(r["metrics"]["citation_legal"]))
                    acc["answer_correct"].append(float(r["metrics"]["answer_correct"]))
                    pr = [rescore(r, a, nd) for a, nd in s["per_rater"].values()]
                    acc["physician_completeness_per_rater"].append(mean(c for c, _ in pr))
                    acc["physician_citation_legal_per_rater"].append(mean(float(l) for _, l in pr))
                    c_m, l_m = rescore(r, *s["majority"])
                    acc["physician_completeness_majority"].append(c_m)
                    acc["physician_citation_legal_majority"].append(float(l_m))
                    per_task[cfg][f"{r['task_id']}@{tag}"] = c_m
                    clusters[f"{r['task_id']}@{tag}"] = str(r.get("patient"))
            if acc:
                cells[cfg] = {"n_rows": len(acc["answer_correct"]), **{k: round(100 * mean(v), 1) for k, v in acc.items()}}
        contrasts = {}
        for a, b in CONTRASTS:
            if per_task.get(a) and per_task.get(b):
                c = paired_cluster_contrast(per_task[a], per_task[b], clusters, n_boot=args.n_boot)
                contrasts[f"{b} - {a}"] = {"delta": round(100 * c["delta"], 1), "ci": [round(100 * c["ci_low"], 1), round(100 * c["ci_high"], 1)], "n": c["n"]}
        agree = [len(v["majority"][1] & v["programmatic_required"]) / len(v["majority"][1] | v["programmatic_required"])
                 for t, v in sets.items() if t in ids and (v["majority"][1] | v["programmatic_required"])]
        result["tiers"][tier] = {"n_tasks": len(ids), "cells": cells, "contrasts_physician_completeness_majority": contrasts,
                                 "needed_vs_programmatic_jaccard_mean": round(mean(agree), 3) if agree else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for tier, d in result["tiers"].items():
        for cfg, c in d["cells"].items():
            print(f"{tier} {cfg:12s} n={c['n_rows']:<4} Ev.C prog={c['programmatic_completeness']:5} phys(maj)={c['physician_completeness_majority']:5} "
                  f"phys(rater)={c['physician_completeness_per_rater']:5}  C-L prog={c['programmatic_citation_legal']:5} phys={c['physician_citation_legal_majority']:5}")
        print(tier, d["contrasts_physician_completeness_majority"], "Jaccard(needed, R_t)", d["needed_vs_programmatic_jaccard_mean"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
