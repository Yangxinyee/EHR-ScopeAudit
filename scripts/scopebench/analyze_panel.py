#!/usr/bin/env python3
"""Cross-checkpoint Tier analysis (plan §9.1–9.4, Tables 2–6, 8, S1–S4, Figure 2 inputs). Loads no model.

Reads <runs-root>/<tag>/<configuration>/predictions.jsonl for every checkpoint and writes one panel.json plus
CSVs. Everything is computed from the saved rows: checkpoint means and their patient-cluster intervals, the
paper's checkpoint-averaged endpoint table, the four prespecified contrasts pooled over checkpoints (rows =
task × checkpoint, clusters = patient) and per checkpoint, the association analyses (Fisher-z intervals over
checkpoint means, per-checkpoint contingency + mutual information, multilevel logistic model), attribution
accounting (answer changed / trace-only), coverage, covered-operation stages, abstention, cost ratios,
strata deltas, the model × task type × scope matrix, the access/citation divergence and the error taxonomy.

Usage: analyze_panel.py --runs-root experiments/scopebench_tierB_runs --out experiments/scopebench_analysis/tierB/panel
       [--tags qwen3_4b ...] [--n-boot 10000] [--seed 2027] [--max-glm-rows 60000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.stats import cluster_bootstrap_mean, contingency, fisher_ci, holm, mutual_information_bits, paired_cluster_contrast, pearson  # noqa: E402
from scopebench.tasktypes import REGISTRY, get as get_task_type  # noqa: E402
from scopebench import tasktypes_tierb  # noqa: E402,F401
from scopebench import tasktypes_tiera, tasktypes_tierc  # noqa: E402,F401  (Tier-A / Tier-C task types: coverage flags, citation parsing)


def answer_value(task_type: str, parsed):
    """The parsed answer with its citations removed (keys named citation(s) and any string equal to a cited id), so
    'answer changed' means the answer VALUE changed, not merely its trace (Table 3's last column)."""
    if not isinstance(parsed, dict):
        return parsed
    # MedAgentBench rows (task type `medagentbench_v1`, scored by scopebench.tierc.medagent) carry no registered task
    # type: their answer and their citations are separate keys, so only the citation keys are stripped
    ids = set(get_task_type(task_type).citations(parsed)) if task_type in REGISTRY else set()

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("citation", "citations")}
        if isinstance(x, list):
            return [strip(v) for v in x if not (isinstance(v, str) and v in ids)]
        return None if (isinstance(x, str) and x in ids) else x
    return strip(parsed)

ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "abstained", "parse_failed", "workflow_compliant"]
CORE = ["prompt_only", "late_filter", "wrong_basis", "scope_aware", "audit", "full", "token_matched", "high_compute"]
ATTRIBUTION = ["scope_aware", "retry_generic", "feedback_schema_only", "audit", "skill_payload_only", "skill_verify_no_replace", "skill_replace_no_payload", "full"]
BASELINES = ["citation_validator", "reranker", "temporal_sql_view", "abac_pdp", "scope_aware", "full"]
HYPOTHESES = {"H1_scope_vs_prompt": ("completeness", "prompt_only", "scope_aware"), "H2_audit_vs_scope": ("completeness", "scope_aware", "audit"),
              "H3_full_vs_audit": ("completeness", "audit", "full"), "H4_scope_vs_wrongbasis_citation": ("citation_legal", "wrong_basis", "scope_aware")}
FAMILY = {"qwen3_4b": "qwen3", "qwen3_32b": "qwen3", "qwen3_30b": "qwen3", "phi4_14b": "phi", "gemma3_27b": "gemma", "medgemma_27b": "gemma", "llama33_70b": "llama", "claude_opus": "claude"}
PARAMS_B = {"qwen3_4b": 4.0, "phi4_14b": 14.7, "gemma3_27b": 27.4, "medgemma_27b": 27.4, "qwen3_32b": 32.8, "qwen3_30b": 30.5, "llama33_70b": 70.6}


def compact(r: dict) -> dict:
    m, im, fg = r["metrics"], r.get("initial_metrics") or {}, r.get("final_guard") or {}
    return {"task_id": r["task_id"], "patient": r["patient"], "task_type": r["task_type"], "scope": r["scope"], "leakage": bool(r.get("leakage")),
            **{ep: float(m.get(ep, 0.0)) for ep in ENDPOINTS},
            "tokens": m["tokens_in"] + m["tokens_out"], "calls": m["calls"], "latency": m["latency_seconds"], "n_illegal_access": m.get("n_illegal_access", 0),
            "initial_correct": bool(im.get("answer_correct", m["answer_correct"])),
            "repaired": bool(r.get("guard_repaired")), "replaced": bool(r.get("guard_deterministic_fallback")),
            "answer_changed": bool(r.get("guard_repaired")) and not r.get("guard_deterministic_fallback")
            and answer_value(r["task_type"], r.get("parsed_initial")) != answer_value(r["task_type"], r.get("parsed_answer")),
            "initial_invalid": bool(r.get("initial_guard_errors")), "final_invalid": bool(r.get("final_guard_errors")),
            # taxonomy inputs come from the scorer's metrics so that guarded and unguarded configurations are coded alike
            "omission": (not m.get("parse_failed")) and float(m.get("completeness", 0.0)) < 1.0,
            "illegal_citation": (not m.get("parse_failed")) and (m.get("n_illegal_citations", 0) + m.get("n_unresolved_citations", 0)) > 0,
            "schema": bool(m.get("parse_failed")) or bool(fg.get("schema_errors")), "gen_error": bool(r.get("generation_errors"))}


def load_panel(runs_root: Path, tags: list[str] | None, exclude: set[str] | None = None, keep_ids: set[str] | None = None) -> dict[str, dict[str, dict[str, dict]]]:
    panel = {}
    exclude = exclude or set()
    for tdir in sorted(runs_root.iterdir()):
        if not tdir.is_dir() or (tags and tdir.name not in tags):
            continue
        cfgs = {}
        for cdir in sorted(tdir.iterdir()):
            f = cdir / "predictions.jsonl"
            if cdir.is_dir() and f.exists() and cdir.name not in exclude and not cdir.name.startswith(("agreement", "sensitivity", "masking", "seed")):
                cfgs[cdir.name] = {r["task_id"]: compact(r) for r in read_jsonl(f) if keep_ids is None or r["task_id"] in keep_ids}
        if cfgs:
            panel[tdir.name] = cfgs
    return panel


def mean_ci(rows: list[dict], ep: str, n_boot: int, seed: int) -> dict:
    point, lo, hi = cluster_bootstrap_mean([r[ep] for r in rows], [r["patient"] for r in rows], n_boot=n_boot, seed=seed)
    return {"mean": round(100 * point, 2), "ci_low": round(100 * lo, 2), "ci_high": round(100 * hi, 2)}


def error_code(r: dict) -> str:
    """Plan §9.4 precedence: omission > out-of-scope citation > operation error > schema. A row is an error when the
    answer is wrong, a required record is uncited, or a citation is out of scope / unresolved; 'none' otherwise.
    Unparseable submissions carry no assessable citations, so they are coded 'schema'."""
    if r["answer_correct"] and not r["omission"] and not r["illegal_citation"] and not r["schema"]:
        return "none"
    if r["schema"] and not (r["omission"] or r["illegal_citation"]):
        return "schema"
    if r["omission"]:
        return "omission"
    if r["illegal_citation"]:
        return "out_of_scope_citation"
    return "operation_error"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max-glm-rows", type=int, default=60000)
    p.add_argument("--exclude-configs", nargs="*", default=[], help="run directories to leave out (e.g. the superseded *_tools_optional controls)")
    p.add_argument("--manifest", type=Path, default=None, help="task manifest directory (tasks.jsonl) — needed for --exclude-splits")
    p.add_argument("--exclude-splits", nargs="*", default=[], help="manifest splits to leave out (Tier-C protocol: primary numbers on train + test, the valid split was the development pool)")
    p.add_argument("--covered-templates", type=Path, default=None, help="Tier-C: skill_validation.json; a task is covered when its manifest template is marked covered (coverage and stages per task, not per task type)")
    args = p.parse_args()
    t0 = time.time()
    keep_ids = None
    if args.exclude_splits:
        if not args.manifest:
            raise SystemExit("--exclude-splits needs --manifest")
        keep_ids = {r["task_id"] for r in read_jsonl(args.manifest / "tasks.jsonl") if r.get("split") not in set(args.exclude_splits)}
    panel = load_panel(args.runs_root, args.tags, set(args.exclude_configs), keep_ids)
    covered_ids: set[str] | None = None
    if args.covered_templates:
        if not args.manifest:
            raise SystemExit("--covered-templates needs --manifest")
        cov_t = {t for t, v in json.loads(args.covered_templates.read_text())["templates"].items() if v.get("covered")}
        covered_ids = {r["task_id"] for r in read_jsonl(args.manifest / "tasks.jsonl") if (r.get("metadata") or {}).get("template") in cov_t}
    if not panel:
        raise SystemExit(f"no runs under {args.runs_root}")
    tags = sorted(panel)
    configs = sorted({c for t in panel.values() for c in t})
    args.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"runs_root": str(args.runs_root), "checkpoints": tags, "configurations": configs, "n_boot": args.n_boot, "seed": args.seed,
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "excluded_splits": sorted(args.exclude_splits), "n_tasks_kept": len(keep_ids) if keep_ids is not None else None}
    print(f"loaded {len(tags)} checkpoints × {len(configs)} configurations ({time.time() - t0:.0f}s)", flush=True)

    # 1. checkpoint × configuration means with cluster intervals; checkpoint-averaged table (Table 2 / S4 / S3 rows)
    ckpt = {t: {c: {ep: mean_ci(list(rows.values()), ep, args.n_boot, args.seed) for ep in ENDPOINTS} | {"n": len(rows), "gen_error_pct": round(100 * sum(r["gen_error"] for r in rows.values()) / len(rows), 2)}
                for c, rows in cfgs.items()} for t, cfgs in panel.items()}
    res["checkpoint_config"] = ckpt
    res["checkpoint_average"] = {c: {ep: round(sum(ckpt[t][c][ep]["mean"] for t in tags if c in ckpt[t]) / max(1, sum(1 for t in tags if c in ckpt[t])), 2) for ep in ENDPOINTS}
                                 | {"n_checkpoints": sum(1 for t in tags if c in ckpt[t])} for c in configs}

    # 2. prespecified contrasts: per checkpoint and pooled (rows = task × checkpoint; clusters = patient)
    hyp = {"per_checkpoint": {}, "pooled": {}}
    raw_p = {}
    for name, (ep, a, b) in HYPOTHESES.items():
        pooled_a, pooled_b, clusters = {}, {}, {}
        for t in tags:
            if a in panel[t] and b in panel[t]:
                ta = {f"{t}|{i}": r[ep] for i, r in panel[t][a].items() if i in panel[t][b]}
                tb = {f"{t}|{i}": panel[t][b][i][ep] for i in panel[t][a] if i in panel[t][b]}
                cl = {f"{t}|{i}": r["patient"] for i, r in panel[t][a].items() if i in panel[t][b]}
                r_ = paired_cluster_contrast(ta, tb, cl, n_boot=args.n_boot, seed=args.seed)
                hyp["per_checkpoint"].setdefault(t, {})[name] = {"delta_pct": round(100 * r_["delta"], 2), "ci_low_pct": round(100 * r_["ci_low"], 2), "ci_high_pct": round(100 * r_["ci_high"], 2), "p_bootstrap": r_["p_bootstrap"], "n": r_["n"]}
                pooled_a.update(ta); pooled_b.update(tb); clusters.update(cl)
        if pooled_a:
            r_ = paired_cluster_contrast(pooled_a, pooled_b, clusters, n_boot=args.n_boot, seed=args.seed)
            hyp["pooled"][name] = {"endpoint": ep, "from": a, "to": b, "delta_pct": round(100 * r_["delta"], 2), "ci_low_pct": round(100 * r_["ci_low"], 2), "ci_high_pct": round(100 * r_["ci_high"], 2), "p_bootstrap": r_["p_bootstrap"], "n": r_["n"], "n_clusters": r_["n_clusters"]}
            raw_p[name] = r_["p_bootstrap"]
    for name, adj in holm(raw_p).items():
        hyp["pooled"][name]["p_holm"] = adj
    for t in hyp["per_checkpoint"]:
        for name, adj in holm({n: h["p_bootstrap"] for n, h in hyp["per_checkpoint"][t].items()}).items():
            hyp["per_checkpoint"][t][name]["p_holm"] = adj
    res["hypotheses"] = hyp

    # 3. associations (§9.2): Fisher-z over checkpoint means (per configuration), per-checkpoint contingency + MI, multilevel model
    assoc: dict = {"checkpoint_level": {}, "task_level_contingency": {}, "multilevel": None}
    for c in configs:
        ts = [t for t in tags if c in ckpt[t]]
        if len(ts) >= 4:
            ans = [ckpt[t][c]["answer_correct"]["mean"] for t in ts]
            for ep in ("completeness", "citation_legal"):
                x = [ckpt[t][c][ep]["mean"] for t in ts]
                if len(set(x)) > 1 and len(set(ans)) > 1:
                    r_ = pearson(ans, x); lo, hi = fisher_ci(r_, len(ts))
                    assoc["checkpoint_level"][f"{c}|answer_correct~{ep}"] = {"r": round(r_, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3), "n_checkpoints": len(ts)}
    for t in tags:
        rows = [r for c in CORE if c in panel[t] for r in panel[t][c].values()]
        if not rows:
            continue
        med = sorted(r["completeness"] for r in rows)[len(rows) // 2]
        assoc["task_level_contingency"][t] = {
            "pooled_core_configurations": [c for c in CORE if c in panel[t]], "completeness_median": med,
            "correct_x_completeness_above_median": contingency([int(r["answer_correct"]) for r in rows], [int(r["completeness"] > med) for r in rows]),
            "correct_x_citation_legal": contingency([int(r["answer_correct"]) for r in rows], [int(r["citation_legal"]) for r in rows])}
    # conditional MI: within (checkpoint, configuration) strata, weighted by stratum size
    def cond_mi(fn) -> float:
        tot, n_all = 0.0, 0
        for t in tags:
            for c in CORE:
                rows = list(panel[t].get(c, {}).values())
                if len(rows) < 30:
                    continue
                med = sorted(r["completeness"] for r in rows)[len(rows) // 2]
                tot += len(rows) * mutual_information_bits([int(r["answer_correct"]) for r in rows], [int(fn(r, med)) for r in rows]); n_all += len(rows)
        return round(tot / n_all, 4) if n_all else float("nan")
    assoc["conditional_mi_bits"] = {"correct;completeness_above_stratum_median|checkpoint,configuration": cond_mi(lambda r, med: r["completeness"] > med),
                                    "correct;citation_legal|checkpoint,configuration": cond_mi(lambda r, med: r["citation_legal"])}
    assoc["multilevel"] = multilevel(panel, tags, args.max_glm_rows, args.seed)
    res["associations"] = assoc

    # 4. attribution accounting (Table 3): answer changed / trace-only, per checkpoint and averaged
    attrib = {}
    for c in ATTRIBUTION:
        per = {}
        for t in tags:
            rows = list(panel[t].get(c, {}).values())
            if rows:
                per[t] = {"answer_changed_pct": round(100 * sum(r["answer_changed"] for r in rows) / len(rows), 2),
                          "trace_only_pct": round(100 * sum(r["initial_invalid"] and not r["final_invalid"] and not r["answer_changed"] for r in rows) / len(rows), 2),
                          "repaired_pct": round(100 * sum(r["repaired"] for r in rows) / len(rows), 2)}
        if per:
            attrib[c] = {"per_checkpoint": per, "mean": {k: round(sum(v[k] for v in per.values()) / len(per), 2) for k in next(iter(per.values()))}}
    res["attribution"] = attrib

    # 5. coverage (Table 4) and covered-operation stages (Table 6) from `full`
    covered_types = {n for n, tt in REGISTRY.items() if tt.covered}
    any_tag = tags[0]; any_cfg = next(iter(panel[any_tag].values()))
    by_type = Counter(r["task_type"] for r in any_cfg.values())
    def is_covered(r: dict) -> bool:
        return (r["task_id"] in covered_ids) if covered_ids is not None else (r["task_type"] in covered_types)
    if covered_ids is not None:   # Tier-C: coverage is per template (validated mapping), measured on the panel's own tasks
        n_cov = sum(1 for r in any_cfg.values() if r["task_id"] in covered_ids)
        res["coverage"] = {"covered_task_types": [], "covered_by": "template (skill_validation.json)", "by_task_type": {tt: round(100 * sum(1 for r in any_cfg.values() if r["task_type"] == tt and r["task_id"] in covered_ids) / n, 2) for tt, n in sorted(by_type.items())},
                           "all_tasks_pct": round(100 * n_cov / len(any_cfg), 2), "n_tasks": len(any_cfg), "n_covered": n_cov}
    else:
        res["coverage"] = {"covered_task_types": sorted(covered_types), "by_task_type": {tt: 100.0 if tt in covered_types else 0.0 for tt in sorted(by_type)},
                           "all_tasks_pct": round(100 * sum(n for tt, n in by_type.items() if tt in covered_types) / sum(by_type.values()), 2), "n_tasks": sum(by_type.values())}
    stages = {}
    for t in tags:
        cov = [r for r in panel[t].get("full", {}).values() if is_covered(r)]
        if cov:
            n = len(cov)
            stages[t] = {"n_covered": n, "initial_valid_pct": round(100 * sum(not r["initial_invalid"] for r in cov) / n, 2),
                         "after_repair_valid_pct": round(100 * sum((not r["initial_invalid"]) or (r["repaired"] and not r["replaced"] and not r["final_invalid"]) for r in cov) / n, 2),
                         "replaced_pct": round(100 * sum(r["replaced"] for r in cov) / n, 2), "final_valid_pct": round(100 * sum(not r["final_invalid"] for r in cov) / n, 2),
                         "initial_correct_pct": round(100 * sum(r["initial_correct"] for r in cov) / n, 2), "final_correct_pct": round(100 * sum(r["answer_correct"] for r in cov) / n, 2),
                         "wrong_corrected_only_by_replacement_pct_of_all_tasks": round(100 * sum(r["replaced"] and not r["initial_correct"] and r["answer_correct"] for r in cov) / len(panel[t]["full"]), 2),
                         "already_correct_replaced_pct_of_covered": round(100 * sum(r["replaced"] and r["initial_correct"] for r in cov) / n, 2)}
    res["stages"] = {"per_checkpoint": stages, "mean": {k: round(sum(v[k] for v in stages.values()) / len(stages), 2) for k in next(iter(stages.values()))} if stages else {}}

    # 6. cost (Table 8): ratios to prompt_only per checkpoint, averaged; wall-clock from run metadata when present
    cost = {}
    for c in configs:
        per = {}
        for t in tags:
            if c in panel[t] and "prompt_only" in panel[t]:
                base, rows = list(panel[t]["prompt_only"].values()), list(panel[t][c].values())
                tok0 = sum(r["tokens"] for r in base) / len(base); tok1 = sum(r["tokens"] for r in rows) / len(rows)
                lat0 = sorted(r["latency"] for r in base)[len(base) // 2]; lat1 = sorted(r["latency"] for r in rows)[len(rows) // 2]
                per[t] = {"tokens_ratio": round(tok1 / tok0, 3) if tok0 else None, "calls": round(sum(r["calls"] for r in rows) / len(rows), 3),
                          "latency_ratio": round(lat1 / lat0, 3) if lat0 else None, "wall_hours_ratio": wall_ratio(args.runs_root / t, c)}
        if per:
            cost[c] = {"per_checkpoint": per, "mean": {k: (round(sum(v[k] for v in per.values() if v[k] is not None) / max(1, sum(1 for v in per.values() if v[k] is not None)), 3)
                                                       if any(v[k] is not None for v in per.values()) else None) for k in ("tokens_ratio", "calls", "latency_ratio", "wall_hours_ratio")}}
    res["cost"] = cost

    # 7. strata (S2) and the model × task type × scope matrix: full − prompt_only, pooled over checkpoints
    strata, matrix = {}, {}
    for key in ("task_type", "scope", "leakage"):
        for level in sorted({str(r[key]) for t in tags for r in panel[t].get("full", {}).values()}):
            a, b, cl = {}, {}, {}
            for t in tags:
                for i, r in panel[t].get("full", {}).items():
                    if str(r[key]) == level and i in panel[t].get("prompt_only", {}):
                        a[f"{t}|{i}"] = panel[t]["prompt_only"][i]; b[f"{t}|{i}"] = r; cl[f"{t}|{i}"] = r["patient"]
            if a:
                strata[f"{key}={level}"] = {ep: delta(a, b, cl, ep, args.n_boot, args.seed) for ep in ("answer_correct", "completeness", "access_legal", "citation_legal", "auditable")} | {"n": len(a)}
    for t in tags:
        for i, r in panel[t].get("full", {}).items():
            if i in panel[t].get("prompt_only", {}):
                matrix.setdefault((t, r["task_type"], r["scope"]), []).append((panel[t]["prompt_only"][i], r))
    cells = {}
    for (t, tt, sc), pairs in matrix.items():
        cells[f"{t}|{tt}|{sc}"] = {"n": len(pairs), "delta_completeness_pct": round(100 * (sum(b["completeness"] for _, b in pairs) - sum(a["completeness"] for a, _ in pairs)) / len(pairs), 2),
                                   "delta_answer_correct_pct": round(100 * (sum(b["answer_correct"] for _, b in pairs) - sum(a["answer_correct"] for a, _ in pairs)) / len(pairs), 2)}
    res["strata"] = strata
    res["matrix"] = {"cells": cells, "min_completeness_gain": min(cells.items(), key=lambda kv: kv[1]["delta_completeness_pct"]) if cells else None,
                     "max_completeness_gain": max(cells.items(), key=lambda kv: kv[1]["delta_completeness_pct"]) if cells else None,
                     "cells_with_nonpositive_completeness_gain": sorted(k for k, v in cells.items() if v["delta_completeness_pct"] <= 0)}

    # 8. Figure 2 / S1 inputs and the parameter-count ordering check
    models = {}
    for t in tags:
        if "prompt_only" in ckpt[t] and "full" in ckpt[t]:
            models[t] = {"prompt_only": {ep: ckpt[t]["prompt_only"][ep] for ep in ("answer_correct", "completeness")}, "full": {ep: ckpt[t]["full"][ep] for ep in ("answer_correct", "completeness")},
                         "delta_answer_correct": round(ckpt[t]["full"]["answer_correct"]["mean"] - ckpt[t]["prompt_only"]["answer_correct"]["mean"], 2),
                         "delta_completeness": round(ckpt[t]["full"]["completeness"]["mean"] - ckpt[t]["prompt_only"]["completeness"]["mean"], 2), "params_b": PARAMS_B.get(t)}
    res["models"] = models
    res["params_vs_completeness_spearman"] = spearman_params(models)

    # 9. access/citation divergence and error taxonomy (raw counts)
    res["access_citation_divergence"] = {c: {t: {"legal_citations_after_illegal_exposure_pct": round(100 * sum(r["citation_legal"] and r["n_illegal_access"] > 0 for r in panel[t][c].values()) / len(panel[t][c]), 2)}
                                             for t in tags if c in panel[t]} for c in configs}
    tax = {}
    for c in configs:
        per = {t: dict(Counter(error_code(r) for r in panel[t][c].values())) for t in tags if c in panel[t]}
        tot = Counter()
        for d in per.values():
            tot.update(d)
        tax[c] = {"per_checkpoint": per, "pooled_counts": dict(tot), "n": sum(tot.values())}
    res["error_taxonomy"] = {"precedence": "omission > out_of_scope_citation > operation_error > schema (correct and valid rows are 'none')", "by_configuration": tax}

    (args.out / "panel.json").write_text(json.dumps(res, indent=1, default=str))
    with open(args.out / "checkpoint_config.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["checkpoint", "configuration", "n"] + [f"{ep}_{s}" for ep in ENDPOINTS for s in ("mean", "ci_low", "ci_high")] + ["gen_error_pct"])
        for t in tags:
            for c, v in ckpt[t].items():
                w.writerow([t, c, v["n"]] + [v[ep][s] for ep in ENDPOINTS for s in ("mean", "ci_low", "ci_high")] + [v["gen_error_pct"]])
    with open(args.out / "matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["checkpoint", "task_type", "scope", "n", "delta_completeness_pct", "delta_answer_correct_pct"])
        for k, v in sorted(cells.items()):
            w.writerow(k.split("|") + [v["n"], v["delta_completeness_pct"], v["delta_answer_correct_pct"]])
    for c in CORE:
        if c in res["checkpoint_average"]:
            a = res["checkpoint_average"][c]
            print(f"{c:26s} ckpts={a['n_checkpoints']}  Ans={a['answer_correct']:>6}  Ev.C={a['completeness']:>6}  A-L={a['access_legal']:>6}  C-L={a['citation_legal']:>6}  Aud={a['auditable']:>6}  abst={a['abstained']:>5}")
    for name, h in hyp["pooled"].items():
        print(f"{name:32s} Δ={h['delta_pct']:>6} [{h['ci_low_pct']}, {h['ci_high_pct']}] p_holm={h.get('p_holm')}  n={h['n']}")
    print(f"written {args.out} ({time.time() - t0:.0f}s)")


def delta(a: dict, b: dict, cl: dict, ep: str, n_boot: int, seed: int) -> dict:
    r_ = paired_cluster_contrast({k: v[ep] for k, v in a.items()}, {k: v[ep] for k, v in b.items()}, cl, n_boot=n_boot, seed=seed)
    return {"delta_pct": round(100 * r_["delta"], 2), "ci_low_pct": round(100 * r_["ci_low"], 2), "ci_high_pct": round(100 * r_["ci_high"], 2)}


def wall_ratio(tag_dir: Path, c: str) -> float | None:
    def hours(cfg: str) -> float | None:
        m = tag_dir / cfg / "metadata.json"
        if not m.exists():
            return None
        d = json.loads(m.read_text())
        try:
            s, e = d["started_at"], d["finished_at"]
            fmt = "%Y-%m-%dT%H:%M:%S%z"
            from datetime import datetime
            return (datetime.strptime(e, fmt) - datetime.strptime(s, fmt)).total_seconds() / 3600 / max(1, d.get("n_tasks_this_run", 1))
        except (KeyError, ValueError):
            return None
    h0, h1 = hours("prompt_only"), hours(c)
    return round(h1 / h0, 3) if h0 and h1 else None


def spearman_params(models: dict) -> dict | None:
    pts = [(v["params_b"], v["full"]["completeness"]["mean"]) for v in models.values() if v.get("params_b") is not None]
    if len(pts) < 4:
        return None
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr([p for p, _ in pts], [c for _, c in pts])
        return {"n": len(pts), "rho": round(float(rho), 3), "p": round(float(pval), 4), "endpoint": "completeness under full", "checkpoints": sorted(models)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def multilevel(panel: dict, tags: list[str], max_rows: int, seed: int) -> dict:
    """Task-level logistic model of correctness on completeness (per 10 points) and citation compliance with variance
    components for patient, task type, scope, checkpoint and model family (BinomialBayesMixedGLM); GEE fallback."""
    rows = []
    for t in tags:
        for c in CORE:
            for r in panel[t].get(c, {}).values():
                rows.append({"y": int(r["answer_correct"]), "completeness10": r["completeness"] * 10, "citation_legal": int(r["citation_legal"]), "configuration": c,
                             "patient": r["patient"], "task_type": r["task_type"], "scope": r["scope"], "checkpoint": t, "family": FAMILY.get(t, t)})
    if len(rows) < 200:
        return {"skipped": f"only {len(rows)} rows"}
    rng = random.Random(seed)
    sub = rows if len(rows) <= max_rows else rng.sample(rows, max_rows)
    out: dict = {"n_rows_total": len(rows), "n_rows_fit": len(sub), "pooled_configurations": CORE}
    try:
        import numpy as np
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        df = pd.DataFrame(sub)
        vc = {"patient": "0 + C(patient)", "task_type": "0 + C(task_type)", "scope": "0 + C(scope)", "checkpoint": "0 + C(checkpoint)", "family": "0 + C(family)"}
        vc = {k: v for k, v in vc.items() if df[k].nunique() > 1}
        model = BinomialBayesMixedGLM.from_formula("y ~ completeness10 + citation_legal + C(configuration)", vc, df)
        fit = model.fit_vb()
        names = list(fit.model.exog_names)
        est = {}
        for i, name in enumerate(names):
            b, se = float(fit.fe_mean[i]), float(fit.fe_sd[i])
            est[name] = {"log_odds": round(b, 4), "sd": round(se, 4), "odds_ratio": round(math.exp(b), 3), "ci_low": round(math.exp(b - 1.96 * se), 3), "ci_high": round(math.exp(b + 1.96 * se), 3)}
        out.update({"method": "BinomialBayesMixedGLM (variational Bayes)", "variance_components": list(vc), "fixed_effects": est,
                    "vc_sd": {k: round(float(np.exp(v)), 4) for k, v in zip(fit.model.vc_names, fit.vcp_mean)}})
        return out
    except Exception as e:  # noqa: BLE001
        out["mixed_glm_error"] = str(e)[:300]
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        df = pd.DataFrame(sub)
        fit = smf.gee("y ~ completeness10 + citation_legal + C(configuration) + C(task_type) + C(scope) + C(checkpoint)", groups="patient", data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        est = {name: {"log_odds": round(float(fit.params[name]), 4), "odds_ratio": round(math.exp(fit.params[name]), 3),
                      "ci_low": round(math.exp(fit.conf_int().loc[name, 0]), 3), "ci_high": round(math.exp(fit.conf_int().loc[name, 1]), 3)} for name in fit.params.index}
        out.update({"method": "GEE (exchangeable, patient clusters; fixed effects for task type, scope, checkpoint, configuration)", "fixed_effects": est})
    except Exception as e:  # noqa: BLE001
        out["gee_error"] = str(e)[:300]
    return out


if __name__ == "__main__":
    main()
