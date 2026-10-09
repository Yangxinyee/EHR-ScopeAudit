#!/usr/bin/env python3
"""Completeness checker for gates G3/G5/G6 (plan §12): every run directory of a checkpoint has the rows, logs and
metadata the protocol requires, and every configuration the panel owes for that checkpoint exists.

Usage: check_runs_complete.py --runs experiments/scopebench_tierB_runs/<tag> [--manifest experiments/scopebench_tierB]
                              [--tier B] [--expect-n 9600] [--json out.json] [--sensitivity]
Exit status 1 on any FAIL. Never modifies anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINTS = ("answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "abstained", "parse_failed")
CORE = ["prompt_only", "late_filter", "wrong_basis", "scope_aware", "audit", "full", "token_matched", "high_compute"]
ARMS = ["retry_generic", "feedback_schema_only", "skill_payload_only", "skill_verify_no_replace", "skill_replace_no_payload"]
BASELINES_B = ["citation_validator", "reranker", "temporal_sql_view", "abac_pdp"]
D5_ARM_TAGS = {"qwen3_4b", "gemma3_27b", "qwen3_30b"}          # DEV-2
NONSEED_CELLS = ["k5", "k20", "bm25", "dense", "no_expansion", "strict_boundaries", "at_window_6h", "extended_rt", "repair_wording_b"]
SEED_CELLS = ["seed13", "seed29", "seed47", "seed71", "seed101"]
MASK_CELLS = ["mask_cited_admissible", "mask_uncited_accessed"]
DERIVED = {"late_filter"}                                       # post-processed from prompt_only: no access_log/stages of its own


def expected_configs(tag: str, tier: str) -> list[str]:
    if tier == "B":
        return CORE + (ARMS if tag in D5_ARM_TAGS else []) + BASELINES_B
    if tier == "A":
        return CORE
    return CORE + ["direct_api", "specialized_tools", "security_label_consent"]


def manifest_ids(manifest: Path | None) -> set[str] | None:
    if manifest is None or not (manifest / "tasks.jsonl").exists():
        return None
    return {json.loads(l)["task_id"] for l in open(manifest / "tasks.jsonl")}


def check_dir(d: Path, expect_n: int | None, ids: set[str] | None, derived: bool) -> dict:
    r: dict = {"dir": str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d), "fail": [], "warn": []}
    pred = d / "predictions.jsonl"
    if not pred.exists():
        r["fail"].append("predictions.jsonl missing"); return r
    rows = [json.loads(l) for l in open(pred)]
    r["n"] = len(rows)
    tids = [x.get("task_id") for x in rows]
    if len(set(tids)) != len(tids):
        r["fail"].append(f"duplicate task ids: {len(tids) - len(set(tids))}")
    if expect_n is not None and len(rows) != expect_n:
        r["fail"].append(f"rows {len(rows)} != expected {expect_n}")
    if ids is not None and expect_n is not None and len(rows) == expect_n and set(tids) != ids:
        r["fail"].append(f"task ids differ from the manifest ({len(set(tids) - ids)} extra, {len(ids - set(tids))} missing)")
    missing_metrics = sum(1 for x in rows if not x.get("metrics") or any(k not in x["metrics"] for k in ENDPOINTS))
    if missing_metrics:
        r["fail"].append(f"{missing_metrics} rows without the full endpoint set")
    r["generation_errors"] = sum(1 for x in rows if x.get("generation_errors"))
    if r["generation_errors"]:
        r["fail"].append(f"{r['generation_errors']} rows carry generation_errors")
    r["parse_failed"] = sum(1 for x in rows if x["metrics"].get("parse_failed"))
    meta = d / "metadata.json"
    if not meta.exists():
        r["fail"].append("metadata.json missing")
    else:
        m = json.load(open(meta)); r["cell_kind"] = m.get("cell_kind")
        cfg_name = m.get("configuration"); cfg_name = cfg_name.get("name") if isinstance(cfg_name, dict) else cfg_name
        r["configuration"] = cfg_name; r["git_commit"] = m.get("git_commit")
        if cfg_name and not derived and cfg_name != d.name and not d.name.startswith("run") and d.parent.name not in ("sensitivity", "masking") \
                and not any(d.parent.parent.name == x for x in ("sensitivity", "masking")):
            r["warn"].append(f"metadata.configuration={cfg_name} but directory is {d.name}")
        if m.get("n_tasks_total") not in (None, len(rows)):
            r["warn"].append(f"metadata n_tasks_total={m.get('n_tasks_total')} vs rows {len(rows)}")
    if not derived:
        for f in ("access_log.jsonl", "stages.jsonl"):
            p = d / f
            if not p.exists():
                r["fail"].append(f"{f} missing")
            elif p.stat().st_size == 0:
                r["fail"].append(f"{f} empty")
        if (d / "access_log.jsonl").exists():
            n_acc = sum(1 for _ in open(d / "access_log.jsonl"))
            r["access_log_rows"] = n_acc
            if n_acc < len(rows):
                r["fail"].append(f"access_log rows {n_acc} < predictions {len(rows)}")
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--tier", default="B", choices=["A", "B", "C"])
    p.add_argument("--expect-n", type=int, default=None)
    p.add_argument("--configs", nargs="*", default=None, help="override the owed configuration list")
    p.add_argument("--sensitivity", action="store_true", help="also check sensitivity/ and masking/ cells")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    runs = a.runs if a.runs.is_absolute() else ROOT / a.runs
    tag = runs.name
    ids = manifest_ids(a.manifest if a.manifest is None or a.manifest.is_absolute() else ROOT / a.manifest)
    expect_n = a.expect_n if a.expect_n is not None else (len(ids) if ids else None)
    report: dict = {"tag": tag, "tier": a.tier, "expect_n": expect_n, "configs": {}, "checkpoint": {"fail": [], "warn": []}}
    owed = a.configs if a.configs is not None else expected_configs(tag, a.tier)
    for cfg in owed:
        d = runs / cfg
        if not d.exists():
            report["configs"][cfg] = {"dir": str(d), "fail": ["directory missing (configuration not run)"], "warn": []}
            continue
        report["configs"][cfg] = check_dir(d, expect_n, ids, cfg in DERIVED)
    extra = sorted(x.name for x in runs.iterdir() if x.is_dir() and x.name not in owed
                   and x.name not in ("agreement", "sensitivity", "masking") and (x / "predictions.jsonl").exists())
    if extra:
        report["checkpoint"]["warn"].append(f"directories present but not owed: {extra}")
    ck = report["checkpoint"]
    ag = runs / "agreement" / "agreement.json"
    if a.tier != "A" or ag.exists():
        if not ag.exists():
            ck["fail"].append("agreement/agreement.json missing (D10 three-run check)")
        else:
            g = json.load(open(ag)); report["agreement"] = g
            for r_ in (1, 2, 3):
                rd = runs / "agreement" / f"run{r_}"
                if not (rd / "predictions.jsonl").exists():
                    ck["fail"].append(f"agreement/run{r_} missing")
    for f in ("freeze_verify.json", "token_budgets.json", "panel.log"):
        if not (runs / f).exists():
            (ck["fail"] if f != "panel.log" else ck["warn"]).append(f"{f} missing")
    fv = runs / "freeze_verify.json"
    if fv.exists():
        try:
            v = json.load(open(fv))
            if v.get("changed_code") or v.get("changed_scripts") or v.get("drift"):
                ck["fail"].append(f"freeze_verify reports drift: {v}")
        except Exception as e:  # noqa: BLE001
            ck["warn"].append(f"freeze_verify.json unreadable: {e}")
    if a.sensitivity:
        sens_owed = (NONSEED_CELLS if tag in D5_ARM_TAGS else []) + SEED_CELLS
        for cell in sens_owed:
            for cfg in ("prompt_only", "full"):
                d = runs / "sensitivity" / cell / cfg
                key = f"sensitivity/{cell}/{cfg}"
                report["configs"][key] = check_dir(d, None, None, False) if d.exists() else {"dir": str(d), "fail": ["missing"], "warn": []}
        for cell in MASK_CELLS:
            for cfg in ("prompt_only", "full"):
                d = runs / "masking" / cell / cfg
                key = f"masking/{cell}/{cfg}"
                report["configs"][key] = check_dir(d, None, None, True) if d.exists() else {"dir": str(d), "fail": ["missing"], "warn": []}
    n_fail = sum(len(v["fail"]) for v in report["configs"].values()) + len(ck["fail"])
    n_warn = sum(len(v["warn"]) for v in report["configs"].values()) + len(ck["warn"])
    report["summary"] = {"n_fail": n_fail, "n_warn": n_warn, "owed": len(owed), "present": sum(1 for c in owed if (runs / c / "predictions.jsonl").exists())}
    for k, v in report["configs"].items():
        status = "FAIL" if v["fail"] else ("warn" if v["warn"] else "ok")
        n = v.get("n", "-"); ge = v.get("generation_errors", "-"); pf = v.get("parse_failed", "-")
        print(f"{status:4s} {k:36s} rows={n!s:>5} errors={ge!s:>3} parse_failed={pf!s:>4} {'; '.join(v['fail'] + v['warn'])}")
    for msg in ck["fail"]:
        print(f"FAIL checkpoint: {msg}")
    for msg in ck["warn"]:
        print(f"warn checkpoint: {msg}")
    if "agreement" in report:
        g = report["agreement"]; keys = [k for k in g if k.startswith("min_")]
        print("agreement:", {k: g[k] for k in keys} if keys else "(see agreement.json)")
    print(f"== {tag}: {report['summary']['present']}/{report['summary']['owed']} owed configurations present, {n_fail} FAIL, {n_warn} warn")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
