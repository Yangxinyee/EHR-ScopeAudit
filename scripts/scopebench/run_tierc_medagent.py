#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# MedAgentBench-related adaptation; Copyright (c) 2025 Stanford Machine Learning Group.
# Upstream license: licenses/MedAgentBench-MIT.txt; changes: THIRD_PARTY_NOTICES.md
"""Run one ScopeBench configuration on MedAgentBench (Tier-C part 2) against the official FHIR server through the logged
HTTP layer, with a serving backend. Resumable; `--retry-errors` re-runs rows that hit an infrastructure error.

Usage: run_tierc_medagent.py --config scope_aware --tag qwen3_4b --backend sglang --base-url http://127.0.0.1:30131 \
         --out-dir experiments/scopebench_tierC_runs/qwen3_4b/medagentbench/scope_aware [--max-tasks N] [--token-budget T]
       run_tierc_medagent.py --derive-late-filter <prompt_only run dir> --out-dir <late_filter run dir>   (no model)
       run_tierc_medagent.py --write-freeze | --verify-freeze   (the MedAgentBench addendum freeze record)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.medagent import CONFIGS, FHIR_BASE, MedAgentSpec, get_json_text, late_filter_row, run_medagent  # noqa: E402

MANIFEST = ROOT / "experiments/scopebench_tierC/medagentbench"
TASK_DATA = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/test_data_v2.json"
FUNCTIONS = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/funcs_v1.json"
FREEZE_RECORD = ROOT / "experiments/SCOPEBENCH_TIERC_MEDAGENT_FREEZE.json"
PANEL_DRIVER = ROOT / "scripts/scopebench/run_tierc_medagent_panel.sh"
CONTAINER = "medagentbench-official-2"
# The formal panel (Tier-C protocol, MedAgentBench addendum): prompt_only is the benchmark's native direct-API mode, so
# `direct_api` (identical configuration) is not run twice; late_filter is derived from prompt_only.
PANEL_CONFIGS = ["prompt_only", "wrong_basis", "scope_aware", "audit", "full", "security_label_consent", "token_matched", "high_compute"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def freeze_record() -> dict:
    """The MedAgentBench addendum freeze: every input of the MedAgentBench runs that the Tier-C record does not hash
    (its manifest, the benchmark's task data, functions, official prompt, grader and compatibility shim, the panel
    driver, the server image), plus the Tier-C record it relies on for the framework code, scripts and protocol."""
    from run_medagentbench_corrected import OFFICIAL_TASK_MODULE, REFSOL_PATH, UTILS_PATH, extract_official_prompt
    tierc = json.loads((ROOT / "experiments/SCOPEBENCH_TIERC_FREEZE.json").read_text())
    try:
        image = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", CONTAINER], capture_output=True, text=True, timeout=60).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        image = None
    shim = ROOT / "scripts/run_medagentbench_corrected.py"
    return {
        "record": "ScopeBench Tier-C MedAgentBench addendum freeze",
        "freeze_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_manifest": {"path": _rel(MANIFEST / "tasks.jsonl"), "sha256": _sha(MANIFEST / "tasks.jsonl"), "n_tasks": sum(1 for _ in open(MANIFEST / "tasks.jsonl")),
                          "construction_stats_sha256": _sha(MANIFEST / "construction_stats.json")},
        "task_data": {"path": _rel(TASK_DATA), "sha256": _sha(TASK_DATA)},
        "functions": {"path": _rel(FUNCTIONS), "sha256": _sha(FUNCTIONS)},
        "official_prompt_sha256": hashlib.sha256(extract_official_prompt(OFFICIAL_TASK_MODULE).encode()).hexdigest(),
        "grader": {"refsol_sha256": _sha(REFSOL_PATH), "utils_sha256": _sha(UTILS_PATH)},
        "compat_shim": {"path": _rel(shim), "sha256": _sha(shim)},
        "panel_driver": {"path": _rel(PANEL_DRIVER), "sha256": _sha(PANEL_DRIVER)},
        "fhir_server": {"container": CONTAINER, "image_id": image},
        "tierc_record": {"freeze_time": tierc["freeze_time"], "protocol_sha256": tierc["protocol"]["sha256"],
                         "framework_tree_sha256": tierc["framework_code"]["tree_sha256"], "scripts_tree_sha256": tierc["scripts"]["tree_sha256"]},
        "configurations": PANEL_CONFIGS + ["late_filter (derived)"],
        "serving": {"context_length": {"default": 32768, "phi4_14b": 16384}, "max_rounds": 8, "max_new_tokens": 1024, "temperature": 0.0, "seed": 2027},   # DEV-25
    }


def derive_late_filter(prompt_only: Path, out_dir: Path, fhir_base: str) -> None:
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")}
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    n = removed = capped = 0
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in read_jsonl(prompt_only / "predictions.jsonl"):
            out = late_filter_row(row, tasks[row["task_id"]], fhir_base, cache)
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1; removed += out["late_filter_removed"]; capped += bool(out["late_filter_access_list_capped"])
    src_meta = json.loads((prompt_only / "metadata.json").read_text()) if (prompt_only / "metadata.json").exists() else {}
    (out_dir / "metadata.json").write_text(json.dumps({"configuration": {"name": "late_filter", "derived_from": "prompt_only"}, "source_run": str(prompt_only),
                                                       "source_metadata": src_meta, "n_tasks_this_run": n, "citations_removed": removed,
                                                       "rows_with_capped_access_list": capped, "fhir_base": fhir_base,
                                                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1, sort_keys=True) + "\n")
    print(f"late_filter: {n} tasks derived into {out_dir} ({removed} citations removed; {capped} rows with a capped access list)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=sorted(CONFIGS))
    p.add_argument("--tag")
    p.add_argument("--derive-late-filter", type=Path, default=None, metavar="PROMPT_ONLY_DIR")
    p.add_argument("--write-freeze", action="store_true")
    p.add_argument("--verify-freeze", action="store_true")
    p.add_argument("--backend", default="sglang", choices=["sglang", "claude"])
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--fhir-base", default=FHIR_BASE)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--repair-wording", default="a", choices=["a", "b"])
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--label", default="")
    p.add_argument("--cell-kind", default="formal", choices=["formal", "sensitivity", "diagnostic", "pilot", "agreement"])
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--claude-batch-size", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    if args.write_freeze or args.verify_freeze:
        rec = freeze_record()
        if args.write_freeze:
            FREEZE_RECORD.write_text(json.dumps(rec, indent=1) + "\n")
            print(f"freeze written: {FREEZE_RECORD} ({rec['task_manifest']['n_tasks']} tasks)")
            return
        frozen = json.loads(FREEZE_RECORD.read_text())
        drift = sorted(k for k in rec if k != "freeze_time" and rec[k] != frozen.get(k))
        print(json.dumps({"drift": drift}, indent=1))
        raise SystemExit(1 if drift else 0)
    if args.out_dir is None:
        p.error("--out-dir is required")
    if args.derive_late_filter is not None:
        if "error" in get_json_text(f"{args.fhir_base}metadata"):
            raise SystemExit(f"FHIR server not reachable at {args.fhir_base}")
        derive_late_filter(args.derive_late_filter, args.out_dir, args.fhir_base)
        return
    if not args.config or not args.tag:
        p.error("--config and --tag are required for a run")
    if CONFIGS[args.config].get("budget") and not args.token_budget:
        raise SystemExit("--token-budget is required for token_matched / high_compute")
    if "error" in get_json_text(f"{args.fhir_base}metadata"):
        raise SystemExit(f"FHIR server not reachable at {args.fhir_base}")
    from run_medagentbench_corrected import OFFICIAL_TASK_MODULE, extract_official_prompt, load_official_grader
    official_prompt = extract_official_prompt(OFFICIAL_TASK_MODULE)
    functions = json.load(open(FUNCTIONS))
    refsol, _utils = load_official_grader()
    refsol.send_get_request = get_json_text
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")}
    cases = [c for c in json.load(open(TASK_DATA)) if c["id"] in tasks]
    if args.categories:
        cases = [c for c in cases if c["id"].split("_", 1)[0] in set(args.categories)]
    cases.sort(key=lambda c: (int(c["id"].split("_", 1)[0][4:]), int(c["id"].split("_", 1)[1])))
    if args.max_tasks:
        cases = cases[: args.max_tasks]
    panel = {e["tag"]: e for e in json.load(open(ROOT / "experiments/SCOPEBENCH_MODEL_PANEL.json"))["this_phase"]}
    if args.backend == "sglang":
        from transformers import AutoTokenizer

        from scopebench.backends.sglang_http import SGLangBackend
        entry = panel[args.tag]
        from scopebench.config import remap_model_path
        tokenizer = AutoTokenizer.from_pretrained(remap_model_path(entry.get("serving_path") or entry["path"]))
        backend = SGLangBackend(base_url=args.base_url, concurrency=args.concurrency, seed=args.seed, temperature=args.temperature, enable_thinking=False)
    else:
        from scopebench.backends.claude_batch import ClaudeBatchBackend
        backend = ClaudeBatchBackend(batch_dir=args.out_dir / "batches", batch_size=args.claude_batch_size)
        tokenizer = None
    spec = MedAgentSpec(configuration=args.config, max_rounds=args.max_rounds, max_new_tokens=args.max_new_tokens, token_budget=args.token_budget, label=args.label,
                        repair_wording=args.repair_wording)
    extra = {"tag": args.tag, "cell_kind": args.cell_kind, "temperature": args.temperature, "sampling_seed": args.seed, "task_manifest": str(MANIFEST.relative_to(ROOT)),
             "official_prompt_sha256": __import__("hashlib").sha256(official_prompt.encode()).hexdigest(), "n_cases_selected": len(cases)}
    meta = run_medagent(spec, cases, tasks, official_prompt, functions, backend, tokenizer, refsol, args.out_dir, fhir_base=args.fhir_base, extra_metadata=extra,
                        retry_errors=args.retry_errors)
    print(f"{args.config}: {meta['n_tasks_this_run']} tasks written to {args.out_dir} ({meta['n_tasks_previously_done']} already done; mean rounds {meta['mean_rounds']})")


if __name__ == "__main__":
    main()
