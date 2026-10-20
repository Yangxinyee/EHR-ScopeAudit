#!/usr/bin/env python3
"""Run one Tier-C configuration on the FHIR-AgentBench manifest (plan §6.1, Tier-C protocol) through the logged
FHIR store and a serving backend. Resumable; `--retry-errors` re-runs rows that hit an infrastructure error.

Usage: run_tierc.py --manifest experiments/scopebench_tierC/fhir_agentbench --config scope_aware --tag qwen3_4b \
         --backend sglang --base-url http://127.0.0.1:30131 --out-dir experiments/scopebench_tierC_runs/qwen3_4b/scope_aware
       [--split valid] [--max-tasks N] [--token-budget T --max-rounds 8] [--adapter native|text] [--concurrency 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.agent import CONFIGS, TierCSpec, run_tierc  # noqa: E402
from scopebench.tierc.store import load_store  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
PATIENT_MAP = PUBLIC_DATA / ".clinehr_cache" / "scopebench_tierc_patients" / "fhir_agentbench_patient_map.json"
SKILL_VALIDATION = ROOT / "experiments/scopebench_tierC/skill_validation_protocol_dev31.json"   # DEV-31: the covered-template record `full` maps (valid-split rule; DEV-26 used skill_validation.json)
NATIVE_TOOL_TEMPLATES = {"qwen3_4b", "qwen3_32b", "qwen3_30b"}   # templates that render tools= and the tool role (checked 2026-09-27)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--tag", required=True, help="panel tag (adapter + tokenizer path from SCOPEBENCH_MODEL_PANEL.json)")
    p.add_argument("--backend", default="sglang", choices=["sglang", "claude"])
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--split", default=None)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--task-ids", type=Path, default=None, help="JSON list of task ids to run (DEV-32 rerun of the affected tasks); recorded by hash in the run metadata")
    p.add_argument("--skill-validation", type=Path, default=SKILL_VALIDATION, help="covered-template record mapped onto `full`")
    p.add_argument("--adapter", default=None, choices=[None, "native", "text"])
    p.add_argument("--mode", default="broad", choices=["broad", "focused"])
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--repair-wording", default="a", choices=["a", "b"])
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--label", default="")
    p.add_argument("--cell-kind", default="formal", choices=["formal", "sensitivity", "diagnostic", "pilot", "agreement"])
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--claude-batch-size", type=int, default=8, help="prompts per frontier batch file (multi-turn prompts are long)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if CONFIGS[args.config].get("budget") and not args.token_budget:
        raise SystemExit("--token-budget is required for token_matched / high_compute")

    panel = {e["tag"]: e for e in json.load(open(ROOT / "experiments/SCOPEBENCH_MODEL_PANEL.json"))["this_phase"]}
    tasks = [Task.from_dict(r) for r in read_jsonl(args.manifest / "tasks.jsonl")]
    if args.split:
        tasks = [t for t in tasks if t.split == args.split]
    task_ids_sha256 = None
    if args.task_ids:
        import hashlib
        keep = set(json.loads(args.task_ids.read_text()))
        task_ids_sha256 = hashlib.sha256(args.task_ids.read_bytes()).hexdigest()
        missing = keep - {t.task_id for t in tasks}
        if missing:
            raise SystemExit(f"--task-ids: {len(missing)} ids not in the manifest")
        tasks = [t for t in tasks if t.task_id in keep]
    tasks = sorted(tasks, key=lambda t: (t.patient, t.task_id))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    patient_map = json.load(open(PATIENT_MAP))
    t0 = time.time()
    store = load_store(FHIR_DIR)
    print(f"store loaded ({sum(store.counts.values()):,} resources) in {time.time() - t0:.0f}s; {len(tasks)} tasks", flush=True)

    tokenizer = None
    adapter = args.adapter
    if args.backend == "sglang":
        from transformers import AutoTokenizer

        from scopebench.backends.sglang_http import SGLangBackend
        entry = panel[args.tag]
        from scopebench.config import remap_model_path
        tokenizer = AutoTokenizer.from_pretrained(remap_model_path(entry.get("serving_path") or entry["path"]))
        adapter = adapter or ("native" if args.tag in NATIVE_TOOL_TEMPLATES else "text")
        backend = SGLangBackend(base_url=args.base_url, concurrency=args.concurrency, seed=args.seed, temperature=args.temperature, enable_thinking=False)
    else:
        from scopebench.backends.claude_batch import ClaudeBatchBackend
        adapter = adapter or "text"
        backend = ClaudeBatchBackend(batch_dir=args.out_dir / "batches", batch_size=args.claude_batch_size)
    spec = TierCSpec(configuration=args.config, adapter=adapter, mode=args.mode, max_rounds=args.max_rounds, max_new_tokens=args.max_new_tokens,
                     token_budget=args.token_budget, label=args.label, repair_wording=args.repair_wording)
    skill_validation_sha256 = None
    if CONFIGS[args.config]["skill"]:
        # DEV-26: the template skills validated in skill_validation.json are mapped onto the loop (coverage per template)
        import hashlib

        from scopebench.tierc.skills import mapped_skills
        spec.skills = mapped_skills(args.skill_validation, patient_map)
        skill_validation_sha256 = hashlib.sha256(args.skill_validation.read_bytes()).hexdigest()
        print(f"skills mapped for {len(spec.skills)} covered templates ({args.skill_validation.name} {skill_validation_sha256[:12]})", flush=True)
    if CONFIGS[args.config].get("plan"):
        # DEV-34: the model's declared evidence plan drives the audit; the catalog is the only skill input
        from scopebench.tierc.plans import CATALOG_PATH, Executor
        spec.plan_executor = Executor(patient_map)
        print(f"evidence-plan catalog: {len(spec.plan_executor.catalog)} templates ({CATALOG_PATH.name})", flush=True)
    extra = {"tag": args.tag, "split": args.split, "cell_kind": args.cell_kind, "temperature": args.temperature, "sampling_seed": args.seed,
             "fhir_dir": FHIR_DIR.name, "tokenizer_path": str(panel[args.tag].get("serving_path") or panel[args.tag]["path"]) if args.tag in panel else None,
             "skill_validation_sha256": skill_validation_sha256, "skill_validation": args.skill_validation.name if skill_validation_sha256 else None,
             "task_ids_file": str(args.task_ids) if args.task_ids else None, "task_ids_sha256": task_ids_sha256}
    meta = run_tierc(spec, tasks, store, patient_map, backend, tokenizer, args.out_dir, manifest_path=args.manifest, extra_metadata=extra, retry_errors=args.retry_errors)
    print(f"{args.config}: {meta['n_tasks_this_run']} tasks written to {args.out_dir} ({meta['n_tasks_previously_done']} already done; mean rounds {meta['mean_rounds']})")


if __name__ == "__main__":
    main()
