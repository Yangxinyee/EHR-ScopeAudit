#!/usr/bin/env python3
"""Run one configuration of ScopeBench over a manifest with a backend (plan §8.2).

Examples
  # ScopePair LabTrend regression (frozen semantics), SGLang server already running on :30000
  .venv-analysis/bin/python scripts/scopebench/run_config.py --loader scopepair_labtrend \
      --manifest experiments/scopepair_fhir/tasks.jsonl --split test --config sp_clinehr_os \
      --backend sglang --base-url http://127.0.0.1:30000 --k 3 --pool-k 3 --skills reference \
      --out-dir experiments/scopebench_m1/regression_qwen3_4b/sp_clinehr_os
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.configurations import CONFIGS  # noqa: E402
from scopebench.runner import RunSpec, run  # noqa: E402
from scopebench.tasktypes import REFERENCE_SKILLS  # noqa: E402


def load(loader: str, manifest: Path, split: str | None):
    if loader == "scopepair_labtrend":
        from scopebench.compat_scopepair import labtrend_tasks

        return labtrend_tasks(manifest, split)
    if loader == "scopebench":
        from scopebench.manifest import load_manifest

        return load_manifest(manifest, split)
    raise SystemExit(f"unknown loader {loader}")


def make_retriever(args):
    from scopebench.retrieval import DenseEncoder, HybridRetriever, TemporalNearestRetriever

    if args.retriever == "temporal":
        return TemporalNearestRetriever()
    encoder = None
    if args.retriever in ("hybrid", "dense"):
        encoder = DenseEncoder(args.encoder_path, device=args.encoder_device)
    return HybridRetriever(encoder=encoder, n_candidates=args.n_candidates, mode=args.retriever, expand_sections=args.expand_sections)


def make_backend(args):
    if args.backend == "sglang":
        from scopebench.backends.sglang_http import SGLangBackend

        return SGLangBackend(base_url=args.base_url, concurrency=args.concurrency, seed=args.sampling_seed if args.sampling_seed is not None else args.seed,
                             temperature=args.temperature, enable_thinking=False)
    if args.backend == "hf":
        from scopebench.backends.hf_local import HFLocalBackend

        return HFLocalBackend(args.model_path, seed=args.seed)
    if args.backend == "oracle":
        from scopebench.backends.oracle_backend import OracleBackend

        return OracleBackend(args._tasks, args._cases)
    if args.backend == "claude":
        from scopebench.backends.claude_batch import ClaudeBatchBackend

        return ClaudeBatchBackend(batch_dir=Path(args.out_dir) / "batches")
    raise SystemExit(args.backend)


def make_skills(name: str):
    if name == "reference":
        return dict(REFERENCE_SKILLS)
    if name == "package":
        from scopebench import skills as pkg

        return dict(pkg.OPERATIONS)  # fn(evidence, contract, params) — the runner supplies contract and params
    if name == "none":
        return {}
    raise SystemExit(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loader", default="scopepair_labtrend", choices=["scopepair_labtrend", "scopebench"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--backend", default="sglang", choices=["sglang", "hf", "claude", "oracle"])
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model-path", default=None)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--retriever", default="temporal", choices=["temporal", "hybrid", "bm25_temporal", "bm25", "dense"])
    p.add_argument("--encoder-path", default="BAAI/bge-m3")
    p.add_argument("--encoder-device", default="cuda")
    p.add_argument("--n-candidates", type=int, default=50)
    p.add_argument("--expand-sections", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--pool-k", type=int, default=None)
    p.add_argument("--skills", default="reference", choices=["reference", "package", "none"])
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--label", default="")
    p.add_argument("--token-budget", type=int, default=None, help="required for tools configurations (token_matched / high_compute)")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--mask", default=None, choices=[None, "cited_admissible", "uncited_accessed"])
    p.add_argument("--mask-source", type=Path, default=None, help="run directory whose citations/access define the masking")
    p.add_argument("--inject", default=None, choices=[None, "future", "inscope"], help="DEV-35 exposure injection arm")
    p.add_argument("--inject-ids", type=Path, default=None, help="DEV-35: JSON {task_id: {arm: [evidence ids]}} fixed before any run")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--retry-errors", action="store_true", help="on resume, drop and re-run rows whose generation hit an infrastructure error")
    # sensitivity suite (plan §8.5) and cell labelling (plan §12 gate G7)
    p.add_argument("--variant", default=None, choices=[None, "strict_boundaries", "at_window_6h", "extended_rt"], help="task transformation applied in memory; the manifest is untouched")
    p.add_argument("--repair-wording", default="a", choices=["a", "b"], help="'b' = the alternative repair wording")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sampling-seed", type=int, default=None, help="decoding seed sent to the backend (default: --seed)")
    p.add_argument("--cell-kind", default="formal", choices=["formal", "sensitivity", "diagnostic", "pilot", "agreement"])
    p.add_argument("--task-ids", type=Path, default=None, help="DEV-34: JSON list of task ids to run; recorded by hash in the run metadata")
    p.add_argument("--stratified-subsample", type=int, default=None, help="D7: N tasks, equal share per (task type, scope) stratum, deterministic (--seed)")
    p.add_argument("--retrieval-cache", type=Path, default=None, help="directory memoizing evidence selections per task/contract/retriever (DEV-4); omit = no cache")
    args = p.parse_args()

    tasks, cases = load(args.loader, args.manifest, args.split)
    tasks = sorted(tasks, key=lambda t: (t.case_id, t.task_id))
    task_ids_sha256 = None
    if args.task_ids:
        import hashlib

        keep = set(json.loads(args.task_ids.read_text()))
        task_ids_sha256 = hashlib.sha256(args.task_ids.read_bytes()).hexdigest()
        missing = keep - {t.task_id for t in tasks}
        if missing:
            raise SystemExit(f"--task-ids: {len(missing)} ids not in the manifest")
        tasks = [t for t in tasks if t.task_id in keep]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    if args.stratified_subsample:
        import hashlib
        from collections import defaultdict

        strata = defaultdict(list)
        for t in tasks:
            strata[(t.task_type, t.contract.scope)].append(t)
        per = max(1, args.stratified_subsample // len(strata))
        chosen = []
        for key in sorted(strata):
            ranked = sorted(strata[key], key=lambda t: hashlib.sha256(f"{args.seed}|{t.task_id}".encode()).hexdigest())
            chosen.extend(ranked[:per])
        tasks = sorted(chosen, key=lambda t: (t.case_id, t.task_id))
    if args.variant:
        from scopebench.sensitivity import apply_variant

        dropped: list[str] = []
        tasks = apply_variant(tasks, args.variant, cases, dropped)
        if dropped:
            print(f"{args.variant}: {len(dropped)} tasks dropped (variant gold is an abstention and cannot be scored): {sorted(dropped)}")
    mask_source = None
    if args.mask:
        from scopebench.schema import read_jsonl

        mask_source = {r["task_id"]: r for r in read_jsonl(args.mask_source / "predictions.jsonl")}
    inject_ids, inject_sha256 = None, None
    if args.inject:
        import hashlib

        if args.inject_ids is None:
            raise SystemExit("--inject needs --inject-ids")
        inject_ids = json.loads(args.inject_ids.read_text())
        inject_sha256 = hashlib.sha256(args.inject_ids.read_bytes()).hexdigest()
        tasks = [t for t in tasks if inject_ids.get(t.task_id, {}).get(args.inject)]   # a task with nothing to inject is the baseline run
    retriever = make_retriever(args)
    cache = None
    if args.retrieval_cache:
        from scopebench.runner import RetrievalCache, sha256_file

        msha = sha256_file(args.manifest / "tasks.jsonl") if (args.manifest / "tasks.jsonl").exists() else None
        cache = RetrievalCache(args.retrieval_cache, msha[:16] if msha else None, retriever)
    spec = RunSpec(configuration=CONFIGS[args.config], retriever=retriever, k=args.k, pool_k=args.pool_k,
                   skills=make_skills(args.skills), max_tokens_override=args.max_tokens, label=args.label, mask=args.mask, mask_source=mask_source,
                   repair_wording=args.repair_wording, retrieval_cache=cache, inject=args.inject, inject_ids=inject_ids)
    extra_mask = {"mask": args.mask, "mask_source": str(args.mask_source)} if args.mask else {}
    args._tasks, args._cases = tasks, cases
    backend = make_backend(args)
    extra = {"loader": args.loader, "split": args.split, "skills_source": args.skills, "cell_kind": args.cell_kind, "variant": args.variant,
             "repair_wording": args.repair_wording, "temperature": args.temperature, "sampling_seed": args.sampling_seed if args.sampling_seed is not None else args.seed,
             "retrieval": {"retriever": args.retriever, "k": args.k, "pool_k": args.pool_k, "n_candidates": args.n_candidates, "expand_sections": args.expand_sections}, **extra_mask,
             **({"task_ids_file": str(args.task_ids), "task_ids_sha256": task_ids_sha256} if args.task_ids else {}),
             **({"inject": args.inject, "inject_ids_file": str(args.inject_ids), "inject_ids_sha256": inject_sha256} if args.inject else {})}
    if spec.configuration.tools:
        if not args.token_budget:
            raise SystemExit("--token-budget is required for tools configurations")
        from scopebench.tools_loop import run_tools

        meta = run_tools(spec, tasks, cases, backend, args.out_dir, token_budget=args.token_budget, max_turns=args.max_turns, manifest_path=args.manifest, extra_metadata=extra,
                         retry_errors=args.retry_errors)
    else:
        meta = run(spec, tasks, cases, backend, args.out_dir, manifest_path=args.manifest, extra_metadata=extra, retry_errors=args.retry_errors)
    print(f"{args.config}: {meta['n_tasks_this_run']} tasks written to {args.out_dir} ({meta['n_tasks_previously_done']} already done)")


if __name__ == "__main__":
    main()
