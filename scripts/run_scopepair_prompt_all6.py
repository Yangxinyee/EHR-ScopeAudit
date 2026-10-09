#!/usr/bin/env python3
"""Run the post-hoc information-complete prompt-only ScopePair baseline."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from run_scopepair_fhir import (
    CudaOnlyModel,
    SYSTEM_PROMPT,
    extract_json,
    load_tasks,
    local_model_revision,
    parse_time,
    sha256,
    text_sha256,
    user_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "experiments" / "scopepair_fhir" / "tasks.jsonl"
DEFAULT_MODEL = Path("/opt/models/Qwen__Qwen3-4B-Instruct-2507")
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_fhir_prompt_all6_qwen3_4b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks, args.split, max_tasks=None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} exists; pass --resume or use a new output directory"
        )
    existing = set()
    if predictions_path.exists():
        existing = {
            json.loads(line)["task_id"]
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    print(f"CUDA-only model load: {args.model}", flush=True)
    model = CudaOnlyModel(args.model, args.max_new_tokens, args.seed)
    started = time.perf_counter()
    for ordinal, task in enumerate(tasks, start=1):
        if task["task_id"] in existing:
            continue
        print(f"[prompt_only_all6] {ordinal}/{len(tasks)} {task['task_id']}", flush=True)
        evidence = sorted(
            task["candidates"],
            key=lambda item: (parse_time(item["issued_time"]), item["id"]),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt(task, evidence, include_skill=False),
            },
        ]
        response, tokens, latency = model.generate(messages)
        record = {
            "task_id": task["task_id"],
            "case_id": task["case_id"],
            "patient_cluster_id": task["patient_cluster_id"],
            "split": task["split"],
            "scope": task["scope"],
            "timestamp_stress": task["timestamp_stress"],
            "system": "prompt_only_all6",
            "selected_evidence_ids": [item["id"] for item in evidence],
            "response": response,
            "repair_response": "",
            "parsed_answer": extract_json(response),
            "initial_guard_errors": [],
            "final_guard_errors": [],
            "guard_repaired": False,
            "guard_deterministic_fallback": False,
            "generated_tokens": tokens,
            "latency_seconds": latency,
        }
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "benchmark": "ScopePair-FHIR",
        "analysis_status": "post-hoc robustness baseline",
        "system": "prompt_only_all6",
        "task_manifest_sha256": sha256(args.tasks),
        "split": args.split,
        "n_tasks": len(tasks),
        "model_path": str(args.model),
        "model_revision": local_model_revision(args.model),
        "temperature": 0.0,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "system_prompt_sha256": text_sha256(SYSTEM_PROMPT),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime": model.runtime_metadata(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
