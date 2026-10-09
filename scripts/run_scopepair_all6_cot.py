#!/usr/bin/env python3
"""Post-hoc reasoning-enabled all-six robustness baseline for both ScopePair families.

The frozen all-six condition asks for a single JSON object and allows 384 new
tokens. A reviewer can reasonably object that its failures reflect that
instruction design rather than the difficulty of executing the contract. This
runner tests that objection directly and is deliberately biased in the
baseline's favour:

* the model is told to reason step by step before answering;
* the generation budget is doubled to 768 new tokens;
* the parser is strictly more permissive than the frozen one -- it prefers the
  last fenced JSON block, then the last balanced JSON object, and only then
  falls back to the frozen extractor.

Everything else is identical to the frozen all-six condition: same tasks, same
candidates, same contract payload, same output schema, same greedy decoding,
same seed, same scorer. No filter, skill, guard, repair, or fallback is used.
Any advantage this condition gains is an advantage for the baseline that
scope-first is compared against.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from run_scopepair_fhir import (
    CudaOnlyModel,
    SYSTEM_PROMPT as LAB_SYSTEM_PROMPT,
    extract_json,
    load_tasks,
    local_model_revision,
    parse_time,
    sha256,
    text_sha256,
)
from run_scopepair_fhir import user_prompt as lab_user_prompt
from run_scopepair_medadmin import SYSTEM_PROMPT as MED_SYSTEM_PROMPT
from run_scopepair_medadmin import user_prompt as med_user_prompt


ROOT = Path(__file__).resolve().parents[1]

REASONING_SUFFIX = """

Before answering, reason step by step in plain text: state which supplied
records satisfy the contract, list them in the required chronological order,
and compute the requested aggregation explicitly. Then output the final answer
as the last thing in your reply, as exactly one JSON object inside a ```json
code fence. Only the fenced JSON object is read."""

FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def last_balanced_object(text: str) -> dict[str, Any] | None:
    """Return the last balanced top-level JSON object in the text, if any."""
    best: dict[str, Any] | None = None
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        best = parsed
    return best


def permissive_extract_json(text: str) -> dict[str, Any] | None:
    """Prefer the last fenced object, then the last balanced object, then the frozen parser."""
    fenced = FENCE.findall(text)
    for block in reversed(fenced):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    balanced = last_balanced_object(text)
    if balanced is not None:
        return balanced
    return extract_json(text)


FAMILIES = {
    "lab": {
        "benchmark": "ScopePair-FHIR",
        "tasks": ROOT / "experiments" / "scopepair_fhir" / "tasks.jsonl",
        "base_prompt": LAB_SYSTEM_PROMPT,
        "user_prompt": lab_user_prompt,
        "time_field": "issued_time",
        "system": "prompt_only_all6_cot",
    },
    "med": {
        "benchmark": "ScopePair-MedAdmin",
        "tasks": ROOT / "experiments" / "scopepair_medadmin" / "tasks.jsonl",
        "base_prompt": MED_SYSTEM_PROMPT,
        "user_prompt": med_user_prompt,
        "time_field": "event_time",
        "system": "prompt_all6_cot",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    family = FAMILIES[args.family]
    tasks_path = args.tasks or family["tasks"]
    system_prompt = family["base_prompt"] + REASONING_SUFFIX
    time_field = family["time_field"]

    tasks = load_tasks(tasks_path, args.split, args.max_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} exists; pass --resume or use a new output directory"
        )
    existing: set[str] = set()
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
        print(
            f"[{family['system']}] {ordinal}/{len(tasks)} {task['task_id']}",
            flush=True,
        )
        evidence = sorted(
            task["candidates"],
            key=lambda item: (parse_time(str(item[time_field])), str(item["id"])),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": family["user_prompt"](task, evidence, include_skill=False),
            },
        ]
        response, tokens, latency = model.generate(messages)
        record = {
            "task_id": task["task_id"],
            "case_id": task["case_id"],
            "patient_cluster_id": task["patient_cluster_id"],
            "split": task["split"],
            "scope": task["scope"],
            "timestamp_stress": task.get("timestamp_stress", False),
            "system": family["system"],
            "selected_evidence_ids": [item["id"] for item in evidence],
            "response": response,
            "repair_response": "",
            "parsed_answer": permissive_extract_json(response),
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
        "benchmark": family["benchmark"],
        "analysis_status": "post-hoc reasoning-enabled robustness baseline",
        "system": family["system"],
        "task_manifest": str(tasks_path),
        "task_manifest_sha256": sha256(tasks_path),
        "split": args.split,
        "n_tasks": len(tasks),
        "model_path": str(args.model),
        "model_revision": local_model_revision(args.model),
        "temperature": 0.0,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "system_prompt_sha256": text_sha256(system_prompt),
        "parser": "permissive: last fenced JSON, then last balanced JSON, then frozen extractor",
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
