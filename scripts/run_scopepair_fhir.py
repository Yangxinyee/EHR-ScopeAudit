#!/usr/bin/env python3
"""Run CUDA-only local models on the prospective ScopePair-FHIR benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "experiments" / "scopepair_fhir" / "tasks.jsonl"
DEFAULT_MODEL = Path("/opt/models/Qwen__Qwen3-4B-Instruct-2507")
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_fhir_dev_qwen3_4b"
SYSTEMS = (
    "unscoped",
    "late_no_refill",
    "wrong_time_scope_first",
    "scope_first",
    "guard_refill",
    "clinehr_os",
)


SYSTEM_PROMPT = """You are an EHR evidence extraction agent. The task contract is binding.

The evidence field `issued_time` is when a laboratory result became available.
The field `effective_time` is when the specimen/event occurred and MUST NOT be
used to decide availability for this task. Use only the supplied evidence.

If exactly three supplied observations satisfy the requested pre/post contract,
return them in chronological `issued_time` order and calculate delta as the last
value minus the first value. Direction is `increasing`, `decreasing`, or `stable`.
If fewer than three supplied observations satisfy the contract, return status
`insufficient`, an empty results list, null delta, and null direction. Never
invent, replace, or cite an observation that is not supplied.

Return exactly one JSON object and no prose:
{"status":"ok|insufficient","results":[{"citation":"id","value":0.0,
"unit":"unit","issued_time":"ISO timestamp"}],"delta":0.0,
"direction":"increasing|decreasing|stable|null"}"""


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_model_revision(model_path: Path) -> str:
    tree_dir = model_path / ".cache" / "huggingface" / "trees"
    tree_files = sorted(tree_dir.glob("*.json")) if tree_dir.exists() else []
    if len(tree_files) == 1:
        return tree_files[0].stem
    parts = []
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        path = model_path / name
        if path.exists():
            parts.append(f"{name}:{sha256(path)}")
    return text_sha256("\n".join(parts)) if parts else "unknown-local-snapshot"


def permitted(evidence: dict[str, Any], task: dict[str, Any], basis: str) -> bool:
    field = "issued_time" if basis == "issued" else "effective_time"
    timestamp = parse_time(str(evidence[field]))
    index = parse_time(str(task["index_time"]))
    if task["scope"] == "pre":
        return timestamp < index
    if task["scope"] == "post":
        return timestamp > index
    raise ValueError(f"Unknown task scope: {task['scope']}")


def nearest(
    candidates: list[dict[str, Any]],
    task: dict[str, Any],
    *,
    basis: str,
    require_permitted: bool,
    k: int = 3,
) -> list[dict[str, Any]]:
    field = "issued_time" if basis == "issued" else "effective_time"
    index = parse_time(str(task["index_time"]))
    pool = [
        evidence
        for evidence in candidates
        if not require_permitted or permitted(evidence, task, basis)
    ]
    selected = sorted(
        pool,
        key=lambda evidence: (
            abs((parse_time(str(evidence[field])) - index).total_seconds()),
            str(evidence["id"]),
        ),
    )[:k]
    return sorted(
        selected,
        key=lambda evidence: (
            parse_time(str(evidence["issued_time"])),
            str(evidence["id"]),
        ),
    )


def select_evidence(task: dict[str, Any], system: str) -> list[dict[str, Any]]:
    candidates = list(task["candidates"])
    unscoped = nearest(
        candidates, task, basis="issued", require_permitted=False, k=3
    )
    if system == "unscoped":
        return unscoped
    if system == "late_no_refill":
        return [
            evidence
            for evidence in unscoped
            if permitted(evidence, task, basis="issued")
        ]
    if system == "wrong_time_scope_first":
        return nearest(
            candidates, task, basis="effective", require_permitted=True, k=3
        )
    scope_first = nearest(
        candidates, task, basis="issued", require_permitted=True, k=3
    )
    if system == "scope_first" or system == "clinehr_os":
        return scope_first
    if system == "guard_refill":
        legal = [
            evidence
            for evidence in unscoped
            if permitted(evidence, task, basis="issued")
        ]
        seen = {str(evidence["id"]) for evidence in legal}
        permitted_remaining = nearest(
            candidates, task, basis="issued", require_permitted=True, k=len(candidates)
        )
        for evidence in permitted_remaining:
            if len(legal) >= 3:
                break
            if str(evidence["id"]) not in seen:
                legal.append(evidence)
                seen.add(str(evidence["id"]))
        legal.sort(
            key=lambda evidence: (
                parse_time(str(evidence["issued_time"])),
                str(evidence["id"]),
            )
        )
        if [item["id"] for item in legal] != [item["id"] for item in scope_first]:
            raise AssertionError("Guard refill and scope-first evidence must be equivalent")
        return legal
    raise ValueError(f"Unknown system: {system}")


def skill_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if len(evidence) != 3:
        return {"status": "insufficient", "provenance": []}
    first_value = float(evidence[0]["value"])
    last_value = float(evidence[-1]["value"])
    delta = last_value - first_value
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        direction = "stable"
    elif delta > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return {
        "status": "ok",
        "ordered_result_ids": [item["id"] for item in evidence],
        "delta_last_minus_first": delta,
        "direction": direction,
        "provenance": [
            {
                "citation": item["id"],
                "value": item["value"],
                "unit": item["unit"],
                "issued_time": item["issued_time"],
            }
            for item in evidence
        ],
    }


def skill_answer(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    summary = skill_summary(evidence)
    if summary["status"] != "ok":
        return {
            "status": "insufficient",
            "results": [],
            "delta": None,
            "direction": None,
        }
    return {
        "status": "ok",
        "results": summary["provenance"],
        "delta": summary["delta_last_minus_first"],
        "direction": summary["direction"],
    }


def user_prompt(
    task: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    include_skill: bool,
) -> str:
    payload: dict[str, Any] = {
        "question": task["question"],
        "contract": {
            "scope": task["scope"],
            "index_time": task["index_time"],
            "timestamp_basis": "issued_time",
            "required_result_count": 3,
        },
        "supplied_evidence": evidence,
    }
    if include_skill:
        payload["deterministic_chartskill"] = skill_summary(evidence)
        payload["skill_instruction"] = (
            "The deterministic skill is provenance-preserving. Verify its cited "
            "records against supplied evidence and use it to format the answer."
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def cited_ids(parsed: dict[str, Any] | None) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    results = parsed.get("results")
    if not isinstance(results, list):
        return []
    citations = []
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("citation"), str):
            citations.append(item["citation"])
    return citations


def guard_errors(
    parsed: dict[str, Any] | None,
    task: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    validate_skill: bool = False,
) -> list[str]:
    if parsed is None:
        return ["submission is not one parseable JSON object"]
    errors = []
    status = parsed.get("status")
    if status not in {"ok", "insufficient"}:
        errors.append("status must be ok or insufficient")
    results = parsed.get("results")
    if not isinstance(results, list):
        errors.append("results must be a list")
        results = []
    citations = cited_ids(parsed)
    allowed = {
        str(candidate["id"])
        for candidate in task["candidates"]
        if permitted(candidate, task, basis="issued")
    }
    illegal = sorted(set(citations) - allowed)
    if illegal:
        errors.append(f"out-of-contract citations: {illegal}")
    supplied = {str(item["id"]) for item in evidence}
    unavailable = sorted(set(citations) - supplied)
    if unavailable:
        errors.append(f"citations not present in supplied evidence: {unavailable}")
    if status == "ok" and (len(results) != 3 or len(set(citations)) != 3):
        errors.append("an ok answer must contain three distinct cited results")
    legal_supplied = [
        item for item in evidence if permitted(item, task, basis="issued")
    ]
    if status == "insufficient" and len(legal_supplied) >= 3:
        errors.append("three permitted supplied observations exist; do not abstain")
    if validate_skill and status == "ok" and len(evidence) == 3:
        expected = skill_summary(evidence)
        expected_results = expected["provenance"]
        if len(results) == 3:
            for index, (observed, gold) in enumerate(
                zip(results, expected_results), start=1
            ):
                if not isinstance(observed, dict):
                    errors.append(f"result {index} must be an object")
                    continue
                if observed.get("citation") != gold["citation"]:
                    errors.append(f"result {index} citation/order disagrees with ChartSkill")
                if observed.get("unit") != gold["unit"]:
                    errors.append(f"result {index} unit disagrees with ChartSkill")
                try:
                    value_matches = math.isclose(
                        float(observed.get("value")),
                        float(gold["value"]),
                        rel_tol=1e-7,
                        abs_tol=1e-7,
                    )
                except (TypeError, ValueError):
                    value_matches = False
                if not value_matches:
                    errors.append(f"result {index} value disagrees with ChartSkill")
                try:
                    time_matches = parse_time(
                        str(observed.get("issued_time"))
                    ) == parse_time(str(gold["issued_time"]))
                except (TypeError, ValueError):
                    time_matches = False
                if not time_matches:
                    errors.append(
                        f"result {index} availability time disagrees with ChartSkill"
                    )
        try:
            delta_matches = math.isclose(
                float(parsed.get("delta")),
                float(expected["delta_last_minus_first"]),
                rel_tol=1e-7,
                abs_tol=1e-7,
            )
        except (TypeError, ValueError):
            delta_matches = False
        if not delta_matches:
            errors.append("delta disagrees with deterministic ChartSkill")
        if parsed.get("direction") != expected["direction"]:
            errors.append("direction disagrees with deterministic ChartSkill")
    return errors


class CudaOnlyModel:
    def __init__(self, model_path: Path, max_new_tokens: int, seed: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for ScopePair-FHIR. Refusing CPU inference."
            )
        if torch.cuda.device_count() < 1:
            raise RuntimeError("No CUDA device found. Refusing CPU inference.")
        torch.cuda.set_device(0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(0)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        parameter_devices = {parameter.device.type for parameter in self.model.parameters()}
        if parameter_devices != {"cuda"}:
            raise RuntimeError(
                f"CPU/offloaded parameters detected: {sorted(parameter_devices)}"
            )
        self.device = next(self.model.parameters()).device

    def generate(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, int, float]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, int(generated.shape[0]), elapsed

    def runtime_metadata(self) -> dict[str, Any]:
        properties = self.torch.cuda.get_device_properties(0)
        return {
            "torch_version": self.torch.__version__,
            "cuda_runtime": self.torch.version.cuda,
            "cuda_available": self.torch.cuda.is_available(),
            "cuda_device_count": self.torch.cuda.device_count(),
            "model_device": str(self.device),
            "gpu_name": properties.name,
            "gpu_compute_capability": list(self.torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
            "gpu_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
            "cpu_inference_allowed": False,
        }


def run_task(
    model: CudaOnlyModel,
    task: dict[str, Any],
    system: str,
) -> dict[str, Any]:
    evidence = select_evidence(task, system)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt(
                task, evidence, include_skill=(system == "clinehr_os")
            ),
        },
    ]
    response, tokens, latency = model.generate(messages)
    parsed = extract_json(response)
    initial_errors = (
        guard_errors(
            parsed,
            task,
            evidence,
            validate_skill=(system == "clinehr_os"),
        )
        if system in {"guard_refill", "clinehr_os"}
        else []
    )
    repair_response = ""
    repaired = False
    total_tokens = tokens
    total_latency = latency
    if initial_errors:
        repaired = True
        repair_evidence = nearest(
            list(task["candidates"]),
            task,
            basis="issued",
            require_permitted=True,
            k=3,
        )
        repair_payload = {
            "scopeguard_errors": initial_errors,
            "instruction": (
                "Repair the submission. Use only the following contract-permitted "
                "evidence and return exactly the required JSON object."
            ),
            "permitted_evidence": repair_evidence,
        }
        if system == "clinehr_os":
            repair_payload["deterministic_chartskill"] = skill_summary(
                repair_evidence
            )
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": json.dumps(
                        repair_payload, ensure_ascii=False, sort_keys=True
                    ),
                },
            ]
        )
        repair_response, repair_tokens, repair_latency = model.generate(messages)
        total_tokens += repair_tokens
        total_latency += repair_latency
        parsed = extract_json(repair_response)
    final_evidence = nearest(
        list(task["candidates"]),
        task,
        basis="issued",
        require_permitted=True,
        k=3,
    )
    final_errors = (
        guard_errors(
            parsed,
            task,
            final_evidence,
            validate_skill=(system == "clinehr_os"),
        )
        if system in {"guard_refill", "clinehr_os"}
        else []
    )
    deterministic_fallback = False
    if system == "clinehr_os" and final_errors:
        deterministic_fallback = True
        parsed = skill_answer(final_evidence)
        final_errors = guard_errors(
            parsed,
            task,
            final_evidence,
            validate_skill=True,
        )
        if final_errors:
            raise AssertionError(
                f"Verifier-generated ChartSkill answer failed validation: {final_errors}"
            )
    return {
        "selected_evidence_ids": [item["id"] for item in evidence],
        "response": response,
        "repair_response": repair_response,
        "parsed_answer": parsed,
        "initial_guard_errors": initial_errors,
        "final_guard_errors": final_errors,
        "guard_repaired": repaired,
        "guard_deterministic_fallback": deterministic_fallback,
        "generated_tokens": total_tokens,
        "latency_seconds": total_latency,
    }


def load_tasks(path: Path, split: str, max_tasks: int | None) -> list[dict[str, Any]]:
    tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = sorted(
        (task for task in tasks if task["split"] == split),
        key=lambda task: task["task_id"],
    )
    if max_tasks is not None:
        selected = selected[:max_tasks]
    if not selected:
        raise ValueError("No tasks selected")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks, args.split, args.max_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} exists; pass --resume or use a new output directory"
        )
    existing: set[tuple[str, str]] = set()
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing.add((record["task_id"], record["system"]))

    print(f"CUDA-only model load: {args.model}", flush=True)
    model = CudaOnlyModel(args.model, args.max_new_tokens, args.seed)
    for system in args.systems:
        for ordinal, task in enumerate(tasks, start=1):
            key = (task["task_id"], system)
            if key in existing:
                continue
            print(
                f"[{system}] {ordinal}/{len(tasks)} {task['task_id']}",
                flush=True,
            )
            result = run_task(model, task, system)
            record = {
                "task_id": task["task_id"],
                "case_id": task["case_id"],
                "patient_cluster_id": task["patient_cluster_id"],
                "split": task["split"],
                "scope": task["scope"],
                "timestamp_stress": task["timestamp_stress"],
                "system": system,
                **result,
            }
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "benchmark": "ScopePair-FHIR",
        "protocol_version": "1.0",
        "task_manifest": str(args.tasks),
        "task_manifest_sha256": sha256(args.tasks),
        "split": args.split,
        "n_tasks": len(tasks),
        "systems": args.systems,
        "model_path": str(args.model),
        "model_revision": local_model_revision(args.model),
        "temperature": 0.0,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "maximum_guard_repairs": 1,
        "seed": args.seed,
        "system_prompt_sha256": text_sha256(SYSTEM_PROMPT),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "runtime": model.runtime_metadata(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
