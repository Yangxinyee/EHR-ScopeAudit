#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# MedAgentBench-related adaptation; Copyright (c) 2025 Stanford Machine Learning Group.
# Upstream license: licenses/MedAgentBench-MIT.txt; changes: THIRD_PARTY_NOTICES.md
"""Run MedAgentBench v1 faithfully against its official Docker environment.

The released task server is tightly coupled to remote chat APIs and contains a
response-type mismatch between ``utils.py`` and the separately distributed
``refsol.py``. This runner preserves the official prompt, task data, eight-turn
protocol, FHIR server, and refsol checks while using a local Hugging Face model.
The only compatibility shim gives refsol the JSON text that its ``json.loads``
calls expect; model-visible GET responses are likewise the unmodified text.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "external_benchmarks" / "MedAgentBench"
TASK_DATA = BENCHMARK_ROOT / "data" / "medagentbench" / "test_data_v2.json"
FUNCTIONS_DATA = BENCHMARK_ROOT / "data" / "medagentbench" / "funcs_v1.json"
OFFICIAL_TASK_MODULE = (
    BENCHMARK_ROOT / "src" / "server" / "tasks" / "medagentbench" / "__init__.py"
)
REFSOL_PATH = (
    BENCHMARK_ROOT / "src" / "server" / "tasks" / "medagentbench" / "refsol.py"
)
UTILS_PATH = (
    BENCHMARK_ROOT / "src" / "server" / "tasks" / "medagentbench" / "utils.py"
)
DEFAULT_MODEL = Path("/opt/models/Qwen__Qwen3-30B-A3B-Instruct-2507")
DEFAULT_OUTPUT = ROOT / "experiments" / "medagentbench_qwen3_30b_official"
DEFAULT_FHIR_BASE = "http://127.0.0.1:8080/fhir/"
BENCHMARK_COMMIT = "99260117137b09f04837a8c18d18a1107efa55ae"
DOCKER_IMAGE_DIGEST = (
    "docker.io/jyxsu6/medagentbench@"
    "sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"
)
QUERY_CATEGORIES = {"task1", "task2", "task4", "task6", "task7"}
ACTION_CATEGORIES = {"task3", "task5", "task8", "task9", "task10"}
ALLOWED_RESOURCE_PATHS = {
    "Condition",
    "MedicationRequest",
    "Observation",
    "Patient",
    "Procedure",
    "ServiceRequest",
}


@dataclass
class HistoryItem:
    role: str
    content: str


@dataclass
class GraderResult:
    history: list[HistoryItem]
    result: str | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_official_prompt(path: Path) -> str:
    """Read the prompt constant from the official source without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "MedAgentBench_prompt"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            break
        return value
    raise ValueError(f"MedAgentBench_prompt not found in {path}")


def load_official_grader() -> tuple[types.ModuleType, types.ModuleType]:
    """Load Box refsol with its relative utils import, avoiding task-server deps."""
    package_name = "_official_medagentbench_grader"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REFSOL_PATH.parent)]
    sys.modules[package_name] = package

    utils_name = f"{package_name}.utils"
    utils_spec = importlib.util.spec_from_file_location(utils_name, UTILS_PATH)
    if utils_spec is None or utils_spec.loader is None:
        raise ImportError(f"Cannot load {UTILS_PATH}")
    utils = importlib.util.module_from_spec(utils_spec)
    sys.modules[utils_name] = utils
    utils_spec.loader.exec_module(utils)

    refsol_name = f"{package_name}.refsol"
    refsol_spec = importlib.util.spec_from_file_location(refsol_name, REFSOL_PATH)
    if refsol_spec is None or refsol_spec.loader is None:
        raise ImportError(f"Cannot load {REFSOL_PATH}")
    refsol = importlib.util.module_from_spec(refsol_spec)
    sys.modules[refsol_name] = refsol
    refsol_spec.loader.exec_module(refsol)
    return refsol, utils


def get_json_text(url: str, timeout: int = 120) -> dict[str, Any]:
    """Compatibility form expected by released refsol: data is JSON text."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return {"status_code": response.status_code, "data": response.text}
    except Exception as exc:
        return {"error": str(exc)}


def strip_model_wrappers(text: str) -> str:
    text = text.strip().split("<|im_end|>", 1)[0].strip()
    text = text.replace("```tool_code", "").replace("```json", "").replace("```", "")
    return text.strip()


def validate_get_url(url: str, fhir_base: str) -> str:
    base = urlsplit(fhir_base)
    parsed = urlsplit(url)
    if (parsed.scheme, parsed.hostname, parsed.port) != (
        base.scheme,
        base.hostname,
        base.port,
    ):
        raise ValueError("GET URL must use the configured local FHIR server")
    base_path = base.path.rstrip("/") + "/"
    if not parsed.path.startswith(base_path):
        raise ValueError("GET URL is outside the configured FHIR base path")
    resource_type = parsed.path[len(base_path) :].strip("/")
    if resource_type not in ALLOWED_RESOURCE_PATHS:
        raise ValueError(f"Unsupported FHIR resource path: {resource_type}")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["_format"] = "json"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), "")
    )


def parse_post(text: str, fhir_base: str) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("POST "):
        raise ValueError("Invalid POST action")
    url = lines[0][5:].strip()
    parsed = urlsplit(url)
    base = urlsplit(fhir_base)
    if (parsed.scheme, parsed.hostname, parsed.port) != (
        base.scheme,
        base.hostname,
        base.port,
    ):
        raise ValueError("POST URL must use the configured local FHIR server")
    resource_type = parsed.path.removeprefix(base.path.rstrip("/") + "/").strip("/")
    if resource_type not in {"Observation", "MedicationRequest", "ServiceRequest"}:
        raise ValueError(f"Unsupported POST resource path: {resource_type}")
    payload = json.loads("\n".join(lines[1:]))
    if not isinstance(payload, dict):
        raise ValueError("POST payload must be one JSON object")
    return url, payload


class LocalQwenText:
    def __init__(self, model_path: Path, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, int, float]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        start = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return strip_model_wrappers(text), int(generated.shape[0]), elapsed

    def runtime_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "torch_version": self.torch.__version__,
            "cuda_runtime": self.torch.version.cuda,
            "cuda_available": self.torch.cuda.is_available(),
            "model_device": str(self.device),
        }
        if self.torch.cuda.is_available():
            properties = self.torch.cuda.get_device_properties(0)
            metadata.update(
                {
                    "gpu_name": properties.name,
                    "gpu_compute_capability": list(
                        self.torch.cuda.get_device_capability(0)
                    ),
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
                    "gpu_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
                }
            )
        return metadata


def local_model_revision(model_path: Path) -> str:
    tree_dir = model_path / ".cache" / "huggingface" / "trees"
    files = sorted(tree_dir.glob("*.json")) if tree_dir.exists() else []
    return files[0].stem if len(files) == 1 else "unknown-local-snapshot"


def execute_case(
    model: LocalQwenText,
    case: dict[str, Any],
    official_prompt: str,
    functions: list[dict[str, Any]],
    fhir_base: str,
    max_rounds: int,
) -> dict[str, Any]:
    prompt = official_prompt.format(
        api_base=fhir_base,
        functions=json.dumps(functions),
        context=case["context"],
        question=case["instruction"],
    )
    messages = [{"role": "user", "content": prompt}]
    history = [HistoryItem(role="user", content=prompt)]
    trace: list[dict[str, Any]] = []
    total_tokens = 0
    total_latency = 0.0
    result: str | None = None
    status = "task_limit_reached"

    for round_index in range(max_rounds):
        response, tokens, latency = model.generate(messages)
        total_tokens += tokens
        total_latency += latency
        history.append(HistoryItem(role="agent", content=response))
        trace.append(
            {
                "round": round_index + 1,
                "role": "agent",
                "content": response,
                "generated_tokens": tokens,
                "latency_seconds": latency,
            }
        )
        messages.append({"role": "assistant", "content": response})

        if response.startswith("GET "):
            try:
                url = validate_get_url(response[4:].strip(), fhir_base)
                get_result = get_json_text(url)
                if "data" in get_result:
                    feedback = (
                        "Here is the response from the GET request:\n"
                        f"{get_result['data']}. Please call FINISH if you have got "
                        "answers for all the questions and finished all the requested tasks"
                    )
                else:
                    feedback = f"Error in sending the GET request: {get_result['error']}"
            except Exception as exc:
                feedback = f"Error in sending the GET request: {type(exc).__name__}: {exc}"
        elif response.startswith("POST "):
            try:
                parse_post(response, fhir_base)
            except Exception:
                feedback = "Invalid POST request"
            else:
                # This mirrors the released task server. Action success is
                # determined by refsol's URL/payload checks over the trace.
                feedback = (
                    "POST request accepted and executed successfully. Please call "
                    "FINISH if you have got answers for all the questions and finished "
                    "all the requested tasks"
                )
        elif response.startswith("FINISH(") and response.endswith(")"):
            result = response[len("FINISH(") : -1]
            status = "completed"
            break
        else:
            status = "invalid_action"
            break

        history.append(HistoryItem(role="user", content=feedback))
        trace.append(
            {
                "round": round_index + 1,
                "role": "environment",
                "content": feedback,
            }
        )
        messages.append({"role": "user", "content": feedback})

    return {
        "result": result,
        "status": status,
        "history": history,
        "trace": trace,
        "generated_tokens": total_tokens,
        "latency_seconds": total_latency,
    }


def bootstrap_ci(
    values: list[float], seed: int, replicates: int = 2_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(values) for _ in range(len(values)))
        for _ in range(replicates)
    ]
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]


def summarize(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    def group_summary(name: str, selected: list[dict[str, Any]], offset: int) -> dict[str, Any]:
        values = [float(record["correct"]) for record in selected]
        ci_low, ci_high = bootstrap_ci(values, seed + offset)
        return {
            "group": name,
            "n": len(selected),
            "success_rate": mean(values),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_rounds": mean(record["rounds"] for record in selected),
            "mean_generated_tokens": mean(
                record["generated_tokens"] for record in selected
            ),
            "mean_latency_seconds": mean(
                record["latency_seconds"] for record in selected
            ),
        }

    query = [record for record in records if record["task_category"] in QUERY_CATEGORIES]
    action = [
        record for record in records if record["task_category"] in ACTION_CATEGORIES
    ]
    return {
        "rows": [
            group_summary("overall_pass_at_1", records, 0),
            group_summary("query_success_rate", query, 1),
            group_summary("action_success_rate", action, 2),
        ]
    }


def fhir_resource_counts(fhir_base: str) -> dict[str, int]:
    counts = {}
    for resource_type in (
        "Patient",
        "Observation",
        "Condition",
        "MedicationRequest",
        "Procedure",
        "ServiceRequest",
    ):
        response = requests.get(
            f"{fhir_base}{resource_type}",
            params={"_summary": "count", "_format": "json"},
            timeout=120,
        )
        response.raise_for_status()
        counts[resource_type] = int(response.json().get("total", 0))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--task-data", type=Path, default=TASK_DATA)
    parser.add_argument("--functions-data", type=Path, default=FUNCTIONS_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fhir-base", default=DEFAULT_FHIR_BASE)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate official tasks, graders, and FHIR counts without loading a model.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} already exists; use --resume or a new output directory"
        )

    official_prompt = extract_official_prompt(OFFICIAL_TASK_MODULE)
    tasks = json.loads(args.task_data.read_text(encoding="utf-8"))
    functions = json.loads(args.functions_data.read_text(encoding="utf-8"))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ValueError("No MedAgentBench tasks selected")

    metadata_response = requests.get(f"{args.fhir_base}metadata", timeout=120)
    metadata_response.raise_for_status()
    counts = fhir_resource_counts(args.fhir_base)
    non_patient_count = sum(
        count for resource_type, count in counts.items() if resource_type != "Patient"
    )
    if non_patient_count != 785_207:
        raise ValueError(
            f"Official environment count mismatch: expected 785207 non-Patient "
            f"records, found {non_patient_count}"
        )

    refsol, utils = load_official_grader()
    utils.send_get_request = get_json_text
    refsol.send_get_request = get_json_text
    categories = sorted({case["id"].split("_", 1)[0] for case in tasks})
    missing_graders = [
        category for category in categories if not callable(getattr(refsol, category, None))
    ]
    if missing_graders:
        raise ValueError(f"Missing official graders: {missing_graders}")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "n_tasks": len(tasks),
                    "categories": categories,
                    "all_official_graders_loaded": True,
                    "fhir_resource_counts": counts,
                    "fhir_non_patient_record_count": non_patient_count,
                    "official_prompt_sha256": hashlib.sha256(
                        official_prompt.encode("utf-8")
                    ).hexdigest(),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing[record["task_id"]] = record

    print(f"Loading model {args.model}", flush=True)
    model = LocalQwenText(args.model, args.max_new_tokens)
    records = list(existing.values())
    by_id = {case["id"]: case for case in tasks}
    for ordinal, case in enumerate(tasks, start=1):
        if case["id"] in existing:
            continue
        print(f"{ordinal}/{len(tasks)} {case['id']}", flush=True)
        output = execute_case(
            model,
            case,
            official_prompt,
            functions,
            args.fhir_base,
            args.max_rounds,
        )
        grader_results = GraderResult(
            history=output["history"],
            result=output["result"],
        )
        task_category = case["id"].split("_", 1)[0]
        grader = getattr(refsol, task_category)
        grader_error = ""
        try:
            correct = bool(grader(case, grader_results, args.fhir_base))
        except Exception as exc:
            correct = False
            grader_error = f"{type(exc).__name__}: {exc}"
        record = {
            "task_id": case["id"],
            "task_category": task_category,
            "benchmark_group": (
                "query" if task_category in QUERY_CATEGORIES else "action"
            ),
            "instruction": case["instruction"],
            "eval_mrn": case.get("eval_MRN"),
            "result": output["result"],
            "status": output["status"],
            "correct": int(correct),
            "rounds": sum(
                item["role"] == "agent" for item in output["trace"]
            ),
            "generated_tokens": output["generated_tokens"],
            "latency_seconds": output["latency_seconds"],
            "grader_error": grader_error,
            "trace": output["trace"],
        }
        records.append(record)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for item in sorted(records, key=lambda value: value["task_id"]):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    selected_records = [record for record in records if record["task_id"] in by_id]
    summary = summarize(selected_records, args.seed)
    import pandas as pd

    pd.DataFrame(summary["rows"]).to_csv(args.output_dir / "summary.csv", index=False)
    per_category = []
    for category in sorted({record["task_category"] for record in selected_records}):
        category_records = [
            record
            for record in selected_records
            if record["task_category"] == category
        ]
        per_category.append(
            {
                "task_category": category,
                "n": len(category_records),
                "success_rate": mean(
                    record["correct"] for record in category_records
                ),
            }
        )
    pd.DataFrame(per_category).to_csv(
        args.output_dir / "per_category.csv", index=False
    )

    metadata = {
        "benchmark": "MedAgentBench v1",
        "benchmark_commit": BENCHMARK_COMMIT,
        "task_data_sha256": sha256(args.task_data),
        "functions_data_sha256": sha256(args.functions_data),
        "refsol_sha256": sha256(REFSOL_PATH),
        "official_prompt_sha256": hashlib.sha256(
            official_prompt.encode("utf-8")
        ).hexdigest(),
        "official_docker_image": DOCKER_IMAGE_DIGEST,
        "fhir_base": args.fhir_base,
        "fhir_resource_counts": counts,
        "fhir_non_patient_record_count": non_patient_count,
        "n_tasks": len(tasks),
        "query_categories": sorted(QUERY_CATEGORIES),
        "action_categories": sorted(ACTION_CATEGORIES),
        "model_path": str(args.model),
        "model_revision": local_model_revision(args.model),
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "max_rounds": args.max_rounds,
        "seed": args.seed,
        "compatibility_shim": (
            "Released utils.py returns response.json(), while Box refsol.py calls "
            "json.loads on data. The shim supplies response.text to both the grader "
            "and the model-visible official GET feedback."
        ),
        "post_semantics": (
            "Mirrors released task server: validate JSON and report acceptance; "
            "refsol scores URL/payload from trace without persisting the POST."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "runtime": model.runtime_metadata(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
