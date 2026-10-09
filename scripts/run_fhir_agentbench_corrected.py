#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Run a faithful local evaluation on the official FHIR-AgentBench split.

Unlike the retired synthetic-HAPI experiment, this runner:

1. reads the official 929k-resource MIMIC-IV-on-FHIR demo;
2. exposes actual patient-linked resources to a real local LLM agent;
3. evaluates only released resource IDs and structured answers;
4. writes predictions, traces, native metrics, and run metadata separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fhir_benchmark_core import (
    SUPPORTED_RESOURCE_TYPES,
    LocalFhirStore,
    canonical_answer_correct,
    extract_final_answer,
    parse_structured_answer,
    parse_tool_calls,
    resource_ids,
    retrieval_precision,
    retrieval_recall,
    run_safe_python,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    ROOT
    / "external_benchmarks"
    / "FHIR-AgentBench"
    / "final_dataset"
    / "questions_answers_sql_fhir.csv"
)
DEFAULT_FHIR_DIR = (
    ROOT.parent
    / "public_data"
    / "mimic-iv-clinical-database-demo-on-fhir-2.1.0"
    / "fhir"
)
DEFAULT_MODEL = Path("/opt/models/Qwen__Qwen3-30B-A3B-Instruct-2507")
DEFAULT_OUT = ROOT / "experiments" / "corrected_fhir_agentbench_qwen3_30b"
SMOKE_TEST_IDS = {
    "01038856e72fffdbf13016be",
    "0298d7de83dc23f17441f202",
    "04e0b332c01bf8b36c4973b5",
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_resources_by_patient_fhir_id",
            "description": "Retrieve real FHIR resources of one type for the task patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "enum": list(SUPPORTED_RESOURCE_TYPES),
                    },
                    "patient_fhir_id": {"type": "string"},
                },
                "required": ["resource_type", "patient_fhir_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_resources_by_resource_id",
            "description": "Follow a FHIR reference by resource type and logical id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "enum": list(SUPPORTED_RESOURCE_TYPES),
                    },
                    "resource_id": {"type": "string"},
                },
                "required": ["resource_type", "resource_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Analyze all resources retrieved so far. The variable "
                "retrieved_resources maps FHIR resource type to a list of full JSON resources."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]


SYSTEM_PROMPT = """You answer questions over real MIMIC-IV FHIR R4 resources.

Use the tools; never guess patient facts. You may retrieve several resource types,
follow references by id, and analyze retrieved_resources with Python. Common links:
MedicationRequest.medicationReference -> Medication; Observation.specimen ->
Specimen; encounter/partOf connect clinical events to hospital or ICU encounters.
Hospital encounters have an identifier system ending in encounter-hosp; ICU
encounters end in encounter-icu. Check codes, displays, values, units, and event
times rather than relying only on free text.

For every temporal, count, aggregate, comparison, first/last, or change question,
you MUST call execute_python_code before answering. Derive codes from the
retrieved code.coding displays; do not invent or hard-code a code that was not
observed in the returned resources. Apply every date condition in Context. For a
simple existence question about labs, procedures, diagnoses, or medications,
inspect the event resource and its timestamp directly; do not follow Specimen or
Encounter references unless the question actually asks about that linked object.
To filter procedures to a particular encounter: retrieve Encounter by patient,
then retrieve Procedure by the SAME patient, and compare each
Procedure.encounter.reference to the chosen Encounter id. Never use an Encounter
id as though it were a Procedure id. In focused mode, Medication references are
resolved automatically and appear under related_resources.

Keep Python concise: at most 8 lines, no comments, and leave the answer as the
last expression. The code environment provides resource_text(resource),
medication_text(request), reference_id(reference), and resource_times(resource),
in addition to retrieved_resources. resource_text is useful for case-insensitive matching;
it returns one plain string, not a dictionary. reference_id strips the
"ResourceType/" prefix. resource_times returns a list of datetime objects.
medication_text directly returns the resolved Medication text for a
MedicationRequest; use it for drug-name filtering.

Follow the benchmark's official encounter pattern. "Hospital encounter" means
an Encounter whose identifier.system contains "encounter-hosp"; do not select an
ED or ICU child encounter merely because it starts earlier. For example:
```python
encs = retrieved_resources['Encounter']
procs = retrieved_resources['Procedure']
hosp = [e for e in encs if any(
    'encounter-hosp' in i.get('system','').lower()
    for i in e.get('identifier', []))]
first = sorted(hosp, key=lambda e: resource_times(e)[0])[0]
ref = f"Encounter/{first['id']}"
matches = [p for p in procs if p.get('encounter',{}).get('reference') == ref]
```

This dataset represents most orders with MedicationRequest.medicationReference.
For example, count a named drug as follows:
```python
requests = retrieved_resources['MedicationRequest']
matches = [r for r in requests
           if 'milk of magnesia' in medication_text(r).strip().lower()
           and r.get('authoredOn','').startswith('2128-09')]
len(matches)
```

When done, emit exactly:
The final answer is: <JSON>

The JSON should match the requested answer only: for example [[1]], [[0]],
[[42]], [["2184-10-10 06:30:00"]], or [] for no answer. Do not include units or
explanation outside the JSON. A tool call and a final answer may not appear in
the same response."""


class LocalQwen:
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

    def generate(self, messages: list[dict[str, Any]]) -> tuple[str, int, float]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
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
        return text, int(generated.shape[0]), elapsed

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
                    "gpu_compute_capability": list(self.torch.cuda.get_device_capability(0)),
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
                    "gpu_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
                }
            )
        return metadata


def merge_resources(
    target: dict[str, list[dict[str, Any]]],
    resource_type: str,
    resources: list[dict[str, Any]],
) -> None:
    by_id = {str(resource.get("id")): resource for resource in target[resource_type]}
    for resource in resources:
        by_id[str(resource.get("id"))] = resource
    target[resource_type] = list(by_id.values())


def execute_agent(
    model: LocalQwen,
    store: LocalFhirStore,
    row: pd.Series,
    *,
    mode: str,
    max_rounds: int,
    model_limit: int,
    retrieval_limit: int,
) -> dict[str, Any]:
    question = str(row["question"])
    assumption = "" if pd.isna(row["assumption"]) else str(row["assumption"])
    patient_id = str(row["patient_fhir_id"])
    user_message = (
        f"Question: {question}\n"
        f"Context: {assumption or 'N/A'}\n"
        f"Patient FHIR ID: {patient_id}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    retrieved: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trace: list[dict[str, Any]] = []
    total_tokens = 0
    total_seconds = 0.0
    final_text = ""
    error = ""
    seen_calls: set[str] = set()

    for round_index in range(max_rounds):
        response, tokens, elapsed = model.generate(messages)
        total_tokens += tokens
        total_seconds += elapsed
        trace.append(
            {
                "round": round_index + 1,
                "role": "assistant",
                "content": response,
                "generated_tokens": tokens,
                "seconds": elapsed,
            }
        )
        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            final_text = extract_final_answer(response)
            break

        messages.append({"role": "assistant", "content": response})
        for tool_call in tool_calls:
            name = tool_call["name"]
            arguments = tool_call["arguments"]
            call_key = json.dumps(
                {"name": name, "arguments": arguments},
                sort_keys=True,
                ensure_ascii=False,
            )
            try:
                if call_key in seen_calls:
                    raise ValueError(
                        "Repeated tool call blocked; use the resources already retrieved "
                        "or choose a different operation."
                    )
                seen_calls.add(call_key)
                if name == "get_resources_by_patient_fhir_id":
                    requested_type = str(arguments["resource_type"])
                    requested_patient = str(arguments["patient_fhir_id"])
                    if requested_patient != patient_id:
                        raise ValueError("Cross-patient access denied")
                    result = store.get_by_patient(
                        requested_type,
                        requested_patient,
                        question=f"{question} {assumption}",
                        mode=mode,
                        model_limit=model_limit,
                        retrieval_limit=retrieval_limit,
                    )
                    merge_resources(retrieved, requested_type, result.resources)
                    related_resources: dict[str, list[dict[str, Any]]] = {}
                    if (
                        mode in {"focused", "reference_aware"}
                        and requested_type == "MedicationRequest"
                    ):
                        linked_medications = []
                        for request in result.resources:
                            reference = request.get("medicationReference", {}).get(
                                "reference", ""
                            )
                            if (
                                not isinstance(reference, str)
                                or not reference.startswith("Medication/")
                            ):
                                continue
                            medication_id = reference.split("/", 1)[1]
                            medication = store.by_type_id.get("Medication", {}).get(
                                medication_id
                            )
                            if medication is not None:
                                linked_medications.append(medication)
                        merge_resources(retrieved, "Medication", linked_medications)
                        related_resources["Medication"] = [
                            {
                                key: value
                                for key, value in medication.items()
                                if key
                                in {"resourceType", "id", "code", "identifier", "status"}
                            }
                            for medication in linked_medications[:model_limit]
                        ]
                    tool_output: Any = {
                        "resource_type": requested_type,
                        "retrieved_count": len(result.resources),
                        "resources": result.returned_to_model,
                        "related_resources": related_resources,
                        "notice": (
                            "Full retrieved resources are available to execute_python_code; "
                            "the response view is relevance-ranked and compact."
                        ),
                    }
                elif name == "get_resources_by_resource_id":
                    requested_type = str(arguments["resource_type"])
                    requested_id = str(arguments["resource_id"])
                    result = store.get_by_id(requested_type, requested_id)
                    merge_resources(retrieved, requested_type, result.resources)
                    tool_output = {
                        "resource_type": requested_type,
                        "retrieved_count": len(result.resources),
                        "resources": result.returned_to_model,
                    }
                elif name == "execute_python_code":
                    code = str(arguments["code"])
                    medication_requests = retrieved.get("MedicationRequest", [])
                    if (
                        "medicationCodeableConcept" in code
                        and medication_requests
                        and any(
                            request.get("medicationReference")
                            for request in medication_requests
                        )
                    ):
                        raise ValueError(
                            "Incomplete FHIR choice-element handling: at least one "
                            "MedicationRequest uses medicationReference. Code that only "
                            "checks medicationCodeableConcept can silently miss orders. "
                            "Resolve the linked Medication and use medication_text(request)."
                        )
                    # Use an ordinary dict so code that reads an unqueried resource
                    # type raises KeyError. Silently materializing an empty list would
                    # turn missing evidence into a false negative.
                    tool_output = run_safe_python(code, dict(retrieved))
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as exc:
                tool_output = {"error": f"{type(exc).__name__}: {exc}"}
            trace.append(
                {
                    "round": round_index + 1,
                    "role": "tool",
                    "name": name,
                    "arguments": arguments,
                    "content": tool_output,
                }
            )
            serialized = json.dumps(tool_output, ensure_ascii=False, default=str)
            messages.append({"role": "tool", "content": serialized[-60_000:]})
    else:
        error = "round_limit"

    if not final_text and not error:
        error = "empty_final_answer"
    return {
        "agent_answer": final_text,
        "retrieved_resources": dict(retrieved),
        "trace": trace,
        "generated_tokens": total_tokens,
        "latency_seconds": total_seconds,
        "error": error,
    }


def bootstrap_ci(values: list[float], seed: int, replicates: int = 2_000) -> tuple[float, float]:
    import random

    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        estimates.append(mean(rng.choice(clean) for _ in range(len(clean))))
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]


def score_prediction(row: pd.Series, prediction: dict[str, Any], mode: str) -> dict[str, Any]:
    gold_by_type = ast.literal_eval(str(row["true_fhir_ids"]))
    gold_ids = [resource_id for ids in gold_by_type.values() for resource_id in ids]
    predicted_ids = resource_ids(prediction["retrieved_resources"], set(gold_by_type))
    return {
        "question_id": row["question_id"],
        "split": row["split"],
        "system": mode,
        "main_table_name": row["main_table_name"],
        "template": row["template"],
        "answer_correct": int(
            not prediction["error"]
            and canonical_answer_correct(prediction["agent_answer"], row["true_answer"])
        ),
        "answer_parseable": int(
            isinstance(parse_structured_answer(prediction["agent_answer"]), (list, tuple))
        ),
        "retrieval_precision": retrieval_precision(predicted_ids, gold_ids),
        "retrieval_recall": retrieval_recall(predicted_ids, gold_ids),
        "n_gold_resources": len(set(gold_ids)),
        "n_retrieved_relevant_type_resources": len(set(predicted_ids)),
        "generated_tokens": prediction["generated_tokens"],
        "latency_seconds": prediction["latency_seconds"],
        "error": prediction["error"],
    }


def summarize(scores: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    summaries = []
    for system, frame in scores.groupby("system", sort=True):
        answer = frame["answer_correct"].astype(float).tolist()
        precision = frame["retrieval_precision"].astype(float).tolist()
        recall = frame["retrieval_recall"].astype(float).tolist()
        answer_ci = bootstrap_ci(answer, seed + 1)
        precision_ci = bootstrap_ci(precision, seed + 2)
        recall_ci = bootstrap_ci(recall, seed + 3)
        summaries.append(
            {
                "benchmark": "FHIR-AgentBench",
                "split": frame["split"].iloc[0],
                "system": system,
                "n": len(frame),
                "canonical_answer_accuracy": mean(answer),
                "answer_ci_low": answer_ci[0],
                "answer_ci_high": answer_ci[1],
                "retrieval_precision": mean(value for value in precision if not math.isnan(value)),
                "precision_ci_low": precision_ci[0],
                "precision_ci_high": precision_ci[1],
                "retrieval_recall": mean(value for value in recall if not math.isnan(value)),
                "recall_ci_low": recall_ci[0],
                "recall_ci_high": recall_ci[1],
                "parse_rate": frame["answer_parseable"].mean(),
                "error_rate": (frame["error"] != "").mean(),
                "mean_generated_tokens": frame["generated_tokens"].mean(),
                "mean_latency_seconds": frame["latency_seconds"].mean(),
                "median_latency_seconds": frame["latency_seconds"].median(),
                "p95_latency_seconds": frame["latency_seconds"].quantile(0.95),
            }
        )
    return summaries


def paired_bootstrap_ci(
    differences: list[float], seed: int, replicates: int = 2_000
) -> tuple[float, float]:
    import random

    clean = [value for value in differences if not math.isnan(value)]
    if not clean:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(clean) for _ in range(len(clean))) for _ in range(replicates)
    ]
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]


def exact_mcnemar_p(broad_only_correct: int, focused_only_correct: int) -> float:
    discordant = broad_only_correct + focused_only_correct
    if discordant == 0:
        return 1.0
    tail = min(broad_only_correct, focused_only_correct)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def paired_comparison(scores: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    systems = set(scores["system"])
    if systems == {"broad", "focused"}:
        comparison_system = "focused"
    elif systems == {"broad", "reference_aware"}:
        comparison_system = "reference_aware"
    else:
        return []
    wide = scores.pivot(index="question_id", columns="system")
    answer_broad = wide["answer_correct"]["broad"].astype(float)
    answer_comparison = wide["answer_correct"][comparison_system].astype(float)
    broad_only = int(((answer_broad == 1) & (answer_comparison == 0)).sum())
    comparison_only = int(((answer_broad == 0) & (answer_comparison == 1)).sum())
    rows = []
    for offset, metric in enumerate(
        ("answer_correct", "retrieval_precision", "retrieval_recall")
    ):
        differences = (
            wide[metric][comparison_system].astype(float)
            - wide[metric]["broad"].astype(float)
        ).tolist()
        clean = [value for value in differences if not math.isnan(value)]
        ci_low, ci_high = paired_bootstrap_ci(clean, seed + offset)
        row = {
            "metric": metric,
            "baseline_system": "broad",
            "comparison_system": comparison_system,
            "n_paired": len(clean),
            "comparison_minus_baseline": mean(clean),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        if metric == "answer_correct":
            row.update(
                {
                    "broad_only_correct": broad_only,
                    "comparison_only_correct": comparison_only,
                    "exact_mcnemar_p": exact_mcnemar_p(
                        broad_only, comparison_only
                    ),
                }
            )
        rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_model_revision(model_path: Path) -> str:
    tree_dir = model_path / ".cache" / "huggingface" / "trees"
    tree_files = sorted(tree_dir.glob("*.json")) if tree_dir.exists() else []
    if len(tree_files) == 1:
        return tree_files[0].stem
    metadata_path = model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata_path.exists():
        first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if re.fullmatch(r"[0-9a-f]{40}", first_line):
            return first_line
    return "unknown-local-snapshot"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--fhir-dir", type=Path, default=DEFAULT_FHIR_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("broad", "focused", "reference_aware"),
        default=["broad", "focused"],
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--skip-tasks",
        type=int,
        default=0,
        help="Skip this many question-id-sorted tasks before applying --max-tasks.",
    )
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--model-limit", type=int, default=8)
    parser.add_argument("--retrieval-limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} already exists; use --resume or a new output directory"
        )
    source = pd.read_csv(args.csv)
    tasks = source[source["split"] == args.split].sort_values("question_id")
    if args.skip_tasks:
        tasks = tasks.iloc[args.skip_tasks :]
    if args.max_tasks:
        tasks = tasks.head(args.max_tasks)
    if tasks.empty:
        raise ValueError("No selected benchmark tasks")

    print(f"Loading official FHIR store from {args.fhir_dir}", flush=True)
    store_started = time.perf_counter()
    store = LocalFhirStore(args.fhir_dir)
    print(
        f"Loaded {sum(store.counts.values()):,} resources in "
        f"{time.perf_counter() - store_started:.1f}s",
        flush=True,
    )
    print(f"Loading model {args.model}", flush=True)
    model = LocalQwen(args.model, args.max_new_tokens)

    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing[(record["question_id"], record["system"])] = record

    predictions: list[dict[str, Any]] = list(existing.values())
    for system in args.systems:
        for ordinal, (_, row) in enumerate(tasks.iterrows(), start=1):
            key = (str(row["question_id"]), system)
            if key in existing:
                continue
            print(
                f"[{system}] {ordinal}/{len(tasks)} {row['question_id']}",
                flush=True,
            )
            result = execute_agent(
                model,
                store,
                row,
                mode=system,
                max_rounds=args.max_rounds,
                model_limit=args.model_limit,
                retrieval_limit=args.retrieval_limit,
            )
            record = {
                "question_id": str(row["question_id"]),
                "system": system,
                "question": str(row["question"]),
                "assumption": "" if pd.isna(row["assumption"]) else str(row["assumption"]),
                "patient_fhir_id": str(row["patient_fhir_id"]),
                "true_answer": str(row["true_answer"]),
                "true_fhir_ids": str(row["true_fhir_ids"]),
                "agent_answer": result["agent_answer"],
                "retrieved_fhir_ids": {
                    resource_type: [resource.get("id") for resource in resources]
                    for resource_type, resources in result["retrieved_resources"].items()
                },
                "trace": result["trace"],
                "generated_tokens": result["generated_tokens"],
                "latency_seconds": result["latency_seconds"],
                "error": result["error"],
            }
            predictions.append(record)
            # Commit each completed trajectory immediately. Appending avoids
            # repeatedly rewriting a large trace file and keeps --resume
            # recoverable after an interrupted long GPU run.
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_by_id = source.set_index("question_id", drop=False)
    score_rows = []
    for prediction in predictions:
        if prediction["system"] not in args.systems:
            continue
        row = source_by_id.loc[prediction["question_id"]]
        reconstructed = {
            "agent_answer": prediction["agent_answer"],
            "retrieved_resources": {
                resource_type: [
                    store.by_type_id[resource_type][resource_id]
                    for resource_id in ids
                    if resource_id in store.by_type_id.get(resource_type, {})
                ]
                for resource_type, ids in prediction["retrieved_fhir_ids"].items()
            },
            "generated_tokens": prediction["generated_tokens"],
            "latency_seconds": prediction["latency_seconds"],
            "error": prediction["error"],
        }
        score_rows.append(score_prediction(row, reconstructed, prediction["system"]))

    scores = pd.DataFrame(score_rows)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    summaries = summarize(scores, args.seed)
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(paired_comparison(scores, args.seed)).to_csv(
        args.output_dir / "paired_comparison.csv", index=False
    )
    if args.split == "test":
        sensitivity_scores = scores[
            ~scores["question_id"].astype(str).isin(SMOKE_TEST_IDS)
        ]
        pd.DataFrame(summarize(sensitivity_scores, args.seed)).to_csv(
            args.output_dir / "summary_excluding_smoke_test_ids.csv", index=False
        )
        pd.DataFrame(paired_comparison(sensitivity_scores, args.seed)).to_csv(
            args.output_dir / "paired_comparison_excluding_smoke_test_ids.csv",
            index=False,
        )

    metadata = {
        "benchmark": "FHIR-AgentBench",
        "benchmark_commit": "bbb42909a5a7eb907d1cd91f72a560729e7037ea",
        "benchmark_csv_sha256": sha256(args.csv),
        "fhir_dataset": "MIMIC-IV Clinical Database Demo on FHIR v2.1.0",
        "fhir_resource_count": sum(store.counts.values()),
        "fhir_resource_counts_by_type": dict(sorted(store.counts.items())),
        "fhir_manifest_verified": True,
        "split": args.split,
        "n_tasks": len(tasks),
        "skip_tasks": args.skip_tasks,
        "systems": args.systems,
        "model_path": str(args.model),
        "model_revision": local_model_revision(args.model),
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "max_rounds": args.max_rounds,
        "model_limit": args.model_limit,
        "focused_retrieval_limit": args.retrieval_limit,
        "seed": args.seed,
        "protocol_frozen_after_validation": "reference_aware" not in args.systems,
        "protocol_timing": (
            "post-test corrective evaluation on disjoint validation holdout"
            if "reference_aware" in args.systems
            else "formal protocol frozen after validation development"
        ),
        "test_sensitivity_exclusion_ids": sorted(SMOKE_TEST_IDS),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "runtime": model.runtime_metadata(),
        "scoring": {
            "retrieval": "released resource IDs; official NaN conventions",
            "answer": "deterministic canonical structured-answer equivalence",
            "official_o4_mini_judge_used": False,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
