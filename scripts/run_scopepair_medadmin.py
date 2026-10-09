#!/usr/bin/env python3
"""Run CUDA-only models on ScopePair-MedAdmin."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

try:
    from run_scopepair_fhir import (
        CudaOnlyModel,
        extract_json,
        local_model_revision,
        parse_time,
        sha256,
        text_sha256,
    )
except ModuleNotFoundError:  # Imported as scripts.run_scopepair_medadmin in tests.
    from scripts.run_scopepair_fhir import (
        CudaOnlyModel,
        extract_json,
        local_model_revision,
        parse_time,
        sha256,
        text_sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "experiments" / "scopepair_medadmin" / "tasks.jsonl"
DEFAULT_MODEL = Path("/opt/models/Qwen__Qwen3-4B-Instruct-2507")
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_medadmin_dev_qwen3_4b"
SYSTEMS = (
    "unscoped",
    "late_no_refill",
    "scope_first",
    "prompt_all6",
    "guard_refill",
    "clinehr_os",
)

SYSTEM_PROMPT = """You are an EHR evidence extraction agent. The task contract is binding.

The evidence field `event_time` is when a medication administration began. Use
only the supplied evidence. If exactly three supplied administrations satisfy
the requested strict pre/post contract, return them in chronological
`event_time` order and calculate both total_dose and mean_dose over those three
dose values. If fewer than three supplied administrations satisfy the contract,
return status `insufficient`, an empty results list, null total_dose, and null
mean_dose. Never invent, replace, or cite an administration that is not supplied.

Return exactly one JSON object and no prose:
{"status":"ok|insufficient","results":[{"citation":"id","dose":0.0,
"unit":"unit","event_time":"ISO timestamp"}],"total_dose":0.0,
"mean_dose":0.0}"""


def permitted(item: dict[str, Any], task: dict[str, Any]) -> bool:
    timestamp = parse_time(str(item["event_time"]))
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
    require_permitted: bool,
    k: int = 3,
) -> list[dict[str, Any]]:
    index = parse_time(str(task["index_time"]))
    pool = [
        item
        for item in candidates
        if not require_permitted or permitted(item, task)
    ]
    chosen = sorted(
        pool,
        key=lambda item: (
            abs((parse_time(str(item["event_time"])) - index).total_seconds()),
            str(item["id"]),
        ),
    )[:k]
    return sorted(
        chosen,
        key=lambda item: (parse_time(str(item["event_time"])), str(item["id"])),
    )


def select_evidence(task: dict[str, Any], system: str) -> list[dict[str, Any]]:
    candidates = list(task["candidates"])
    unscoped = nearest(candidates, task, require_permitted=False)
    if system == "unscoped":
        return unscoped
    if system == "late_no_refill":
        return [item for item in unscoped if permitted(item, task)]
    if system == "prompt_all6":
        return sorted(
            candidates,
            key=lambda item: (
                parse_time(str(item["event_time"])),
                str(item["id"]),
            ),
        )
    scoped = nearest(candidates, task, require_permitted=True)
    if system in {"scope_first", "clinehr_os"}:
        return scoped
    if system == "guard_refill":
        legal = [item for item in unscoped if permitted(item, task)]
        seen = {str(item["id"]) for item in legal}
        for item in nearest(
            candidates, task, require_permitted=True, k=len(candidates)
        ):
            if len(legal) >= 3:
                break
            if str(item["id"]) not in seen:
                legal.append(item)
                seen.add(str(item["id"]))
        legal.sort(
            key=lambda item: (
                parse_time(str(item["event_time"])),
                str(item["id"]),
            )
        )
        if [item["id"] for item in legal] != [item["id"] for item in scoped]:
            raise AssertionError("Guard refill and scope-first evidence differ")
        return legal
    raise ValueError(f"Unknown system: {system}")


def skill_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if len(evidence) != 3:
        return {"status": "insufficient", "provenance": []}
    total = math.fsum(float(item["dose"]) for item in evidence)
    return {
        "status": "ok",
        "ordered_administration_ids": [item["id"] for item in evidence],
        "total_dose": total,
        "mean_dose": total / 3,
        "provenance": [
            {
                "citation": item["id"],
                "dose": item["dose"],
                "unit": item["unit"],
                "event_time": item["event_time"],
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
            "total_dose": None,
            "mean_dose": None,
        }
    return {
        "status": "ok",
        "results": summary["provenance"],
        "total_dose": summary["total_dose"],
        "mean_dose": summary["mean_dose"],
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
            "timestamp_basis": "event_time",
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


def cited_ids(parsed: dict[str, Any] | None) -> list[str]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        return []
    return [
        item["citation"]
        for item in parsed["results"]
        if isinstance(item, dict) and isinstance(item.get("citation"), str)
    ]


def close_number(actual: Any, expected: Any) -> bool:
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-7
        )
    except (TypeError, ValueError):
        return False


def guard_errors(
    parsed: dict[str, Any] | None,
    task: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    validate_skill: bool,
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
        if permitted(candidate, task)
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
    legal_supplied = [item for item in evidence if permitted(item, task)]
    if status == "insufficient" and len(legal_supplied) >= 3:
        errors.append("three permitted supplied administrations exist; do not abstain")
    if not validate_skill or status != "ok" or len(legal_supplied) != 3:
        return errors
    expected = skill_summary(legal_supplied)
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
            if not close_number(observed.get("dose"), gold["dose"]):
                errors.append(f"result {index} dose disagrees with ChartSkill")
            try:
                time_match = parse_time(
                    str(observed.get("event_time"))
                ) == parse_time(str(gold["event_time"]))
            except (TypeError, ValueError):
                time_match = False
            if not time_match:
                errors.append(f"result {index} event time disagrees with ChartSkill")
    if not close_number(parsed.get("total_dose"), expected["total_dose"]):
        errors.append("total dose disagrees with deterministic ChartSkill")
    if not close_number(parsed.get("mean_dose"), expected["mean_dose"]):
        errors.append("mean dose disagrees with deterministic ChartSkill")
    return errors


def run_task(
    model: CudaOnlyModel, task: dict[str, Any], system: str
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
            list(task["candidates"]), task, require_permitted=True
        )
        repair_payload: dict[str, Any] = {
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
        list(task["candidates"]), task, require_permitted=True
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
            parsed, task, final_evidence, validate_skill=True
        )
        if final_errors:
            raise AssertionError(
                f"Verifier-generated answer failed validation: {final_errors}"
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


def load_tasks(
    path: Path, split: str, max_tasks: int | None
) -> list[dict[str, Any]]:
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
            f"{predictions_path} exists; pass --resume or use another directory"
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
            if (task["task_id"], system) in existing:
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
                "task_family": task["task_family"],
                "split": task["split"],
                "scope": task["scope"],
                "dose_variation": task["dose_variation"],
                "system": system,
                **result,
            }
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "benchmark": "ScopePair-MedAdmin",
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
