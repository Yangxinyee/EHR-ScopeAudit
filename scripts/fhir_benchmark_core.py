#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Core utilities for faithful local FHIR-AgentBench experiments.

The module deliberately keeps benchmark-native concepts separate:

* retrieval is scored against the released FHIR resource identifiers;
* answers are scored against the released structured SQL answers;
* no synthetic scope or auditability score is inferred from task metadata.

The local store reads the official MIMIC-IV-on-FHIR demo NDJSON files.  It
implements the same two resource-access operations exposed by the public
FHIR-AgentBench agents, while avoiding a dependency on a particular cloud
FHIR server.
"""

from __future__ import annotations

import ast
import contextlib
import gzip
import io
import json
import math
import re
import signal
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_RESOURCE_TYPES = (
    "Patient",
    "Encounter",
    "Condition",
    "MedicationRequest",
    "Procedure",
    "Observation",
    "MedicationAdministration",
    "Location",
    "Specimen",
    "Medication",
)

SAFE_IMPORTS = {"collections", "datetime", "json", "math", "re", "statistics"}
BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
PYTHON_CALL_RE = re.compile(
    r"execute_python_code\(\s*(?:\"\"\"|''')(?P<code>.*?)(?:\"\"\"|''')\s*\)",
    re.DOTALL,
)
PYTHON_STRING_CALL_RE = re.compile(
    r"execute_python_code\(\s*(?P<quote>[\"'])(?P<code>.*)(?P=quote)\s*\)",
    re.DOTALL,
)
FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+final\s+answer\s+is|final\s+answer)\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _reference_id(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    reference = value.get("reference")
    if not isinstance(reference, str) or "/" not in reference:
        return None
    return reference.split("/", 1)[1]


def patient_id_for_resource(resource: dict[str, Any]) -> str | None:
    """Return the patient FHIR id referenced by a resource, when present."""
    if resource.get("resourceType") == "Patient":
        return str(resource.get("id", "")) or None
    for field in ("subject", "patient", "beneficiary"):
        patient_id = _reference_id(resource.get(field))
        if patient_id:
            return patient_id
    return None


def scalar_strings(value: Any) -> Iterable[str]:
    """Yield searchable scalar strings without exposing object keys as terms."""
    if isinstance(value, dict):
        for child in value.values():
            yield from scalar_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from scalar_strings(child)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield str(value)


def lexical_tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1}


def resource_times(resource: dict[str, Any]) -> list[datetime]:
    """Extract common FHIR event timestamps as timezone-aware UTC datetimes."""
    candidates: list[Any] = []
    for key in (
        "authoredOn",
        "birthDate",
        "dateAsserted",
        "effectiveDateTime",
        "issued",
        "occurrenceDateTime",
        "performedDateTime",
        "recordedDate",
        "whenHandedOver",
    ):
        candidates.append(resource.get(key))
    for key in (
        "effectivePeriod",
        "occurrencePeriod",
        "performedPeriod",
        "period",
    ):
        period = resource.get(key)
        if isinstance(period, dict):
            candidates.extend((period.get("start"), period.get("end")))
    collection = resource.get("collection")
    if isinstance(collection, dict):
        candidates.append(collection.get("collectedDateTime"))

    parsed: list[datetime] = []
    for value in candidates:
        if not isinstance(value, str):
            continue
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            parsed.append(dt.astimezone(timezone.utc))
        except ValueError:
            continue
    return parsed


def compact_resource(resource: dict[str, Any]) -> dict[str, Any]:
    """Prune fields that are large and unhelpful while preserving FHIR semantics."""
    keep = {
        "authoredOn",
        "birthDate",
        "category",
        "class",
        "code",
        "component",
        "context",
        "dateAsserted",
        "dispenseRequest",
        "dosage",
        "dosageInstruction",
        "effectiveDateTime",
        "effectivePeriod",
        "encounter",
        "gender",
        "id",
        "identifier",
        "location",
        "medicationCodeableConcept",
        "medicationReference",
        "name",
        "occurrenceDateTime",
        "occurrencePeriod",
        "partOf",
        "performedDateTime",
        "performedPeriod",
        "period",
        "resourceType",
        "route",
        "specimen",
        "status",
        "subject",
        "type",
        "valueCodeableConcept",
        "valueDateTime",
        "valueInteger",
        "valueQuantity",
        "valueString",
        "whenHandedOver",
    }
    return {key: value for key, value in resource.items() if key in keep}


@dataclass
class SearchResult:
    resources: list[dict[str, Any]]
    returned_to_model: list[dict[str, Any]]


class LocalFhirStore:
    """In-memory index over official compressed FHIR NDJSON resources."""

    def __init__(self, fhir_dir: Path):
        self.fhir_dir = Path(fhir_dir)
        self.by_type_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.by_patient_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)
        self._load()

    def _load(self) -> None:
        paths = sorted(self.fhir_dir.glob("*.ndjson.gz"))
        if not paths:
            raise FileNotFoundError(f"No *.ndjson.gz files under {self.fhir_dir}")
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    resource = json.loads(line)
                    resource_type = str(resource["resourceType"])
                    resource_id = str(resource["id"])
                    self.by_type_id[resource_type][resource_id] = resource
                    self.counts[resource_type] += 1
                    patient_id = patient_id_for_resource(resource)
                    if patient_id:
                        self.by_patient_type[(patient_id, resource_type)].append(resource)

    def get_by_id(self, resource_type: str, resource_id: str) -> SearchResult:
        resource = self.by_type_id.get(resource_type, {}).get(resource_id)
        resources = [resource] if resource is not None else []
        return SearchResult(resources=resources, returned_to_model=[compact_resource(r) for r in resources])

    def get_by_patient(
        self,
        resource_type: str,
        patient_id: str,
        *,
        question: str,
        mode: str,
        model_limit: int,
        retrieval_limit: int,
    ) -> SearchResult:
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            raise ValueError(f"Unsupported FHIR resource type: {resource_type}")
        resources = list(self.by_patient_type.get((patient_id, resource_type), []))
        if mode == "focused":
            resources = self._rank(resources, question)[:retrieval_limit]
        elif mode not in {"broad", "reference_aware"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        model_view = self._rank(resources, question)[:model_limit]
        return SearchResult(
            resources=resources,
            returned_to_model=[compact_resource(resource) for resource in model_view],
        )

    def _rank(self, resources: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
        query_tokens = lexical_tokens(question)

        def score(resource: dict[str, Any]) -> tuple[float, str]:
            searchable = list(scalar_strings(compact_resource(resource)))
            # MedicationRequest usually stores only a reference; resolving that
            # reference is necessary for lexical search by medication name.
            medication_id = _reference_id(resource.get("medicationReference"))
            if medication_id:
                medication = self.by_type_id.get("Medication", {}).get(medication_id)
                if medication:
                    searchable.extend(scalar_strings(compact_resource(medication)))
            text_tokens = lexical_tokens(" ".join(searchable))
            overlap = len(query_tokens & text_tokens)
            coverage = overlap / max(1, len(query_tokens))
            return (-coverage, str(resource.get("id", "")))

        return sorted(resources, key=score)


class UnsafeCodeError(ValueError):
    pass


class _CodeGuard(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] not in SAFE_IMPORTS:
                raise UnsafeCodeError(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module or node.module.split(".", 1)[0] not in SAFE_IMPORTS:
            raise UnsafeCodeError(f"Import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise UnsafeCodeError(f"Private attribute not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise UnsafeCodeError(f"Call not allowed: {node.func.id}")
        self.generic_visit(node)


def run_safe_python(code: str, retrieved_resources: dict[str, list[dict[str, Any]]]) -> str:
    """Execute model-generated analysis code with an AST allowlist and timeout."""
    tree = ast.parse(code, mode="exec")
    _CodeGuard().visit(tree)
    # Models often leave the desired value as the final notebook-style
    # expression instead of explicitly printing it. Preserve that expression
    # and report its repr without changing the analysis itself.
    result_name = "__clinehr_result"
    final_is_print = bool(
        tree.body
        and isinstance(tree.body[-1], ast.Expr)
        and isinstance(tree.body[-1].value, ast.Call)
        and isinstance(tree.body[-1].value.func, ast.Name)
        and tree.body[-1].value.func.id == "print"
    )
    has_result_expression = bool(
        tree.body and isinstance(tree.body[-1], ast.Expr) and not final_is_print
    )
    if has_result_expression:
        final_expression = tree.body[-1]
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=final_expression.value,
        )
        ast.fix_missing_locations(tree)
    compiled = compile(tree, "<fhir-agent-analysis>", "exec")
    output = io.StringIO()
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": __import__,
    }

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Analysis code exceeded 10 seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        def plain_resource_text(resource: dict[str, Any]) -> str:
            return " ".join(scalar_strings(resource))

        def medication_text(request: dict[str, Any]) -> str:
            medication_id = _reference_id(request.get("medicationReference"))
            if not medication_id:
                return ""
            for medication in retrieved_resources.get("Medication", []):
                if str(medication.get("id")) == medication_id:
                    return plain_resource_text(medication)
            return ""

        def resource_text(resource: dict[str, Any]) -> str:
            text = plain_resource_text(resource)
            if resource.get("resourceType") == "MedicationRequest":
                linked_text = medication_text(resource)
                if linked_text:
                    text = f"{text} {linked_text}"
            return text

        local_scope = {
            "__builtins__": safe_builtins,
            "retrieved_resources": retrieved_resources,
            "resource_text": resource_text,
            "medication_text": medication_text,
            "reference_id": _reference_id,
            "resource_times": resource_times,
        }
        with contextlib.redirect_stdout(output):
            # A shared globals/locals namespace is required so variables created
            # by one statement are visible inside later comprehensions.
            exec(compiled, local_scope, local_scope)
            if has_result_expression:
                print(repr(local_scope.get(result_name)))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    text = output.getvalue().strip()
    return text[-12_000:] if text else "(code executed without printed output)"


def _normalize_tool_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"name": payload["name"], "arguments": arguments}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse every Qwen tool call in one assistant turn, preserving order."""
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        python_match = PYTHON_CALL_RE.search(text)
        if python_match:
            return [{
                "name": "execute_python_code",
                "arguments": {"code": python_match.group("code").strip()},
            }]
        string_match = PYTHON_STRING_CALL_RE.search(text)
        if string_match:
            code = string_match.group("code").strip()
            code = code.replace(r"\"", '"').replace(r"\'", "'").replace(r"\n", "\n")
            return [{
                "name": "execute_python_code",
                "arguments": {"code": code},
            }]
        return []
    calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        call = _normalize_tool_payload(payload)
        if call is not None:
            calls.append(call)
    return calls


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Backward-compatible single-call accessor used by unit tests and callers."""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_RE.search(text)
    answer = match.group(1).strip() if match else text.strip()
    answer = answer.split("<|im_end|>", 1)[0].strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.strip("`").strip()
    return answer


def parse_structured_answer(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    bracket = re.search(r"(\[\s*.*\s*\])", text, re.DOTALL)
    if bracket:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(bracket.group(1))
            except Exception:
                continue
        # Released SQL answers are sometimes serialized from NumPy without
        # commas between rows, e.g. ``[[67.]\n [42.]]``. The benchmark scorer
        # ultimately compares flattened values, so recover those scalars.
        tokens = re.findall(
            r"""(?:"([^"]*)"|'([^']*)'|([-+]?\d+(?:\.\d*)?)|\b(None|null|true|false)\b)""",
            bracket.group(1),
            flags=re.IGNORECASE,
        )
        recovered: list[Any] = []
        for double_quoted, single_quoted, numeric, literal in tokens:
            if double_quoted or single_quoted:
                recovered.append(double_quoted or single_quoted)
            elif numeric:
                recovered.append(float(numeric) if "." in numeric else int(numeric))
            elif literal.casefold() in {"none", "null"}:
                recovered.append(None)
            else:
                recovered.append(literal.casefold() == "true")
        if recovered:
            return recovered
    return text


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        output: list[Any] = []
        for child in value:
            output.extend(_flatten(child))
        return output
    return [value]


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*(?:[A-Za-z/%\[\]]+)?\s*", value)
        if match:
            return float(match.group(1))
    return None


def _datetime_value(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed


def _scalar_equal(predicted: Any, gold: Any) -> bool:
    pred_num = _numeric(predicted)
    gold_num = _numeric(gold)
    if pred_num is not None and gold_num is not None:
        return round(pred_num) == round(gold_num)
    pred_date = _datetime_value(predicted)
    gold_date = _datetime_value(gold)
    if pred_date is not None and gold_date is not None:
        if pred_date.tzinfo is None or gold_date.tzinfo is None:
            # SQL gold timestamps in the released benchmark are naive local
            # wall-clock times, whereas FHIR timestamps carry offsets.
            pred_date = pred_date.replace(tzinfo=None)
            gold_date = gold_date.replace(tzinfo=None)
        else:
            pred_date = pred_date.astimezone(timezone.utc)
            gold_date = gold_date.astimezone(timezone.utc)
        return abs((pred_date - gold_date).total_seconds()) <= 60
    pred_text = re.sub(r"\s+", " ", str(predicted)).strip().casefold()
    gold_text = re.sub(r"\s+", " ", str(gold)).strip().casefold()
    aliases = {
        "female": "f",
        "male": "m",
        "true": "1",
        "yes": "1",
        "false": "0",
        "no": "0",
    }
    return aliases.get(pred_text, pred_text) == aliases.get(gold_text, gold_text)


def canonical_answer_correct(predicted: Any, gold: Any) -> bool:
    """Deterministic structured-answer scorer modeled on the official rubric."""
    predicted = parse_structured_answer(predicted)
    gold = parse_structured_answer(gold)
    pred_values = _flatten(predicted)
    gold_values = _flatten(gold)

    no_answer = {"", "[]", "none", "null", "no answer", "cannot find", "not found"}
    if not gold_values or all(str(value).strip().casefold() in no_answer for value in gold_values):
        return not pred_values or all(
            str(value).strip().casefold() in no_answer for value in pred_values
        )
    if len(pred_values) != len(gold_values):
        return False

    unmatched = list(pred_values)
    for gold_value in gold_values:
        for index, predicted_value in enumerate(unmatched):
            if _scalar_equal(predicted_value, gold_value):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def resource_ids(resources: dict[str, list[dict[str, Any]]], relevant_types: set[str]) -> list[str]:
    return [
        str(resource.get("id"))
        for resource_type, typed_resources in resources.items()
        if resource_type in relevant_types
        for resource in typed_resources
        if resource.get("id")
    ]


def retrieval_precision(predicted: list[str], gold: list[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold and predicted:
        return 0.0
    if gold and not predicted:
        return math.nan
    gold_set = set(gold)
    return sum(resource_id in gold_set for resource_id in set(predicted)) / len(set(predicted))


def retrieval_recall(predicted: list[str], gold: list[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold and predicted:
        return math.nan
    if gold and not predicted:
        return 0.0
    predicted_set = set(predicted)
    return sum(resource_id in predicted_set for resource_id in set(gold)) / len(set(gold))
