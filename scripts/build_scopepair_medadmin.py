#!/usr/bin/env python3
"""Build the prospective ScopePair medication-administration benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FHIR_DIR = (
    ROOT.parent
    / "public_data"
    / "mimic-iv-clinical-database-demo-on-fhir-2.1.0"
    / "fhir"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_medadmin"
ADMIN_SOURCES = (
    "MimicMedicationAdministration.ndjson.gz",
    "MimicMedicationAdministrationICU.ndjson.gz",
)
MEDICATION_SOURCE = "MimicMedication.ndjson.gz"
ORACLE_VERSION = "scopepair-medadmin-oracle-v1"


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hex(*parts: object, seed: int) -> str:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_id(reference: Any) -> str | None:
    if not isinstance(reference, dict):
        return None
    value = reference.get("reference")
    if not isinstance(value, str) or "/" not in value:
        return None
    return value.split("/", 1)[1]


def first_coding(resource: dict[str, Any]) -> dict[str, Any] | None:
    coding = resource.get("medicationCodeableConcept", {}).get("coding", [])
    if not isinstance(coding, list) or not coding:
        return None
    return coding[0] if isinstance(coding[0], dict) else None


def medication_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            resource = json.loads(line)
            formulary = None
            name = None
            for identifier in resource.get("identifier", []):
                if not isinstance(identifier, dict):
                    continue
                system = str(identifier.get("system", ""))
                value = identifier.get("value")
                if not isinstance(value, str):
                    continue
                if system.endswith("medication-formulary-drug-cd"):
                    formulary = value
                elif system.endswith("medication-name"):
                    name = value
            if formulary and name and formulary not in names:
                names[formulary] = name
    return names


@dataclass(frozen=True)
class Administration:
    resource_id: str
    patient_id: str
    medication_key: str
    medication_code: str
    medication_display: str
    dose: float
    unit: str
    event_time: datetime
    source: str

    def compact(self) -> dict[str, Any]:
        return {
            "id": self.resource_id,
            "medication_code": self.medication_code,
            "medication_display": self.medication_display,
            "dose": self.dose,
            "unit": self.unit,
            "event_time": iso_utc(self.event_time),
            "source": self.source,
        }


@dataclass(frozen=True)
class Case:
    patient_id: str
    medication_key: str
    medication_code: str
    medication_display: str
    unit: str
    index_time: datetime
    candidates: tuple[Administration, ...]
    dose_variation: bool


def load_administrations(
    fhir_dir: Path,
) -> tuple[dict[tuple[str, str, str], list[Administration]], dict[str, int]]:
    name_map = medication_names(fhir_dir / MEDICATION_SOURCE)
    groups: dict[tuple[str, str, str], list[Administration]] = defaultdict(list)
    counts = {"valid_administrations": 0}
    for filename in ADMIN_SOURCES:
        source_count = 0
        with gzip.open(fhir_dir / filename, "rt", encoding="utf-8") as handle:
            for line in handle:
                resource = json.loads(line)
                if resource.get("status") != "completed":
                    continue
                patient_id = reference_id(resource.get("subject"))
                coding = first_coding(resource)
                dose = resource.get("dosage", {}).get("dose", {})
                value = dose.get("value")
                unit = dose.get("unit") or dose.get("code")
                event_raw = resource.get("effectiveDateTime")
                if event_raw is None:
                    event_raw = resource.get("effectivePeriod", {}).get("start")
                if (
                    not patient_id
                    or coding is None
                    or not coding.get("code")
                    or not coding.get("system")
                    or not unit
                    or not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    or not isinstance(event_raw, str)
                    or not resource.get("id")
                ):
                    continue
                try:
                    event_time = parse_time(event_raw)
                except ValueError:
                    continue
                code = str(coding["code"])
                system = str(coding["system"])
                display = str(coding.get("display") or name_map.get(code) or code)
                medication_key = f"{system}|{code}"
                item = Administration(
                    resource_id=str(resource["id"]),
                    patient_id=patient_id,
                    medication_key=medication_key,
                    medication_code=code,
                    medication_display=display,
                    dose=float(value),
                    unit=str(unit),
                    event_time=event_time,
                    source=filename,
                )
                groups[(patient_id, medication_key, str(unit))].append(item)
                source_count += 1
        counts[filename] = source_count
        counts["valid_administrations"] += source_count
    for items in groups.values():
        items.sort(key=lambda item: (item.event_time, item.resource_id))
    return groups, counts


def permitted(item: Administration, index: datetime, scope: str) -> bool:
    if scope == "pre":
        return item.event_time < index
    if scope == "post":
        return item.event_time > index
    raise ValueError(f"Unknown scope: {scope}")


def select_nearest(
    candidates: Iterable[Administration],
    index: datetime,
    scope: str,
    k: int = 3,
) -> list[Administration]:
    legal = [item for item in candidates if permitted(item, index, scope)]
    nearest = sorted(
        legal,
        key=lambda item: (
            abs((item.event_time - index).total_seconds()),
            item.resource_id,
        ),
    )[:k]
    return sorted(nearest, key=lambda item: (item.event_time, item.resource_id))


def candidate_case(
    observations: list[Administration],
    *,
    seed: int,
    patient_id: str,
    medication_key: str,
    unit: str,
) -> Case | None:
    split_points = [
        point
        for point in range(3, len(observations) - 2)
        if observations[point - 1].event_time < observations[point].event_time
    ]
    if not split_points:
        return None
    point = min(
        split_points,
        key=lambda value: stable_hex(
            patient_id,
            medication_key,
            unit,
            iso_utc(observations[value - 1].event_time),
            iso_utc(observations[value].event_time),
            seed=seed,
        ),
    )
    candidates = tuple(observations[point - 3 : point + 3])
    gap = observations[point].event_time - observations[point - 1].event_time
    index = observations[point - 1].event_time + gap / 2
    if any(len(select_nearest(candidates, index, scope)) != 3 for scope in ("pre", "post")):
        raise AssertionError("Each medication case must have three rows per side")
    first = observations[0]
    return Case(
        patient_id=patient_id,
        medication_key=medication_key,
        medication_code=first.medication_code,
        medication_display=first.medication_display,
        unit=unit,
        index_time=index,
        candidates=candidates,
        dose_variation=len({item.dose for item in candidates}) > 1,
    )


def one_case_per_patient(
    groups: dict[tuple[str, str, str], list[Administration]], seed: int
) -> dict[str, Case]:
    by_patient: dict[str, list[Case]] = defaultdict(list)
    for (patient_id, medication_key, unit), observations in groups.items():
        if len(observations) < 6:
            continue
        case = candidate_case(
            observations,
            seed=seed,
            patient_id=patient_id,
            medication_key=medication_key,
            unit=unit,
        )
        if case is not None:
            by_patient[patient_id].append(case)
    selected = {}
    for patient_id, cases in by_patient.items():
        selected[patient_id] = min(
            cases,
            key=lambda case: (
                not case.dose_variation,
                stable_hex(
                    patient_id,
                    case.medication_key,
                    case.unit,
                    "case",
                    seed=seed,
                ),
            ),
        )
    return selected


def gold_answer(selected: list[Administration]) -> dict[str, Any]:
    total = math.fsum(item.dose for item in selected)
    return {
        "status": "ok",
        "results": [
            {
                "citation": item.resource_id,
                "dose": item.dose,
                "unit": item.unit,
                "event_time": iso_utc(item.event_time),
            }
            for item in selected
        ],
        "total_dose": total,
        "mean_dose": total / len(selected),
    }


def task_records(
    selected_cases: list[Case], dev_cases: int, seed: int
) -> list[dict[str, Any]]:
    dev_patients = {
        case.patient_id
        for case in sorted(
            selected_cases,
            key=lambda case: stable_hex(case.patient_id, "dev", seed=seed),
        )[:dev_cases]
    }
    records = []
    for case in selected_cases:
        patient_cluster_id = stable_hex(case.patient_id, "patient", seed=seed)[:16]
        case_id = stable_hex(
            case.patient_id,
            case.medication_key,
            case.unit,
            iso_utc(case.index_time),
            seed=seed,
        )[:20]
        split = "dev" if case.patient_id in dev_patients else "test"
        for scope in ("pre", "post"):
            selected = select_nearest(case.candidates, case.index_time, scope)
            relation = "strictly before" if scope == "pre" else "strictly after"
            records.append(
                {
                    "task_id": f"{case_id}-{scope}",
                    "case_id": case_id,
                    "patient_cluster_id": patient_cluster_id,
                    "task_family": "medication_administration_dose",
                    "split": split,
                    "scope": scope,
                    "timestamp_basis": "administration_start",
                    "index_time": iso_utc(case.index_time),
                    "medication_code": case.medication_code,
                    "medication_display": case.medication_display,
                    "unit": case.unit,
                    "dose_variation": case.dose_variation,
                    "question": (
                        f"Return the three {case.medication_display} administration "
                        f"records whose administrations began nearest to but {relation} "
                        f"{iso_utc(case.index_time)}. Order them chronologically, then "
                        "report their total administered dose and mean dose."
                    ),
                    "candidates": [item.compact() for item in case.candidates],
                    "gold_answer": gold_answer(selected),
                    "oracle_version": ORACLE_VERSION,
                }
            )
    return sorted(records, key=lambda record: record["task_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fhir-dir", type=Path, default=DEFAULT_FHIR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-cases", type=int, default=80)
    parser.add_argument("--dev-cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if not 0 < args.dev_cases < args.n_cases:
        raise ValueError("--dev-cases must be between zero and --n-cases")
    for filename in (*ADMIN_SOURCES, MEDICATION_SOURCE):
        if not (args.fhir_dir / filename).exists():
            raise FileNotFoundError(args.fhir_dir / filename)

    groups, counts = load_administrations(args.fhir_dir)
    eligible = {
        patient_id: case
        for patient_id, case in one_case_per_patient(groups, args.seed).items()
        if case.dose_variation
    }
    if len(eligible) < args.n_cases:
        raise ValueError(
            f"Only {len(eligible)} eligible patients for {args.n_cases} requested"
        )
    selected = [
        eligible[patient_id]
        for patient_id in sorted(
            eligible,
            key=lambda patient_id: stable_hex(
                patient_id, "selected", seed=args.seed
            ),
        )[: args.n_cases]
    ]
    records = task_records(selected, args.dev_cases, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "tasks.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "benchmark": "ScopePair-MedAdmin",
        "protocol_version": "1.0",
        "oracle_version": ORACLE_VERSION,
        "source_files": [
            str(args.fhir_dir / name)
            for name in (*ADMIN_SOURCES, MEDICATION_SOURCE)
        ],
        "source_sha256": {
            name: sha256(args.fhir_dir / name)
            for name in (*ADMIN_SOURCES, MEDICATION_SOURCE)
        },
        "seed": args.seed,
        **counts,
        "eligible_patients": len(eligible),
        "n_cases": len(selected),
        "n_tasks": len(records),
        "dev_cases": args.dev_cases,
        "test_cases": args.n_cases - args.dev_cases,
        "dose_variation_cases": sum(case.dose_variation for case in selected),
        "task_manifest_sha256": sha256(manifest),
        "patient_ids_exported": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
