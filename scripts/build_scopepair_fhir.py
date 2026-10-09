#!/usr/bin/env python3
"""Build the prospective paired-scope FHIR benchmark manifest.

The builder streams only the public MIMIC-IV-on-FHIR laboratory observations.
It selects at most one case per patient and hashes patient identifiers before
writing the manifest. No model is loaded by this script.
"""

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
DEFAULT_OUTPUT = ROOT / "experiments" / "scopepair_fhir"
SOURCE_FILE = "MimicObservationLabevents.ndjson.gz"
ORACLE_VERSION = "scopepair-fhir-oracle-v1"


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hex(*parts: object, seed: int) -> str:
    joined = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def reference_id(reference: Any) -> str | None:
    if not isinstance(reference, dict):
        return None
    value = reference.get("reference")
    if not isinstance(value, str) or "/" not in value:
        return None
    return value.split("/", 1)[1]


def coding(resource: dict[str, Any]) -> tuple[str, str] | None:
    entries = resource.get("code", {}).get("coding", [])
    if not isinstance(entries, list):
        return None
    for entry in entries:
        code = entry.get("code")
        display = entry.get("display")
        if code and display:
            return str(code), str(display)
    return None


@dataclass(frozen=True)
class Lab:
    resource_id: str
    patient_id: str
    code: str
    display: str
    value: float
    unit: str
    effective: datetime
    issued: datetime

    def compact(self) -> dict[str, Any]:
        return {
            "id": self.resource_id,
            "display": self.display,
            "value": self.value,
            "unit": self.unit,
            "effective_time": iso_utc(self.effective),
            "issued_time": iso_utc(self.issued),
        }


@dataclass(frozen=True)
class Case:
    patient_id: str
    code: str
    display: str
    unit: str
    index_time: datetime
    candidates: tuple[Lab, ...]
    timestamp_stress: bool


def load_labs(path: Path) -> tuple[dict[tuple[str, str, str], list[Lab]], int]:
    groups: dict[tuple[str, str, str], list[Lab]] = defaultdict(list)
    valid = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            resource = json.loads(line)
            patient_id = reference_id(resource.get("subject"))
            code_display = coding(resource)
            quantity = resource.get("valueQuantity")
            issued_raw = resource.get("issued")
            effective_raw = resource.get("effectiveDateTime")
            if (
                not patient_id
                or code_display is None
                or not isinstance(quantity, dict)
                or not isinstance(issued_raw, str)
                or not isinstance(effective_raw, str)
            ):
                continue
            value = quantity.get("value")
            unit = quantity.get("unit") or quantity.get("code")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not unit
            ):
                continue
            try:
                effective = parse_time(effective_raw)
                issued = parse_time(issued_raw)
            except ValueError:
                continue
            code, display = code_display
            lab = Lab(
                resource_id=str(resource["id"]),
                patient_id=patient_id,
                code=code,
                display=display,
                value=float(value),
                unit=str(unit),
                effective=effective,
                issued=issued,
            )
            groups[(patient_id, code, str(unit))].append(lab)
            valid += 1
    for observations in groups.values():
        observations.sort(key=lambda item: (item.issued, item.resource_id))
    return groups, valid


def permitted(lab: Lab, index: datetime, scope: str, basis: str) -> bool:
    timestamp = lab.issued if basis == "issued" else lab.effective
    if scope == "pre":
        return timestamp < index
    if scope == "post":
        return timestamp > index
    raise ValueError(f"Unknown scope: {scope}")


def select_nearest(
    candidates: Iterable[Lab],
    index: datetime,
    scope: str,
    basis: str,
    k: int = 3,
) -> list[Lab]:
    allowed = [lab for lab in candidates if permitted(lab, index, scope, basis)]
    timestamp = (lambda lab: lab.issued) if basis == "issued" else (lambda lab: lab.effective)
    nearest = sorted(
        allowed,
        key=lambda lab: (abs((timestamp(lab) - index).total_seconds()), lab.resource_id),
    )[:k]
    return sorted(nearest, key=lambda lab: (timestamp(lab), lab.resource_id))


def candidate_variants_for_group(
    observations: list[Lab],
    *,
    seed: int,
    patient_id: str,
    code: str,
    unit: str,
) -> list[Case]:
    split_points = [
        point
        for point in range(3, len(observations) - 2)
        if observations[point - 1].issued < observations[point].issued
    ]
    if not split_points:
        return []
    cases = []
    for point in split_points:
        left = observations[point - 3 : point]
        right = observations[point : point + 3]
        candidates = tuple(left + right)
        gap = observations[point].issued - observations[point - 1].issued
        index = observations[point - 1].issued + gap / 2
        issued_sets = {
            scope: {
                lab.resource_id
                for lab in select_nearest(candidates, index, scope, "issued")
            }
            for scope in ("pre", "post")
        }
        effective_sets = {
            scope: {
                lab.resource_id
                for lab in select_nearest(candidates, index, scope, "effective")
            }
            for scope in ("pre", "post")
        }
        stress = any(
            issued_sets[scope] != effective_sets[scope] for scope in ("pre", "post")
        )
        cases.append(
            Case(
                patient_id=patient_id,
                code=code,
                display=observations[0].display,
                unit=unit,
                index_time=index,
                candidates=candidates,
                timestamp_stress=stress,
            )
        )
    # Preserve at most one ordinary and one disagreement window per concept;
    # choose_cases() later applies the patient-level stress quota.
    stress_cases = [case for case in cases if case.timestamp_stress]
    ordinary_cases = [case for case in cases if not case.timestamp_stress]
    variants = []
    for pool in (ordinary_cases, stress_cases):
        if pool:
            variants.append(
                min(
                    pool,
                    key=lambda case: stable_hex(
                        patient_id,
                        code,
                        unit,
                        iso_utc(case.index_time),
                        seed=seed,
                    ),
                )
            )
    return variants


def one_case_per_patient(
    groups: dict[tuple[str, str, str], list[Lab]], seed: int
) -> dict[str, dict[str, Case | None]]:
    by_patient: dict[str, list[Case]] = defaultdict(list)
    for (patient_id, code, unit), observations in groups.items():
        if len(observations) < 6:
            continue
        variants = candidate_variants_for_group(
            observations,
            seed=seed,
            patient_id=patient_id,
            code=code,
            unit=unit,
        )
        by_patient[patient_id].extend(variants)
    selected: dict[str, dict[str, Case | None]] = {}
    for patient_id, cases in by_patient.items():
        stress_cases = [case for case in cases if case.timestamp_stress]
        ordinary_cases = [case for case in cases if not case.timestamp_stress]
        general_pool = ordinary_cases or cases
        general = min(
            general_pool,
            key=lambda case: stable_hex(
                patient_id, case.code, case.unit, "general-concept", seed=seed
            ),
        )
        stress = (
            min(
                stress_cases,
                key=lambda case: stable_hex(
                    patient_id, case.code, case.unit, "stress-concept", seed=seed
                ),
            )
            if stress_cases
            else None
        )
        selected[patient_id] = {"general": general, "stress": stress}
    return selected


def choose_cases(
    options: dict[str, dict[str, Case | None]], n_cases: int, seed: int
) -> list[Case]:
    if len(options) < n_cases:
        raise ValueError(
            f"Only {len(options)} eligible patients for {n_cases} requested cases"
        )
    stress_patients = [
        patient_id
        for patient_id, variants in options.items()
        if variants["stress"] is not None
    ]
    stress_quota = min(n_cases // 4, len(stress_patients))
    chosen_stress_patients = set(
        sorted(
            stress_patients,
            key=lambda patient_id: stable_hex(patient_id, "stress", seed=seed),
        )[:stress_quota]
    )
    stress = [
        options[patient_id]["stress"]
        for patient_id in chosen_stress_patients
        if options[patient_id]["stress"] is not None
    ]
    remaining_patients = sorted(
        (
            patient_id
            for patient_id in options
            if patient_id not in chosen_stress_patients
        ),
        key=lambda patient_id: (
            bool(options[patient_id]["general"].timestamp_stress),
            stable_hex(patient_id, "general", seed=seed),
        ),
    )[: n_cases - len(stress)]
    remaining = [options[patient_id]["general"] for patient_id in remaining_patients]
    return sorted(
        stress + remaining,
        key=lambda case: stable_hex(case.patient_id, "final", seed=seed),
    )


def gold_answer(selected: list[Lab]) -> dict[str, Any]:
    first = selected[0]
    last = selected[-1]
    delta = last.value - first.value
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        direction = "stable"
    elif delta > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return {
        "status": "ok",
        "results": [
            {
                "citation": lab.resource_id,
                "value": lab.value,
                "unit": lab.unit,
                "issued_time": iso_utc(lab.issued),
            }
            for lab in selected
        ],
        "delta": delta,
        "direction": direction,
    }


def task_records(
    selected_cases: list[Case], dev_cases: int, seed: int
) -> list[dict[str, Any]]:
    dev_ids = {
        case.patient_id
        for case in sorted(
            selected_cases,
            key=lambda case: (
                not case.timestamp_stress,
                stable_hex(case.patient_id, "dev", seed=seed),
            ),
        )[:dev_cases]
    }
    records = []
    for case in selected_cases:
        patient_cluster_id = stable_hex(case.patient_id, "patient", seed=seed)[:16]
        case_id = stable_hex(
            case.patient_id, case.code, case.unit, iso_utc(case.index_time), seed=seed
        )[:20]
        split = "dev" if case.patient_id in dev_ids else "test"
        for scope in ("pre", "post"):
            selected = select_nearest(
                case.candidates, case.index_time, scope, basis="issued", k=3
            )
            if len(selected) != 3:
                raise AssertionError("Oracle construction must yield three observations")
            direction_phrase = "strictly before" if scope == "pre" else "strictly after"
            records.append(
                {
                    "task_id": f"{case_id}-{scope}",
                    "case_id": case_id,
                    "patient_cluster_id": patient_cluster_id,
                    "split": split,
                    "scope": scope,
                    "timestamp_basis": "issued",
                    "index_time": iso_utc(case.index_time),
                    "lab_code": case.code,
                    "lab_display": case.display,
                    "unit": case.unit,
                    "timestamp_stress": case.timestamp_stress,
                    "question": (
                        f"Return the three {case.display} results whose results became "
                        f"available nearest to but {direction_phrase} "
                        f"{iso_utc(case.index_time)}. Order them chronologically, then "
                        "report the last-minus-first numeric change and whether it is "
                        "increasing, decreasing, or stable."
                    ),
                    "candidates": [lab.compact() for lab in case.candidates],
                    "gold_answer": gold_answer(selected),
                    "oracle_version": ORACLE_VERSION,
                }
            )
    return sorted(records, key=lambda record: record["task_id"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    source = args.fhir_dir / SOURCE_FILE
    if not source.exists():
        raise FileNotFoundError(source)

    groups, valid_observations = load_labs(source)
    eligible = one_case_per_patient(groups, args.seed)
    selected = choose_cases(eligible, args.n_cases, args.seed)
    records = task_records(selected, args.dev_cases, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "tasks.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "benchmark": "ScopePair-FHIR",
        "protocol_version": "1.0",
        "oracle_version": ORACLE_VERSION,
        "source_file": str(source),
        "source_sha256": sha256(source),
        "seed": args.seed,
        "valid_numeric_labs": valid_observations,
        "eligible_patients": len(eligible),
        "n_cases": len(selected),
        "n_tasks": len(records),
        "dev_cases": args.dev_cases,
        "test_cases": args.n_cases - args.dev_cases,
        "timestamp_stress_cases": sum(case.timestamp_stress for case in selected),
        "task_manifest_sha256": sha256(manifest),
        "patient_ids_exported": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
