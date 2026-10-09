"""Core objects (plan §4.1): evidence objects with an explicit time basis, task contracts, tasks, cases.

One evidence object = one citable row/line/resource. It carries up to three
timestamps, named by *basis*:
  occurred   - when the event happened (specimen drawn, drug given, exam done)
  available  - when the information became available (result stored/issued)
  documented - when it was written (note charted)
A contract names which basis governs legality for the task.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

SCOPES = ("pre", "at", "post", "full")
BASES = ("occurred", "available", "documented")


def parse_time(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stable_hex(*parts: object, n: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


@dataclass(frozen=True)
class EvidenceObject:
    id: str
    source: str
    patient: str
    occurred: str | None = None
    available: str | None = None
    documented: str | None = None
    occurred_end: str | None = None
    row_ref: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def time(self, basis: str) -> datetime | None:
        if basis not in BASES:
            raise ValueError(f"unknown basis {basis!r}")
        return parse_time(getattr(self, basis))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EvidenceObject":
        return cls(
            id=str(row["id"]),
            source=str(row["source"]),
            patient=str(row.get("patient", "")),
            occurred=row.get("occurred"),
            available=row.get("available"),
            documented=row.get("documented"),
            occurred_end=row.get("occurred_end"),
            row_ref=str(row.get("row_ref", "")),
            fields=dict(row.get("fields", {})),
        )


@dataclass(frozen=True)
class Contract:
    """kappa_t = (index event, permitted sources, basis, scope) plus window parameters."""

    index_time: str
    scope: str
    basis: str
    permitted_sources: tuple[str, ...] | None = None
    window_hours: float | None = None
    at_tolerance_hours: float = 0.0
    index_evidence_ids: tuple[str, ...] = ()
    strict_boundaries: bool = False   # sensitivity variant (plan §8.5): window edges and the at-index tolerance exclusive

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}")
        if self.basis not in BASES:
            raise ValueError(f"unknown basis {self.basis!r}")
        parse_time(self.index_time)

    @property
    def index(self) -> datetime:
        return parse_time(self.index_time)  # type: ignore[return-value]

    def with_basis(self, basis: str) -> "Contract":
        return Contract(
            index_time=self.index_time,
            scope=self.scope,
            basis=basis,
            permitted_sources=self.permitted_sources,
            window_hours=self.window_hours,
            at_tolerance_hours=self.at_tolerance_hours,
            index_evidence_ids=self.index_evidence_ids,
            strict_boundaries=self.strict_boundaries,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["permitted_sources"] = list(self.permitted_sources) if self.permitted_sources else None
        d["index_evidence_ids"] = list(self.index_evidence_ids)
        if not self.strict_boundaries:
            d.pop("strict_boundaries")   # frozen manifests keep their exact serialization
        return d

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Contract":
        ps = row.get("permitted_sources")
        return cls(
            index_time=str(row["index_time"]),
            scope=str(row["scope"]),
            basis=str(row["basis"]),
            permitted_sources=tuple(ps) if ps else None,
            window_hours=row.get("window_hours"),
            at_tolerance_hours=float(row.get("at_tolerance_hours", 0.0)),
            index_evidence_ids=tuple(row.get("index_evidence_ids", ())),
            strict_boundaries=bool(row.get("strict_boundaries", False)),
        )


@dataclass
class Task:
    task_id: str
    case_id: str
    patient: str
    tier: str
    task_type: str
    question: str
    contract: Contract
    gold: dict[str, Any]
    required_ids: tuple[str, ...]
    split: str = "test"
    leakage: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        return self.contract.scope

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["contract"] = self.contract.to_dict()
        d["required_ids"] = list(self.required_ids)
        return d

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Task":
        return cls(
            task_id=str(row["task_id"]),
            case_id=str(row["case_id"]),
            patient=str(row.get("patient", "")),
            tier=str(row.get("tier", "")),
            task_type=str(row["task_type"]),
            question=str(row["question"]),
            contract=Contract.from_dict(row["contract"]),
            gold=dict(row.get("gold", {})),
            required_ids=tuple(row.get("required_ids", ())),
            split=str(row.get("split", "test")),
            leakage=bool(row.get("leakage", False)),
            metadata=dict(row.get("metadata", {})),
        )


@dataclass
class Case:
    case_id: str
    patient: str
    index_event: dict[str, Any]
    evidence: list[EvidenceObject]

    def by_id(self) -> dict[str, EvidenceObject]:
        return {e.id: e for e in self.evidence}


def read_jsonl(path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n
