"""Tier-A anchors and seeded case selection (plan §7.2, mirroring `tierb/sampling.py`).

An anchor is (modality, cohort, MRN, index exam date). Selection is deterministic: anchors are ordered by
SHA-256(seed | modality | mrn | date) within a modality, modalities are drawn round-robin so the quota fills evenly,
one case per patient across all modalities and splits, and the family of a case rotates through the six families in
selection order within its modality (so every modality × family cell is populated). Raw MRNs never leave the enclave:
the builder records only hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .config import MODALITIES
from .loader import TierALoader
from .timeshift import parse_any, shifted_iso

FAMILIES = ("evidence_retrieval", "outcome_abstraction", "cohort_abstraction", "claim_verification", "report_reconstruction", "summarization")


@dataclass(frozen=True)
class AnchorA:
    modality: str
    cohort: str
    mrn: str
    dos_raw: str            # as exported (never written to a manifest)
    index_time: str         # shifted ISO

    @property
    def key(self) -> tuple:
        return (self.modality, self.cohort, self.dos_raw)


def order_key(seed: int, anchor: AnchorA) -> str:
    return hashlib.sha256(f"{seed}|{anchor.modality}|{anchor.mrn}|{anchor.dos_raw}".encode()).hexdigest()


def discover_anchors(loader: TierALoader, modalities: tuple[str, ...] = MODALITIES) -> dict[str, list[AnchorA]]:
    out: dict[str, list[AnchorA]] = {}
    for m in modalities:
        rows = loader.index_events(m)
        anchors = []
        for r in rows:
            if parse_any(r["dos"]) is None:
                continue
            iso = shifted_iso(r["dos"], loader.shift(r["mrn"]))
            if iso:
                anchors.append(AnchorA(m, r["cohort"], str(r["mrn"]).strip(), str(r["dos"]).strip(), iso))
        out[m] = anchors
    return out


def select(anchors_by_modality: dict[str, list[AnchorA]], seed: int, per_modality: int, exclude_mrns: set[str] = frozenset(),
           families: tuple[str, ...] = FAMILIES) -> list[tuple[AnchorA, str]]:
    """Up to `per_modality` anchors per modality in seeded order, one per patient overall, round-robin across
    modalities, family rotating in selection order within each modality. Returns (anchor, family) pairs."""
    queues = {m: sorted(a, key=lambda x: order_key(seed, x)) for m, a in anchors_by_modality.items()}
    taken_mrn: set[str] = set(exclude_mrns)
    picked: dict[str, list[AnchorA]] = {m: [] for m in queues}
    pos = {m: 0 for m in queues}
    progress = True
    while progress:
        progress = False
        for m in queues:
            if len(picked[m]) >= per_modality:
                continue
            q = queues[m]
            while pos[m] < len(q):
                a = q[pos[m]]; pos[m] += 1
                if a.mrn in taken_mrn:
                    continue
                taken_mrn.add(a.mrn); picked[m].append(a); progress = True
                break
    out: list[tuple[AnchorA, str]] = []
    for m, lst in picked.items():
        offset = int(hashlib.sha256(f"{seed}|family_offset|{m}".encode()).hexdigest(), 16) % len(families)
        for i, a in enumerate(lst):
            out.append((a, families[(i + offset) % len(families)]))
    return out


def summary(pairs: list[tuple[AnchorA, str]]) -> dict[str, Any]:
    from collections import Counter
    return {"n": len(pairs), "by_modality": dict(Counter(a.modality for a, _ in pairs)),
            "by_modality_family": {f"{m}|{f}": n for (m, f), n in sorted(Counter((a.modality, f) for a, f in pairs).items())}}
