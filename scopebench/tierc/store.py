# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Logged FHIR store for Tier-C (plan §6.1, Tier-C protocol): the benchmark-native `LocalFhirStore` behind a
tool layer that (a) applies the task contract's predicate to every result when the configuration enforces
scope, (b) records A_t — every resource id returned to the model view or to the Python analysis environment,
per call, with its legality — and (c) refuses cross-patient access exactly as the released agent does.

Enforcement modes:
  none            benchmark-native results (prompt_only, token/compute controls, direct_api, specialized_tools)
  contract        results filtered with legal(resource, contract) before the model view and before Python
  wrong_basis     the same with the availability-basis contract
  security_label  Security Label + Consent baseline: out-of-scope resources carry a `meta.security` label and the
                  tool layer honours it (equivalent to `contract` for access; the label is visible in the trace)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..predicate import legal, wrong_basis
from ..schema import Contract, EvidenceObject
from .contracts import REFERENCE_TYPES, resource_to_evidence

MODEL_VIEW_LIMIT = 8          # frozen FHIR-AgentBench setting (model_limit)
FOCUSED_RETRIEVAL_LIMIT = 128  # frozen focused-mode retrieval_limit (broad mode returns every resource of the type)


@dataclass
class AccessEntry:
    round: int
    tool: str
    arguments: dict[str, Any]
    returned_ids: list[str]            # ids the call handed to the model view or to Python (A_t contribution)
    illegal_ids: list[str]             # subset outside the contract (only non-empty when enforcement is 'none')
    filtered_out_ids: list[str]        # ids the enforcement removed before returning (never reach the model)
    error: str | None = None


@dataclass
class TaskStore:
    """One task's view of the store: contract, enforcement, access log, and the resources retrieved so far."""
    store: Any                         # scripts.fhir_benchmark_core.LocalFhirStore
    patient_fhir_id: str
    patient_hash: str
    contract: Contract
    enforcement: str = "none"
    mode: str = "broad"                # benchmark retrieval mode: broad | focused
    question: str = ""
    retrieved: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    log: list[AccessEntry] = field(default_factory=list)
    _evidence_cache: dict[str, EvidenceObject] = field(default_factory=dict)
    _index: dict[str, dict[str, Any]] = field(default_factory=dict)   # id -> retrieved resource (O(1); a broad call returns ~25k ids)
    _py_copies: dict[str, dict[str, Any]] = field(default_factory=dict)   # DEV-28: this episode's working copies for execute_python_code
    _enc_periods: dict[str, tuple[str | None, str | None]] | None = None   # DEV-32: encounter id -> (start, end)
    _referrers: dict[str, list[dict[str, Any]]] | None = None              # DEV-32: reference-resource id -> resources citing it

    # ----------------------------------------------------------------- legality
    def evidence(self, resource: dict[str, Any]) -> EvidenceObject:
        """DEV-32: the same time construction as the manifest builder and the skills. A Condition without an occurrence
        time takes its encounter's period (occurrence = start, availability = end, as `build_tierc_fhiragentbench.py`
        and `skills.patient_pairs` do); an untimed reference resource (Medication, Patient, Location, …) takes the times
        of a resource of the same patient that references it — the first such resource inside the contract, else the
        earliest — so a Medication is in scope exactly when an in-scope MedicationRequest names it (the construction
        `skills.legal_set_with_references` uses). Before DEV-32 both kinds carried no time and were dropped by every
        pre/post contract."""
        rid = str(resource["id"])
        if rid not in self._evidence_cache:
            e = resource_to_evidence(resource, self.patient_hash)
            rtype = str(resource.get("resourceType"))
            if e.occurred is None and e.available is None:
                if rtype == "Condition":
                    st, en = self._encounter_periods().get(_ref_id((resource.get("encounter") or {}).get("reference")), (None, None))
                    e = replace(e, occurred=st, available=en or st, occurred_end=en)
                elif rtype in REFERENCE_TYPES:
                    e = self._inherit_time(e, rid)
            self._evidence_cache[rid] = e
        return self._evidence_cache[rid]

    def _patient_resources(self) -> list[dict[str, Any]]:
        return [r for (pp, _), lst in self.store.by_patient_type.items() if pp == self.patient_fhir_id for r in lst]

    def _encounter_periods(self) -> dict[str, tuple[str | None, str | None]]:
        if self._enc_periods is None:
            self._enc_periods = {str(r["id"]): ((r.get("period") or {}).get("start"), (r.get("period") or {}).get("end"))
                                 for r in self.store.by_patient_type.get((self.patient_fhir_id, "Encounter"), [])}
        return self._enc_periods

    def _inherit_time(self, e: EvidenceObject, rid: str) -> EvidenceObject:
        if self._referrers is None:
            refs: dict[str, list[dict[str, Any]]] = {}
            for r in self._patient_resources():
                if str(r.get("resourceType")) in REFERENCE_TYPES:
                    continue
                for ref in _references(r):
                    refs.setdefault(ref, []).append(r)
            self._referrers = refs
        timed = [x for x in (self.evidence(r) for r in self._referrers.get(rid, [])) if x.occurred is not None or x.available is not None]
        if not timed:
            return e
        as_ref = [replace(x, id=e.id, source=e.source, row_ref=e.row_ref, fields=e.fields) for x in timed]
        chosen = next((x for x in as_ref if legal(x, self.contract)), None)
        if chosen is None:
            chosen = min(as_ref, key=lambda x: (x.time("occurred") is None, x.time("occurred") or x.time("available")))
        return chosen

    def effective_contract(self) -> Contract:
        return wrong_basis(self.contract) if self.enforcement == "wrong_basis" else self.contract

    def is_legal(self, resource: dict[str, Any]) -> bool:
        return legal(self.evidence(resource), self.contract)

    def enforce(self, resources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        if self.enforcement == "none":
            return resources, []
        c = self.effective_contract()
        kept, dropped = [], []
        for r in resources:
            (kept if legal(self.evidence(r), c) else dropped).append(r)
        if self.enforcement == "security_label":
            for r in kept:   # the label the Consent baseline exposes: within-scope resources are marked as such
                r = dict(r); r.setdefault("meta", {})
        return kept, [str(r["id"]) for r in dropped]

    # ----------------------------------------------------------------- tools
    def _record(self, round_index: int, tool: str, arguments: dict[str, Any], returned: list[dict[str, Any]], dropped: list[str], error: str | None = None) -> None:
        ids = [str(r["id"]) for r in returned]
        illegal = [str(r["id"]) for r in returned if not self.is_legal(r)]
        self.log.append(AccessEntry(round=round_index, tool=tool, arguments=arguments, returned_ids=ids, illegal_ids=illegal, filtered_out_ids=dropped, error=error))
        if returned:
            rtype = returned[0]["resourceType"]
            by_id = {str(r["id"]): r for r in self.retrieved.get(rtype, [])}
            for r in returned:
                by_id[str(r["id"])] = r
            self.retrieved[rtype] = list(by_id.values())
            self._index.update(by_id)

    def get_by_patient(self, round_index: int, resource_type: str, patient_fhir_id: str) -> dict[str, Any]:
        args = {"resource_type": resource_type, "patient_fhir_id": patient_fhir_id}
        if patient_fhir_id != self.patient_fhir_id:
            self.log.append(AccessEntry(round_index, "get_resources_by_patient_fhir_id", args, [], [], [], error="Cross-patient access denied"))
            raise ValueError("Cross-patient access denied")
        result = self.store.get_by_patient(resource_type, patient_fhir_id, question=self.question, mode=self.mode, model_limit=MODEL_VIEW_LIMIT, retrieval_limit=FOCUSED_RETRIEVAL_LIMIT)
        kept, dropped = self.enforce(result.resources)
        self._record(round_index, "get_resources_by_patient_fhir_id", args, kept, dropped)
        view = self.store._rank(kept, self.question)[:MODEL_VIEW_LIMIT]
        from scripts.fhir_benchmark_core import compact_resource  # local import: scripts/ is on sys.path in the runner
        return {"resource_type": resource_type, "retrieved_count": len(kept), "resources": [compact_resource(r) for r in view],
                "notice": "Full retrieved resources are available to execute_python_code; the response view is relevance-ranked and compact."
                          + (" Resources outside the task's scope contract are not returned." if self.enforcement != "none" else "")}

    def get_by_id(self, round_index: int, resource_type: str, resource_id: str) -> dict[str, Any]:
        args = {"resource_type": resource_type, "resource_id": resource_id}
        result = self.store.get_by_id(resource_type, resource_id)
        kept, dropped = self.enforce(result.resources)
        self._record(round_index, "get_resources_by_resource_id", args, kept, dropped)
        from scripts.fhir_benchmark_core import compact_resource
        return {"resource_type": resource_type, "retrieved_count": len(kept), "resources": [compact_resource(r) for r in kept]}

    def linked_medications(self, round_index: int, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Focused-mode convenience of the released agent: resolve MedicationRequest.medicationReference."""
        meds = []
        for req in requests:
            ref = (req.get("medicationReference") or {}).get("reference", "")
            if isinstance(ref, str) and ref.startswith("Medication/"):
                m = self.store.by_type_id.get("Medication", {}).get(ref.split("/", 1)[1])
                if m is not None:
                    meds.append(m)
        if meds:
            kept, dropped = self.enforce(meds)
            self._record(round_index, "resolve_medication_reference", {"n": len(requests)}, kept, dropped)
            return kept
        return []

    def python_view(self) -> dict[str, list[dict[str, Any]]]:
        """DEV-28: the resources handed to `execute_python_code`. Model-written code may mutate what it receives (the released
        loop hands out the store's own dicts, so one episode's edits — e.g. a date string replaced by a datetime object —
        leaked into every later episode of the patient in the same process). Each episode now works on its own copies,
        made on first exposure and kept for the episode, so its edits still persist across its own calls while the shared
        store stays pristine."""
        view: dict[str, list[dict[str, Any]]] = {}
        for rtype, rs in self.retrieved.items():
            lst = []
            for r in rs:
                rid = str(r["id"])
                c = self._py_copies.get(rid)
                if c is None:
                    c = json.loads(json.dumps(r))
                    self._py_copies[rid] = c
                lst.append(c)
            view[rtype] = lst
        return view

    # ----------------------------------------------------------------- A_t
    def access_ids(self) -> list[str]:
        seen, out = set(), []
        for e in self.log:
            for i in e.returned_ids:
                if i not in seen:
                    seen.add(i); out.append(i)
        return out

    def illegal_access_ids(self) -> list[str]:
        return sorted({i for e in self.log for i in e.illegal_ids})

    def resource_by_id(self, rid: str) -> dict[str, Any] | None:
        return self._index.get(rid)

    def record_skill_provenance(self, pairs: list[tuple[dict[str, Any], EvidenceObject | None]]) -> None:
        """DEV-26: the resources a template skill cites enter the access log as one `skill_provenance` call (round 0: after
        the episode). The skill's own evidence objects are cached first, so the pool and the legality checks see the
        same objects the skill validation used; a resource outside the contract would be logged as illegal access."""
        for r, e in pairs:
            if e is not None:
                self._evidence_cache[str(r["id"])] = e
        by_type: dict[str, list[dict[str, Any]]] = {}
        for r, _ in pairs:
            by_type.setdefault(str(r.get("resourceType")), []).append(r)
        for rtype, rs in by_type.items():   # one entry per resource type: `retrieved` is keyed by type
            self._record(0, "skill_provenance", {"resource_type": rtype, "n": len(rs)}, rs, [])

    def legal_required_resources(self, required_ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        """Resources of R_t that lie inside the contract and were not handed to the model yet (the repair refill)."""
        seen = set(self.access_ids())
        out = []
        for rid in required_ids:
            if rid in seen:
                continue
            for rtype, table in self.store.by_type_id.items():
                r = table.get(rid)
                if r is not None:
                    if self.is_legal(r):
                        out.append(r)
                    break
        return out


def _ref_id(ref: Any) -> str:
    return ref.split("/")[-1] if isinstance(ref, str) else ""


def _references(x: Any) -> list[str]:
    """Ids of every `{"reference": "Type/id"}` inside a resource."""
    out: list[str] = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "reference" and isinstance(v, str) and "/" in v:
                out.append(v.split("/")[-1])
            else:
                out += _references(v)
    elif isinstance(x, list):
        for v in x:
            out += _references(v)
    return out


def load_store(fhir_dir: Path):
    from scripts.fhir_benchmark_core import LocalFhirStore
    return LocalFhirStore(fhir_dir)
