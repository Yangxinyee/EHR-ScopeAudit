"""DEV-32: the Tier-C runtime store gives untimed resources the same times as the manifest builder and the skills
(integration over the real FHIR demo store; skipped without the data). Before DEV-32 a Condition without its own time
and every untimed reference resource (Medication, Patient, Location) were dropped by every pre/post contract, so
diagnosis and prescription tasks lost their evidence under enforcement although the builder counted R_t inside L_t."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.predicate import legal  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
MANIFEST = ROOT / "experiments/scopebench_tierC/fhir_agentbench"
PMAP = PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json"

pytestmark = pytest.mark.skipif(not (FHIR_DIR.exists() and MANIFEST.exists() and PMAP.exists()), reason="FHIR demo / Tier-C manifest not present")


@pytest.fixture(scope="module")
def env():
    from scopebench.tierc.store import load_store
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")}
    return load_store(FHIR_DIR), tasks, json.load(open(PMAP))


def _ts(env, task):
    from scopebench.tierc.store import TaskStore
    store, _, pmap = env
    return TaskStore(store=store, patient_fhir_id=pmap[task.patient], patient_hash=task.patient, contract=task.contract, enforcement="contract", question=task.question)


def _resource(store, rid):
    for table in store.by_type_id.values():
        if rid in table:
            return table[rid]
    return None


def test_conditions_take_the_encounter_period(env):
    store, tasks, _ = env
    t = tasks["fab-39f61d8cb6ee0e63eed2c0c1"]   # frontier loss: all 11 gold Conditions were filtered out at runtime
    ts = _ts(env, t)
    gold = [_resource(store, i) for i in t.required_ids]
    assert gold and all(g is not None and g["resourceType"] == "Condition" for g in gold)
    assert all(ts.is_legal(g) for g in gold)
    kept, _ = ts.enforce(gold)
    assert len(kept) == len(gold)


def test_medication_is_in_scope_when_an_in_scope_request_names_it(env):
    store, tasks, _ = env
    t = tasks["fab-5d890144fe08088c541a7324"]   # frontier loss: the drug name behind medicationReference was unreachable
    ts = _ts(env, t)
    reqs = [r for r in store.by_patient_type.get((ts.patient_fhir_id, "MedicationRequest"), []) if ts.is_legal(r)]
    assert reqs
    meds = {(r.get("medicationReference") or {}).get("reference", "").split("/")[-1] for r in reqs}
    meds.discard("")
    assert meds
    for mid in meds:
        m = store.by_type_id["Medication"][mid]
        assert ts.is_legal(m), mid
    # a Medication that no request of this patient names keeps no time and stays out of a pre/post contract
    other = next(m for i, m in store.by_type_id["Medication"].items() if i not in {(r.get("medicationReference") or {}).get("reference", "").split("/")[-1]
                                                                                     for r in store.by_patient_type.get((ts.patient_fhir_id, "MedicationRequest"), [])})
    assert t.contract.scope in ("pre", "post", "at") and not ts.is_legal(other)


def test_runtime_legal_set_covers_the_skill_construction(env):
    """On every manifest task with R_t inside L_t, every required resource is legal at runtime, and every resource the
    skills treat as legal (contract set + referenced Medications) is legal at runtime too."""
    from scopebench.tierc.skills import legal_set_with_references, patient_pairs
    store, tasks, pmap = env
    cache = {}
    bad_required, bad_skill, n = [], [], 0
    for t in tasks.values():
        if t.contract.scope == "full" or not t.metadata.get("rt_in_lt"):
            continue
        n += 1
        ts = _ts(env, t)
        for rid in t.required_ids:
            r = _resource(store, rid)
            if r is not None and not ts.is_legal(r):
                bad_required.append((t.task_id, rid))
        pid = pmap[t.patient]
        if pid not in cache:
            cache[pid] = patient_pairs(store, pid, t.patient)
        legal_rs, extra = legal_set_with_references(store, cache[pid], t.contract, pid)
        for r in legal_rs + [x for x in extra if x["resourceType"] == "Medication"]:
            if not ts.is_legal(r):
                bad_skill.append((t.task_id, str(r["id"])))
    assert n > 1000
    assert not bad_required, bad_required[:5]
    assert not bad_skill, bad_skill[:5]
