#!/usr/bin/env python3
"""DEV-32: the FHIR-AgentBench tasks whose admissible set changes under the corrected runtime store, and a control sample.

A task is affected when, for its contract or for the wrong-basis contract, some resource of a type the contract permits
(Condition and the untimed reference types) is legal under the corrected `TaskStore` and was not legal under the
pre-DEV-32 construction (`resource_to_evidence` without imputation), or vice versa. Full-chart contracts admit every
resource and are never affected. The control sample (patient-clustered, seed 2027) is drawn from the unaffected tasks; it
is re-run with the affected ones so that the corrected store can be shown to leave unaffected episodes unchanged.

Writes experiments/scopebench_tierC/dev32/{affected,control,rerun,frontier_affected,frontier_rerun}_task_ids.json and
summary.json. Loads no model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.predicate import legal, wrong_basis  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.contracts import REFERENCE_TYPES, resource_to_evidence  # noqa: E402
from scopebench.tierc.store import TaskStore, _references, load_store  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
PMAP = PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json"
TC = ROOT / "experiments/scopebench_tierC"
UNTIMED = {"Condition"} | set(REFERENCE_TYPES)


def candidates(store, pid: str, rtype: str) -> list[dict]:
    if rtype == "Medication":   # Medications are not patient-indexed: the ones this patient's requests name
        ids = {str((q.get("medicationReference") or {}).get("reference", "")).split("/")[-1] for q in store.by_patient_type.get((pid, "MedicationRequest"), [])}
        return [store.by_type_id["Medication"][i] for i in sorted(ids) if i in store.by_type_id.get("Medication", {})]
    if rtype == "Patient":
        p = store.by_type_id.get("Patient", {}).get(pid)
        return [p] if p is not None else []
    if rtype in REFERENCE_TYPES:   # Location / Organization / Practitioner: the ones this patient's resources reference
        refs = {ref for (pp, _), lst in store.by_patient_type.items() if pp == pid for r in lst for ref in _references(r)}
        table = store.by_type_id.get(rtype, {})
        return [table[i] for i in sorted(refs) if i in table]
    return list(store.by_patient_type.get((pid, rtype), []))


def changed(store, pmap: dict, t: Task) -> bool:
    types = set(t.contract.permitted_sources or []) & UNTIMED
    if t.contract.scope == "full" or not types:
        return False
    pid = pmap[t.patient]
    for contract in (t.contract, wrong_basis(t.contract)):
        ts = TaskStore(store=store, patient_fhir_id=pid, patient_hash=t.patient, contract=contract, enforcement="contract")
        for rtype in types:
            for r in candidates(store, pid, rtype):
                if legal(resource_to_evidence(r, t.patient), contract) != ts.is_legal(r):
                    return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    store = load_store(FHIR_DIR)
    pmap = json.load(open(PMAP))
    tasks = [Task.from_dict(r) for r in read_jsonl(TC / "fhir_agentbench/tasks.jsonl")]
    affected = sorted(t.task_id for t in tasks if changed(store, pmap, t))
    aff = set(affected)
    # control: whole patients drawn at random among tasks that are unaffected, until control_n tasks are reached
    rng = random.Random(args.seed)
    by_patient: dict[str, list[str]] = {}
    for t in sorted(tasks, key=lambda t: (t.patient, t.task_id)):
        if t.task_id not in aff:
            by_patient.setdefault(t.patient, []).append(t.task_id)
    patients = sorted(by_patient)
    rng.shuffle(patients)
    control: list[str] = []
    for p in patients:
        if len(control) >= args.control_n:
            break
        control += by_patient[p][: args.control_n - len(control)]
    frontier = {r["task_id"] for r in read_jsonl(TC / "fhir_agentbench_frontier300/tasks.jsonl")}
    out = TC / "dev32"
    out.mkdir(parents=True, exist_ok=True)
    sets = {"affected": affected, "control": sorted(control), "rerun": sorted(aff | set(control)),
            "frontier_affected": sorted(aff & frontier), "frontier_rerun": sorted((aff | set(control)) & frontier)}
    for name, ids in sets.items():
        (out / f"{name}_task_ids.json").write_text(json.dumps(ids) + "\n")
    split = {t.task_id: t.split for t in tasks}
    table = {t.task_id: t.metadata.get("main_table") for t in tasks}
    summary = {"n_tasks": len(tasks), **{f"n_{k}": len(v) for k, v in sets.items()},
               "affected_by_split": dict(Counter(split[i] for i in affected)), "affected_by_main_table": dict(Counter(table[i] for i in affected)),
               "control_seed": args.seed, "rule": __doc__.split("\n\n")[1].replace("\n", " ")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
