#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Tier-C manifest for FHIR-AgentBench (plan §6.1): every released task (train/valid/test, 2,931) gets a
scope contract derived from its assumption and time filters, R_t = the released gold FHIR ids, and a
check that R_t lies inside the legal set under the native (occurrence) basis. Evidence pools are not
materialized (a patient has thousands of resources); the runner queries the logged FHIR store.

Writes experiments/scopebench_tierC/fhir_agentbench/{tasks.jsonl, construction_stats.json, metadata.json}.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench import __version__  # noqa: E402
from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.predicate import legal  # noqa: E402
from scopebench.schema import EvidenceObject, Task, write_jsonl  # noqa: E402
from scopebench.tierc.contracts import REFERENCE_TYPES, derive_contract, resource_to_evidence, wallclock  # noqa: E402

CSV = ROOT / "external_benchmarks/FHIR-AgentBench/final_dataset/questions_answers_sql_fhir.csv"
FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
OUT = ROOT / "experiments/scopebench_tierC/fhir_agentbench"


def _dt(s: str) -> datetime:
    # wall-clock semantics (see scopebench.tierc.contracts.wallclock): the offset is dropped, not applied
    return datetime.fromisoformat(wallclock(s).replace("Z", "+00:00"))


def skill_params(val_dict: str) -> dict:
    """The task's structured specification from the benchmark's placeholder values (never the answer or the SQL):
    entity values (vital/lab/drug/specimen names, thresholds, routes), operators (comparison, aggregation, sort) and
    the natural-language time filters. Tier-C skills (protocol §Skills) take these as their parameters."""
    try:
        v = ast.literal_eval(val_dict) if val_dict else {}
    except (ValueError, SyntaxError):
        return {}
    out: dict = {"values": {}, "ops": {}, "time": {}}
    for k, val in (v.get("val_placeholder") or {}).items():
        out["values"][k] = str(val)
    for group, key in (("op_placeholder", "ops"), ("time_placeholder", "time")):
        for k, val in (v.get(group) or {}).items():
            if isinstance(val, dict):
                out[key][k] = {kk: str(vv) for kk, vv in val.items() if kk in ("nlq", "type", "sql") and vv is not None}
            else:
                out[key][k] = {"nlq": str(val)}
    return out


def load_visits() -> dict[str, dict[str, list[tuple[datetime, datetime | None]]]]:
    """patient fhir id -> {'hospital': [(start, end)...], 'icu': [...]} sorted by start."""
    visits: dict[str, dict[str, list]] = defaultdict(lambda: {"hospital": [], "icu": []})
    for name, kind in (("MimicEncounter.ndjson.gz", "hospital"), ("MimicEncounterICU.ndjson.gz", "icu")):
        with gzip.open(FHIR_DIR / name, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pid = str(r.get("subject", {}).get("reference", "")).split("/")[-1]
                p = r.get("period") or {}
                if pid and p.get("start"):
                    visits[pid][kind].append((_dt(p["start"]), _dt(p["end"]) if p.get("end") else None))
    for pid in visits:
        for kind in visits[pid]:
            visits[pid][kind].sort()
    return visits


def main() -> None:
    from fhir_benchmark_core import LocalFhirStore  # noqa: E402

    t0 = time.time()
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    visits = load_visits()
    print(f"rows={len(rows)} patients_with_visits={len(visits)} ({time.time() - t0:.0f}s); loading FHIR store…", flush=True)
    store = LocalFhirStore(FHIR_DIR)
    print(f"store loaded: {sum(store.counts.values()):,} resources ({time.time() - t0:.0f}s)", flush=True)
    # per-patient evidence (occurrence/availability times) for the legality check
    tasks: list[Task] = []
    stats = Counter()
    rules = Counter()
    unmapped = Counter()
    rt_sizes = []
    lt_sizes = []
    by_patient_cache: dict[str, dict] = {}
    for r in rows:
        pid = str(r["patient_fhir_id"])
        phash = hashlib.sha256(f"scopebench-tierc|{pid}".encode()).hexdigest()[:16]
        if pid not in by_patient_cache:
            evid = {}
            encounter_period: dict[str, tuple[str | None, str | None]] = {}
            for (p, rtype), resources in store.by_patient_type.items():
                if p == pid and rtype == "Encounter":
                    for res in resources:
                        per = res.get("period") or {}
                        encounter_period[str(res["id"])] = (per.get("start"), per.get("end"))
            for (p, rtype), resources in store.by_patient_type.items():
                if p == pid:
                    for res in resources:
                        e = resource_to_evidence(res, phash)
                        if rtype == "Condition" and e.occurred is None:
                            # diagnoses carry no time of their own: occurrence = admission start, availability = discharge (coded at discharge)
                            enc = str((res.get("encounter") or {}).get("reference", "")).split("/")[-1]
                            st, en = encounter_period.get(enc, (None, None))
                            e = EvidenceObject(id=e.id, source=e.source, patient=e.patient, occurred=st, available=en or st, occurred_end=en, row_ref=e.row_ref, fields=e.fields)
                        evid[e.id] = e
            latest = max((e.time("occurred") for e in evid.values() if e.time("occurred")), default=None)
            by_patient_cache[pid] = {"evidence": evid, "latest": latest}
        cache = by_patient_cache[pid]
        contract, notes = derive_contract(r, visits.get(pid, {"hospital": [], "icu": []}), cache["latest"])
        if contract is None:
            unmapped[notes.get("reason", "?")[:40]] += 1
            stats["unmapped"] += 1
            continue
        try:
            gold_ids = ast.literal_eval(r["true_fhir_ids"]) if r["true_fhir_ids"] else {}
        except (ValueError, SyntaxError):
            gold_ids = {}
        required = tuple(i for rtype, ids in gold_ids.items() if rtype not in REFERENCE_TYPES for i in ids) if isinstance(gold_ids, dict) else ()
        pool = cache["evidence"]
        legal_ids = {i for i, e in pool.items() if legal(e, contract)}
        rt_in_lt = all(i in legal_ids for i in required)
        stats["rt_in_lt" if rt_in_lt else "rt_outside_lt"] += 1
        rules[notes.get("rule")] += 1
        rt_sizes.append(len(required)); lt_sizes.append(len(legal_ids))
        try:
            gold_answer = ast.literal_eval(r["true_answer"])
        except (ValueError, SyntaxError):
            gold_answer = r["true_answer"]
        tasks.append(Task(task_id=f"fab-{r['question_id']}", case_id=f"fab-{r['question_id']}", patient=phash, tier="C", task_type="fhir_agentbench_v1",
                          question=r["question"], contract=contract, gold={"answer": gold_answer, "fhir_ids": gold_ids, "sql": r.get("sql_query")},
                          required_ids=required, split=r["split"], leakage=False,
                          metadata={"family": "fhir_agentbench", "template": r["template"], "assumption": r.get("assumption"), "patient_fhir_id_hash": phash,
                                    "contract_notes": notes, "rt_in_lt": rt_in_lt, "n_legal": len(legal_ids), "n_patient_resources": len(pool), "main_table": r.get("main_table_name"),
                                    "skill_params": skill_params(r.get("val_dict", ""))}))
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "tasks.jsonl", (t.to_dict() for t in tasks))
    # raw patient-id map kept outside the repository
    raw_dir = PUBLIC_DATA / ".clinehr_cache" / "scopebench_tierc_patients"; raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "fhir_agentbench_patient_map.json").write_text(json.dumps({hashlib.sha256(f"scopebench-tierc|{p}".encode()).hexdigest()[:16]: p for p in by_patient_cache}, indent=1))
    scope_counts = Counter(t.contract.scope for t in tasks)
    s = {"n_rows": len(rows), "n_tasks": len(tasks), "unmapped": dict(unmapped), "rules": dict(rules), "scopes": dict(scope_counts),
         "rt_in_lt": stats["rt_in_lt"], "rt_outside_lt": stats["rt_outside_lt"], "splits": dict(Counter(t.split for t in tasks)),
         "median_required": sorted(rt_sizes)[len(rt_sizes) // 2] if rt_sizes else None, "median_legal": sorted(lt_sizes)[len(lt_sizes) // 2] if lt_sizes else None,
         "seconds": round(time.time() - t0)}
    (OUT / "construction_stats.json").write_text(json.dumps(s, indent=1))
    (OUT / "metadata.json").write_text(json.dumps({"scopebench_version": __version__, "source_csv": str(CSV.relative_to(ROOT)), "fhir_dir": FHIR_DIR.name, "basis": "occurred (benchmark-native); wrong-basis control = available (Observation.issued)"}, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
