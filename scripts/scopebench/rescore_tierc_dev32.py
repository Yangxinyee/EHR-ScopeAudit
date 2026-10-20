#!/usr/bin/env python3
"""DEV-32: re-score the UNSCOPED Tier-C configurations under the corrected runtime store, without any model call.

Without enforcement the store hands the agent every resource, so the DEV-32 change (untimed Condition / reference
resources now take the builder's times) leaves the episodes untouched and changes only legality: which accessed objects
are admissible (A-L) and which cited ones are (C-L, workflow compliance). Each recorded row is re-scored with the shared
scorer over a pool built by the corrected `TaskStore`.

A_t: rows keep their accessed-id list when it has at most ID_LIST_CAP ids; larger rows keep only their illegal ids
(capped). For a capped row the corrected legality is decided on the recorded illegal ids; when that list was itself
truncated and none of its ids stays illegal, legality cannot be decided from the record and the row is kept illegal and
counted (`undecidable_capped`).

`--old-semantics` re-scores with the pre-DEV-32 construction instead; it must reproduce the stored metrics exactly and
is the validation of this script. Writes <out-root>/<tag>/<config>/predictions.jsonl + metadata.json and a summary.
Usage: rescore_tierc_dev32.py [--tags ...] [--configs ...] [--old-semantics] [--out-root DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.predicate import legal  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.scoring import score  # noqa: E402
from scopebench.tasktypes import get as get_task_type  # noqa: E402
from scopebench import tasktypes_tierc  # noqa: E402,F401
from scopebench.tierc import store as store_mod  # noqa: E402
from scopebench.tierc.contracts import resource_to_evidence  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
PMAP = PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json"
RUNS = ROOT / "experiments/scopebench_tierC_runs"
TAGS = ["gemma3_27b", "medgemma_27b", "phi4_14b", "qwen3_30b", "qwen3_32b", "qwen3_4b"]
UNSCOPED = ["prompt_only", "token_matched", "high_compute", "direct_api", "specialized_tools"]
FRONTIER_UNSCOPED = ["prompt_only"]
ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "workflow_compliant", "abstained", "parse_failed"]
CAP = 2000


def old_evidence(self, resource):   # the pre-DEV-32 TaskStore.evidence
    rid = str(resource["id"])
    if rid not in self._evidence_cache:
        self._evidence_cache[rid] = resource_to_evidence(resource, self.patient_hash)
    return self._evidence_cache[rid]


def rescore_row(row: dict, task: Task, store, index: dict, pmap: dict, stats: Counter) -> dict:
    tt = get_task_type("fhir_agentbench_v1")
    ts = store_mod.TaskStore(store=store, patient_fhir_id=pmap[task.patient], patient_hash=task.patient, contract=task.contract, enforcement="none")
    acc = list(row.get("access_ids") or [])
    capped = int(row.get("n_access_ids", len(acc))) > len(acc)
    ids = set(acc) | set(tt.citations(row.get("parsed_answer"))) | set(tt.citations(row.get("parsed_initial"))) | set(task.required_ids)
    if capped:
        ids |= set(row.get("illegal_access_ids") or [])
    pool = {i: ts.evidence(index[i]) for i in ids if i in index}
    out = copy.deepcopy(row)
    for key, parsed in (("metrics", row.get("parsed_answer")), ("initial_metrics", row.get("parsed_initial"))):
        m_old = row.get(key)
        if not m_old:
            continue
        access = set(acc) if not capped else set(row.get("illegal_access_ids") or [])
        m = score(parsed, task, tt, pool, access, tokens_in=m_old["tokens_in"], tokens_out=m_old["tokens_out"], calls=m_old["calls"], latency=m_old["latency_seconds"])
        if capped:
            still = [i for i in (row.get("illegal_access_ids") or []) if i in pool and not legal(pool[i], task.contract)]
            truncated = len(row.get("illegal_access_ids") or []) < int(row.get("n_illegal_access_ids", 0))
            if not still and truncated:
                m["access_legal"] = False
                if key == "metrics":
                    stats["undecidable_capped"] += 1
            else:
                m["access_legal"] = not still
            m["n_illegal_access"] = len(still) + (int(row.get("n_illegal_access_ids", 0)) - len(row.get("illegal_access_ids") or []) if truncated else 0)
            m["workflow_compliant"] = bool(m["substantive"] and m["access_legal"] and m["citation_legal"])
        out[key] = m
    if not capped:
        out["illegal_access_ids"] = sorted(i for i in acc if i in pool and not legal(pool[i], task.contract))[:CAP]
        out["n_illegal_access_ids"] = sum(1 for i in acc if i in pool and not legal(pool[i], task.contract))
    else:
        out["illegal_access_ids"] = [i for i in (row.get("illegal_access_ids") or []) if i in pool and not legal(pool[i], task.contract)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=TAGS + ["frontier"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--old-semantics", action="store_true")
    ap.add_argument("--out-root", type=Path, default=ROOT / "experiments/scopebench_tierC_runs_dev32_rescored")
    args = ap.parse_args()
    if args.old_semantics:
        store_mod.TaskStore.evidence = old_evidence
    from fhir_benchmark_core import LocalFhirStore  # noqa: E402
    store = LocalFhirStore(FHIR_DIR)
    index = {rid: r for table in store.by_type_id.values() for rid, r in table.items()}
    pmap = json.load(open(PMAP))
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(ROOT / "experiments/scopebench_tierC/fhir_agentbench/tasks.jsonl")}
    summary = {"semantics": "pre-DEV-32" if args.old_semantics else "DEV-32 corrected store", "cells": {}}
    for tag in args.tags:
        src_root = RUNS / "frontier" / "fhir_agentbench" if tag == "frontier" else RUNS / tag
        configs = args.configs or (FRONTIER_UNSCOPED if tag == "frontier" else UNSCOPED)
        for cfg in configs:
            f = src_root / cfg / "predictions.jsonl"
            if not f.exists():
                continue
            stats: Counter = Counter()
            rows = list(read_jsonl(f))
            new = [rescore_row(r, tasks[r["task_id"]], store, index, pmap, stats) for r in rows]
            changed = {ep: sum(float(a["metrics"][ep]) != float(b["metrics"][ep]) for a, b in zip(rows, new)) for ep in ENDPOINTS}
            cell = {"n": len(rows), "rows_changed": changed, "undecidable_capped": stats["undecidable_capped"],
                    "means_old": {ep: round(100 * sum(float(r["metrics"][ep]) for r in rows) / len(rows), 2) for ep in ENDPOINTS},
                    "means_new": {ep: round(100 * sum(float(r["metrics"][ep]) for r in new) / len(new), 2) for ep in ENDPOINTS}}
            summary["cells"][f"{tag}/{cfg}"] = cell
            print(tag, cfg, "changed:", {k: v for k, v in changed.items() if v}, "undecidable:", stats["undecidable_capped"], flush=True)
            if not args.old_semantics:
                d = args.out_root / ("frontier/fhir_agentbench" if tag == "frontier" else tag) / cfg
                d.mkdir(parents=True, exist_ok=True)
                with open(d / "predictions.jsonl", "w") as fh:
                    for r in new:
                        fh.write(json.dumps(r, default=str) + "\n")
                meta = json.loads((src_root / cfg / "metadata.json").read_text()) if (src_root / cfg / "metadata.json").exists() else {}
                meta["dev32_rescored"] = {"by": "scripts/scopebench/rescore_tierc_dev32.py", "source": str(f.relative_to(ROOT))}
                (d / "metadata.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    name = "rescore_validation_old_semantics.json" if args.old_semantics else "rescore_summary.json"
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / name).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
