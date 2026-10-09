#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Tier-C skill validation and coverage (Tier-C protocol §Skills): for every FHIR-AgentBench task, run the
deterministic skill over the task's LEGAL resource set (contract predicate on the patient's resources, navigation
types included) and compare the answer with the released one using the benchmark's own scorer. A template is
covered when the skill reproduces the released answer on every task whose R_t lies inside the contract; the
per-template agreement table is written for the protocol. Loads no model.

Usage: validate_tierc_skills.py [--manifest experiments/scopebench_tierC/fhir_agentbench] [--out experiments/scopebench_tierC/skill_validation.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.predicate import legal  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.contracts import resource_to_evidence  # noqa: E402
from scopebench.tierc.skills import legal_set_with_references, patient_pairs, solve  # noqa: E402
from scopebench.tierc.store import load_store  # noqa: E402
from fhir_benchmark_core import canonical_answer_correct  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
PMAP = PUBLIC_DATA / ".clinehr_cache" / "scopebench_tierc_patients" / "fhir_agentbench_patient_map.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/scopebench_tierC/fhir_agentbench")
    p.add_argument("--out", type=Path, default=ROOT / "experiments/scopebench_tierC/skill_validation.json")
    p.add_argument("--max-tasks", type=int, default=None)
    args = p.parse_args()
    t0 = time.time()
    store = load_store(FHIR_DIR)
    pmap = json.load(open(PMAP))
    tasks = [Task.from_dict(r) for r in read_jsonl(args.manifest / "tasks.jsonl")]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    # patient resources (all types) with their evidence objects, cached per patient — the construction the loop's
    # mapped skills use (scopebench.tierc.skills.patient_pairs / legal_set_with_references, DEV-26)
    per_patient: dict[str, list[tuple[dict, object]]] = {}
    stats = defaultdict(lambda: {"n": 0, "checked": 0, "agree": 0, "uncovered": 0, "insufficient": 0, "disagree_examples": []})
    per_task = []
    for t in tasks:
        pid = pmap[t.patient]
        if pid not in per_patient:
            per_patient[pid] = patient_pairs(store, pid, t.patient)
        legal_rs, extra = legal_set_with_references(store, per_patient[pid], t.contract, pid)
        res = solve(t.metadata, legal_rs + extra)
        tmpl = t.metadata["template"]
        s = stats[tmpl]; s["n"] += 1
        row = {"task_id": t.task_id, "template": tmpl, "status": res.get("status"), "op": res.get("op"), "rt_in_lt": t.metadata.get("rt_in_lt")}
        if res.get("status") == "uncovered":
            s["uncovered"] += 1; row["reason"] = res.get("reason")
        elif res.get("status") != "ok":
            s["insufficient"] += 1
        elif t.metadata.get("rt_in_lt", True):
            s["checked"] += 1
            gold = t.gold.get("answer")
            ok = bool(canonical_answer_correct(json.dumps(res["answer"]), gold if isinstance(gold, str) else json.dumps(gold)))
            row["agree"] = ok; row["skill_answer"] = res["answer"]; row["gold"] = gold
            s["agree"] += ok
            if not ok and len(s["disagree_examples"]) < 3:
                s["disagree_examples"].append({"task_id": t.task_id, "question": t.question[:120], "skill": res["answer"], "gold": gold})
        per_task.append(row)
    table = {}
    for tmpl, s in stats.items():
        covered = s["uncovered"] == 0 and s["checked"] > 0 and s["agree"] == s["checked"]
        table[tmpl] = {**s, "agreement": round(s["agree"] / s["checked"], 4) if s["checked"] else None, "covered": covered}
    n_cov = sum(v["n"] for v in table.values() if v["covered"])
    out = {"n_tasks": len(tasks), "n_templates": len(table), "covered_templates": sum(1 for v in table.values() if v["covered"]),
           "covered_tasks": n_cov, "coverage_pct": round(100 * n_cov / len(tasks), 2), "templates": table, "seconds": round(time.time() - t0)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    (args.out.with_suffix(".per_task.jsonl")).write_text("".join(json.dumps(r, default=str) + "\n" for r in per_task))
    for tmpl, v in sorted(table.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{v['n']:4d} {'COV' if v['covered'] else '   '} agree={v['agree']}/{v['checked']} unc={v['uncovered']} ins={v['insufficient']}  {tmpl[:95]}")
    print(f"\ncovered templates {out['covered_templates']}/{out['n_templates']}, covered tasks {n_cov}/{len(tasks)} = {out['coverage_pct']} % ({out['seconds']}s)")


if __name__ == "__main__":
    main()
