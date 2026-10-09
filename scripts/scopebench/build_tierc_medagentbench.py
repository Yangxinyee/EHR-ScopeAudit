#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# MedAgentBench-related adaptation; Copyright (c) 2025 Stanford Machine Learning Group.
# Upstream license: licenses/MedAgentBench-MIT.txt; changes: THIRD_PARTY_NOTICES.md
"""Build the ScopeBench task manifest for MedAgentBench (Tier-C part 2; plan §6.2, Tier-C protocol addendum).

Reads the released test_data_v2.json (300 tasks), derives one contract per task from the category rules, and R_t from the
released grader's own rule evaluated against the official FHIR server (which must be running at --fhir-base).
Writes experiments/scopebench_tierC/medagentbench/{tasks.jsonl, construction_stats.json, metadata.json}.
The data are synthetic (MedAgentBench); nothing here is PHI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import __version__  # noqa: E402
from scopebench.schema import write_jsonl  # noqa: E402
from scopebench.tierc.medagent import CATEGORY_RULES, FHIR_BASE, QUERY_CATEGORIES, get_json_text, make_task  # noqa: E402

TASK_DATA = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/test_data_v2.json"
OUT = ROOT / "experiments/scopebench_tierC/medagentbench"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fhir-base", default=FHIR_BASE)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    if "error" in get_json_text(f"{a.fhir_base}metadata"):
        raise SystemExit(f"FHIR server not reachable at {a.fhir_base}")
    cases = json.load(open(TASK_DATA))
    t0 = time.time()
    tasks, stats = [], {"n_cases": len(cases), "by_category": {}, "no_contract": 0}
    for c in cases:
        t = make_task(c, a.fhir_base)
        if t is None:
            stats["no_contract"] += 1
            continue
        tasks.append(t)
    for cat in sorted({t.metadata["category"] for t in tasks}, key=lambda s: int(s[4:])):
        rows = [t for t in tasks if t.metadata["category"] == cat]
        stats["by_category"][cat] = {"n": len(rows), "scope": CATEGORY_RULES[cat]["scope"], "window_hours": CATEGORY_RULES[cat]["window_hours"],
                                     "sources": list(CATEGORY_RULES[cat]["sources"]), "rt_rule": CATEGORY_RULES[cat]["rt"], "group": "query" if cat in QUERY_CATEGORIES else "action",
                                     "n_rt_nonempty": sum(1 for t in rows if t.required_ids), "mean_rt": round(sum(len(t.required_ids) for t in rows) / len(rows), 2),
                                     "n_rt_errors": sum(1 for t in rows if "error" in t.gold)}
    stats.update({"n_tasks": len(tasks), "scopes": dict(Counter(t.contract.scope for t in tasks)), "n_rt_nonempty": sum(1 for t in tasks if t.required_ids),
                  "seconds": round(time.time() - t0)})
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out / "tasks.jsonl", (t.to_dict() for t in sorted(tasks, key=lambda t: (int(t.metadata["category"][4:]), t.task_id))))
    (a.out / "construction_stats.json").write_text(json.dumps(stats, indent=1))
    (a.out / "metadata.json").write_text(json.dumps({"scopebench_version": __version__, "tier": "C", "benchmark": "MedAgentBench v1", "source": str(TASK_DATA.relative_to(ROOT)),
                                                      "source_sha256": hashlib.sha256(TASK_DATA.read_bytes()).hexdigest(), "fhir_base": a.fhir_base,
                                                      "rules": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in CATEGORY_RULES.items()}}, indent=1))
    print(f"wrote {len(tasks)} tasks to {a.out}: {json.dumps(stats['by_category'])}")


if __name__ == "__main__":
    main()
