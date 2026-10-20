#!/usr/bin/env python3
"""DEV-34 offline checks of the evidence-plan auditor (experiments/SCOPEBENCH_PLAN_PROTOCOL.md §7). Loads no model.

  identity  executing the reference plan reproduces the reference skill exactly: same result, same provenance (R̂_t),
            for every covered task — Tier-B dev + test (MIMIC-derived: aggregate counts only), Tier-C all splits.
  rules     rule-routed skill-only: a deterministic parser maps each request to a plan, the plan's answer is submitted
            (scopebench/plan_rules.py; built on the dev splits). Reports plan accuracy and the scored endpoints.

Output: experiments/scopebench_analysis/plans_dev34/<command>.json (aggregates only).
Usage: plans_offline.py identity|rules [--tiers B C] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench import plans  # noqa: E402
from scopebench import tasktypes_tierb  # noqa: E402,F401  (registers the Tier-B task types)
from scopebench.skills import OPERATIONS  # noqa: E402
from scopebench.tasktypes import get as task_type  # noqa: E402

E = ROOT / "experiments"
OUT = E / "scopebench_analysis" / "plans_dev34"
TIER_B = {"dev": E / "scopebench_tierB_dev", "test": E / "scopebench_tierB"}
TIER_C = E / "scopebench_tierC" / "fhir_agentbench"


def public(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------- Tier-B
def tier_b_tasks():
    from scopebench.manifest import load_manifest

    for split, manifest in TIER_B.items():
        tasks, cases = load_manifest(manifest, None)
        yield split, tasks, cases


def identity_b() -> dict:
    from scopebench.runner import _call_skill, legal_all

    out: dict = {}
    for split, tasks, cases in tier_b_tasks():
        c: Counter = Counter()
        for t in tasks:
            tt = task_type(t.task_type)
            if not tt.covered:
                c["uncovered"] += 1
                if plans.reference_plan(t).operation is not None:
                    c["uncovered_with_plan"] += 1
                continue
            evidence = legal_all(t, cases[t.case_id])
            ref = _call_skill(OPERATIONS[tt.skill_op], evidence, t)          # the reference system's skill call
            res = plans.execute(plans.reference_plan(t), evidence, t, OPERATIONS)
            c["covered"] += 1
            c["summary_identical"] += res is not None and res.summary == ref
            c["required_equals_R_t"] += res is not None and res.required == list(t.required_ids)
            c["required_set_equals_R_t"] += res is not None and set(res.required) == set(t.required_ids)
            c["plan_answer_correct"] += res is not None and res.ok and tt.answer_correct(tt.skill_answer(res.summary, t, evidence), t)
        out[split] = dict(c)
    return out


# ------------------------------------------------------------------------------------------- Tier-C
def tier_c_context():
    from scopebench.schema import Task, read_jsonl
    from scopebench.tierc.store import load_store
    from run_tierc import FHIR_DIR, PATIENT_MAP, SKILL_VALIDATION

    tasks = [Task.from_dict(r) for r in read_jsonl(TIER_C / "tasks.jsonl")]
    t0 = time.time()
    store = load_store(FHIR_DIR)
    print(f"Tier-C store loaded in {time.time() - t0:.0f}s", flush=True)
    patient_map = json.load(open(PATIENT_MAP))
    return tasks, SimpleNamespace(store=store), patient_map, SKILL_VALIDATION


def identity_c() -> dict:
    from scopebench.tierc.plans import Executor, reference_plan
    from scopebench.tierc.skills import mapped_skills

    tasks, ts, patient_map, validation = tier_c_context()
    skills = mapped_skills(validation, patient_map)
    ex = Executor(patient_map)
    by_split: dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        c = by_split[t.split]
        skill = skills.get(t.metadata.get("template", ""))
        plan = reference_plan(t, ex.catalog)
        if skill is None:
            c["uncovered"] += 1
            c["uncovered_with_plan"] += plan.operation is not None
            continue
        c["covered"] += 1
        ref = skill(ts, t)
        ref_pairs = [r.get("id") for r, _ in ref.pop("_provenance_pairs", [])]
        res = ex(plan, ts, t, t.metadata.get("assumption"))
        if res is None:
            c["no_plan"] += 1
            continue
        got_pairs = [r.get("id") for r, _ in res.summary.get("_provenance_pairs", [])]
        c["summary_identical"] += public(res.summary) == public(ref)
        c["provenance_pairs_identical"] += got_pairs == ref_pairs
        c["required_set_equals_R_t"] += set(res.required) == set(t.required_ids)
    return {s: dict(c) for s, c in sorted(by_split.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["identity", "rules"])
    ap.add_argument("--tiers", nargs="*", default=["B", "C"])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"command": args.command, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.command == "identity":
        if "B" in args.tiers:
            result["tier_b"] = identity_b()
        if "C" in args.tiers:
            result["tier_c"] = identity_c()
    else:
        from plan_rules_eval import evaluate   # noqa: E402  (scripts/scopebench/plan_rules_eval.py)

        result.update(evaluate(args.tiers))
    path = args.out_dir / f"{args.command}.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
