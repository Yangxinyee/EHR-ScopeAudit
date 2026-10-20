#!/usr/bin/env python3
"""DEV-35 (experiments/SCOPEBENCH_EXPOSURE_PROTOCOL.md): fix, before any run, the records each exposure arm appends to the
scope-aware context. Loads no model.

Task set: the pre-index side of the three leakage-decidable Tier-B families (test split) —
  reconstruct_v1 / pre   counterfactual = the paired `at` task (the index report, illegal under `pre`)
  cohort_v1 / at         counterfactual = the paired `full` task (records after the admission)
  lab_trend_k3 / pre     counterfactual = the same question on the occurrence basis (results drawn before the index but
                         available only after it)
For every task:
  future   = the out-of-scope part of the counterfactual answer's required evidence (R_cf minus L_t), nearest to the index
             first, at most 10; when that is empty, the 3 out-of-scope records of the task's permitted sources nearest to
             the index (recorded as `nearest`)
  inscope  = as many contract-permitted records the scoped selection did not show, nearest to the index first
The scoped selection is the formal `scope_aware` one (deterministic and model-independent: the retrieval cache is shared
by all checkpoints; checked here against a second checkpoint).

Writes (MIMIC-derived; under the gitignored Tier-B tree): <out>/task_ids.json, <out>/inject_ids.json,
<out>/counterfactual.json (per task: family, leakage, source of the future set, counterfactual answer and R_cf), and
prints a summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import oracle  # noqa: E402
from scopebench.manifest import load_manifest  # noqa: E402
from scopebench.predicate import filter_legal, legal  # noqa: E402
from scopebench.schema import Contract, parse_time, read_jsonl  # noqa: E402

FAMILIES = {("reconstruct_v1", "pre"): "at", ("cohort_v1", "at"): "full", ("lab_trend_k3", "pre"): None}
MAX_FUTURE, NEAREST_FALLBACK = 10, 3


def _distance(e, contract: Contract) -> float:
    t = e.time(contract.basis)
    return abs((t - parse_time(contract.index_time)).total_seconds()) if t is not None else float("inf")


def _nearest(items, contract: Contract) -> list:
    return sorted(items, key=lambda e: (_distance(e, contract), e.id))


def _selected(path: Path) -> dict[str, list[str]]:
    return {r["task_id"]: r["selected_evidence_ids"] for r in read_jsonl(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/scopebench_tierB")
    ap.add_argument("--selection", type=Path, default=ROOT / "experiments/scopebench_tierB_runs/qwen3_4b/scope_aware/predictions.jsonl")
    ap.add_argument("--selection-check", type=Path, default=ROOT / "experiments/scopebench_tierB_runs/qwen3_30b/scope_aware/predictions.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/scopebench_tierB/exposure_dev35")
    args = ap.parse_args()

    tasks, cases = load_manifest(args.manifest, "test")
    by_id = {t.task_id: t for t in tasks}
    shown, check = _selected(args.selection), _selected(args.selection_check)
    chosen = sorted((t for t in tasks if (t.task_type, t.scope) in FAMILIES), key=lambda t: t.task_id)
    mismatch = sum(1 for t in chosen if shown.get(t.task_id) != check.get(t.task_id))
    if mismatch:
        raise SystemExit(f"scoped selections differ between checkpoints on {mismatch} tasks")

    inject, cf_rows, stats = {}, {}, Counter()
    for t in chosen:
        case = cases[t.case_id]
        legal_ids = {e.id for e in filter_legal(case.evidence, t.contract)}
        partner_scope = FAMILIES[(t.task_type, t.scope)]
        if partner_scope is not None:
            partner = by_id[f"{t.case_id}-{partner_scope}"]
            cf_required, cf_answer = list(partner.required_ids), partner.gold
        else:
            wb = Contract(index_time=t.contract.index_time, scope="pre", basis="occurred", permitted_sources=t.contract.permitted_sources)
            g = oracle.OPERATIONS["lab_trend_k3"](filter_legal(case.evidence, wb), wb, {"k": 3})
            if g["status"] != "ok":
                cf_required, cf_answer = [], {"status": g["status"]}
            else:
                cf_required = list(g["ordered_result_ids"])
                cf_answer = {"status": "ok", "results": g["provenance"], "delta": g["delta_last_minus_first"], "direction": g["direction"]}
        pool = case.by_id()
        future = _nearest([pool[i] for i in cf_required if i in pool and i not in legal_ids], t.contract)[:MAX_FUTURE]
        source = "counterfactual"
        if not future:
            permitted = set(t.contract.permitted_sources) if t.contract.permitted_sources else None
            outside = [e for e in case.evidence if e.id not in legal_ids and (permitted is None or e.source in permitted)]
            future, source = _nearest(outside, t.contract)[:NEAREST_FALLBACK], "nearest"
        unseen = [e for e in case.evidence if e.id in legal_ids and e.id not in set(shown[t.task_id])]
        inscope = _nearest(unseen, t.contract)[:len(future)]
        assert all(not legal(e, t.contract) for e in future) and all(legal(e, t.contract) for e in inscope)
        inject[t.task_id] = {"future": [e.id for e in future], "inscope": [e.id for e in inscope]}
        cf_rows[t.task_id] = {"family": t.task_type, "scope": t.scope, "leakage": t.leakage, "future_source": source,
                              "cf_answer": cf_answer, "cf_required_ids": cf_required}
        stats[(t.task_type, t.leakage, source)] += 1
        stats[("n_future", len(future))] += 1
        stats[("inscope_short", len(inscope) < len(future))] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    ids = sorted(inject)
    (args.out / "task_ids.json").write_text(json.dumps(ids) + "\n")
    (args.out / "inject_ids.json").write_text(json.dumps(inject, sort_keys=True) + "\n")
    (args.out / "counterfactual.json").write_text(json.dumps(cf_rows, sort_keys=True) + "\n")
    print(f"{len(ids)} tasks; scoped selections identical across the two checkpoints")
    for k, v in sorted(stats.items(), key=str):
        print(" ", k, v)
    for name in ("task_ids.json", "inject_ids.json", "counterfactual.json"):
        print(f"  {name} sha256 {hashlib.sha256((args.out / name).read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
