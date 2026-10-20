#!/usr/bin/env python3
"""DEV-36 analysis (experiments/SCOPEBENCH_CONTROLS_PROTOCOL.md §4): the deployable `plan_full` against the scope-aware
pipeline with more inference compute — five seeds (each seed, self-consistency, pass@5), the scoped top-50 and
`self_revise` — on the DEV-34 Tier-B sample, per checkpoint and pooled. Loads no model.

Self-consistency: per task the most frequent answer over the five seeds (answer = the parsed JSON without its top-level
`citations` and `evidence_plan`, canonical form), ties broken by the earliest seed in the protocol order; the row of the
earliest seed giving that answer is scored. pass@5: correct in at least one seed (an upper bound for any selector).
Tier C (protocol addendum 1): the five seed cells on FHIR-AgentBench train + test against DEV-34 `plan_full` and the
DEV-32 formal rows; the vote is over the benchmark-native `answer` field; coverage as in analyze_plan_dev34.py.
Usage: analyze_controls_dev36.py [--tier B|C] [--tags ...] [--out FILE] [--n-boot N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/scopebench"))

from analyze_plan_dev34 import contrast, endpoint_cells  # noqa: E402
from scopebench import tasktypes_tierb  # noqa: E402,F401
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tasktypes import get as task_type  # noqa: E402

E = ROOT / "experiments"
SEEDS = ("seed13", "seed29", "seed47", "seed71", "seed101")
CELLS = SEEDS + ("k50", "self_revise")
TAGS = ("qwen3_4b", "qwen3_30b", "gemma3_27b")


def load(path: Path, keep: set[str]) -> dict[str, dict]:
    p = path / "predictions.jsonl"
    return {r["task_id"]: r for r in read_jsonl(p) if r["task_id"] in keep} if p.exists() else {}


def answer_key(parsed: Any, tier: str = "B") -> str:
    if tier == "C":
        return json.dumps(parsed.get("answer") if isinstance(parsed, dict) else None, sort_keys=True, ensure_ascii=False)
    if isinstance(parsed, dict):
        parsed = {k: v for k, v in parsed.items() if k not in ("citations", "evidence_plan")}
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)


def self_consistency(seed_rows: dict[str, dict[str, dict]], tier: str = "B") -> tuple[dict[str, dict], dict[str, dict]]:
    common = set.intersection(*(set(r) for r in seed_rows.values()))
    vote, best = {}, {}
    for i in sorted(common):
        rows = [seed_rows[s][i] for s in SEEDS]
        keys = [answer_key(r.get("parsed_answer"), tier) for r in rows]
        counts = Counter(keys)
        top = max(counts.values())
        winner = next(k for k in keys if counts[k] == top)          # earliest seed among the most frequent answers
        vote[i] = rows[keys.index(winner)]
        hit = next((r for r in rows if r["metrics"].get("answer_correct")), rows[0])
        best[i] = {**hit, "metrics": {**hit["metrics"], "answer_correct": any(r["metrics"].get("answer_correct") for r in rows)}}
    return vote, best


def compute(rows: dict[str, dict]) -> dict[str, Any]:
    vals = list(rows.values())
    if not vals:
        return {}
    return {"tokens_per_task": round(mean(r["metrics"]["tokens_in"] + r["metrics"]["tokens_out"] for r in vals)),
            "calls_per_task": round(mean(r["metrics"]["calls"] for r in vals), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", default="B", choices=["B", "C"])
    ap.add_argument("--tags", nargs="*", default=list(TAGS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    out = args.out or E / f"scopebench_analysis/controls_dev36/tier{args.tier}.json"
    if args.tier == "B":
        keep = set(json.loads((E / "scopebench_tierB/plan_dev34_task_ids.json").read_text()))
        tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(E / "scopebench_tierB/tasks.jsonl") if r["task_id"] in keep}
        covered = {i: task_type(t.task_type).covered for i, t in tasks.items()}
        plan_root, ref_root, ctl_root, cells = E / "scopebench_tierB_runs_dev34", E / "scopebench_tierB_runs", E / "scopebench_tierB_runs_dev36", CELLS
    else:
        from scopebench.tierc.plans import load_catalog, reference_plan

        keep = set(json.loads((E / "scopebench_tierC/plan_dev34/traintest_task_ids.json").read_text()))
        tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(E / "scopebench_tierC/fhir_agentbench/tasks.jsonl") if r["task_id"] in keep}
        cat = load_catalog()
        covered = {i: reference_plan(t, cat).operation is not None for i, t in tasks.items()}
        plan_root, ref_root, ctl_root, cells = E / "scopebench_tierC_runs_dev34", E / "scopebench_tierC_runs_dev32", E / "scopebench_tierC_runs_dev36", SEEDS
    result: dict[str, Any] = {"tier": args.tier, "n_tasks": len(tasks), "n_covered": sum(covered.values()), "checkpoints": {}}
    pooled: dict[str, dict[str, dict]] = defaultdict(dict)
    for tag in args.tags:
        rows = {"plan_full": load(plan_root / tag / "plan_full", keep),
                "scope_aware": load(ref_root / tag / "scope_aware", keep),
                "full": load(ref_root / tag / "full", keep)}
        rows.update({c: load(ctl_root / tag / c, keep) for c in cells})
        seed_rows = {s: rows[s] for s in SEEDS if rows.get(s)}
        # DEV-36 addendum 2: seed cells are independent samples only if different seeds give different responses (the
        # Tier-C raw path ignored the seed before the fix); a cell whose first two seeds agree on > 90 % of tasks is
        # reported per seed, without self-consistency or pass@5
        same = (sum(seed_rows["seed13"][i]["response"] == seed_rows["seed29"].get(i, {}).get("response") for i in seed_rows["seed13"]) / len(seed_rows["seed13"])
                if "seed13" in seed_rows and "seed29" in seed_rows else None)
        independent = same is not None and same <= 0.9
        if len(seed_rows) == len(SEEDS) and independent:
            rows["self_consistency@5"], rows["pass@5"] = self_consistency(seed_rows, args.tier)
        rows = {c: r for c, r in rows.items() if r}
        if "plan_full" not in rows:
            continue
        comp = {c: compute(r) for c, r in rows.items() if c not in ("self_consistency@5", "pass@5")}
        if len(seed_rows) == len(SEEDS) and independent:
            comp["self_consistency@5"] = {"tokens_per_task": sum(comp[s]["tokens_per_task"] for s in SEEDS), "calls_per_task": 5.0}
        result["checkpoints"][tag] = {
            "endpoints": {c: endpoint_cells(r, tasks, covered, args.n_boot) for c, r in rows.items()},
            "contrasts": {f"plan_full - {c}": contrast(rows[c], rows["plan_full"], tasks, covered, args.n_boot) for c in rows if c != "plan_full"},
            "compute": comp, "seeds_identical_response_rate_13_29": round(100 * same, 1) if same is not None else None, "seeds_independent": independent}
        for c, r in rows.items():
            pooled[c].update({f"{i}@{tag}": v for i, v in r.items()})
    if len(result["checkpoints"]) > 1:
        ptasks = {k: tasks[k.split("@")[0]] for r in pooled.values() for k in r}
        pcov = {k: covered[k.split("@")[0]] for k in ptasks}
        complete = [c for c in pooled if all(c in e["endpoints"] for e in result["checkpoints"].values())]
        result["pooled"] = {"checkpoints": sorted(result["checkpoints"]),
                            "endpoints": {c: endpoint_cells(pooled[c], ptasks, pcov, args.n_boot) for c in complete},
                            "contrasts": {f"plan_full - {c}": contrast(pooled[c], pooled["plan_full"], ptasks, pcov, args.n_boot) for c in complete if c != "plan_full"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for tag, e in result["checkpoints"].items():
        for c, cell in e["endpoints"].items():
            k = e["compute"].get(c, {})
            print(f"{tag:11s} {c:20s} n={cell['all']['n']:<5} answer={cell['all']['answer_correct']['mean']:>5}  covered={cell['covered']['answer_correct']['mean']:>5}"
                  f"  uncovered={cell['uncovered']['answer_correct']['mean']:>5}  tokens={k.get('tokens_per_task', '-')}  calls={k.get('calls_per_task', '-')}")
    for k, v in result.get("pooled", {}).get("contrasts", {}).items():
        a = v["all"].get("answer_correct", {})
        print(f"pooled {k:34s} {a.get('delta', 0):+.1f} [{a.get('ci', ['', ''])[0]}, {a.get('ci', ['', ''])[1]}]")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
