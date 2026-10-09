#!/usr/bin/env python3
"""GPU regression report (plan §12 M1): a ScopeBench run of a ScopePair-compat configuration versus
the frozen ScopePair run of the same system and checkpoint.

Per task: parsed-answer agreement, answer-correct agreement (both sides scored with the ScopeBench
scorer), guard-repaired / replaced flag agreement. Prints and writes a JSON summary.
Usage: compare_to_frozen.py --run <scopebench run dir> --frozen <frozen predictions.jsonl> --system <frozen system name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import scoring  # noqa: E402
from scopebench.compat_scopepair import labtrend_tasks  # noqa: E402
from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.tasktypes import get as get_task_type  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/scopepair_fhir/tasks.jsonl")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    tasks, cases = labtrend_tasks(args.manifest, split=None)
    task_by_id = {t.task_id: t for t in tasks}
    ours = {r["task_id"]: r for r in read_jsonl(args.run / "predictions.jsonl")}
    theirs = {r["task_id"]: r for r in read_jsonl(args.frozen) if r["system"] == args.system}
    common = sorted(set(ours) & set(theirs))
    agree = {"parsed": 0, "correct": 0, "repaired": 0, "replaced": 0}
    ours_correct = theirs_correct = 0
    diffs = []
    for tid in common:
        o, t = ours[tid], theirs[tid]
        task = task_by_id[tid]
        tt = get_task_type(task.task_type)
        pool = cases[task.case_id].by_id()
        o_correct = bool(o["metrics"]["answer_correct"])
        t_correct = bool(tt.answer_correct(t["parsed_answer"], task))
        ours_correct += o_correct
        theirs_correct += t_correct
        agree["parsed"] += o["parsed_answer"] == t["parsed_answer"]
        agree["correct"] += o_correct == t_correct
        agree["repaired"] += bool(o["guard_repaired"]) == bool(t["guard_repaired"])
        agree["replaced"] += bool(o["guard_deterministic_fallback"]) == bool(t["guard_deterministic_fallback"])
        if o["parsed_answer"] != t["parsed_answer"] and len(diffs) < 10:
            diffs.append({"task_id": tid, "ours": o["parsed_answer"], "frozen": t["parsed_answer"], "ours_correct": o_correct, "frozen_correct": t_correct})
    n = len(common)
    summary = {
        "system": args.system, "n_common": n, "n_ours": len(ours), "n_frozen": len(theirs),
        "agreement_pct": {k: round(100.0 * v / n, 2) for k, v in agree.items()} if n else {},
        "answer_correct_pct": {"scopebench_run": round(100.0 * ours_correct / n, 2) if n else None, "frozen_run": round(100.0 * theirs_correct / n, 2) if n else None},
        "examples": diffs,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
