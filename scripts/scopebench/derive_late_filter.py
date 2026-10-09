#!/usr/bin/env python3
"""Derive the late-filtering control from a prompt_only run (plan §4.2): drop illegal citations from
each prompt-only answer without refill, re-score, and write a run directory that looks like any other
configuration's output. Loads no model.

Usage: derive_late_filter.py --manifest <dir> [--loader scopebench|scopepair_labtrend] --prompt-only <run dir> --out <run dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import __version__, scoring  # noqa: E402
from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.tasktypes import get as get_task_type  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--loader", default="scopebench", choices=["scopebench", "scopepair_labtrend", "tierc"])
    p.add_argument("--prompt-only", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cases = None
    store = None
    if args.loader == "scopebench":
        from scopebench.manifest import load_manifest

        tasks, cases = load_manifest(args.manifest)
    elif args.loader == "tierc":
        # Tier-C rows carry no evidence pool: legality is decided against the FHIR store for the ids the row touches
        from scopebench import tasktypes_tierc  # noqa: F401
        from scopebench.config import PUBLIC_DATA
        from scopebench.schema import Task
        from scopebench.tierc.contracts import resource_to_evidence
        from scopebench.tierc.store import load_store

        tasks = [Task.from_dict(r) for r in read_jsonl(args.manifest / "tasks.jsonl")]
        store = load_store(PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir")
    else:
        from scopebench.compat_scopepair import labtrend_tasks

        tasks, cases = labtrend_tasks(args.manifest)
    task_by_id = {t.task_id: t for t in tasks}
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in read_jsonl(args.prompt_only / "predictions.jsonl"):
            task = task_by_id[row["task_id"]]
            tt = get_task_type(task.task_type)
            if cases is not None:
                pool = cases[task.case_id].by_id()
            else:
                pool = {}
                for rid in set(row.get("access_ids", [])) | set(tt.citations(row.get("parsed_answer"))) | set(task.required_ids):
                    for table in store.by_type_id.values():
                        if rid in table:
                            pool[rid] = resource_to_evidence(table[rid], task.patient)
                            break
            filtered = scoring.late_filter(row["parsed_answer"], task, tt, pool)
            m = row["metrics"]
            capped = False
            if args.loader == "tierc":
                # DEV-27: a Tier-C row keeps its id list only below the cap; the kept citations count as accessed when it was capped
                from scopebench.tierc.agent import late_filter_access
                access, capped = late_filter_access(row, tt.citations(filtered))
            else:
                access = set(row.get("access_ids", []))
            metrics = scoring.score(filtered, task, tt, pool, access, tokens_in=m["tokens_in"], tokens_out=m["tokens_out"], calls=m["calls"], latency=m["latency_seconds"])
            if args.loader == "tierc":
                # A_t is what the episode accessed; late filtering changes citations only, so exposure is the original row's
                metrics["access_legal"] = bool(m["access_legal"]); metrics["n_illegal_access"] = m.get("n_illegal_access", metrics["n_illegal_access"])
                metrics["workflow_compliant"] = bool(metrics["substantive"] and metrics["access_legal"] and metrics["citation_legal"])
            out = dict(row)
            out.update({"configuration": "late_filter", "parsed_answer": filtered, "metrics": metrics, "derived_from": str(args.prompt_only.name),
                        "late_filter_removed": len(tt.citations(row["parsed_answer"])) - len(tt.citations(filtered)), "late_filter_access_list_capped": capped})
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    src_meta = json.loads((args.prompt_only / "metadata.json").read_text()) if (args.prompt_only / "metadata.json").exists() else {}
    (args.out / "metadata.json").write_text(json.dumps({"scopebench_version": __version__, "configuration": {"name": "late_filter", "derived_from": "prompt_only"},
                                                        "source_run": str(args.prompt_only), "source_metadata": src_meta, "n_tasks_this_run": n,
                                                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1, sort_keys=True) + "\n")
    print(f"late_filter: {n} tasks derived into {args.out}")


if __name__ == "__main__":
    main()
