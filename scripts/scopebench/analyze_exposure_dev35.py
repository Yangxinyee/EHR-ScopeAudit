#!/usr/bin/env python3
"""DEV-35 analysis (experiments/SCOPEBENCH_EXPOSURE_PROTOCOL.md §4): paired effects of appending out-of-scope records
(`future`) or contract-permitted unseen records (`inscope`) to the scope-aware context, against `baseline` on the same
task. Loads no model.

Per task and arm: answer changed (parsed answer without top-level `citations` differs from the baseline's), correct
(as in the formal panel), adopted (correct under the counterfactual gold and not under the true gold: the task type's
own `answer_correct` with the gold replaced by the counterfactual answer), and cited-injected (the answer cites an
injected record). Cells: family x decisive (the generator's leakage flag) x arm, per checkpoint and pooled; primary
contrast future - inscope on the length-matched tasks (|inscope| = |future|), secondary future - baseline on all tasks.
Patient-cluster bootstrap. Also reports the agreement of `baseline` with the formal panel's scope_aware rows.
Usage: analyze_exposure_dev35.py [--tags ...] [--out FILE] [--n-boot N]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import tasktypes_tierb  # noqa: E402,F401
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.stats import cluster_bootstrap_mean, paired_cluster_contrast  # noqa: E402
from scopebench.tasktypes import get as task_type  # noqa: E402

E = ROOT / "experiments"
EXP = E / "scopebench_tierB/exposure_dev35"
TAGS = ("qwen3_4b", "phi4_14b", "qwen3_30b", "gemma3_27b", "medgemma_27b", "qwen3_32b")
ARMS = ("baseline", "future", "inscope")
MEASURES = ("changed", "correct", "adopted", "cites_injected")


def answer_key(parsed: Any) -> str:
    if isinstance(parsed, dict):
        parsed = {k: v for k, v in parsed.items() if k != "citations"}
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)


def per_task(rows: dict[str, dict[str, dict]], tasks: dict[str, Task], cf: dict[str, dict], inject: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """arm -> task -> measures (0/1); `baseline` defines `changed`."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    base = rows.get("baseline", {})
    for arm, arm_rows in rows.items():
        for i, r in arm_rows.items():
            t = tasks[i]
            tt = task_type(t.task_type)
            parsed = r.get("parsed_answer")
            cf_task = copy.copy(t)
            cf_task.gold = cf[i]["cf_answer"]
            correct = bool(r["metrics"]["answer_correct"])
            cf_correct = bool(cf[i]["cf_answer"].get("status", "ok") == "ok" and tt.answer_correct(parsed, cf_task))
            injected = set(inject[i].get(arm, [])) if arm != "baseline" else set()
            out[arm][i] = {"changed": float(i in base and answer_key(parsed) != answer_key(base[i].get("parsed_answer"))),
                           "correct": float(correct), "adopted": float(cf_correct and not correct),
                           "cites_injected": float(bool(injected & set(tt.citations(parsed))))}
    return out


def cells(meas: dict[str, dict[str, dict]], tasks: dict[str, Task], cf: dict[str, dict], n_boot: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for arm, by_task in meas.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for i in by_task:
            fam = cf[i.split("@")[0]]
            groups[f"{fam['family']}|{'decisive' if fam['leakage'] else 'non-decisive'}"].append(i)
            groups["all"].append(i)
        for g, ids in groups.items():
            cell = {"n": len(ids)}
            for m in MEASURES:
                m_, lo, hi = cluster_bootstrap_mean([by_task[i][m] for i in ids], [tasks[i].patient for i in ids], n_boot=n_boot)
                cell[m] = {"mean": round(100 * m_, 1), "ci": [round(100 * lo, 1), round(100 * hi, 1)]}
            res.setdefault(arm, {})[g] = cell
    return res


def contrasts(meas: dict[str, dict[str, dict]], tasks: dict[str, Task], cf: dict[str, dict], inject: dict[str, dict], n_boot: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    matched = {i for i in meas.get("future", {}) if len(inject[i.split("@")[0]]["inscope"]) == len(inject[i.split("@")[0]]["future"])}
    pairs = [("future - inscope (length-matched)", "inscope", "future", matched), ("future - baseline", "baseline", "future", None),
             ("inscope - baseline", "baseline", "inscope", None)]
    for name, a, b, restrict in pairs:
        if a not in meas or b not in meas:
            continue
        groups: dict[str, set[str]] = defaultdict(set)
        for i in set(meas[a]) & set(meas[b]):
            if restrict is not None and i not in restrict:
                continue
            fam = cf[i.split("@")[0]]
            groups[f"{fam['family']}|{'decisive' if fam['leakage'] else 'non-decisive'}"].add(i)
            groups["decisive" if fam["leakage"] else "non-decisive"].add(i)
        for g, ids in groups.items():
            for m in ("changed", "correct", "adopted"):
                va = {i: meas[a][i][m] for i in ids}
                vb = {i: meas[b][i][m] for i in ids}
                c = paired_cluster_contrast(va, vb, {i: tasks[i].patient for i in ids}, n_boot=n_boot)
                res.setdefault(name, {}).setdefault(g, {})[m] = {"delta": round(100 * c["delta"], 1), "ci": [round(100 * c["ci_low"], 1), round(100 * c["ci_high"], 1)],
                                                                 "p": c["p_bootstrap"], "n": c["n"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="*", default=list(TAGS))
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/exposure_dev35/tierB.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    cf = json.loads((EXP / "counterfactual.json").read_text())
    inject = json.loads((EXP / "inject_ids.json").read_text())
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(E / "scopebench_tierB/tasks.jsonl") if r["task_id"] in cf}
    result: dict[str, Any] = {"n_tasks": len(tasks), "checkpoints": {}}
    pooled: dict[str, dict[str, dict]] = defaultdict(dict)
    for tag in args.tags:
        rows = {a: {r["task_id"]: r for r in read_jsonl(p)} for a in ARMS if (p := E / "scopebench_tierB_runs_dev35" / tag / a / "predictions.jsonl").exists()}
        if "baseline" not in rows:
            continue
        panel = {r["task_id"]: r for r in read_jsonl(E / "scopebench_tierB_runs" / tag / "scope_aware/predictions.jsonl") if r["task_id"] in rows["baseline"]}
        agree = sum(1 for i, r in rows["baseline"].items() if i in panel and r["response"] == panel[i]["response"])
        meas = per_task(rows, tasks, cf, inject)
        result["checkpoints"][tag] = {"baseline_vs_panel_identical_responses": [agree, len(rows["baseline"])],
                                      "cells": cells(meas, tasks, cf, args.n_boot), "contrasts": contrasts(meas, tasks, cf, inject, args.n_boot)}
        for arm, by_task in meas.items():
            pooled[arm].update({f"{i}@{tag}": v for i, v in by_task.items()})
    if len(result["checkpoints"]) > 1:
        ptasks = {k: tasks[k.split("@")[0]] for r in pooled.values() for k in r}
        result["pooled"] = {"checkpoints": sorted(result["checkpoints"]), "cells": cells(pooled, ptasks, cf, args.n_boot),
                            "contrasts": contrasts(pooled, ptasks, cf, inject, args.n_boot)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for tag, e in list(result["checkpoints"].items()) + ([("POOLED", result["pooled"])] if "pooled" in result else []):
        if tag != "POOLED":
            print(f"{tag}: baseline identical to the panel's scope_aware response on {e['baseline_vs_panel_identical_responses']}")
        for name, groups in e["contrasts"].items():
            for g in ("decisive", "non-decisive"):
                if g in groups:
                    s = "  ".join(f"{m}={v['delta']:+.1f} [{v['ci'][0]},{v['ci'][1]}]" for m, v in groups[g].items())
                    print(f"  {tag:12s} {name:34s} {g:13s} n={groups[g]['changed']['n']:<5} {s}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
