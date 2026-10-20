#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# MedAgentBench-related adaptation; Copyright (c) 2025 Stanford Machine Learning Group.
# Upstream license: licenses/MedAgentBench-MIT.txt; changes: THIRD_PARTY_NOTICES.md
"""DEV-37: re-grade the MedAgentBench `full` rows whose answer the reference skill replaced, with the released grader's
answer shapes (task4 [value]; task10 [value, the recorded effectiveDateTime string]). Before DEV-37 the skill returned
the two shapes swapped, so every replaced task4 answer and every replaced task10 answer with a value was graded wrong.

The skill answer enters a `full` episode only at replacement (finalize), after the last model call, so the corrected
episode is the recorded one with the replaced answer reshaped; nothing is re-generated. The grader also reads the
episode's POST history (task4 and recent task10 fail on any agent message containing "POST"; old task10 needs the
order). The full history is not stored, so a replaced row is re-graded only when its access log has no POST attempt
(valid or invalid) and neither recorded response mentions POST: then the grader sees no POST and its verdict depends on
the answer alone, exactly as in the episode. Other replaced rows stay as recorded and are listed as unresolved. As a
diagnostic, every row is also re-graded with its recorded answer from the stored accepted POSTs (rows with POSTs the
released parser reads differently from the run's parser show up there; none of them is re-graded).

Needs the local MedAgentBench FHIR server (the grader queries it). Loads no model.
Writes <runs>/<tag>/medagentbench/full/predictions_dev37.jsonl (all rows; changed rows carry "dev37") and a summary.
Usage: rescore_medagent_dev37.py [--tags ...] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/scopebench"))

from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.tierc.medagent import FHIR_BASE, _GraderResult, _HistoryItem, get_json_text  # noqa: E402

E = ROOT / "experiments"
RUNS = E / "scopebench_tierC_runs"
TAGS = ("qwen3_4b", "phi4_14b", "gemma3_27b", "medgemma_27b", "qwen3_32b", "qwen3_30b", "frontier")


def history(row: dict) -> list[_HistoryItem]:
    """The grader-visible POST history: each accepted POST and the server's acceptance (the only history refsol reads
    for the covered categories, besides check_has_post, which the task4 filter handles)."""
    h = []
    for p in row.get("posts") or []:
        h.append(_HistoryItem("agent", f"POST {p['url']}\n{json.dumps(p['payload'])}"))
        h.append(_HistoryItem("user", "POST request accepted and executed successfully."))
    return h


def grade(refsol, case: dict, row: dict, answer_text: str | None) -> bool:
    try:
        return bool(getattr(refsol, row["category"])(case, _GraderResult(history=history(row), result=answer_text), FHIR_BASE))
    except Exception:  # noqa: BLE001 — the run's grade() maps exceptions to False as well
        return False


def corrected_answer(row: dict) -> list | None:
    s = row["skill_summary"]
    if not s or s.get("status") != "ok" or not s.get("provenance"):
        return None
    if row["category"] == "task4":
        return s["answer"][:1]
    if row["category"] == "task10":
        obs = json.loads(get_json_text(f"{FHIR_BASE}Observation/{s['provenance'][0]}?_format=json")["data"])
        return [s["answer"][0], obs["effectiveDateTime"]]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", default=list(TAGS))
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/medagent_dev37/rescore.json")
    args = ap.parse_args()
    from run_medagentbench_corrected import load_official_grader
    from run_tierc_medagent import TASK_DATA
    refsol, _ = load_official_grader()
    refsol.send_get_request = get_json_text
    cases = {c["id"]: c for c in json.load(open(TASK_DATA))}
    out = {}
    for tag in args.tags:
        d = RUNS / tag / "medagentbench/full"
        if not (d / "predictions.jsonl").exists():
            continue
        rows = list(read_jsonl(d / "predictions.jsonl"))
        posts_tried = {r["task_id"]: any(e["tool"] == "POST" for e in r["entries"]) for r in read_jsonl(d / "access_log.jsonl")}
        mismatch = [r["task_id"] for r in rows
                    if grade(refsol, cases[r["case_id"]], r, (r["parsed_answer"] or {}).get("answer_text")) != r["metrics"]["answer_correct"]]
        changed, unresolved, new_rows = [], [], []
        for r in rows:
            if r["guard_deterministic_fallback"] and r["category"] in ("task4", "task10"):
                ans = corrected_answer(r)
                mentions_post = posts_tried.get(r["task_id"], False) or "POST" in (r.get("response") or "") or "POST" in (r.get("repair_response") or "")
                if ans is None or mentions_post:
                    unresolved.append(r["task_id"])
                else:
                    text = json.dumps(ans)
                    ok = grade(refsol, cases[r["case_id"]], r, text)
                    changed.append({"task_id": r["task_id"], "category": r["category"], "before": r["parsed_answer"]["answer"], "after": ans,
                                    "correct_before": r["metrics"]["answer_correct"], "correct_after": ok})
                    r = {**r, "parsed_answer": {**r["parsed_answer"], "answer": ans, "answer_text": text},
                         "skill_summary": {**r["skill_summary"], "answer": ans},
                         # workflow compliance = correct and R_t within the legal citations; `full` rows access nothing illegal
                         "metrics": {**r["metrics"], "answer_correct": ok, "workflow_compliant": ok and r["metrics"]["completeness"] == 1.0},
                         "benchmark_native": {**r["benchmark_native"], "correct": ok}, "dev37": True}
            new_rows.append(r)
        n = len(rows)
        acc0 = sum(r["metrics"]["answer_correct"] for r in rows) / n
        acc1 = sum(r["metrics"]["answer_correct"] for r in new_rows) / n
        out[tag] = {"n": n, "diagnostic_post_history_mismatches": mismatch, "replaced_task4_task10": len(changed) + len(unresolved), "unresolved": unresolved,
                    "now_correct": sum(c["correct_after"] for c in changed), "answer_correct_pct_before": round(100 * acc0, 2),
                    "answer_correct_pct_after": round(100 * acc1, 2), "changed": changed}
        with open(d / "predictions_dev37.jsonl", "w") as f:
            for r in new_rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"{tag:13s} diagnostic mismatches {len(mismatch)}  replaced t4/t10 {out[tag]['replaced_task4_task10']}  now correct {out[tag]['now_correct']}"
              f"  unresolved {len(unresolved)}  answer_correct {out[tag]['answer_correct_pct_before']} -> {out[tag]['answer_correct_pct_after']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
