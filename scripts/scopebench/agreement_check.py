#!/usr/bin/env python3
"""Run-to-run agreement of one configuration repeated in fresh output directories on the same tasks
(protocol: D10 three-run verification per checkpoint — Qwen/Phi in deterministic-triton mode must be
byte-identical; the Gemma class in normal fa4 mode reports its measured noise floor).

Usage: agreement_check.py <dir-with-run1..runN> [--json out.json]
Each run directory holds a predictions.jsonl; rows are matched by task_id. Reports, for every pair of
runs, byte-identical `response` share and parsed-answer-identical share, plus answer_correct per run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402


def load(d: Path) -> dict[str, dict]:
    return {r["task_id"]: r for r in read_jsonl(d / "predictions.jsonl")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", type=Path)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists())
    if len(dirs) < 2:
        raise SystemExit(f"need at least two run directories under {args.runs_dir}")
    runs = {d.name: load(d) for d in dirs}
    common = set.intersection(*(set(r) for r in runs.values()))
    out = {"runs_dir": str(args.runs_dir), "runs": sorted(runs), "n_common_tasks": len(common), "per_run": {}, "pairs": {}}
    for name, rows in runs.items():
        rs = [rows[t] for t in sorted(common)]
        out["per_run"][name] = {"n": len(rs), "answer_correct": sum(bool(r["metrics"]["answer_correct"]) for r in rs),
                                "completeness_mean": round(100 * sum(float(r["metrics"]["completeness"]) for r in rs) / max(1, len(rs)), 2),
                                "parse_failed": sum(bool(r["metrics"]["parse_failed"]) for r in rs)}
    for a, b in itertools.combinations(sorted(runs), 2):
        ra, rb = runs[a], runs[b]
        byte_same = sum(1 for t in common if (ra[t].get("response") or "") == (rb[t].get("response") or ""))
        parsed_same = sum(1 for t in common if ra[t].get("parsed_answer") == rb[t].get("parsed_answer"))
        correct_same = sum(1 for t in common if bool(ra[t]["metrics"]["answer_correct"]) == bool(rb[t]["metrics"]["answer_correct"]))
        out["pairs"][f"{a}~{b}"] = {"byte_identical": byte_same, "parsed_identical": parsed_same, "correctness_identical": correct_same, "n": len(common)}
    out["min_byte_identical_share"] = round(min(v["byte_identical"] / max(1, v["n"]) for v in out["pairs"].values()), 4)
    out["min_parsed_identical_share"] = round(min(v["parsed_identical"] / max(1, v["n"]) for v in out["pairs"].values()), 4)
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
