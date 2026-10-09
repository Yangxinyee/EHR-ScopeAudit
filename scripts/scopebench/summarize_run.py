#!/usr/bin/env python3
"""Summarize one or more ScopeBench run directories: endpoint means overall and by scope, stage counts.
Loads no model. Usage: summarize_run.py <run_dir> [<run_dir> ...] [--json out.json]"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402

ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "workflow_compliant", "abstained", "parse_failed"]


def summarize(run_dir: Path) -> dict:
    rows = list(read_jsonl(run_dir / "predictions.jsonl"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[f"scope={r['scope']}"].append(r)
    out = {"run": str(run_dir), "configuration": rows[0]["configuration"] if rows else None, "n": len(rows), "groups": {}}
    for name, rs in groups.items():
        m = {ep: 100.0 * sum(float(r["metrics"][ep]) for r in rs) / len(rs) for ep in ENDPOINTS}
        m["initial_answer_correct"] = 100.0 * sum(float(r["initial_metrics"]["answer_correct"]) for r in rs) / len(rs)
        m["repaired"] = 100.0 * sum(bool(r["guard_repaired"]) for r in rs) / len(rs)
        m["replaced"] = 100.0 * sum(bool(r["guard_deterministic_fallback"]) for r in rs) / len(rs)
        m["generation_errors"] = 100.0 * sum(bool(r.get("generation_errors")) for r in rs) / len(rs)
        m["mean_tokens_out"] = sum(r["metrics"]["tokens_out"] for r in rs) / len(rs)
        m["mean_calls"] = sum(r["metrics"]["calls"] for r in rs) / len(rs)
        out["groups"][name] = {k: round(v, 2) for k, v in m.items()}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    results = [summarize(d) for d in args.run_dirs]
    for res in results:
        g = res["groups"].get("all", {})
        print(f"{res['configuration']:<28} n={res['n']:<5} correct={g.get('answer_correct'):>6}  init={g.get('initial_answer_correct'):>6}  "
              f"compl={g.get('completeness'):>6}  A-L={g.get('access_legal'):>6}  C-L={g.get('citation_legal'):>6}  aud={g.get('auditable'):>6}  "
              f"abst={g.get('abstained'):>5}  parsefail={g.get('parse_failed'):>5}  repaired={g.get('repaired'):>5}  replaced={g.get('replaced'):>5}  generr={g.get('generation_errors'):>5}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
