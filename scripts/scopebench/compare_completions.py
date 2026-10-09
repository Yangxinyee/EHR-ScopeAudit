#!/usr/bin/env python3
"""Compare two completion sets on the same prompts (e.g. SGLang backend X vs the transformers reference):
byte-identical count, parsed-JSON-identical count (LabTrend schema), length statistics, and examples.
Inputs: JSONL rows with task_id and completion (a determinism_runs.jsonl may be filtered by --run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import parser  # noqa: E402


def load(path: Path, run: int | None) -> dict[str, str]:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if run is not None and r.get("run") != run:
            continue
        out[r["task_id"]] = r["completion"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--a-run", type=int, default=None)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--b-run", type=int, default=None)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    a, b = load(args.a, args.a_run), load(args.b, args.b_run)
    ids = sorted(set(a) & set(b))
    byte = parsed = 0
    ex = []
    la = lb = 0
    for i in ids:
        la += len(a[i]); lb += len(b[i])
        pa, pb = parser.extract_json(a[i]), parser.extract_json(b[i])
        byte += a[i] == b[i]
        same = pa == pb
        parsed += same
        if not same and len(ex) < 5:
            ex.append({"task_id": i, args.label_a: a[i][:300], args.label_b: b[i][:300]})
    res = {"n": len(ids), "byte_identical": byte, "parsed_identical": parsed, f"mean_chars_{args.label_a}": round(la / max(1, len(ids))),
           f"mean_chars_{args.label_b}": round(lb / max(1, len(ids))), "examples": ex}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
    for e in ex:
        print("  DIFF", e["task_id"], "|", args.label_a, ":", repr(e[args.label_a][:160]), "|", args.label_b, ":", repr(e[args.label_b][:160]))
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
