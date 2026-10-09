#!/usr/bin/env python3
"""List the frontier batch files that still need a completion file (plan §3.7 file-queue protocol).

A harness run with `--backend claude` writes `<out-dir>/batches/<prefix>_<n>.jsonl` and waits for the matching
`<prefix>_<n>.out.jsonl`. The session that drives the `frontier-llm` subagents calls this script, spawns one subagent per
pending batch (≤ 8 at a time), and repeats until the harness reports its configuration done.

Usage: frontier_pending.py <run dir or batches dir> [...] [--json]
Prints one absolute batch path per line (oldest first); with --json a list of {batch, out, n_rows, age_seconds}.
A row may occupy several lines (DEV-21 continuation lines, `"append"`); `n_rows` counts rows, not lines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def n_rows(batch: Path) -> int:
    return count(batch)[0]


def count(batch: Path) -> tuple[int, int]:
    """(rows, lines) of a batch file; the driving session names both in the subagent's task."""
    rows = lines = 0
    for line in open(batch, encoding="utf-8"):
        line = line.strip()
        if line:
            lines += 1
            if "append" not in json.loads(line):
                rows += 1
    return rows, lines


def pending(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        d = p / "batches" if (p / "batches").exists() else p
        for b in sorted(d.glob("*.jsonl")):
            if b.name.endswith(".out.jsonl"):
                continue
            o = b.with_suffix(".out.jsonl")
            n, n_lines = count(b)
            done = 0
            if o.exists():
                done = sum(1 for l in open(o, encoding="utf-8") if l.strip())
            if done < n:
                out.append({"batch": str(b.resolve()), "out": str(o.resolve()), "n_rows": n, "n_lines": n_lines, "rows_done": done,
                            "age_seconds": round(time.time() - b.stat().st_mtime)})
    return sorted(out, key=lambda x: -x["age_seconds"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = pending([Path(p) for p in a.paths])
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{r['batch']}\t{r['n_rows']} rows over {r['n_lines']} lines\t{r['rows_done']} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
