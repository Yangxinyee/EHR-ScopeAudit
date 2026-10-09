#!/usr/bin/env python3
"""DEV-23: relabel MedAgentBench episodes that ended with `FINISH(...)` followed by their Citations line.

Before DEV-23 the loop's FINISH pattern required the response to end at the closing parenthesis, so a final turn that
carried the Citations line the ScopeBench addendum asks for was filed as `invalid_action` although it was graded as the
final answer (`parse_final` strips the Citations line first). Only the label was wrong: `benchmark_native.status` in
predictions.jsonl and `status` in stages.jsonl. This script applies the corrected pattern to the stored `response` of
every row whose status is `invalid_action`, rewrites those two fields in place (write to a temporary file, then
replace), and records the counts in metadata.json under `status_relabel_dev23`. Nothing else is touched; rows that were
graded, scored or filtered keep every other value.

Usage: relabel_medagent_status.py <run dir> [<run dir> ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scopebench.tierc.medagent import FINISH_RE, first_action  # noqa: E402


def _rewrite(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    os.replace(tmp, path)


def relabel(run_dir: Path) -> dict:
    pred_path, stages_path, meta_path = run_dir / "predictions.jsonl", run_dir / "stages.jsonl", run_dir / "metadata.json"
    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    changed: set[str] = set()
    for r in rows:
        bn = r["benchmark_native"]
        if bn.get("status") == "invalid_action" and FINISH_RE.search(first_action(r.get("response") or "")):
            bn["status"] = "completed"
            changed.add(r["task_id"])
    n_stages = 0
    if stages_path.exists():
        stages = [json.loads(l) for l in open(stages_path, encoding="utf-8") if l.strip()]
        for s in stages:
            if s["task_id"] in changed and s.get("status") == "invalid_action":
                s["status"] = "completed"
                n_stages += 1
        if n_stages:
            _rewrite(stages_path, stages)
    if changed:
        _rewrite(pred_path, rows)
    record = {"applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows_relabelled": len(changed), "stages_relabelled": n_stages,
              "still_invalid_action": sum(1 for r in rows if r["benchmark_native"].get("status") == "invalid_action"), "n_rows": len(rows)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["status_relabel_dev23"] = record
        _rewrite_meta = meta_path.with_name(meta_path.name + ".tmp")
        _rewrite_meta.write_text(json.dumps(meta, indent=1, sort_keys=True))
        os.replace(_rewrite_meta, meta_path)
    return record


def main() -> None:
    for d in sys.argv[1:]:
        print(d, json.dumps(relabel(Path(d))))


if __name__ == "__main__":
    main()
