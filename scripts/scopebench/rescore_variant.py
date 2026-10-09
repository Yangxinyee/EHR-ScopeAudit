#!/usr/bin/env python3
"""Re-score a Tier-B run against the corrected sensitivity-variant tasks (DEV-14). Loads no model.

  in place:  rescore_variant.py --variant strict_boundaries --run-dir <runs>/<tag>/sensitivity/strict_boundaries/prompt_only
             (prompt_only only: its generation does not depend on the gold answer or R_t, so re-scoring equals re-running;
             the original directory is kept as <config>_v0_pre_dev14)
  derived:   rescore_variant.py --variant extended_rt --source <runs>/<tag>/full --run-dir <runs>/<tag>/sensitivity/extended_rt/full
             (extended_rt is a scoring denominator: the runner sees the reference tasks, so the cell equals the reference
             run re-scored with the five-result R_t; `derived_from.json` records the source)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.manifest import load_manifest  # noqa: E402
from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.sensitivity import apply_variant, extended_scoring_tasks, rescore_rows  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["strict_boundaries", "at_window_6h", "extended_rt"])
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/scopebench_tierB")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, default=None, help="derive the run dir from this reference run (extended_rt)")
    args = p.parse_args()
    tasks, cases = load_manifest(args.manifest)
    dropped: list[str] = []
    vtasks = extended_scoring_tasks(tasks, cases) if args.variant == "extended_rt" else apply_variant(tasks, args.variant, cases, dropped)
    tmap = {t.task_id: t for t in vtasks}
    src = args.source or args.run_dir
    rows = list(read_jsonl(src / "predictions.jsonl"))
    if args.source is None:
        if args.run_dir.name != "prompt_only":
            raise SystemExit("in-place re-scoring is valid for prompt_only only (its generation ignores gold and R_t)")
        backup = args.run_dir.parent / f"{args.run_dir.name}_v0_pre_dev14"
        if not backup.exists():
            shutil.copytree(args.run_dir, backup)
    else:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("access_log.jsonl", "stages.jsonl"):
            if (src / name).exists():
                shutil.copyfile(src / name, args.run_dir / name)
    new = rescore_rows(rows, tmap, cases)
    with open(args.run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    note = {"deviation": "DEV-14", "variant": args.variant, "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_rows": len(new),
            "n_source_rows": len(rows), "dropped_tasks": sorted(dropped),
            "source_run": str(src), "source_predictions_sha256": _sha(src / "predictions.jsonl") if args.source else None,
            "method": ("derived: reference run re-scored with the five-result R_t (the runner sees the reference tasks)" if args.variant == "extended_rt"
                       else "prompt_only re-scored against the corrected variant tasks (gold and R_t recomputed); generation unchanged")}
    (args.run_dir / ("derived_from.json" if args.source else "rescored_dev14.json")).write_text(json.dumps(note, indent=1) + "\n")
    if args.source:
        meta = json.loads((src / "metadata.json").read_text()) if (src / "metadata.json").exists() else {}
        meta.update({"variant": args.variant, "cell_kind": "sensitivity", "derived": note})
        (args.run_dir / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    old_by_id = {r["task_id"]: r for r in rows}
    changed = sum(1 for b in new if old_by_id[b["task_id"]]["metrics"] != b["metrics"])
    print(f"{args.variant}: {len(new)} rows re-scored into {args.run_dir} ({changed} rows with changed metrics; {len(rows) - len(new)} rows dropped with their tasks)")


if __name__ == "__main__":
    main()
