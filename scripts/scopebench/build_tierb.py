#!/usr/bin/env python3
"""Build the Tier-B manifests (plan §5.5): dev pool and test set, per family, seeded, one case per patient.

Usage:
  build_tierb.py --split dev  --per-family 84  --out experiments/scopebench_tierB_dev
  build_tierb.py --split test --per-family 800 --out experiments/scopebench_tierB --exclude-patients experiments/scopebench_tierB_dev/patients.json
Writes tasks.jsonl, evidence/<case_id>.jsonl, patients.json (hashed ids + raw subject_ids kept OUTSIDE the repo
in <public_data>/.clinehr_cache/), construction_stats.json, metadata.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import __version__, oracle  # noqa: E402
import scopebench.tasktypes_tierb  # noqa: E402,F401  (registers the Tier-B task types)
from scopebench.config import PUBLIC_DATA, SEED  # noqa: E402
from scopebench.schema import write_jsonl  # noqa: E402
from scopebench.tierb.families import BUILDERS  # noqa: E402
from scopebench.tierb.sampling import FAMILY_QUERIES, select  # noqa: E402
from scopebench.tierb.sources import SourceLoader  # noqa: E402

DB = PUBLIC_DATA / ".clinehr_cache" / "scopebench_mimic.duckdb"


def main() -> None:
    import duckdb

    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--per-family", type=int, required=True)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude-patients", type=Path, nargs="*", default=[])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--oversample", type=float, default=2.0, help="anchors selected per family = per_family * oversample (builders may reject; leakage enrichment needs surplus)")
    p.add_argument("--leakage-fraction", type=float, default=0.25)
    args = p.parse_args()

    seed = args.seed if args.seed is not None else (SEED + 1 if args.split == "dev" else SEED)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("PRAGMA threads=8")
    loader = SourceLoader(con)
    used: set[int] = set()
    for path in args.exclude_patients:
        used |= set(json.loads(Path(path).read_text())["subject_ids"])
    families = args.families or list(BUILDERS)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence").mkdir(exist_ok=True)
    stats: dict = {"split": args.split, "seed": seed, "per_family": args.per_family, "families": {}}
    all_tasks = []
    subject_ids: list[int] = []
    for family in families:
        t0 = time.time()
        anchors = FAMILY_QUERIES[family](con)
        chosen = select(anchors, seed, family, int(args.per_family * args.oversample), used)
        rejected = 0
        candidates_built = []   # (anchor, case, tasks) in seeded order
        for anchor in chosen:
            result = BUILDERS[family](loader, anchor, args.split)
            if result is None:
                rejected += 1
                used.discard(anchor.subject_id)
                continue
            candidates_built.append((anchor, *result))
        # leakage enrichment (plan §5.4): up to 25 % of the family's cases are leakage-flagged, taken in seeded order,
        # the rest are non-leakage cases in seeded order; surplus anchors release their patients
        target_leak = int(round(args.leakage_fraction * args.per_family))
        leaky = [c for c in candidates_built if c[2][0].leakage][:target_leak]
        clean = [c for c in candidates_built if not c[2][0].leakage][: args.per_family - len(leaky)]
        keep = leaky + clean
        if len(keep) < args.per_family:
            extra = [c for c in candidates_built if c not in keep][: args.per_family - len(keep)]
            keep += extra
        keep_ids = {c[0].subject_id for c in keep}
        for anchor, _, _ in candidates_built:
            if anchor.subject_id not in keep_ids:
                used.discard(anchor.subject_id)
        built = 0
        for anchor, case, tasks in sorted(keep, key=lambda c: c[1].case_id):
            write_jsonl(args.out / "evidence" / f"{case.case_id}.jsonl", (e.to_dict() for e in case.evidence))
            all_tasks.extend(tasks)
            subject_ids.append(anchor.subject_id)
            built += 1
        stats.setdefault("leakage", {})[family] = {"available_leaky": sum(1 for c in candidates_built if c[2][0].leakage), "kept_leaky": len(leaky), "target": target_leak}
        stats["families"][family] = {"candidates": len(anchors), "selected": len(chosen), "built": built, "rejected": rejected,
                                     "n_tasks": built * 2, "seconds": round(time.time() - t0, 1)}
        print(f"{family:24s} candidates={len(anchors):>8,} built={built:>5} rejected={rejected:>4} ({time.time() - t0:.0f}s)", flush=True)
    write_jsonl(args.out / "tasks.jsonl", (t.to_dict() for t in sorted(all_tasks, key=lambda t: (t.case_id, t.task_id))))
    # patient lists: hashed in the repo, raw subject_ids outside it (credentialed identifiers)
    (args.out / "patients.json").write_text(json.dumps({"n": len(subject_ids), "patient_hashes": sorted({t.patient for t in all_tasks})}, indent=1))
    raw_dir = PUBLIC_DATA / ".clinehr_cache" / "scopebench_tierb_patients"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{args.split}_{seed}.json").write_text(json.dumps({"split": args.split, "seed": seed, "subject_ids": sorted(subject_ids)}))
    stats["n_tasks"] = len(all_tasks)
    stats["n_cases"] = len(subject_ids)
    stats["leakage_fraction"] = round(sum(t.leakage for t in all_tasks) / max(1, len(all_tasks)), 4)
    (args.out / "construction_stats.json").write_text(json.dumps(stats, indent=1))
    (args.out / "metadata.json").write_text(json.dumps({"scopebench_version": __version__, "oracle_spec_version": oracle.SPEC_VERSION, "db": str(DB.name),
                                                        "split": args.split, "seed": seed, "families": families, "raw_patient_list": str(raw_dir / f"{args.split}_{seed}.json")}, indent=1))
    print(f"wrote {len(all_tasks)} tasks / {len(subject_ids)} cases to {args.out}; leakage fraction {stats['leakage_fraction']}")


if __name__ == "__main__":
    main()
