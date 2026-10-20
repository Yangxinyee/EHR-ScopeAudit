#!/usr/bin/env python3
"""DEV-34: the Tier-C evidence-plan catalog — the covered FHIR-AgentBench templates as the operations a model may
declare (experiments/SCOPEBENCH_PLAN_PROTOCOL.md §2).

Built from the dev split only: the template list is the DEV-31 coverage record (decided on `valid`), and each
template's main table is read from the `valid` tasks of that template. Per template the catalog keeps what a model
needs to name it and fill it: an id, the template text, the value placeholders `{...}` and the operator slots
`[...]` it carries. No answer, SQL or task-level value enters the catalog. Loads no model.

Usage: build_tierc_plan_catalog.py [--manifest DIR] [--coverage JSON] [--out JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.tierc.plans import OPERATOR_SLOTS  # noqa: E402

E = ROOT / "experiments" / "scopebench_tierC"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=E / "fhir_agentbench")
    ap.add_argument("--coverage", type=Path, default=E / "skill_validation_protocol_dev31.json")
    ap.add_argument("--out", type=Path, default=E / "plan_catalog.json")
    args = ap.parse_args()

    covered = sorted(t for t, v in json.loads(args.coverage.read_text())["templates"].items() if v.get("covered"))
    tasks = args.manifest / "tasks.jsonl"
    tables: dict[str, set[str]] = {}
    for line in tasks.open():
        row = json.loads(line)
        if row.get("split") != "valid":
            continue
        m = row["metadata"]
        if m.get("template") in covered:
            tables.setdefault(m["template"], set()).add(m["main_table"])
    missing = [t for t in covered if t not in tables]
    ambiguous = {t: sorted(s) for t, s in tables.items() if len(s) != 1}
    if missing or ambiguous:
        sys.exit(f"refusing: covered templates without a single valid-split main table: missing={missing} ambiguous={ambiguous}")

    entries = []
    for i, t in enumerate(sorted(covered, key=lambda t: (next(iter(tables[t])), t)), 1):
        entries.append({
            "id": f"C{i:02d}",
            "template": t,
            "main_table": next(iter(tables[t])),
            "values": sorted(set(re.findall(r"\{(\w+)\}", t))),
            "operators": [arg for slot, arg in OPERATOR_SLOTS.items() if f"[{slot}]" in t],
        })
    record = {
        "record": "Tier-C evidence-plan catalog (DEV-34)",
        "built_from": {"coverage": str(args.coverage.relative_to(ROOT)), "coverage_sha256": sha(args.coverage),
                       "manifest": str(tasks.relative_to(ROOT)), "manifest_sha256": sha(tasks), "split": "valid"},
        "n_templates": len(entries),
        "templates": entries,
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(entries)} templates -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
