#!/usr/bin/env python3
"""DEV-34 task sets (experiments/SCOPEBENCH_PLAN_PROTOCOL.md §4). Loads no model.

  Tier-B formal: a stratified sample of the test split — equal share per (task type, scope) stratum, ranked within a
                 stratum by sha256("2027|<task_id>") (the rule of run_config.py --stratified-subsample) —
                 2,000 // 12 strata = 166 per stratum = 1,992 tasks. Written next to the Tier-B data (MIMIC-derived
                 task ids stay out of git); counts go to the analysis directory.
  Tier-C formal: FHIR-AgentBench train + test (2,507 tasks), the primary Tier-C panel.
  Development:   the whole Tier-B dev manifest (1,008) and the Tier-C `valid` split (424).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402

E = ROOT / "experiments"
SEED, N = 2027, 2000
OUT_B = E / "scopebench_tierB" / "plan_dev34_task_ids.json"
OUT_C = E / "scopebench_tierC" / "plan_dev34"
SUMMARY = E / "scopebench_analysis" / "plans_dev34" / "task_sets.json"


def main() -> None:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in read_jsonl(E / "scopebench_tierB" / "tasks.jsonl"):
        strata[(r["task_type"], r["contract"]["scope"])].append(r["task_id"])
    per = N // len(strata)
    chosen = sorted(i for key in sorted(strata) for i in sorted(strata[key], key=lambda t: hashlib.sha256(f"{SEED}|{t}".encode()).hexdigest())[:per])
    OUT_B.write_text(json.dumps(chosen) + "\n")

    rows = list(read_jsonl(E / "scopebench_tierC" / "fhir_agentbench" / "tasks.jsonl"))
    OUT_C.mkdir(parents=True, exist_ok=True)
    traintest = sorted(r["task_id"] for r in rows if r["split"] in ("train", "test"))
    valid = sorted(r["task_id"] for r in rows if r["split"] == "valid")
    (OUT_C / "traintest_task_ids.json").write_text(json.dumps(traintest) + "\n")
    (OUT_C / "valid_task_ids.json").write_text(json.dumps(valid) + "\n")

    summary = {
        "tier_b_formal": {"file": str(OUT_B.relative_to(ROOT)), "sha256": hashlib.sha256(OUT_B.read_bytes()).hexdigest(), "n": len(chosen),
                          "per_stratum": per, "strata": {f"{k[0]}|{k[1]}": per for k in sorted(strata)}, "rule": f"sha256('{SEED}|<task_id>'), first {per} per stratum"},
        "tier_c_formal": {"file": str((OUT_C / "traintest_task_ids.json").relative_to(ROOT)), "n": len(traintest),
                          "by_split": dict(Counter(r["split"] for r in rows if r["split"] in ("train", "test")))},
        "tier_c_dev": {"file": str((OUT_C / "valid_task_ids.json").relative_to(ROOT)), "n": len(valid)},
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
