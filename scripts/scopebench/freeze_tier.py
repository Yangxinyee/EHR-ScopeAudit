#!/usr/bin/env python3
"""Write (or verify) a tier freeze record (plan §1.2): SHA-256 of the task manifest, every evidence file
(as one manifest hash), the framework code, the scripts, SKILL_SPEC, the task-type system prompts, the
protocol, the tests, plus the model panel and serving flags. Formal runs may start only after the
record exists; `--verify` re-hashes and exits non-zero on any drift.

Usage: freeze_tier.py --tier B --manifest experiments/scopebench_tierB --protocol experiments/SCOPEBENCH_TIERB_PROTOCOL.md [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import tasktypes, tasktypes_tierb  # noqa: E402,F401


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(paths: list[Path], base: Path = ROOT) -> tuple[str, dict[str, str]]:
    per = {str(p.relative_to(base)): sha(p) for p in sorted(paths)}
    combined = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in per.items()).encode()).hexdigest()
    return combined, per


# Scripts on the data-generating path are gating (any change after the freeze is a deviation); analysis,
# table and verification scripts are written after the freeze and are recorded per run, not gated here.
EXECUTION_SCRIPTS = {"run_config.py", "serve.sh", "run_tierb_panel.sh", "run_tierb_queue.sh", "run_tierb_sensitivity.sh", "wait_gpu_free.sh", "freeze_tier.py", "derive_late_filter.py",
                     "build_tierb.py", "build_tierc_fhiragentbench.py", "ingest_mimic_duckdb.py", "sglang_m0_check.sh", "sglang_backend_probe.sh"}


# Modules that take part in one tier's runs only: they are hashed by that tier's freeze and left out of the others'
# (Tier-C development must not invalidate the Tier-B record while the Tier-B panel runs).
TIER_ONLY_MODULES = {"C": ("tierc", "tasktypes_tierc.py"), "A": ("tiera", "tasktypes_tiera.py")}
TIER_ONLY_SCRIPTS = {"C": ("run_tierc.py", "build_tierc_fhiragentbench.py", "run_tierc_panel.sh", "run_tierc_dev32.sh", "build_tierc_frontier_subsample.py",
                           "run_tierc_medagent.py", "build_tierc_medagentbench.py"),
                     "A": ("build_tiera.py", "ingest_ccda_duckdb.py", "run_tiera_panel.sh", "build_tiera_surrogate.py", "phi_scan.py")}
EXECUTION_SCRIPTS |= {s for names in TIER_ONLY_SCRIPTS.values() for s in names}
# Serving context per tier (Tier-B protocol v1.0: 16,384; Tier-C protocol v1.0: 32,768 — eight tool rounds exceed 16k tokens)
CONTEXT_LENGTH = {"A": 16384, "B": 16384, "C": 32768}


def _tier_code_files(tier: str) -> list[Path]:
    files = [p for p in (ROOT / "scopebench").rglob("*.py") if "__pycache__" not in p.parts]
    for other, parts in TIER_ONLY_MODULES.items():
        if other != tier:
            files = [p for p in files if not any(part in p.relative_to(ROOT / "scopebench").parts for part in parts)]
    return sorted(files)


def _path_label(p: Path) -> str:
    """Repository-relative path; the Tier-A manifest lives in the PHI enclave outside the repository and is recorded by its
    absolute path (a directory name, no content)."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# formal run trees per tier (Tier-A lives in the enclave: pass --runs-dir); used to state honestly whether formal
# execution had begun when a record is (re-)issued
RUNS_DIRS = {"B": ROOT / "experiments/scopebench_tierB_runs", "C": ROOT / "experiments/scopebench_tierC_runs"}


def formal_runs_state(tier: str, runs_dir: Path | None) -> dict:
    """Formal prediction files present at issue time (superseded `_*` trees excluded) and the earliest one's mtime."""
    root = runs_dir or RUNS_DIRS.get(tier[:1])
    if root is None or not root.exists():
        return {"runs_dir": str(root) if root else None, "n_prediction_files": None, "earliest_mtime": None}
    files = [f for f in root.rglob("predictions.jsonl") if not any(part.startswith("_") for part in f.relative_to(root).parts)]
    first = min((f.stat().st_mtime for f in files), default=None)
    return {"runs_dir": _path_label(root), "n_prediction_files": len(files),
            "earliest_mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(first)) if first else None}


def build_record(tier: str, manifest: Path, protocol: Path, runs_dir: Path | None = None) -> dict:
    if tier == "A":
        from scopebench import tasktypes_tiera  # noqa: F401  (registers outcome_count_v1 so its prompt hash is recorded)
    code = _tier_code_files(tier)
    other_tier_scripts = {s for t, names in TIER_ONLY_SCRIPTS.items() if t != tier for s in names}
    scripts = sorted(p for p in (ROOT / "scripts/scopebench").iterdir() if p.name in EXECUTION_SCRIPTS and p.name not in other_tier_scripts)
    tests = sorted((ROOT / "tests/scopebench").glob("*.py"))
    evidence_files = sorted((manifest / "evidence").glob("*.jsonl")) if (manifest / "evidence").exists() else []
    code_hash, code_per = tree_hash(code)
    scripts_hash, scripts_per = tree_hash(scripts)
    tests_hash, _ = tree_hash(tests)
    # evidence paths relative to the repository (Tier-B/C) or, for a manifest outside it (Tier-A enclave), to the manifest's parent
    evidence_hash, _ = tree_hash(evidence_files, ROOT if manifest.is_relative_to(ROOT) else manifest.parent) if evidence_files else ("", {})
    prompts = {name: tt.system_prompt_sha256 for name, tt in tasktypes.REGISTRY.items()}
    panel = json.loads((ROOT / "experiments/SCOPEBENCH_MODEL_PANEL.json").read_text())
    return {
        "record": f"ScopeBench Tier-{tier} formal freeze",
        "tier": tier,
        "freeze_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # DEV-33: was hard-coded False, also in records re-issued after formal runs had begun
        "formal_execution_started_at_freeze": bool(formal_runs_state(tier, runs_dir)["n_prediction_files"]),
        "formal_runs_at_freeze": formal_runs_state(tier, runs_dir),
        "protocol": {"path": str(protocol.relative_to(ROOT)), "sha256": sha(protocol)},
        "task_manifest": {"path": _path_label(manifest / "tasks.jsonl"), "sha256": sha(manifest / "tasks.jsonl"),
                          "n_tasks": sum(1 for _ in open(manifest / "tasks.jsonl")), "construction_stats_sha256": sha(manifest / "construction_stats.json") if (manifest / "construction_stats.json").exists() else None},
        "evidence_files": {"n": len(evidence_files), "tree_sha256": evidence_hash},
        "skill_spec": {"path": "scopebench/SKILL_SPEC.md", "sha256": sha(ROOT / "scopebench/SKILL_SPEC.md")},
        "system_prompts_sha256": prompts,
        "framework_code": {"tree_sha256": code_hash, "files": code_per},
        "scripts": {"tree_sha256": scripts_hash, "files": scripts_per},
        "tests": {"tree_sha256": tests_hash},
        "model_panel": {"path": "experiments/SCOPEBENCH_MODEL_PANEL.json", "sha256": sha(ROOT / "experiments/SCOPEBENCH_MODEL_PANEL.json"),
                        "checkpoints": [{"tag": e["tag"], "revision": e["revision"], "config_sha256": e.get("config_sha256")} for e in panel["this_phase"]]},
        "serving": {"engine": "sglang 0.5.19 (.venv-serve)", "requirements_sha256": sha(ROOT / "requirements-scopebench-serve.txt"),
                    "mode_by_class": {"qwen3_4b": "formal (deterministic triton)", "phi4_14b": "formal (deterministic triton)", "qwen3_32b": "formal (deterministic triton)",
                                      "qwen3_30b": "formal (deterministic triton)", "gemma3_27b": "fast (fa4, normal mode)", "medgemma_27b": "fast (fa4, normal mode)"},
                    "decoding": {"temperature": 0.0, "top_p": 1.0, "seed": 2027, "enable_thinking": False, "context_length": CONTEXT_LENGTH.get(tier, 16384), "max_repairs": 1}},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--runs-dir", type=Path, default=None, help="formal run tree (default per tier; required for the Tier-A enclave)")
    args = p.parse_args()
    out = ROOT / f"experiments/SCOPEBENCH_TIER{args.tier}_FREEZE.json"
    rec = build_record(args.tier, args.manifest.resolve(), args.protocol.resolve(), args.runs_dir)
    if args.verify:
        frozen = json.loads(out.read_text())
        drift = []
        for key in ("protocol", "task_manifest", "skill_spec", "model_panel"):
            if frozen[key]["sha256"] != rec[key]["sha256"]:
                drift.append(key)
        if frozen["evidence_files"]["tree_sha256"] != rec["evidence_files"]["tree_sha256"]:
            drift.append("evidence_files")
        if frozen["system_prompts_sha256"] != rec["system_prompts_sha256"]:
            drift.append("system_prompts")
        changed_code = [f for f, h in rec["framework_code"]["files"].items() if frozen["framework_code"]["files"].get(f) != h]
        changed_scripts = [f for f, h in rec["scripts"]["files"].items() if frozen["scripts"]["files"].get(f) != h]
        print(json.dumps({"drift": drift, "changed_code": changed_code, "changed_scripts": changed_scripts}, indent=1))
        sys.exit(1 if (drift or changed_code or changed_scripts) else 0)
    if out.exists():   # DEV-33: a re-issue keeps the history of earlier issues
        prev = json.loads(out.read_text())
        rec["first_freeze_time"] = prev.get("first_freeze_time", prev.get("freeze_time"))
        rec["previous_freeze_time"] = prev.get("freeze_time")
        rec["issue_count"] = int(prev.get("issue_count", 1)) + 1
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"freeze written: {out} (manifest {rec['task_manifest']['n_tasks']} tasks, {rec['evidence_files']['n']} evidence files, {len(rec['framework_code']['files'])} code files)")


if __name__ == "__main__":
    main()
