#!/usr/bin/env python3
"""Check the Git publication payload and generate/verify its integrity manifest.

This structural and pattern audit complements the release's manual source and
license review. It reports filenames and issue types, never credential values.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"
DATASETS = {"experiments/scopepair_fhir/tasks.jsonl", "experiments/scopepair_medadmin/tasks.jsonl"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "api_key": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}\b"),
    "author_machine_path": re.compile(r"/(?:media/extradrive|data2/xinye|home/(?:baksoy|wush))(?:/|\b)"),
}
BLOCKED_NAMES = {"secret_salt.txt", ".env", "predictions.jsonl", "access_log.jsonl", "stages.jsonl", "ANSWER_KEY.jsonl", "patients.json", "patient_map.json"}
BLOCKED_PARTS = {".aws", ".gcloud-config", ".docker-config", ".runtime", ".vendor", ".claude", ".codex", ".agents", "external_benchmarks", "local_data", "models", "runs", "outputs", "revisions", "batches"}
BINARY_ASSETS = {"docs/figures/" + name for name in ("architecture.pdf", "architecture.png", "scope_teaser.pdf", "scope_teaser.png", "gap.pdf", "operation_results.pdf", "operation_results.png")}


def publication_files():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, stdout=subprocess.PIPE)
    return sorted({name for name in result.stdout.decode().split("\0") if name})


def scan(names):
    problems = []
    for name in names:
        path = ROOT / name
        relative = Path(name)
        if path.is_symlink() or not path.is_file():
            problems.append((name, "symlink or missing file"))
            continue
        if relative.name in BLOCKED_NAMES or relative.name.startswith(".env.") or set(relative.parts) & BLOCKED_PARTS:
            problems.append((name, "excluded resource"))
        if any(part.startswith(".venv") for part in relative.parts):
            problems.append((name, "environment"))
        if relative.suffix in {".jsonl", ".ndjson"} and name not in DATASETS:
            problems.append((name, "unapproved row-level dataset"))
        if relative.suffix in {".gz", ".zip", ".duckdb", ".parquet", ".xlsx", ".pem", ".key"}:
            problems.append((name, "unapproved archive, database, export, or key"))
        if relative.suffix in {".png", ".pdf"}:
            if name not in BINARY_ASSETS:
                problems.append((name, "unapproved binary"))
            if relative.suffix == ".pdf":
                content = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
            else:
                continue
        else:
            try:
                content = path.read_text()
            except UnicodeDecodeError:
                problems.append((name, "unreviewed binary"))
                continue
        for kind, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append((name, kind))
    for name in DATASETS:
        path = ROOT / name
        if name not in names:
            problems.append((name, "approved public dataset missing"))
            continue
        metadata = json.loads(path.with_name("metadata.json").read_text())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if sha != metadata["task_manifest_sha256"] or len(rows) != 160:
            problems.append((name, "public dataset integrity mismatch"))
        if metadata.get("data_license") != "ODbL-1.0" or metadata.get("source_url") != "https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/":
            problems.append((name, "source attribution missing"))
        if any("patient_id" in row or "subject_id" in row or "hadm_id" in row for row in rows):
            problems.append((name, "unexpected raw identity field"))
        if {row["split"] for row in rows} != {"dev", "test"}:
            problems.append((name, "task splits mismatch"))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    names = publication_files()
    problems = scan(names)
    files = {name: {"sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), "bytes": (ROOT / name).stat().st_size}
             for name in names if name != MANIFEST.name and (ROOT / name).is_file() and not (ROOT / name).is_symlink()}
    if args.verify:
        expected = json.loads(MANIFEST.read_text())["files"]
        if files != expected:
            for name in sorted(set(files) | set(expected)):
                if files.get(name) != expected.get(name):
                    problems.append((name, "release manifest mismatch"))
    if problems:
        for name, kind in problems:
            print(f"FAIL {name}: {kind}")
        raise SystemExit(1)
    if args.write:
        MANIFEST.write_text(json.dumps({"release": "EHR-AgentGuard 0.1.0", "audit": "Explicit allowlist, license review, pattern scan, and dataset integrity checks.",
                                       "excluded": ["restricted clinical data and derivatives", "patient-level model outputs", "institutional records and salts", "credentials and infrastructure", "old Git history and manuscript backups", "model weights and third-party datasets"],
                                       "files": files}, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {len(files)} publication files; only the two approved ODbL task databases; no detected secret or author-machine-path patterns")


if __name__ == "__main__":
    main()
