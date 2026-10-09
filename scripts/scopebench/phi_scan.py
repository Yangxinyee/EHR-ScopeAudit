#!/usr/bin/env python3
"""PHI scan for Tier-A artefacts (plan §12 gate G6): identifier patterns and free-text fields in a directory tree.

Modes:
  --release   (default) a text-free artefact (release manifests, analysis outputs, generated tables): any free-text field
              (`text`, `first_impression_line`, `claim`, `response`, `repair_response`, `narrative`) and any identifier
              pattern is a failure.
  --allow-text an enclave artefact (full manifests, run outputs): free text is expected; only identifier patterns fail.
Patterns: MRN-like tokens (two letters + 7–9 digits, e.g. the export's BV/HC/E1 prefixes), 10-digit encounter (CSN)
numbers, accession-like tokens, the literal words MRN / DOB / SSN followed by a value, US social-security format,
raw e-mail addresses. Prints per-file counts (never the matched text) and exits 1 on any hit.
Usage: phi_scan.py <dir or file> [...] [--allow-text] [--json report.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PATTERNS = {
    "mrn_like": re.compile(r"\b[A-Z]{1,2}\d{7,9}\b"),
    "csn_like": re.compile(r"(?<![\w.-])\d{10}(?![\w.-])"),
    "accession_like": re.compile(r"\b(ACC|ACCESSION)[A-Z]*[ _:#-]*\d{6,}\b", re.I),
    "mrn_word": re.compile(r"\bMRN\s*[:#=-]?\s*[A-Z0-9]{5,}", re.I),
    "dob_word": re.compile(r"\bDOB\s*[:=-]?\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", re.I),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}
TEXT_KEYS = {"text", "first_impression_line", "claim", "response", "repair_response", "narrative", "note_text", "question"}
EXTS = {".json", ".jsonl", ".txt", ".tex", ".csv", ".md", ".log"}


def scan_text(s: str) -> Counter:
    c = Counter()
    for name, pat in PATTERNS.items():
        n = len(pat.findall(s))
        if n:
            c[name] += n
    return c


def walk_json(obj, hits: Counter, allow_text: bool, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in TEXT_KEYS and isinstance(v, str) and v.strip():
                if not allow_text and k != "question":
                    hits[f"text_field:{k}"] += 1
                hits.update(scan_text(v))
            else:
                walk_json(v, hits, allow_text, f"{path}.{k}")
    elif isinstance(obj, list):
        for x in obj:
            walk_json(x, hits, allow_text, path)
    elif isinstance(obj, str):
        hits.update(scan_text(obj))


def scan_file(p: Path, allow_text: bool) -> Counter:
    hits: Counter = Counter()
    if p.suffix == ".jsonl":
        with open(p, errors="replace") as f:
            for line in f:
                try:
                    walk_json(json.loads(line), hits, allow_text)
                except json.JSONDecodeError:
                    hits.update(scan_text(line))
    elif p.suffix == ".json":
        try:
            walk_json(json.load(open(p, errors="replace")), hits, allow_text)
        except json.JSONDecodeError:
            hits.update(scan_text(p.read_text(errors="replace")))
    else:
        hits.update(scan_text(p.read_text(errors="replace")))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--allow-text", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    files = []
    for s in a.paths:
        p = Path(s)
        files.extend([p] if p.is_file() else [q for q in p.rglob("*") if q.is_file() and q.suffix in EXTS and "__pycache__" not in q.parts])
    report = {}; total = Counter()
    for f in sorted(files):
        h = scan_file(f, a.allow_text)
        if h:
            report[str(f)] = dict(h); total.update(h)
    for f, h in report.items():
        print(f"HIT {f}: {h}")
    print(f"scanned {len(files)} files; hits: {dict(total) if total else 'none'}")
    if a.json:
        a.json.write_text(json.dumps({"n_files": len(files), "allow_text": a.allow_text, "hits": report, "total": dict(total)}, indent=1))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
