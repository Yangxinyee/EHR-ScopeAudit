#!/usr/bin/env python3
"""Materialize the MIMIC tables Tier-B needs into one DuckDB file (plan §5.1), outside the repo.

Default output: <public_data>/.clinehr_cache/scopebench_mimic.duckdb (never committed; contains
credentialed data). Every table keeps its official column names; two derived views add the
ScopeBench time bases (`occurred`, `available`). Idempotent: existing tables are skipped unless
--force. Records row counts and SHA-256 of the source files into <out>.manifest.json.
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

from scopebench.config import PUBLIC_DATA  # noqa: E402

TABLES = {
    # name: (relative path, columns or None for all)
    "patients": ("mimiciv/3.1/hosp/patients.csv.gz", None),
    "admissions": ("mimiciv/3.1/hosp/admissions.csv.gz", None),
    "transfers": ("mimiciv/3.1/hosp/transfers.csv.gz", None),
    "diagnoses_icd": ("mimiciv/3.1/hosp/diagnoses_icd.csv.gz", None),
    "d_icd_diagnoses": ("mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz", None),
    "d_labitems": ("mimiciv/3.1/hosp/d_labitems.csv.gz", None),
    "labevents": ("mimiciv/3.1/hosp/labevents.csv.gz",
                  ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority"]),
    "microbiologyevents": ("mimiciv/3.1/hosp/microbiologyevents.csv.gz",
                           ["microevent_id", "subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime", "spec_type_desc", "test_name", "org_name", "ab_name", "interpretation", "storedate", "storetime"]),
    "emar": ("mimiciv/3.1/hosp/emar.csv.gz", ["subject_id", "hadm_id", "emar_id", "emar_seq", "poe_id", "pharmacy_id", "charttime", "medication", "event_txt", "scheduletime", "storetime"]),
    "prescriptions": ("mimiciv/3.1/hosp/prescriptions.csv.gz",
                      ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime", "stoptime", "drug_type", "drug", "formulary_drug_cd", "prod_strength", "dose_val_rx", "dose_unit_rx", "route"]),
    "procedures_icd": ("mimiciv/3.1/hosp/procedures_icd.csv.gz", None),
    "icustays": ("mimiciv/3.1/icu/icustays.csv.gz", None),
    "radiology": ("mimic-iv-note/2.2/note/radiology.csv.gz", None),
    "radiology_detail": ("mimic-iv-note/2.2/note/radiology_detail.csv.gz", None),
    "discharge": ("mimic-iv-note/2.2/note/discharge.csv.gz", None),
    "discharge_detail": ("mimic-iv-note/2.2/note/discharge_detail.csv.gz", None),
    "edstays": ("mimic-iv-ed/2.2/ed/edstays.csv.gz", None),
    "triage": ("mimic-iv-ed/2.2/ed/triage.csv.gz", None),
    "vitalsign": ("mimic-iv-ed/2.2/ed/vitalsign.csv.gz", None),
    "pyxis": ("mimic-iv-ed/2.2/ed/pyxis.csv.gz", None),
    "medrecon": ("mimic-iv-ed/2.2/ed/medrecon.csv.gz", None),
    "ed_diagnosis": ("mimic-iv-ed/2.2/ed/diagnosis.csv.gz", None),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import duckdb

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=PUBLIC_DATA / ".clinehr_cache" / "scopebench_mimic.duckdb")
    p.add_argument("--force", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.out))
    con.execute(f"PRAGMA threads={args.threads}")
    existing = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (rel, cols) in TABLES.items():
        if args.only and name not in args.only:
            continue
        src = PUBLIC_DATA / rel
        if not src.exists():
            print(f"!! missing {src}", flush=True)
            continue
        if name in existing and not args.force:
            print(f"-- {name}: exists ({manifest.get(name, {}).get('rows')} rows)", flush=True)
            continue
        t0 = time.time()
        select = ", ".join(cols) if cols else "*"
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT {select} FROM read_csv_auto('{src}', header=true, sample_size=200000, all_varchar=false)")
        rows = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        manifest[name] = {"source": rel, "rows": rows, "sha256": sha256(src), "seconds": round(time.time() - t0, 1), "columns": cols}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"== {name}: {rows:,} rows in {time.time() - t0:.0f}s", flush=True)
    # Derived views: ScopeBench time bases per source (plan §5.1)
    con.execute("""
        CREATE OR REPLACE VIEW ev_labevents AS
        SELECT 'labevents' AS source, labevent_id AS row_id, subject_id, hadm_id, itemid AS concept_id,
               charttime AS occurred, storetime AS available, value, valuenum, valueuom, flag
        FROM labevents""")
    con.execute("""
        CREATE OR REPLACE VIEW ev_emar AS
        SELECT 'emar' AS source, emar_id AS row_id, subject_id, hadm_id, medication AS concept,
               charttime AS occurred, storetime AS available, event_txt
        FROM emar""")
    con.execute("""
        CREATE OR REPLACE VIEW ev_radiology AS
        SELECT 'radiology' AS source, note_id AS row_id, subject_id, hadm_id, note_seq,
               charttime AS occurred, storetime AS available
        FROM radiology""")
    print("views: ev_labevents, ev_emar, ev_radiology")
    con.close()
    print("DONE", args.out, f"{args.out.stat().st_size / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
