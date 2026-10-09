#!/usr/bin/env python3
"""Ingest the CCDA cohort exports into the Tier-A enclave DuckDB (plan §7.2; `scopebench/tiera/sources.py`).

Streams every recognised file/sheet into parquet parts under <enclave>/parts/<table>/, then materialises the canonical
tables with SELECT DISTINCT (per-anchor duplicates collapse) and `index_candidates` (the sampling anchors). Writes a
discovery report (file names, header signatures, row counts — no data values) to the enclave and a copy to
experiments/scopebench_tierA_build/discovery.json. Resumable per file: parts that already exist are kept
(pass --rebuild to start over).

Usage: ingest_ccda_duckdb.py [--rebuild] [--only KIND ...] [--limit-files N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.tiera import sources as S  # noqa: E402
from scopebench.tiera.config import DUCKDB, ENCLAVE, secret_salt  # noqa: E402

BUILD_DIR = ROOT / "experiments/scopebench_tierA_build"


def write_parts(sf: S.SourceFile, out_dir: Path, anchors_dir: Path) -> tuple[int, int]:
    table, _ = S.CANONICAL[sf.kind]
    cols = S.TABLE_COLUMNS[table] + ["path_id"]
    stem = sf.path_id.replace("/", "__").replace("::", "__").replace(" ", "_")
    n_rows = n_anchor = 0
    for i, chunk in enumerate(S.iter_rows(sf)):
        recs, anchors = S.canonical_rows(sf, chunk)
        n_rows += len(recs); n_anchor += len(anchors)
        if recs:
            arr = pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in recs], schema=pa.schema([(c, pa.string()) for c in cols]))
            pq.write_table(arr, out_dir / f"{stem}__{i:04d}.parquet", compression="zstd")
        if anchors:
            acols = S.TABLE_COLUMNS["index_candidates"]
            arr = pa.Table.from_pylist([{c: a.get(c) for c in acols} for a in anchors], schema=pa.schema([(c, pa.string()) for c in acols]))
            pq.write_table(arr, anchors_dir / f"{stem}__{i:04d}.parquet", compression="zstd")
    (out_dir / f"{stem}.done").write_text(json.dumps({"rows": n_rows, "anchors": n_anchor}))
    return n_rows, n_anchor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these kinds")
    p.add_argument("--limit-files", type=int, default=None)
    a = p.parse_args()
    secret_salt()                                   # creates the enclave (mode 700) and the salt on first use
    parts = S.parts_dir(); parts.mkdir(parents=True, exist_ok=True)
    if a.rebuild:
        shutil.rmtree(parts); parts.mkdir()
        if DUCKDB.exists():
            DUCKDB.unlink()
    files, skipped = S.discover()
    if a.only:
        files = [f for f in files if f.kind in a.only]
    if a.limit_files:
        files = files[: a.limit_files]
    report = {"n_recognised": len(files), "n_skipped": len(skipped), "skipped": skipped,
              "files": [{"path": f.path_id, "kind": f.kind, "table": S.CANONICAL[f.kind][0], "cohort": f.cohort, "modality": f.modality, "header": f.header} for f in files]}
    print(f"recognised {len(files)} files/sheets, skipped {len(skipped)}", flush=True)
    t0 = time.time()
    for k, sf in enumerate(files, 1):
        table = S.CANONICAL[sf.kind][0]
        out_dir = parts / table; out_dir.mkdir(exist_ok=True)
        anchors_dir = parts / "index_candidates"; anchors_dir.mkdir(exist_ok=True)
        stem = sf.path_id.replace("/", "__").replace("::", "__").replace(" ", "_")
        done = out_dir / f"{stem}.done"
        if done.exists():
            continue
        t1 = time.time()
        n_rows, n_anchor = write_parts(sf, out_dir, anchors_dir)
        print(f"[{k}/{len(files)}] {sf.kind:18s} {sf.cohort:16s} {sf.path.name[:40]:40s} {sf.sheet or '':26s} rows={n_rows:>9,} anchors={n_anchor:>8,} {time.time() - t1:5.0f}s", flush=True)
    # ---- materialise
    import duckdb
    con = duckdb.connect(str(DUCKDB))
    counts = {}
    for table, cols in S.TABLE_COLUMNS.items():
        d = parts / table
        if not d.exists() or not any(d.glob("*.parquet")):
            continue
        sel = ", ".join(cols)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT DISTINCT {sel} FROM read_parquet('{d}/*.parquet', union_by_name=true)")
        counts[table] = con.execute(f"SELECT count(*), count(DISTINCT mrn) FROM {table}").fetchone() if "mrn" in cols else (con.execute(f"SELECT count(*) FROM {table}").fetchone()[0], None)
    con.execute("CREATE OR REPLACE VIEW cohort_summary AS SELECT cohort, modality, count(*) AS n_index_rows, count(DISTINCT mrn) AS n_mrn FROM index_candidates GROUP BY 1, 2")
    summary = [dict(zip(["cohort", "modality", "n_index_rows", "n_mrn"], r)) for r in con.execute("SELECT * FROM cohort_summary ORDER BY 2, 1").fetchall()]
    con.close()
    report["tables"] = {t: {"rows": c[0], "n_mrn": c[1]} for t, c in counts.items()}
    report["cohorts"] = summary
    report["seconds"] = round(time.time() - t0)
    ENCLAVE.mkdir(parents=True, exist_ok=True)
    (ENCLAVE / "discovery.json").write_text(json.dumps(report, indent=1))
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    (BUILD_DIR / "discovery.json").write_text(json.dumps(report, indent=1))
    os.chmod(DUCKDB, 0o600)
    print("tables:", json.dumps(report["tables"]))
    print("cohorts:", json.dumps(summary))
    print(f"done in {report['seconds']}s -> {DUCKDB}")


if __name__ == "__main__":
    main()
