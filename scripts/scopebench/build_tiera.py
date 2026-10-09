#!/usr/bin/env python3
"""Build a Tier-A manifest (plan §7.2) from the enclave DuckDB: seeded anchors per modality, one case per patient, the
family rotating within a modality, four tasks per case (pre / at / post / full).

Usage:
  build_tiera.py --split dev  --per-modality 15  --out-name tierA_dev
  build_tiera.py --split test --per-modality 154 --out-name tierA --exclude tierA_dev
Everything with text or patient linkage is written inside the enclave (`scopebench.tiera.config.ENCLAVE`):
  <enclave>/<out-name>/{tasks.jsonl, evidence/<case_id>.jsonl, patients.json, construction_stats.json, metadata.json}
  <enclave>/<out-name>/patients_raw.json   (mrn -> hash, mode 600; never leaves the enclave)
  <enclave>/release/<out-name>/            (text-free copy for the gated package; still a limited data set)
Only the aggregate construction statistics are copied to the repository (experiments/scopebench_tierA_build/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import __version__, oracle  # noqa: E402
from scopebench.config import SEED  # noqa: E402
from scopebench.schema import write_jsonl  # noqa: E402
from scopebench.tiera.config import DUCKDB, ENCLAVE, MODALITIES, secret_salt  # noqa: E402
from scopebench.tiera.families import BUILDERS  # noqa: E402
from scopebench.tiera.loader import TierALoader  # noqa: E402
from scopebench.tiera.sampling import FAMILIES, discover_anchors, select  # noqa: E402

BUILD_DIR = ROOT / "experiments/scopebench_tierA_build"
TEXT_KEYS = ("text", "first_impression_line")


def ensure_indexes(con) -> None:
    for t in ("labevents", "emar", "prescriptions", "diagnoses", "admissions", "transfers", "procedures", "component_value", "narrative_line"):
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS ix_{t}_mrn ON {t}(mrn)")
        except Exception as e:  # noqa: BLE001
            print(f"index on {t}: {type(e).__name__}", flush=True)


def strip_text(obj):
    """Text-free copy: drop every free-text field (report lines, gold text) while keeping ids, times, sections and values."""
    if isinstance(obj, dict):
        return {k: strip_text(v) for k, v in obj.items() if k not in TEXT_KEYS and k != "claim"}
    if isinstance(obj, list):
        return [strip_text(x) for x in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--per-modality", type=int, required=True)
    p.add_argument("--out-name", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="enclave manifest names whose patients are excluded (e.g. tierA_dev)")
    p.add_argument("--oversample", type=float, default=40.0, help="anchors selected per modality = per_modality × oversample (report families build for a minority of anchors; attempts are cheap)")
    p.add_argument("--leakage-fraction", type=float, default=0.25)
    p.add_argument("--modalities", nargs="*", default=list(MODALITIES))
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    seed = a.seed if a.seed is not None else (SEED + 1 if a.split == "dev" else SEED)
    salt = secret_salt()
    con = duckdb.connect(str(DUCKDB), read_only=False)
    ensure_indexes(con)
    loader = TierALoader(con, salt)
    loader.ensure_report_links()      # accession -> patient links for the radiology reports (unambiguous links only)
    exclude: set[str] = set()
    for name in a.exclude:
        raw = ENCLAVE / name / "patients_raw.json"
        if raw.exists():
            exclude |= set(json.load(open(raw))["mrn_to_hash"])
    t0 = time.time()
    anchors = discover_anchors(loader, tuple(a.modalities))
    print("anchors per modality:", {m: len(v) for m, v in anchors.items()}, f"({time.time() - t0:.0f}s)", flush=True)
    pairs = select(anchors, seed, int(a.per_modality * a.oversample), exclude)
    # build in selection order, per modality, until every (modality, family) quota is met
    quota = {m: {} for m in a.modalities}
    for m in a.modalities:
        base, rem = divmod(a.per_modality, len(FAMILIES))
        for i, f in enumerate(FAMILIES):
            quota[m][f] = base + (1 if i < rem else 0)
    built: dict[tuple[str, str], list] = defaultdict(list)
    attempts = Counter(); rejected = Counter()
    for k, (anchor, family) in enumerate(pairs):
        need = quota[anchor.modality][family]
        have = built[(anchor.modality, family)]
        if len(have) >= need * 2 + 2:   # enough candidates for this cell (leakage preference and the modality top-up need a surplus)
            continue
        attempts[(anchor.modality, family)] += 1
        try:
            res = BUILDERS[family](loader, anchor, a.split)
        except Exception as e:  # noqa: BLE001 — a builder failure on one anchor is logged, never fatal
            rejected[(anchor.modality, family, type(e).__name__)] += 1
            continue
        if res is None:
            rejected[(anchor.modality, family, "none")] += 1
            continue
        have.append((anchor, res))
        if (k + 1) % 50 == 0:
            done = sum(min(len(v), quota[m][f]) for (m, f), v in built.items())
            print(f"  {k + 1}/{len(pairs)} anchors tried; cells filled {done}/{a.per_modality * len(a.modalities)} ({time.time() - t0:.0f}s)", flush=True)
    # keep per cell: leaky candidates first up to the leakage fraction (unless the family is leaky by construction), then clean ones
    kept: list = []
    stats: dict = {"split": a.split, "seed": seed, "per_modality": a.per_modality, "quota": quota, "cells": {}, "attempts": {}, "rejected": {}}
    for m in a.modalities:
        for f in FAMILIES:
            cands = built[(m, f)]
            q = quota[m][f]
            leaky = [c for c in cands if c[1][1][0].leakage]
            clean = [c for c in cands if not c[1][1][0].leakage]
            if f == "report_reconstruction" or not clean:
                chosen = cands[:q]
            else:
                n_leak = min(len(leaky), int(round(a.leakage_fraction * q)))
                chosen = (leaky[:n_leak] + clean)[:q]
            kept.extend(chosen)
            stats["cells"][f"{m}|{f}"] = {"quota": q, "built": len(cands), "kept": len(chosen), "leaky_kept": sum(1 for c in chosen if c[1][1][0].leakage),
                                          "surplus": [c for c in cands if c not in chosen]}
        # modality top-up (plan §7.2 "where the sources allow"): a family cell short of its quota releases its share to the
        # other families of the same modality, round-robin over the families with the most surplus, so the modality still
        # yields per_modality cases; the realised family distribution is recorded in the statistics
        deficit = a.per_modality - sum(stats["cells"][f"{m}|{f}"]["kept"] for f in FAMILIES)
        stats.setdefault("top_up", {})[m] = 0
        while deficit > 0:
            donors = sorted((f for f in FAMILIES if stats["cells"][f"{m}|{f}"]["surplus"]), key=lambda f: -len(stats["cells"][f"{m}|{f}"]["surplus"]))
            if not donors:
                break
            for f in donors:
                if deficit <= 0:
                    break
                c = stats["cells"][f"{m}|{f}"]["surplus"].pop(0)
                kept.append(c); stats["cells"][f"{m}|{f}"]["kept"] += 1; stats["cells"][f"{m}|{f}"]["leaky_kept"] += int(c[1][1][0].leakage)
                stats["top_up"][m] += 1; deficit -= 1
    for cell in stats["cells"].values():
        cell["surplus"] = len(cell["surplus"])
    stats["attempts"] = {f"{m}|{f}": n for (m, f), n in sorted(attempts.items())}
    stats["rejected"] = {f"{m}|{f}|{r}": n for (m, f, r), n in sorted(rejected.items())}
    # ---- write (enclave)
    out = ENCLAVE / a.out_name
    import shutil
    for d in (out / "evidence", ENCLAVE / "release" / a.out_name / "evidence"):
        if d.exists():
            shutil.rmtree(d)          # a rebuild never leaves evidence files of cases that are no longer selected
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    all_tasks = []; mrn_to_hash = {}
    for anchor, (case, tasks) in kept:
        write_jsonl(out / "evidence" / f"{case.case_id}.jsonl", (e.to_dict() for e in case.evidence))
        all_tasks.extend(tasks)
        mrn_to_hash[anchor.mrn] = case.patient
    all_tasks.sort(key=lambda t: (t.case_id, t.task_id))
    write_jsonl(out / "tasks.jsonl", (t.to_dict() for t in all_tasks))
    (out / "patients.json").write_text(json.dumps({"n": len(mrn_to_hash), "patient_hashes": sorted(set(mrn_to_hash.values()))}, indent=1))
    (out / "patients_raw.json").write_text(json.dumps({"mrn_to_hash": mrn_to_hash}, indent=1)); os.chmod(out / "patients_raw.json", 0o600)
    n_cases = len(kept)
    stats.update({"n_cases": n_cases, "n_tasks": len(all_tasks), "n_patients": len(set(mrn_to_hash.values())),
                  "by_modality": dict(Counter(a_.modality for a_, _ in kept)), "by_family": dict(Counter(t.metadata["family"] for t in all_tasks)),
                  "by_task_type": dict(Counter(t.task_type for t in all_tasks)), "by_scope": dict(Counter(t.contract.scope for t in all_tasks)),
                  "leakage_fraction": round(sum(t.leakage for t in all_tasks) / max(1, len(all_tasks)), 4),
                  "rt_nonempty": all(bool(t.required_ids) for t in all_tasks), "seconds": round(time.time() - t0)})
    (out / "construction_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "metadata.json").write_text(json.dumps({"scopebench_version": __version__, "oracle_spec_version": oracle.SPEC_VERSION, "tier": "A", "split": a.split, "seed": seed,
                                                    "db": DUCKDB.name, "modalities": a.modalities, "families": list(FAMILIES), "scopes": ["pre", "at", "post", "full"],
                                                    "time_shift": "per-patient whole-day offset from the enclave salt (|offset| in [30, 400] days)",
                                                    "identifiers": "hashed with the enclave salt; no MRN, CSN, accession or order id in any object"}, indent=1))
    # ---- text-free release copy (still inside the enclave)
    rel = ENCLAVE / "release" / a.out_name
    (rel / "evidence").mkdir(parents=True, exist_ok=True)
    write_jsonl(rel / "tasks.jsonl", (strip_text(t.to_dict()) for t in all_tasks))
    for _, (case, _) in kept:
        write_jsonl(rel / "evidence" / f"{case.case_id}.jsonl", (strip_text(e.to_dict()) for e in case.evidence))
    for name in ("patients.json", "construction_stats.json", "metadata.json"):
        (rel / name).write_text((out / name).read_text())
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    (BUILD_DIR / f"construction_stats_{a.out_name}.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {len(all_tasks)} tasks / {n_cases} cases / {stats['n_patients']} patients to {out} (release copy: {rel}); "
          f"by modality {stats['by_modality']}; leakage {stats['leakage_fraction']}; rejected {sum(rejected.values())} of {sum(attempts.values())} attempts; {stats['seconds']}s")


if __name__ == "__main__":
    main()
