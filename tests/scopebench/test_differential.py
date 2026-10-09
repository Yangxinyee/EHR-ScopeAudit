"""Differential test (SKILL_SPEC §9): oracle vs skills on >= 5,000 random inputs per operation.

Run: .venv-analysis/bin/python -m pytest -q tests/scopebench/test_differential.py -p no:cacheprovider
Set SCOPEBENCH_DIFF_N to change the sample size. The test writes a JSON summary next to the
oracle package (experiments/scopebench_m1/differential_<spec>.json) with per-operation agreement.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scopebench.predicate import filter_legal  # noqa: E402
from scopebench.schema import Contract, EvidenceObject  # noqa: E402

oracle = pytest.importorskip("scopebench.oracle")
skills = pytest.importorskip("scopebench.skills")

N = int(os.environ.get("SCOPEBENCH_DIFF_N", "5000"))
OUT = ROOT / "experiments/scopebench_m1"
INDEX = datetime(2150, 6, 1, 12, 0, tzinfo=timezone.utc)
SOURCES = ["labevents", "transfers", "admissions", "edstays", "emar", "diagnoses_icd", "triage", "lab"]
CONCEPTS = ["creatinine", "hemoglobin", "potassium"]
DRUGS = ["heparin", "warfarin", "insulin"]


def iso(t: datetime | None) -> str | None:
    return None if t is None else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def random_pool(rng: random.Random, n: int) -> list[EvidenceObject]:
    pool = []
    times = [INDEX + timedelta(minutes=rng.randint(-5000, 5000)) for _ in range(max(1, n // 2))]
    for i in range(n):
        occ = rng.choice(times) if rng.random() < 0.7 else INDEX + timedelta(minutes=rng.randint(-5000, 5000))
        avail = occ + timedelta(minutes=rng.randint(0, 600)) if rng.random() < 0.9 else None
        if rng.random() < 0.05:
            occ = None
        src = rng.choice(SOURCES)
        value = rng.choice([round(rng.uniform(0, 20), 2), rng.randint(0, 9), "abc", None, "3.5"]) if rng.random() < 0.9 else rng.uniform(-5, 5)
        fields = {"value": value, "unit": rng.choice(["mg/dL", "g/dL", "sec"]), "concept": rng.choice(CONCEPTS),
                  "dose": rng.choice([rng.uniform(0.5, 10), "x", None]), "medication": rng.choice(DRUGS + ["Heparin ", None]),
                  "code": rng.choice(["I50.9", "N17", "E11.65", None]), "acuity": rng.choice([1, 2, 3, "2", None]),
                  "careunit": rng.choice(["MICU", "Med", None]), "discharge_location": rng.choice(["HOME", "SNF", None]),
                  "disposition": rng.choice(["ADMITTED", "HOME", None])}
        pool.append(EvidenceObject(id=f"e{i:03d}", source=src, patient="p",  # ids are unique per pool (SKILL_SPEC §0)
                                   occurred=iso(occ), available=iso(avail), fields=fields))
    return pool


def random_contract(rng: random.Random) -> Contract:
    return Contract(index_time=iso(INDEX), scope=rng.choice(["pre", "at", "post", "full"]), basis=rng.choice(["available", "occurred"]),
                    window_hours=rng.choice([None, 24, 72, 720]), at_tolerance_hours=rng.choice([0.0, 6.0]))


def params_for(op: str, rng: random.Random, contract: Contract) -> dict:
    basis = contract.basis
    if op in ("lab_trend_k3", "med_dose_k3"):
        return {"k": 3}
    if op == "nearest_k":
        return {"k": rng.randint(1, 5), "basis": basis}
    if op in ("first_after", "last_before"):
        return {"basis": basis, "which": op}
    if op in ("window_count", "window_sum", "window_mean"):
        return {"basis": basis, "agg": op.split("_")[1]}
    if op == "outcome_lookup":
        target = rng.choice(["first_careunit", "discharge_disposition", "ed_disposition", "administration_count"])
        field = {"first_careunit": "careunit", "discharge_disposition": "discharge_location", "ed_disposition": "disposition", "administration_count": "medication"}[target]
        return {"target": target, "field": field, "basis": basis, "drug": rng.choice(DRUGS)}
    if op == "cohort_criteria":
        crits = []
        for _ in range(rng.randint(1, 3)):
            kind = rng.choice(["lab_threshold", "code_present", "exposure", "acuity"])
            if kind == "lab_threshold":
                crits.append({"kind": kind, "concept": rng.choice(CONCEPTS), "op": rng.choice([">=", "<=", ">", "<"]), "threshold": round(rng.uniform(0, 15), 1), "basis": basis})
            elif kind == "code_present":
                crits.append({"kind": kind, "prefix": rng.choice(["I50", "N17", "E11", "Z99"])})
            elif kind == "exposure":
                crits.append({"kind": kind, "drug": rng.choice(DRUGS)})
            else:
                crits.append({"kind": kind, "max_acuity": rng.randint(1, 3)})
        return {"criteria": crits}
    raise ValueError(op)


def normalize(obj):
    if isinstance(obj, float):
        return round(obj, 6) if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj


OPS = ["lab_trend_k3", "med_dose_k3", "nearest_k", "first_after", "last_before", "window_count", "window_sum", "window_mean", "outcome_lookup", "cohort_criteria"]


def test_operation_sets_match():
    assert set(oracle.OPERATIONS) == set(OPS) == set(skills.OPERATIONS)
    assert oracle.SPEC_VERSION == skills.SPEC_VERSION


@pytest.mark.parametrize("op", OPS)
def test_differential(op: str):
    rng = random.Random(f"{op}-{N}")
    failures = []
    exceptions = []
    for i in range(N):
        pool = random_pool(rng, rng.randint(0, 40))
        contract = random_contract(rng)
        params = params_for(op, rng, contract)
        legal_pool = filter_legal(pool, contract)
        if op in ("lab_trend_k3", "med_dose_k3", "nearest_k"):
            legal_pool = [e for e in legal_pool if e.time(contract.basis if op != "lab_trend_k3" else "available") is not None]
        try:
            a = oracle.OPERATIONS[op](list(legal_pool), contract, dict(params))
        except Exception as exc:  # noqa: BLE001
            a = {"__exception__": f"{type(exc).__name__}: {exc}"}
        try:
            b = skills.OPERATIONS[op](list(legal_pool), contract, dict(params))
        except Exception as exc:  # noqa: BLE001
            b = {"__exception__": f"{type(exc).__name__}: {exc}"}
        if "__exception__" in a or "__exception__" in b:
            exceptions.append({"i": i, "oracle": a.get("__exception__"), "skills": b.get("__exception__"), "params": params, "n": len(legal_pool)})
            if a.get("__exception__") and b.get("__exception__"):
                continue
        if normalize(a) != normalize(b):
            failures.append({"i": i, "params": params, "contract": contract.to_dict(), "n_legal": len(legal_pool),
                             "pool": [e.to_dict() for e in legal_pool][:12], "oracle": a, "skills": b})
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / f"differential_{op}.json"
    summary_path.write_text(json.dumps({"op": op, "n": N, "spec_version": oracle.SPEC_VERSION, "disagreements": len(failures),
                                        "exceptions": len(exceptions), "examples": failures[:5], "exception_examples": exceptions[:5]}, indent=1, default=str))
    assert not exceptions, f"{op}: {len(exceptions)} exceptions, first: {exceptions[0]}"
    assert not failures, f"{op}: {len(failures)}/{N} disagreements, first: {json.dumps(failures[0], default=str)[:1500]}"
