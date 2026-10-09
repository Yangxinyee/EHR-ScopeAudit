"""The SQL view and the ABAC PDP must reproduce scopebench.predicate.legal exactly (paper S4: same predicate)."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.policy_engines import abac_legal_ids, sql_view_legal_ids  # noqa: E402
from scopebench.predicate import legal_set  # noqa: E402
from scopebench.schema import Contract, EvidenceObject  # noqa: E402

INDEX = datetime(2150, 6, 1, 12, 0, tzinfo=timezone.utc)


def iso(t):
    return None if t is None else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_engines_agree_with_predicate():
    rng = random.Random(7)
    n_checked = 0
    for _ in range(300):
        pool = []
        for i in range(rng.randint(0, 30)):
            occ = INDEX + timedelta(minutes=rng.randint(-5000, 5000)) if rng.random() < 0.9 else None
            avail = (occ + timedelta(minutes=rng.randint(0, 600))) if occ and rng.random() < 0.9 else None
            pool.append(EvidenceObject(id=f"e{i:03d}", source=rng.choice(["labevents", "emar", "radiology_line"]), patient="p", occurred=iso(occ), available=iso(avail)))
        c = Contract(index_time=iso(INDEX), scope=rng.choice(["pre", "at", "post", "full"]), basis=rng.choice(["available", "occurred"]),
                     permitted_sources=rng.choice([None, ("labevents",), ("emar", "radiology_line")]), window_hours=rng.choice([None, 24, 720]),
                     at_tolerance_hours=rng.choice([0.0, 6.0]), index_evidence_ids=tuple(e.id for e in pool[:2]) if pool and rng.random() < 0.3 else ())
        ref = legal_set(pool, c)
        assert sql_view_legal_ids(pool, c) == ref, (c, sorted(ref), sorted(sql_view_legal_ids(pool, c)))
        assert abac_legal_ids(pool, c) == ref, (c, sorted(ref), sorted(abac_legal_ids(pool, c)))
        n_checked += 1
    assert n_checked == 300
