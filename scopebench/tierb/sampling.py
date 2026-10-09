"""Candidate discovery and seeded selection for Tier-B (plan §5.2).

Every family exposes a SQL query returning one row per *anchor* (patient + index event [+ concept]).
Selection is deterministic: anchors are ordered by SHA-256(seed, family, anchor key) and taken in
that order subject to (a) one case per patient across ALL families and both splits, (b) the
requested count. The development pool is drawn first from its own seed so test patients never
appear in it; the builder records both seeds and every anchor key hash in construction_stats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable

LAB_ITEMS = {50813: "Lactate", 50868: "Anion Gap", 50882: "Bicarbonate", 50893: "Calcium, Total", 50912: "Creatinine", 50931: "Glucose",
             50971: "Potassium", 50983: "Sodium", 51221: "Hematocrit", 51222: "Hemoglobin", 51265: "Platelet Count", 51301: "White Blood Cells"}


def order_key(seed: int, family: str, key: tuple) -> str:
    return hashlib.sha256(f"{seed}|{family}|{'|'.join(str(k) for k in key)}".encode()).hexdigest()


@dataclass(frozen=True)
class Anchor:
    family: str
    subject_id: int
    key: tuple
    payload: dict[str, Any]


# --- candidate queries -----------------------------------------------------------------------
# Each returns rows with subject_id first; the family's key/payload are built from the row.

def q_retrieval(con) -> list[Anchor]:
    items = ",".join(map(str, LAB_ITEMS))
    rows = con.execute(f"""
        WITH icu AS (SELECT subject_id, hadm_id, min(intime) AS idx FROM transfers WHERE careunit ILIKE '%ICU%' GROUP BY 1,2),
        lab AS (SELECT l.subject_id, l.hadm_id, l.itemid, i.idx,
                       sum(CASE WHEN l.storetime < i.idx THEN 1 ELSE 0 END) AS n_pre,
                       sum(CASE WHEN l.storetime > i.idx THEN 1 ELSE 0 END) AS n_post,
                       count(*) AS n_all
                FROM labevents l JOIN icu i USING (subject_id, hadm_id)
                WHERE l.itemid IN ({items}) AND l.valuenum IS NOT NULL AND l.storetime IS NOT NULL AND l.charttime IS NOT NULL
                GROUP BY 1,2,3,4)
        SELECT subject_id, hadm_id, itemid, idx, n_pre, n_post, n_all FROM lab WHERE n_pre >= 3 AND n_post >= 3 AND n_all <= 60""").fetchall()
    return [Anchor("evidence_retrieval", int(r[0]), (int(r[1]), int(r[2])), {"hadm_id": int(r[1]), "itemid": int(r[2]), "index_time": str(r[3]), "n_pre": int(r[4]), "n_post": int(r[5])}) for r in rows]


def q_outcome(con) -> list[Anchor]:
    rows = con.execute("""
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.discharge_location, t.n
        FROM admissions a JOIN (SELECT hadm_id, count(*) n FROM transfers WHERE eventtype IN ('transfer','ED','admit') GROUP BY 1) t USING (hadm_id)
        WHERE a.discharge_location IS NOT NULL AND t.n >= 2 AND a.dischtime > a.admittime""").fetchall()
    return [Anchor("outcome_abstraction", int(r[0]), (int(r[1]),), {"hadm_id": int(r[1]), "index_time": str(r[2]), "dischtime": str(r[3])}) for r in rows]


def q_cohort(con) -> list[Anchor]:
    rows = con.execute("""
        WITH ranked AS (SELECT subject_id, hadm_id, admittime, dischtime, row_number() OVER (PARTITION BY subject_id ORDER BY admittime) AS k FROM admissions)
        SELECT r.subject_id, r.hadm_id, r.admittime, r.dischtime, r.k FROM ranked r WHERE r.k >= 2""").fetchall()
    return [Anchor("cohort_abstraction", int(r[0]), (int(r[1]),), {"hadm_id": int(r[1]), "index_time": str(r[2]), "dischtime": str(r[3]), "prior_admissions": int(r[4]) - 1}) for r in rows]


def q_radiology(con, family: str) -> list[Anchor]:
    rows = con.execute("""
        SELECT r.subject_id, r.hadm_id, r.note_id, r.charttime, r.storetime, d.field_value AS exam_name, n.n_notes
        FROM radiology r
        JOIN radiology_detail d ON d.note_id = r.note_id AND d.field_name = 'exam_name'
        JOIN (SELECT hadm_id, count(*) n_notes FROM radiology WHERE hadm_id IS NOT NULL GROUP BY 1) n USING (hadm_id)
        WHERE r.hadm_id IS NOT NULL AND r.text ILIKE '%IMPRESSION:%' AND r.storetime IS NOT NULL AND n.n_notes BETWEEN 2 AND 12""").fetchall()
    return [Anchor(family, int(r[0]), (int(r[1]), str(r[2])), {"hadm_id": int(r[1]), "note_id": str(r[2]), "index_time": str(r[3]), "storetime": str(r[4]), "exam_name": r[5], "n_notes": int(r[6])}) for r in rows]


def q_summarization(con) -> list[Anchor]:
    rows = con.execute("""
        WITH x AS (SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
                          sum(CASE WHEN r.storetime < a.admittime THEN 1 ELSE 0 END) AS n_pre,
                          sum(CASE WHEN r.storetime > a.admittime THEN 1 ELSE 0 END) AS n_post
                   FROM admissions a JOIN radiology r ON r.subject_id = a.subject_id AND r.storetime IS NOT NULL
                   GROUP BY 1,2,3,4)
        SELECT subject_id, hadm_id, admittime, dischtime, n_pre, n_post FROM x WHERE n_pre BETWEEN 2 AND 8 AND n_post BETWEEN 1 AND 8""").fetchall()
    return [Anchor("summarization", int(r[0]), (int(r[1]),), {"hadm_id": int(r[1]), "index_time": str(r[2]), "dischtime": str(r[3]), "n_pre": int(r[4]), "n_post": int(r[5])}) for r in rows]


FAMILY_QUERIES: dict[str, Callable[[Any], list[Anchor]]] = {
    "evidence_retrieval": q_retrieval,
    "outcome_abstraction": q_outcome,
    "cohort_abstraction": q_cohort,
    "claim_verification": lambda con: q_radiology(con, "claim_verification"),
    "report_reconstruction": lambda con: q_radiology(con, "report_reconstruction"),
    "summarization": q_summarization,
}


def select(anchors: Iterable[Anchor], seed: int, family: str, n: int, used_patients: set[int]) -> list[Anchor]:
    """Seeded selection: hash order, one case per patient (also across families/splits via used_patients)."""
    ordered = sorted(anchors, key=lambda a: order_key(seed, family, (a.subject_id,) + a.key))
    out: list[Anchor] = []
    for a in ordered:
        if a.subject_id in used_patients:
            continue
        used_patients.add(a.subject_id)
        out.append(a)
        if len(out) >= n:
            break
    return out
