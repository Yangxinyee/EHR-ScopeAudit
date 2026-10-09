"""MIMIC sources -> EvidenceObjects with explicit time bases (plan §5.1).

| source          | occurred                 | available            | fields                                           |
|-----------------|--------------------------|----------------------|--------------------------------------------------|
| labevents       | charttime (specimen)     | storetime            | concept, value, unit, flag, itemid               |
| microbiology    | charttime                 | storetime            | concept (test_name), organism, interpretation    |
| emar            | charttime (administration)| storetime            | medication, event_txt                            |
| prescriptions   | starttime                | starttime            | drug, dose, unit, route, stoptime                |
| transfers       | intime                   | intime               | eventtype, careunit, outtime                     |
| admissions      | admittime                | dischtime            | admission_type, discharge_location, dischtime    |
| diagnoses_icd   | admittime of the stay    | dischtime (coded at discharge) | code, icd_version, title, seq_num      |
| radiology_line  | charttime of the note    | storetime            | note_id, line, section, text, exam_name          |
| discharge_line  | charttime                | storetime            | note_id, line, section, text                     |
| edstays         | intime                   | outtime              | disposition, arrival_transport                   |
| triage          | edstay intime            | edstay intime        | acuity, temperature, heartrate, ...              |
| vitalsign       | charttime                | charttime            | temperature, heartrate, resprate, o2sat, sbp, dbp|
| pyxis           | charttime                | charttime            | medication (name)                                |
| medrecon        | charttime                | charttime            | medication (name), etccode                       |

Identifiers: evidence ids are stable SHA-256 prefixes of (source, row key); patient ids are hashed.
Raw MIMIC identifiers never enter the manifests (plan §1.5); `row_ref` keeps a hashed row key.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..schema import EvidenceObject, stable_hex

SALT = "scopebench-tierb-2027"


def hash_patient(subject_id: int | str) -> str:
    return hashlib.sha256(f"{SALT}|patient|{subject_id}".encode()).hexdigest()[:16]


def eid(source: str, key: object) -> str:
    return stable_hex(SALT, source, key, n=24)


def ts(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    if s in ("", "NaT", "None"):
        return None
    return s.replace(" ", "T") + ("Z" if len(s) == 19 else "")


@dataclass
class SourceLoader:
    """All per-patient extraction lives here so the builder and the surrogate generator share it."""

    con: Any  # duckdb connection

    def _rows(self, sql: str, *params) -> list[dict[str, Any]]:
        cur = self.con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ------------------------------------------------------------------ structured hosp tables
    def labevents(self, subject_id: int, itemids: Iterable[int] | None = None) -> list[EvidenceObject]:
        where = "AND l.itemid IN (" + ",".join(str(int(i)) for i in itemids) + ")" if itemids else ""
        rows = self._rows(f"""
            SELECT l.labevent_id, l.hadm_id, l.itemid, d.label, l.charttime, l.storetime, l.value, l.valuenum, l.valueuom, l.flag
            FROM labevents l JOIN d_labitems d USING (itemid)
            WHERE l.subject_id = ? {where}""", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("labevents", r["labevent_id"]), source="labevents", patient=p,
                               occurred=ts(r["charttime"]), available=ts(r["storetime"]), row_ref=eid("labevents_row", r["labevent_id"]),
                               fields={"concept": r["label"], "itemid": int(r["itemid"]), "value": r["valuenum"], "value_text": r["value"],
                                       "unit": r["valueuom"], "flag": r["flag"], "hadm": _h(r["hadm_id"])})
                for r in rows]

    def microbiology(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM microbiologyevents WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("microbiologyevents", r["microevent_id"]), source="microbiologyevents", patient=p,
                               occurred=ts(r["charttime"] or r["chartdate"]), available=ts(r["storetime"] or r["storedate"]),
                               row_ref=eid("micro_row", r["microevent_id"]),
                               fields={"concept": r["test_name"], "specimen": r["spec_type_desc"], "organism": r["org_name"],
                                       "antibiotic": r["ab_name"], "interpretation": r["interpretation"], "hadm": _h(r["hadm_id"])})
                for r in rows]

    def emar(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM emar WHERE subject_id = ? AND event_txt = 'Administered'", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("emar", r["emar_id"]), source="emar", patient=p,
                               occurred=ts(r["charttime"]), available=ts(r["storetime"]), row_ref=eid("emar_row", r["emar_id"]),
                               fields={"medication": r["medication"], "event_txt": r["event_txt"], "hadm": _h(r["hadm_id"])})
                for r in rows]

    def prescriptions(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM prescriptions WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("prescriptions", (r["pharmacy_id"], r["poe_id"], r["starttime"], r["drug"])), source="prescriptions", patient=p,
                               occurred=ts(r["starttime"]), available=ts(r["starttime"]), occurred_end=ts(r["stoptime"]),
                               row_ref=eid("rx_row", (r["pharmacy_id"], r["poe_id"])),
                               fields={"drug": r["drug"], "dose": r["dose_val_rx"], "unit": r["dose_unit_rx"], "route": r["route"], "hadm": _h(r["hadm_id"])})
                for r in rows]

    def transfers(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM transfers WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("transfers", r["transfer_id"]), source="transfers", patient=p,
                               occurred=ts(r["intime"]), available=ts(r["intime"]), occurred_end=ts(r["outtime"]), row_ref=eid("transfer_row", r["transfer_id"]),
                               fields={"eventtype": r["eventtype"], "careunit": r["careunit"], "hadm": _h(r["hadm_id"])})
                for r in rows]

    def admissions(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM admissions WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("admissions", r["hadm_id"]), source="admissions", patient=p,
                               occurred=ts(r["admittime"]), available=ts(r["dischtime"]), occurred_end=ts(r["dischtime"]), row_ref=eid("adm_row", r["hadm_id"]),
                               fields={"admission_type": r["admission_type"], "admission_location": r["admission_location"],
                                       "discharge_location": r["discharge_location"], "dischtime": ts(r["dischtime"]),
                                       "hospital_expire_flag": r["hospital_expire_flag"], "hadm": _h(r["hadm_id"])})
                for r in rows]

    def diagnoses(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("""
            SELECT d.hadm_id, d.seq_num, d.icd_code, d.icd_version, t.long_title, a.admittime, a.dischtime
            FROM diagnoses_icd d JOIN admissions a USING (hadm_id)
            LEFT JOIN d_icd_diagnoses t ON t.icd_code = d.icd_code AND t.icd_version = d.icd_version
            WHERE d.subject_id = ?""", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("diagnoses_icd", (r["hadm_id"], r["seq_num"])), source="diagnoses_icd", patient=p,
                               occurred=ts(r["admittime"]), available=ts(r["dischtime"]), row_ref=eid("dx_row", (r["hadm_id"], r["seq_num"])),
                               fields={"code": r["icd_code"], "icd_version": int(r["icd_version"]), "title": r["long_title"], "seq_num": int(r["seq_num"]), "hadm": _h(r["hadm_id"])})
                for r in rows]

    # ------------------------------------------------------------------ notes (line level)
    def radiology_lines(self, subject_id: int, note_ids: Iterable[str] | None = None) -> list[EvidenceObject]:
        where = "AND note_id IN (" + ",".join(f"'{n}'" for n in note_ids) + ")" if note_ids else ""
        notes = self._rows(f"SELECT note_id, hadm_id, note_seq, charttime, storetime, text FROM radiology WHERE subject_id = ? {where}", subject_id)
        details = self._rows("SELECT note_id, field_name, field_value FROM radiology_detail WHERE subject_id = ? AND field_name IN ('exam_name','exam_code','addendum_note_id','parent_note_id')", subject_id)
        meta: dict[str, dict[str, str]] = {}
        for d in details:
            meta.setdefault(d["note_id"], {})[d["field_name"]] = d["field_value"]
        p = hash_patient(subject_id)
        out: list[EvidenceObject] = []
        for n in notes:
            m = meta.get(n["note_id"], {})
            for line_no, section, text in split_note(n["text"]):
                out.append(EvidenceObject(
                    id=eid("radiology_line", (n["note_id"], line_no)), source="radiology_line", patient=p,
                    occurred=ts(n["charttime"]), available=ts(n["storetime"]), row_ref=eid("rad_note", n["note_id"]),
                    fields={"note": eid("rad_note", n["note_id"]), "line": line_no, "section": section, "text": text,
                            "exam_name": m.get("exam_name"), "exam_code": m.get("exam_code"), "is_addendum": bool(m.get("parent_note_id")),
                            "hadm": _h(n["hadm_id"])}))
        return out

    def discharge_lines(self, subject_id: int) -> list[EvidenceObject]:
        notes = self._rows("SELECT note_id, hadm_id, charttime, storetime, text FROM discharge WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        out: list[EvidenceObject] = []
        for n in notes:
            for line_no, section, text in split_note(n["text"]):
                out.append(EvidenceObject(
                    id=eid("discharge_line", (n["note_id"], line_no)), source="discharge_line", patient=p,
                    occurred=ts(n["charttime"]), available=ts(n["storetime"]), row_ref=eid("ds_note", n["note_id"]),
                    fields={"note": eid("ds_note", n["note_id"]), "line": line_no, "section": section, "text": text, "hadm": _h(n["hadm_id"])}))
        return out

    # ------------------------------------------------------------------ ED
    def edstays(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM edstays WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("edstays", r["stay_id"]), source="edstays", patient=p,
                               occurred=ts(r["intime"]), available=ts(r["outtime"]), occurred_end=ts(r["outtime"]), row_ref=eid("ed_row", r["stay_id"]),
                               fields={"disposition": r["disposition"], "arrival_transport": r["arrival_transport"], "stay": eid("ed_row", r["stay_id"]), "hadm": _h(r["hadm_id"])})
                for r in rows]

    def triage(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT t.*, e.intime FROM triage t JOIN edstays e USING (stay_id) WHERE t.subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("triage", r["stay_id"]), source="triage", patient=p,
                               occurred=ts(r["intime"]), available=ts(r["intime"]), row_ref=eid("triage_row", r["stay_id"]),
                               fields={"acuity": r["acuity"], "temperature": r["temperature"], "heartrate": r["heartrate"], "resprate": r["resprate"],
                                       "o2sat": r["o2sat"], "sbp": r["sbp"], "dbp": r["dbp"], "pain": r["pain"], "stay": eid("ed_row", r["stay_id"])})
                for r in rows]

    def vitalsign(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM vitalsign WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("vitalsign", (r["stay_id"], r["charttime"])), source="vitalsign", patient=p,
                               occurred=ts(r["charttime"]), available=ts(r["charttime"]), row_ref=eid("vs_row", (r["stay_id"], r["charttime"])),
                               fields={"temperature": r["temperature"], "heartrate": r["heartrate"], "resprate": r["resprate"], "o2sat": r["o2sat"],
                                       "sbp": r["sbp"], "dbp": r["dbp"], "rhythm": r["rhythm"], "stay": eid("ed_row", r["stay_id"])})
                for r in rows]

    def pyxis(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM pyxis WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("pyxis", (r["stay_id"], r["charttime"], r["med_rn"])), source="pyxis", patient=p,
                               occurred=ts(r["charttime"]), available=ts(r["charttime"]), row_ref=eid("pyxis_row", (r["stay_id"], r["charttime"], r["med_rn"])),
                               fields={"medication": r["name"], "stay": eid("ed_row", r["stay_id"])})
                for r in rows]

    def medrecon(self, subject_id: int) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM medrecon WHERE subject_id = ?", subject_id)
        p = hash_patient(subject_id)
        return [EvidenceObject(id=eid("medrecon", (r["stay_id"], r["charttime"], r["name"])), source="medrecon", patient=p,
                               occurred=ts(r["charttime"]), available=ts(r["charttime"]), row_ref=eid("medrecon_row", (r["stay_id"], r["charttime"], r["name"])),
                               fields={"medication": r["name"], "etccode": r["etccode"], "etcdescription": r["etcdescription"], "stay": eid("ed_row", r["stay_id"])})
                for r in rows]


def _h(hadm_id: Any) -> str | None:
    return None if hadm_id is None else eid("hadm", int(hadm_id))


_SECTION = re.compile(r"^\s*([A-Z][A-Z /&\-]{2,40}):\s*(.*)$")


def split_note(text: str | None) -> list[tuple[int, str, str]]:
    """Line-level rows: (line_no, section, text). Section = the last 'HEADER:' seen (MIMIC note style)."""
    if not text:
        return []
    out: list[tuple[int, str, str]] = []
    section = "HEADER"
    line_no = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SECTION.match(line)
        if m:
            section = m.group(1).strip()
            rest = m.group(2).strip()
            line_no += 1
            out.append((line_no, section, line if rest else section + ":"))
            continue
        line_no += 1
        out.append((line_no, section, line))
    return out
