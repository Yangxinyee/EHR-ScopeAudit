"""Enclave DuckDB -> EvidenceObjects for one patient, in the shared source vocabulary (plan §7.2).

The Tier-A loader emits the same `source` names and field keys as the Tier-B loader wherever the CCDA table is the
same kind of record, so the frozen oracle operations, skills, prompts, parser and scorer apply unchanged:

| CCDA table          | source           | occurred                | available     | fields                                        |
|---------------------|------------------|-------------------------|---------------|-----------------------------------------------|
| labevents           | labevents        | collection_time         | result_time   | concept (base name), value, value_text, unit, flag, ref_low, ref_high |
| emar                | emar             | date_administered       | same          | medication, dose, route, event_txt=Administered |
| prescriptions       | prescriptions    | ordering_date           | same          | drug, dose, route=None, status                |
| diagnoses           | diagnoses_icd    | first_noted             | same          | code, icd_version=10, title, status           |
| admissions (Outcome)| admissions       | admit_time (or DOS)     | dischtime     | admission_type (visit type), discharge_location=None, dischtime, hospital_expire_flag, los_days |
| transfers (ADT)     | transfers        | in_time                 | same          | eventtype=transfer, careunit, location, icu   |
| procedures          | procedures       | proc_time               | same          | name, code, kind, status                      |
| component_value     | ekg              | result_time             | same          | concept (component), value, value_text, exam (procedure name) |
| narrative_line      | radiology_line / echo_line / cath_line / note_line / pathology_line | result_time | same | note_id, line, section, text, exam_name |

CCDA reports carry one timestamp (the result time), so for narrative sources occurred == available and the wrong-basis
control changes nothing for them; the difference lives in laboratory objects (collection vs result time). Every time is
shifted by the patient's secret offset; ids are salted hashes of (table, cohort, canonical row); no raw MRN, CSN,
accession or order id enters an object.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from ..schema import EvidenceObject
from .timeshift import hash_patient, offset_days, parse_any, shifted_iso

# exam-description patterns per modality (upper-case substring / regex over narrative_line.description); the index report
# of a case is the narrative whose description matches its modality and whose result time is within INDEX_WINDOW_DAYS of DOS
MODALITY_EXAM_PATTERNS: dict[str, str] = {
    "ctpe": r"CT.*(ANGIO|PULMONARY|PE\b|CHEST)",
    "ruq_us": r"(US|ULTRASOUND).*(ABD|RUQ|RIGHT UPPER|GALLBLADDER|LIVER)",
    "fdg_pet_ct": r"PET",
    "head_ct": r"CT.*(HEAD|BRAIN)|CTA HEAD|CT PERFUSION",
    "coronary_ct": r"CT.*(CORONARY|CARDIAC|HEART)",
    "cardiac_mri": r"MR.*(CARDIAC|HEART)",
    "cardiac_cath": r"(CATH|CORONARY ANGIO|PCI|LEFT HEART|RIGHT HEART)",
}
# Cohorts whose exports carry no radiology report text (CTPE, coronary CT, cardiac MRI — verified 2026-09-29 on the
# enclave database: the "PFT Narrative" sheets hold echocardiography / PFT reports only). For these modalities the
# report families use the echocardiography / catheterisation report nearest the index exam (within ±3 days) as the
# index report; the manifest records `index_report_kind` so the two situations are never pooled silently.
MODALITIES_WITHOUT_RADIOLOGY_TEXT = {"ctpe", "coronary_ct", "cardiac_mri"}
INDEX_WINDOW_DAYS = 3
ECHO_PATTERN = r"ECHO|TRANSTHORACIC|TRANSESOPHAGEAL|TEE\b|TTE\b"
_MISSING = {"", "NAN", "NULL", "NONE", "N/A", "NAT"}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return None if s.upper() in _MISSING else s


# Administrative identifier lines of the report exports (patient name / MRN / DOB / phone / staff names / accession) are
# removed from the evidence text at load time — they carry no clinical content and would otherwise reach prompts and
# model outputs. Line numbers of the remaining lines are preserved. Recorded in the Tier-A protocol.
ADMIN_LINE = re.compile(r"^\s*(MRN|MEDICAL RECORD|DOB|DATE OF BIRTH|BIRTH DATE|NAME|PATIENT NAME|PATIENT|PT NAME|TEL|PHONE|FAX|ADDRESS|SONOGRAPHER|FELLOW|"
                        r"INTERPRETING PHYSICIAN|READING PHYSICIAN|ELECTRONICALLY SIGNED BY|SIGNED BY|DICTATED BY|TRANSCRIBED BY|PROCEDURE ATTENDING|ATTENDING|"
                        r"REFERRING (PHYSICIAN|PROVIDER)|ORDERING (PHYSICIAN|PROVIDER)|DISTRIBUTION LIST|ACCESSION( NUMBER| #)?|ACC ?#|CSN|ENCOUNTER|ORDER ?#|"
                        r"RACE|SSN|INSURANCE|LOCATION|WARD|BILLING TIMES|FINAL DATE|CASE (START|END) TIME)\s*[:#]", re.I)


def is_admin_line(text: str) -> bool:
    return bool(ADMIN_LINE.match(text or ""))


# Identifier tokens embedded inside report text are masked at load time (MRN-like tokens, 10-digit encounter numbers,
# DOB / SSN patterns); names cannot be masked by pattern and remain inside the enclave copy only.
_MASKS = ((re.compile(r"\b[A-Z]{1,2}\d{7,9}\b"), "[ID]"), (re.compile(r"(?<![\w.-])\d{10}(?![\w.-])"), "[ID]"),
          (re.compile(r"\b(DOB|DATE OF BIRTH|BIRTH DATE)\s*[:=-]?\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", re.I), "[DOB]"),
          (re.compile(r"\b(MRN|MEDICAL RECORD (NUMBER|#))\s*[:#=-]?\s*[A-Z0-9]{5,}", re.I), "MRN [ID]"),
          (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ID]"), (re.compile(r"\b(ACC|ACCESSION)[A-Z]*[ _:#-]*\d{6,}\b", re.I), "[ACC]"))


def mask_identifiers(text: str) -> str:
    for pat, rep in _MASKS:
        text = pat.sub(rep, text)
    return text


def _dedupe_earliest(objs: list[EvidenceObject]) -> list[EvidenceObject]:
    """The exports repeat a report under several date formats / index sheets (the RUQ index sheet exists as CSV and
    XLSX; ordering dates appear as text and as serial numbers), so the same line can arrive twice with different
    times. One object per id, the earliest availability time wins (deterministic)."""
    best: dict[str, EvidenceObject] = {}
    for e in objs:
        cur = best.get(e.id)
        if cur is None or (e.available or "") < (cur.available or ""):
            best[e.id] = e
    return list(best.values())



def _num(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    m = re.match(r"^[<>]?=?\s*(-?\d+(\.\d+)?)", s)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


@dataclass
class TierALoader:
    con: Any
    salt: str

    # ------------------------------------------------------------------ identity
    def patient(self, mrn: str) -> str:
        return hash_patient(mrn, self.salt)

    def shift(self, mrn: str) -> int:
        return offset_days(mrn, self.salt)

    def eid(self, table: str, cohort: str, key: tuple) -> str:
        return hashlib.sha256(f"{self.salt}|{table}|{cohort}|{'|'.join('' if k is None else str(k) for k in key)}".encode()).hexdigest()[:24]

    def _rows(self, sql: str, *params) -> list[dict[str, Any]]:
        cur = self.con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _ts(self, mrn: str, value: Any) -> str | None:
        return shifted_iso(value, self.shift(mrn))

    # ------------------------------------------------------------------ structured
    def labevents(self, mrn: str, cohort: str | None = None) -> list[EvidenceObject]:
        where = "AND cohort = ?" if cohort else ""
        rows = self._rows(f"SELECT * FROM labevents WHERE mrn = ? {where}", mrn, *([cohort] if cohort else []))
        p = self.patient(mrn)
        out = []
        for r in rows:
            key = (r["csn"], r["result_time"], r["collection_time"], r["lab_name"], r["base_name"], r["value"], r["units"])
            v = _num(r["value"])
            flag = None
            lo, hi = _num(r["ref_low"]), _num(r["ref_high"])
            if v is not None and lo is not None and v < lo:
                flag = "abnormal"
            elif v is not None and hi is not None and v > hi:
                flag = "abnormal"
            out.append(EvidenceObject(id=self.eid("labevents", r["cohort"], key), source="labevents", patient=p,
                                      occurred=self._ts(mrn, r["collection_time"]) or self._ts(mrn, r["result_time"]),
                                      available=self._ts(mrn, r["result_time"]) or self._ts(mrn, r["collection_time"]),
                                      row_ref=self.eid("labevents_row", r["cohort"], key),
                                      fields={"concept": (_clean(r["base_name"]) or _clean(r["lab_name"]) or "").upper(), "lab_name": _clean(r["lab_name"]), "value": v,
                                              "value_text": r["value"], "unit": r["units"], "flag": flag, "ref_low": lo, "ref_high": hi,
                                              "hadm": self.eid("csn", r["cohort"], (r["csn"],))[:12] if r["csn"] else None}))
        return out

    def emar(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM emar WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        return [EvidenceObject(id=self.eid("emar", r["cohort"], (r["csn"], r["date_administered"], r["medication"], r["dose"], r["route"])), source="emar", patient=p,
                               occurred=self._ts(mrn, r["date_administered"]), available=self._ts(mrn, r["date_administered"]),
                               row_ref=self.eid("emar_row", r["cohort"], (r["csn"], r["date_administered"], r["medication"], r["dose"], r["route"])),
                               fields={"medication": r["medication"], "dose": r["dose"], "route": r["route"], "event_txt": "Administered",
                                       "hadm": self.eid("csn", r["cohort"], (r["csn"],))[:12] if r["csn"] else None})
                for r in rows if r["date_administered"]]

    def prescriptions(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM prescriptions WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        return [EvidenceObject(id=self.eid("prescriptions", r["cohort"], (r["csn"], r["ordering_date"], r["medication"], r["dose"], r["status"])), source="prescriptions", patient=p,
                               occurred=self._ts(mrn, r["ordering_date"]), available=self._ts(mrn, r["ordering_date"]),
                               row_ref=self.eid("rx_row", r["cohort"], (r["csn"], r["ordering_date"], r["medication"], r["dose"], r["status"])),
                               fields={"drug": r["medication"], "dose": r["dose"], "unit": None, "route": None, "status": r["status"],
                                       "hadm": self.eid("csn", r["cohort"], (r["csn"],))[:12] if r["csn"] else None})
                for r in rows if r["ordering_date"]]

    def diagnoses(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM diagnoses WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        return [EvidenceObject(id=self.eid("diagnoses", r["cohort"], (r["code"], r["title"], r["first_noted"], r["status"])), source="diagnoses_icd", patient=p,
                               occurred=self._ts(mrn, r["first_noted"]), available=self._ts(mrn, r["first_noted"]),
                               row_ref=self.eid("dx_row", r["cohort"], (r["code"], r["title"], r["first_noted"], r["status"])),
                               fields={"code": (r["code"] or "").replace(".", "").strip().upper(), "code_raw": r["code"], "icd_version": 10, "title": r["title"], "status": r["status"]})
                for r in rows if r["first_noted"]]

    def admissions(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM admissions WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        out = []
        for r in rows:
            admit = parse_any(r["admit_time"]) or parse_any(r["date_of_service"])
            if admit is None:
                continue
            los = _num(r["los_days"])
            disch = None
            if los is not None:
                from datetime import timedelta
                disch = admit + timedelta(days=los)
            death = parse_any(r["death_date"])
            expire = bool(death and disch and admit <= death <= disch)
            key = (r["csn"], r["admit_time"], r["date_of_service"], r["visit_type"], r["los_days"])
            out.append(EvidenceObject(id=self.eid("admissions", r["cohort"], key), source="admissions", patient=p,
                                      occurred=self._ts(mrn, admit), available=self._ts(mrn, disch) if disch else self._ts(mrn, admit),
                                      occurred_end=self._ts(mrn, disch) if disch else None, row_ref=self.eid("adm_row", r["cohort"], key),
                                      fields={"admission_type": (r["visit_type"] or "").strip().upper() or None, "admission_location": None, "discharge_location": None,
                                              "dischtime": self._ts(mrn, disch) if disch else None, "los_days": los, "hospital_expire_flag": int(expire),
                                              "hadm": self.eid("csn", r["cohort"], (r["csn"],))[:12] if r["csn"] else None}))
        return out

    def transfers(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM transfers WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        return [EvidenceObject(id=self.eid("transfers", r["cohort"], (r["csn"], r["in_time"], r["department"], r["location"])), source="transfers", patient=p,
                               occurred=self._ts(mrn, r["in_time"]), available=self._ts(mrn, r["in_time"]), occurred_end=self._ts(mrn, r["out_time"]),
                               row_ref=self.eid("transfer_row", r["cohort"], (r["csn"], r["in_time"], r["department"], r["location"])),
                               fields={"eventtype": "transfer", "careunit": (r["department"] or "").strip().upper() or None, "location": r["location"],
                                       "icu": str(r["icu_flag"] or "").strip().upper() in ("Y", "YES", "1", "TRUE"),
                                       "hadm": self.eid("csn", r["cohort"], (r["csn"],))[:12] if r["csn"] else None})
                for r in rows if r["in_time"]]

    def procedures(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM procedures WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        return [EvidenceObject(id=self.eid("procedures", r["cohort"], (r["csn"], r["proc_time"], r["name"], r["code"], r["kind"])), source="procedures", patient=p,
                               occurred=self._ts(mrn, r["proc_time"]), available=self._ts(mrn, r["proc_time"]),
                               row_ref=self.eid("proc_row", r["cohort"], (r["csn"], r["proc_time"], r["name"], r["code"], r["kind"])),
                               fields={"name": (r["name"] or "").strip().upper() or None, "code": r["code"], "kind": r["kind"], "status": r["status"]})
                for r in rows if r["proc_time"]]

    def ekg_values(self, mrn: str) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM component_value WHERE mrn = ?", mrn)
        p = self.patient(mrn)
        return [EvidenceObject(id=self.eid("ekg", r["cohort"], (r["csn"], r["result_time"], r["procedure_name"], r["component"], r["line"], r["value"])), source="ekg", patient=p,
                               occurred=self._ts(mrn, r["result_time"]), available=self._ts(mrn, r["result_time"]),
                               row_ref=self.eid("ekg_row", r["cohort"], (r["csn"], r["result_time"], r["procedure_name"], r["component"], r["line"])),
                               fields={"concept": (r["component"] or "").strip().upper(), "value": _num(r["value"]), "value_text": r["value"], "unit": None,
                                       "exam": r["procedure_name"]})
                for r in rows if r["result_time"]]

    # ------------------------------------------------------------------ narrative lines
    @staticmethod
    def line_source(kind: str | None, description: str | None) -> str:
        """Source of a narrative_line row. Radiology text reaches the loader only through the accession-keyed reports
        (`radiology_lines`); the cohort workbooks' narrative sheets hold echocardiography and pulmonary-function reports."""
        d = (description or "").upper()
        if kind == "cath" or kind == "cath_findings":
            return "cath_line"
        if kind == "note" or kind == "study_note":
            return "note_line"
        if kind == "pathology":
            return "pathology_line"
        if re.search(ECHO_PATTERN, d):
            return "echo_line"
        if re.search(r"PULMONARY FUNCTION|SPIROMETRY|PFT", d):
            return "pft_line"
        return "other_line"

    # ------------------------------------------------------------------ radiology reports keyed by accession
    def ensure_report_links(self) -> None:
        """Materialise `report_link(accession, mrn, cohort, exam_dos)` once: the RUQ index sheet links accession -> MRN
        directly (exam date included); the head-CT and PET/CT exports carry only an encounter number (CSN), which is
        resolved through the other tables and used ONLY when it maps to exactly one MRN (9 % of CSNs are ambiguous in
        the exports and are dropped, never guessed)."""
        exists = self._rows("SELECT count(*) AS n FROM information_schema.tables WHERE table_name = 'report_link'")[0]["n"]
        if exists:
            return
        self.con.execute("""
            CREATE TABLE report_link AS
            WITH cm AS (
              SELECT csn, min(mrn) AS mrn FROM (
                SELECT DISTINCT mrn, csn FROM (
                  SELECT mrn, csn FROM labevents UNION ALL SELECT mrn, csn FROM emar UNION ALL SELECT mrn, csn FROM narrative_line
                  UNION ALL SELECT mrn, csn FROM admissions UNION ALL SELECT mrn, csn FROM component_value)
                WHERE csn IS NOT NULL AND csn <> '' AND mrn IS NOT NULL AND mrn <> '')
              GROUP BY csn HAVING count(DISTINCT mrn) = 1)
            SELECT DISTINCT accession, mrn, cohort, exam_dos, link FROM (
              SELECT e.accession, e.mrn, e.cohort, e.dos AS exam_dos, 'exam_index' AS link FROM exam_index e WHERE e.accession IS NOT NULL
              UNION ALL
              SELECT r.accession, cm.mrn, r.cohort, NULL AS exam_dos, 'csn' AS link
              FROM (SELECT DISTINCT cohort, csn, accession FROM report_by_accession WHERE csn IS NOT NULL) r JOIN cm ON cm.csn = r.csn
              WHERE r.accession NOT IN (SELECT accession FROM exam_index WHERE accession IS NOT NULL))""")
        self.con.execute("CREATE INDEX IF NOT EXISTS ix_report_link_mrn ON report_link(mrn)")

    def radiology_lines(self, mrn: str) -> list[EvidenceObject]:
        """Lines of the accession-keyed radiology reports of one patient (source `radiology_line`): time = the exam date
        when the index sheet gives it, else the ordering date; both bases equal (one timestamp per report)."""
        rows = self._rows("""SELECT r.cohort, r.accession, r.csn, r.ordering_date, r.text, r.line, r.description, l.exam_dos
                             FROM report_by_accession r JOIN report_link l ON l.accession = r.accession WHERE l.mrn = ? AND r.text IS NOT NULL AND r.text <> '' AND upper(r.text) <> 'NULL'""", mrn)
        p = self.patient(mrn)
        out = []
        for r in rows:
            if is_admin_line(r["text"]):
                continue
            when = _clean(r["exam_dos"]) or _clean(r["ordering_date"])
            t = self._ts(mrn, when)
            if t is None:
                continue
            note_id = self.eid("rad_note", r["cohort"], (r["accession"],))[:16]
            line_no = _num(r["line"])
            key = (r["accession"], r["line"], r["text"][:200])
            out.append(EvidenceObject(id=self.eid("radiology_line", r["cohort"], key), source="radiology_line", patient=p, occurred=t, available=t,
                                      row_ref=self.eid("radiology_line_row", r["cohort"], key),
                                      fields={"note": note_id, "line": int(line_no) if line_no is not None else 0, "section": None, "text": mask_identifiers(r["text"].strip()),
                                              "exam_name": (_clean(r["description"]) or "").upper() or None, "kind": "radiology"}))
        return _dedupe_earliest(out)

    def narrative_lines(self, mrn: str, sources: tuple[str, ...] | None = None) -> list[EvidenceObject]:
        rows = self._rows("SELECT * FROM narrative_line WHERE mrn = ? AND text IS NOT NULL AND text <> '' AND upper(text) <> 'NULL' AND result_time IS NOT NULL", mrn)
        p = self.patient(mrn)
        out = self.radiology_lines(mrn) if (sources is None or "radiology_line" in sources) else []
        for r in rows:
            src = self.line_source(r["kind"], r["description"])
            if sources and src not in sources:
                continue
            if is_admin_line(r["text"]):
                continue
            note_key = (r["csn"], r["result_time"], r["description"], r["order_id"], r["accession"])
            note_id = self.eid("note", r["cohort"], note_key)[:16]
            line_no = _num(r["line"])
            key = note_key + (r["section"], r["line"], r["text"][:200])
            out.append(EvidenceObject(id=self.eid(src, r["cohort"], key), source=src, patient=p,
                                      occurred=self._ts(mrn, r["result_time"]), available=self._ts(mrn, r["result_time"]),
                                      row_ref=self.eid(f"{src}_row", r["cohort"], key),
                                      fields={"note": note_id, "line": int(line_no) if line_no is not None else 0, "section": (r["section"] or "").strip().upper() or None,
                                              "text": mask_identifiers(r["text"].strip()), "exam_name": (r["description"] or "").strip().upper() or None, "kind": r["kind"]}))
        return _dedupe_earliest(out)

    def index_report_lines(self, mrn: str, modality: str, dos_iso: str, lines: list[EvidenceObject] | None = None) -> tuple[list[EvidenceObject], str | None]:
        """(lines of the index report, kind). Radiology modalities: the accession-keyed report whose exam name matches the
        modality pattern, within −1 … +INDEX_WINDOW_DAYS days of the (shifted) index date, nearest first. Catheterisation:
        the cath report at the anchor. Modalities without radiology text (MODALITIES_WITHOUT_RADIOLOGY_TEXT): the
        echocardiography / catheterisation report nearest the index within ±INDEX_WINDOW_DAYS ('echo_or_cath_nearest')."""
        from datetime import timedelta
        from ..schema import parse_time
        lines = lines if lines is not None else self.narrative_lines(mrn, ("radiology_line", "echo_line", "cath_line"))
        idx = parse_time(dos_iso)
        if modality == "cardiac_cath":
            cands, kind, lo, hi = [e for e in lines if e.source == "cath_line"], "cath", 1, INDEX_WINDOW_DAYS
        elif modality in MODALITIES_WITHOUT_RADIOLOGY_TEXT:
            cands, kind, lo, hi = [e for e in lines if e.source in ("echo_line", "cath_line")], "echo_or_cath_nearest", INDEX_WINDOW_DAYS, INDEX_WINDOW_DAYS
        else:
            pat = re.compile(MODALITY_EXAM_PATTERNS[modality])
            cands, kind, lo, hi = [e for e in lines if e.source == "radiology_line" and e.fields.get("exam_name") and pat.search(e.fields["exam_name"])], "radiology", 1, INDEX_WINDOW_DAYS
        cands = [e for e in cands if e.time("available") is not None and idx - timedelta(days=lo) <= e.time("available") <= idx + timedelta(days=hi)]
        if not cands:
            return [], None
        nearest = min(cands, key=lambda e: (abs((e.time("available") - idx).total_seconds()), e.fields["note"])).fields["note"]
        return sorted([e for e in cands if e.fields["note"] == nearest], key=lambda e: (e.fields.get("line") or 0, e.id)), kind

    # ------------------------------------------------------------------ anchors
    def index_events(self, modality: str) -> list[dict[str, Any]]:
        """Distinct (cohort, mrn, dos) sampling anchors of a modality; the date is returned raw (shifted later per patient).
        Imaging modalities: the cohort's sampling anchor `DOS` (the index exam). Cardiac catheterisation (D6): the
        procedure itself — the result time of a cath narrative whose description matches the cath pattern (the cath
        sheets' `DOS` column is the parent cohort's imaging anchor, not the procedure)."""
        if modality == "cardiac_cath":
            rows = self._rows("SELECT DISTINCT cohort, mrn, description, result_time AS dos FROM narrative_line "
                              "WHERE kind = 'cath' AND result_time IS NOT NULL AND mrn IS NOT NULL")
            pat = re.compile(MODALITY_EXAM_PATTERNS["cardiac_cath"])
            seen = set(); out = []
            for r in rows:
                if not pat.search((r["description"] or "").upper()):
                    continue
                d = parse_any(r["dos"])
                if d is None:
                    continue
                key = (r["cohort"], r["mrn"], d.strftime("%Y-%m-%d"))
                if key in seen:
                    continue
                seen.add(key); out.append({"cohort": r["cohort"], "mrn": r["mrn"], "dos": d.strftime("%Y-%m-%d %H:%M:%S")})
            return out
        return self._rows("SELECT DISTINCT cohort, mrn, dos FROM index_candidates WHERE modality = ? AND dos IS NOT NULL AND mrn IS NOT NULL", modality)

    def all_evidence(self, mrn: str) -> list[EvidenceObject]:
        return (self.labevents(mrn) + self.emar(mrn) + self.prescriptions(mrn) + self.diagnoses(mrn) + self.admissions(mrn) + self.transfers(mrn)
                + self.procedures(mrn) + self.ekg_values(mrn) + self.narrative_lines(mrn))
