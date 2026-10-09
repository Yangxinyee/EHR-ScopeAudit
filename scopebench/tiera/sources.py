"""CCDA cohort exports -> canonical DuckDB tables in the enclave (plan §7.2, WP4).

The corpus is a set of per-cohort workbooks and delimited exports with eight recurring header signatures. Files are
recognised by their header signature (never by sheet name), tagged with the cohort folder and the imaging modality,
streamed into parquet parts, and materialised as canonical tables with `SELECT DISTINCT` over the non-anchor columns
(every export repeats a patient's rows once per sampling anchor `DOS`; the anchor itself only feeds `index_candidates`).

Canonical tables (all columns VARCHAR; times are parsed and shifted later, at evidence-object construction):
  narrative_line   cohort, modality, kind, mrn, csn, description, result_time, section, line, text, accession, order_id
  component_value  cohort, modality, mrn, csn, procedure_name, result_time, component, line, value
  labevents        cohort, modality, mrn, csn, result_time, collection_time, lab_name, base_name, value, units, ref_low, ref_high
  emar             cohort, modality, mrn, csn, date_administered, medication, dose, route
  prescriptions    cohort, modality, mrn, csn, medication, dose, ordering_date, status
  diagnoses        cohort, modality, mrn, code, title, first_noted, status
  admissions       cohort, modality, mrn, csn, date_of_service, admit_time, los_days, death_date, visit_type
  patients         cohort, modality, mrn, age, gender, race, ethnicity, status, death_date, bmi
  transfers        cohort, modality, mrn, csn, department, location, in_time, out_time, icu_flag
  procedures       cohort, modality, mrn, csn, proc_time, name, code, status, kind
  index_candidates cohort, modality, mrn, csn, dos, origin
Nothing here prints or logs a data value.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import CCDA_ROOT, ENCLAVE

# ----------------------------------------------------------------------------------------------- discovery
MODALITY_BY_FOLDER = {"CTPE": "ctpe", "Ultrasound_RUQ": "ruq_us", "FDG_PET_CT": "fdg_pet_ct", "CT_Head": "head_ct",
                      "CT_Cardiac": "coronary_ct", "Cardiac_MRI": "cardiac_mri"}
ALL_KINDS_MODALITY = {"Cardiac Cath": "cardiac_cath"}          # other All_kinds folders are supplementary sources
SKIP_DIR_PARTS = {"__MACOSX", "data-test", "data-test-v2"}
EXTS = {".xlsx", ".txt", ".csv", ".tsv"}

# header signature (normalised, order-free) -> kind
SIGNATURES: dict[str, set[str]] = {
    "narrative": {"MRN", "CSN", "DESCRIPTION", "RESULT_TIME", "RAD_RESULT_TYPE", "LINE", "NARRATIVE"},
    "cath_narrative": {"MRN", "CSN", "DESCRIPTION", "RESULT_TIME", "ACCESSION_NUMBER", "RAD_RESULT_TYPE", "LINE", "NARRATIVE", "ORDER_PROC_ID"},
    "cath_findings": {"MRN", "CSN", "RESULT_TIME", "ORDER_PROC_ID", "NAME", "FINDINGS_TEXT"},
    "cath_diagnoses": {"MRN", "RESULT_TIME", "PAT_ENC_CSN_ID", "CURRENT_ICD10_LIST", "DX_NAME"},
    "study_notes": {"PATIENT_MRN", "EXAM_STARTED_DATE", "STUDY_NOTE_DATE", "NOTE_ID", "LINE", "NOTE_TEXT"},
    "report_by_accession": {"ACC_NUM", "PAT_ENC_CSN_ID", "ORDERING_DATE", "IMPRESSION", "LINE", "DESCRIPTION"},
    "component_value": {"MRN", "CSN", "PROCEDURE_NAME", "RESULT_TIME", "COMPONENT_NAME", "LINE", "RESULT_VALUE"},
    "labs": {"MRN", "CSN", "RESULT_TIME", "COLLECTION_TIME", "LAB_NAME", "BASE_NAME", "VALUE", "UNITS"},
    "emar": {"MRN", "CSN", "DATE_ADMINISTERED", "MEDICATION_NAME", "DOSE", "ROUTE"},
    "prescriptions": {"MRN", "ENCOUNTER_CSN", "MEDICATION_NAME", "HV_DISCRETE_DOSE", "ORDERING_DATE", "ORDER_STATUS"},
    "diagnoses": {"MRN", "DIAGNOSIS_CODE", "DIAGNOSIS_NAME", "FIRST_NOTED_DATE", "STATUS"},
    "admissions": {"MRN", "CSN", "HOSP_ADMSN_TIME", "LENGTH_OF_STAY_DAYS", "DEATH_DATE", "HOSPITAL_VISIT_TYPE"},
    "patients": {"MRN", "AGE", "GENDER", "ETHNICITY", "STATUS", "DEATH_DATE", "BMI"},
    "transfers": {"PATIENT_MRN", "ADT_DEPARTMENT_NAME", "IN_DTTM", "PAT_OUT_DTTM", "ICU_DEPT_YN"},
    "proc_ir": {"PATIENT_MRN", "PERFORMING_CSN_ID", "BEGIN_EXAM_DTTM", "PROC_CODE", "PROC_NAME", "STATUS"},
    "proc_surgery": {"PATIENT_MRN", "LOG_ID", "IN_OR_DTTM", "ALL_PROC_CODE_ID", "PROC_DISPLAY_NAME", "LOG_STATUS_NM"},
    "proc_cardiac": {"PATIENT_MRN", "ORDER_PROC_ID", "RESULT_TIME", "PROC_ID", "DESCRIPTION", "NAME"},
    "surgical_case": {"PAT_MRN_ID", "LOG_ID", "PROC_DATE", "SERVICE_NM", "PRIMARY_PROCEDURE_ID", "PRIMARY_PROCEDURE_NM"},
    "exam_index": {"PAT_MRN_ID", "DOS", "ACCESSION_NUM", "PROC_CODE", "PROC_NAME"},
    "followup": {"PAT_MRN_ID", "EFFECTIVE_DATE_DTTM", "RN"},
    "followup2": {"PATIENT_MRN", "EFFECTIVE_DATE_DTTM", "RN"},
    "weights": {"PATIENT_MRN", "RECORDED_TIME", "WEIGHT_IN_KG"},
    # supplementary All_kinds / FDG exports (same tables, PAT_MRN_ID / PATIENT_MRN variants)
    "transfers2": {"PAT_MRN_ID", "ADT_DEPARTMENT_NAME", "IN_DTTM", "PAT_OUT_DTTM", "ICU_DEPT_YN"},
    "note_line": {"PAT_MRN_ID", "PAT_ENC_CSN_ID", "CREATE_INSTANT_DTTM", "LINE", "NOTE_TEXT"},
    "pathology": {"ORDER_PROC_ID", "ORDERING_DATE", "DESCRIPTION", "LINE_COMMENT", "RESULTS_COMP_CMT"},
    "proc_ir2": {"PAT_MRN_ID", "PERFORMING_CSN_ID", "BEGIN_EXAM_DTTM", "PROC_CODE", "PROC_NAME"},
    "ultrasound_cpt": {"PAT_ENC_CSN_ID", "CPT_CODE", "CPT_CODE_DESC", "SERVICE_DATE"},
}
# delimited exports without a header row (the RUQ cohort): field names by filename pattern, accepted only when the
# file's column count matches. Their first line is a data row and is never reported anywhere.
HEADERLESS_BY_SUFFIX: dict[str, list[str]] = {
    "_All_RX.txt": ["MRN", "ENCOUNTER_CSN", "MEDICATION_NAME", "HV_DISCRETE_DOSE", "ORDERING_DATE", "ORDER_STATUS"],
    "_Echo.txt": ["MRN", "CSN", "DESCRIPTION", "DOS", "RESULT_TIME", "RAD_RESULT_TYPE", "LINE", "NARRATIVE", "CLOSEST_TO_CT"],
    "_EKG.txt": ["MRN", "CSN", "PROCEDURE_NAME", "DOS", "RESULT_TIME", "COMPONENT_NAME", "LINE", "RESULT_VALUE", "CLOSEST_TO_CT"],
    "_Lab_Results.txt": ["MRN", "DOS", "CSN", "RESULT_TIME", "COLLECTION_TIME", "LAB_NAME", "BASE_NAME", "VALUE", "UNITS", "REFERENCE_RANGE_LOW", "REFERENCE_RANGE_HIGH", "RN"],
}


def headerless_fields(name: str, n_cols: int) -> list[str] | None:
    for suffix, names in HEADERLESS_BY_SUFFIX.items():
        if name.endswith(suffix) and len(names) == n_cols:
            return names
    return None


def _looks_like_column_names(header: list[str]) -> bool:
    return bool(header) and len(header) <= 40 and all(re.fullmatch(r"[A-Z_][A-Z0-9_]{0,40}", h or "") for h in header)


def norm_header(h: Any) -> str:
    s = str(h if h is not None else "").strip().upper()
    s = re.sub(r"\(.*", "", s).strip()
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s


def classify(header: list[str]) -> str | None:
    cols = set(header)
    best = None
    for kind, sig in SIGNATURES.items():
        if sig <= cols and (best is None or len(sig) > len(SIGNATURES[best])):
            best = kind
    return best


@dataclass
class SourceFile:
    path: Path
    sheet: str | None
    kind: str
    cohort: str
    modality: str
    header: list[str]
    fieldnames: list[str] | None = None   # forced names for header-less exports
    part_paths: list[Path] = field(default_factory=list)

    @property
    def path_id(self) -> str:
        return f"{self.path.relative_to(CCDA_ROOT)}::{self.sheet or ''}"


def _cohort_and_modality(path: Path) -> tuple[str, str] | None:
    rel = path.relative_to(CCDA_ROOT).parts
    top = rel[0]
    if top in MODALITY_BY_FOLDER:
        cohort = rel[1] if len(rel) > 2 else top
        cohort = re.sub(r"\s+-.*", "", cohort).strip()          # "29 - pulmonary embolism" -> "29"
        if top == "CTPE" and len(rel) > 3:
            cohort = f"{cohort}/{rel[2]}"                        # "29/has_embolism"
        return cohort, MODALITY_BY_FOLDER[top]
    if top == "All_kinds" and len(rel) > 1:
        folder = rel[1]
        modality = ALL_KINDS_MODALITY.get(folder, "supplementary")
        stem = re.sub(r"\.(xlsx|txt|csv|tsv)$", "", path.name, flags=re.I)
        m = re.match(r"^(\d+[ab]?)_", stem)
        cohort = m.group(1) if m else folder
        return f"{cohort}", modality
    return None


def _xlsx_headers(path: Path) -> list[tuple[str, list[str]]]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        first = next(ws.iter_rows(values_only=True), None)
        if first:
            out.append((ws.title, [norm_header(h) if h is not None else f"COL{i}" for i, h in enumerate(first)]))
    wb.close()
    return out


def _delimited_header(path: Path) -> tuple[str, list[str], bool]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    delim = "\t" if first.count("\t") >= first.count(",") else ","
    header = [norm_header(h) for h in next(csv.reader([first], delimiter=delim))]
    headerish = sum(1 for h in header if re.fullmatch(r"[A-Z_][A-Z0-9_]*", h)) >= max(1, int(0.7 * len(header)))
    return delim, header, headerish


def discover(root: Path = CCDA_ROOT) -> tuple[list[SourceFile], list[dict[str, Any]]]:
    """Recognised files by header signature (headers only are read) and the list of skipped files with the reason."""
    found, skipped = [], []
    candidates = []
    for top in list(MODALITY_BY_FOLDER) + ["All_kinds"]:
        base = root / top
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith("._") and not (set(p.parts) & SKIP_DIR_PARTS):
                candidates.append(p)
    for p in candidates:
        cm = _cohort_and_modality(p)
        if cm is None:
            skipped.append({"path": str(p.relative_to(root)), "reason": "unmapped folder"}); continue
        cohort, modality = cm
        if p.suffix.lower() == ".xlsx":
            try:
                sheets = _xlsx_headers(p)
            except Exception as e:  # noqa: BLE001
                skipped.append({"path": str(p.relative_to(root)), "reason": f"unreadable xlsx: {type(e).__name__}"}); continue
            for sheet, header in sheets:
                kind = classify(header)
                if kind is None:
                    skipped.append({"path": f"{p.relative_to(root)}::{sheet}", "reason": "unknown header signature",
                                    "header": header[:12] if _looks_like_column_names(header) else None, "n_cols": len(header)}); continue
                found.append(SourceFile(p, sheet, kind, cohort, modality, header))
        else:
            delim, header, headerish = _delimited_header(p)
            if not headerish or not _looks_like_column_names(header):
                # first line is data (or free text): report nothing but the column count
                names = headerless_fields(p.name, len(header))
                if names is None:
                    skipped.append({"path": str(p.relative_to(root)), "reason": "no header row and no known field list", "n_cols": len(header)}); continue
                kind = classify(names)
                found.append(SourceFile(p, None, kind, cohort, modality, names, fieldnames=names))
                continue
            kind = classify(header)
            if kind is None:
                skipped.append({"path": str(p.relative_to(root)), "reason": "unknown header signature", "header": header[:12]}); continue
            found.append(SourceFile(p, None, kind, cohort, modality, header))
    return found, skipped


# ----------------------------------------------------------------------------------------------- streaming readers
def iter_rows(sf: SourceFile, chunk: int = 200_000) -> Iterator[list[list[Any]]]:
    """Yield lists of rows (as lists, positional with sf.header); values are left as read."""
    if sf.sheet is not None:
        import openpyxl
        wb = openpyxl.load_workbook(sf.path, read_only=True, data_only=True)
        ws = wb[sf.sheet]
        it = ws.iter_rows(values_only=True)
        next(it, None)
        buf: list[list[Any]] = []
        for row in it:
            buf.append(list(row))
            if len(buf) >= chunk:
                yield buf; buf = []
        if buf:
            yield buf
        wb.close()
    else:
        import pandas as pd
        with open(sf.path, "r", encoding="utf-8-sig", errors="replace") as f:
            first = f.readline()
        delim = "\t" if first.count("\t") >= first.count(",") else ","
        kw: dict[str, Any] = dict(sep=delim, dtype=str, chunksize=chunk, on_bad_lines="skip", encoding="utf-8-sig",
                                  encoding_errors="replace", engine="python" if delim not in ("\t", ",") else "c", quoting=csv.QUOTE_MINIMAL)
        if sf.fieldnames:
            kw.update(header=None, names=sf.fieldnames)
        for df in pd.read_csv(sf.path, **kw):
            df.columns = [norm_header(c) for c in df.columns] if not sf.fieldnames else sf.fieldnames
            yield df.reindex(columns=sf.header).values.tolist()


# ----------------------------------------------------------------------------------------------- canonical mapping
# kind -> (canonical table, {canonical column: source column or callable})
def _c(*names: str):
    """First present column among the candidates."""
    def pick(row: dict[str, Any]) -> Any:
        for n in names:
            if n in row and row[n] not in (None, ""):
                return row[n]
        return None
    return pick


CANONICAL: dict[str, tuple[str, dict[str, Any]]] = {
    "narrative": ("narrative_line", {"kind": lambda r: "narrative", "mrn": _c("MRN"), "csn": _c("CSN"), "description": _c("DESCRIPTION"),
                                     "result_time": _c("RESULT_TIME"), "section": _c("RAD_RESULT_TYPE"), "line": _c("LINE"), "text": _c("NARRATIVE"),
                                     "accession": lambda r: None, "order_id": lambda r: None}),
    "cath_narrative": ("narrative_line", {"kind": lambda r: "cath", "mrn": _c("MRN"), "csn": _c("CSN"), "description": _c("DESCRIPTION"),
                                          "result_time": _c("RESULT_TIME"), "section": _c("RAD_RESULT_TYPE"), "line": _c("LINE"), "text": _c("NARRATIVE"),
                                          "accession": _c("ACCESSION_NUMBER"), "order_id": _c("ORDER_PROC_ID")}),
    "cath_findings": ("narrative_line", {"kind": lambda r: "cath_findings", "mrn": _c("MRN"), "csn": _c("CSN"), "description": _c("NAME"),
                                         "result_time": _c("RESULT_TIME"), "section": lambda r: "FINDINGS", "line": _c("LINE"), "text": _c("FINDINGS_TEXT"),
                                         "accession": lambda r: None, "order_id": _c("ORDER_PROC_ID")}),
    "study_notes": ("narrative_line", {"kind": lambda r: "study_note", "mrn": _c("PATIENT_MRN"), "csn": _c("NOTE_CSN_ID"), "description": lambda r: "STUDY NOTE",
                                       "result_time": _c("STUDY_NOTE_DATE"), "section": lambda r: None, "line": _c("LINE"), "text": _c("NOTE_TEXT"),
                                       "accession": lambda r: None, "order_id": _c("NOTE_ID")}),
    "report_by_accession": ("report_by_accession", {"accession": _c("ACC_NUM"), "csn": _c("PAT_ENC_CSN_ID"), "ordering_date": _c("ORDERING_DATE"),
                                                    "text": _c("IMPRESSION"), "line": _c("LINE"), "description": _c("DESCRIPTION")}),
    "component_value": ("component_value", {"mrn": _c("MRN"), "csn": _c("CSN"), "procedure_name": _c("PROCEDURE_NAME"), "result_time": _c("RESULT_TIME"),
                                            "component": _c("COMPONENT_NAME"), "line": _c("LINE"), "value": _c("RESULT_VALUE")}),
    "labs": ("labevents", {"mrn": _c("MRN"), "csn": _c("CSN"), "result_time": _c("RESULT_TIME"), "collection_time": _c("COLLECTION_TIME"),
                           "lab_name": _c("LAB_NAME"), "base_name": _c("BASE_NAME"), "value": _c("VALUE"), "units": _c("UNITS"),
                           "ref_low": _c("REFERENCE_RANGE_LOW"), "ref_high": _c("REFERENCE_RANGE_HIGH")}),
    "emar": ("emar", {"mrn": _c("MRN"), "csn": _c("CSN"), "date_administered": _c("DATE_ADMINISTERED"), "medication": _c("MEDICATION_NAME"),
                      "dose": _c("DOSE"), "route": _c("ROUTE")}),
    "prescriptions": ("prescriptions", {"mrn": _c("MRN"), "csn": _c("ENCOUNTER_CSN"), "medication": _c("MEDICATION_NAME"), "dose": _c("HV_DISCRETE_DOSE"),
                                        "ordering_date": _c("ORDERING_DATE"), "status": _c("ORDER_STATUS")}),
    "diagnoses": ("diagnoses", {"mrn": _c("MRN"), "code": _c("DIAGNOSIS_CODE"), "title": _c("DIAGNOSIS_NAME"), "first_noted": _c("FIRST_NOTED_DATE"), "status": _c("STATUS")}),
    "cath_diagnoses": ("diagnoses", {"mrn": _c("MRN"), "code": _c("CURRENT_ICD10_LIST"), "title": _c("DX_NAME"), "first_noted": _c("RESULT_TIME"), "status": lambda r: "cath"}),
    "admissions": ("admissions", {"mrn": _c("MRN"), "csn": _c("CSN"), "date_of_service": _c("DATE_OF_SERVICE", "EXAM_STARTED_DATE"), "admit_time": _c("HOSP_ADMSN_TIME"),
                                  "los_days": _c("LENGTH_OF_STAY_DAYS"), "death_date": _c("DEATH_DATE"), "visit_type": _c("HOSPITAL_VISIT_TYPE")}),
    "patients": ("patients", {"mrn": _c("MRN"), "age": _c("AGE"), "gender": _c("GENDER"), "race": _c("RACE"), "ethnicity": _c("ETHNICITY"),
                              "status": _c("STATUS"), "death_date": _c("DEATH_DATE"), "bmi": _c("BMI")}),
    "transfers": ("transfers", {"mrn": _c("PATIENT_MRN"), "csn": _c("PAT_ENC_CSN_ID"), "department": _c("ADT_DEPARTMENT_NAME"), "location": _c("ADT_LOC_NAME"),
                                "in_time": _c("IN_DTTM"), "out_time": _c("PAT_OUT_DTTM"), "icu_flag": _c("ICU_DEPT_YN")}),
    "proc_ir": ("procedures", {"mrn": _c("PATIENT_MRN"), "csn": _c("PERFORMING_CSN_ID"), "proc_time": _c("BEGIN_EXAM_DTTM"), "name": _c("PROC_NAME"),
                               "code": _c("PROC_CODE"), "status": _c("STATUS"), "kind": lambda r: "ir"}),
    "proc_surgery": ("procedures", {"mrn": _c("PATIENT_MRN"), "csn": _c("LOG_ID"), "proc_time": _c("IN_OR_DTTM"), "name": _c("PROC_DISPLAY_NAME"),
                                    "code": _c("ALL_PROC_CODE_ID"), "status": _c("LOG_STATUS_NM"), "kind": lambda r: "surgery"}),
    "proc_cardiac": ("procedures", {"mrn": _c("PATIENT_MRN"), "csn": _c("ORDER_PROC_ID"), "proc_time": _c("RESULT_TIME"), "name": _c("DESCRIPTION", "NAME"),
                                    "code": _c("PROC_ID"), "status": lambda r: None, "kind": lambda r: "cardiac"}),
    "surgical_case": ("procedures", {"mrn": _c("PAT_MRN_ID"), "csn": _c("LOG_ID"), "proc_time": _c("PROC_DATE"), "name": _c("PRIMARY_PROCEDURE_NM"),
                                     "code": _c("PRIMARY_PROCEDURE_ID"), "status": _c("SERVICE_NM"), "kind": lambda r: "surgery"}),
    "exam_index": ("exam_index", {"mrn": _c("PAT_MRN_ID"), "dos": _c("DOS"), "accession": _c("ACCESSION_NUM"), "proc_code": _c("PROC_CODE"), "proc_name": _c("PROC_NAME")}),
    "followup": ("followup", {"mrn": _c("PAT_MRN_ID"), "effective": _c("EFFECTIVE_DATE_DTTM")}),
    "followup2": ("followup", {"mrn": _c("PATIENT_MRN"), "effective": _c("EFFECTIVE_DATE_DTTM")}),
    "weights": ("weights", {"mrn": _c("PATIENT_MRN"), "recorded": _c("RECORDED_TIME"), "kg": _c("WEIGHT_IN_KG")}),
    "transfers2": ("transfers", {"mrn": _c("PAT_MRN_ID"), "csn": _c("PAT_ENC_CSN_ID"), "department": _c("ADT_DEPARTMENT_NAME"), "location": _c("ADT_LOC_NAME"),
                                 "in_time": _c("IN_DTTM"), "out_time": _c("PAT_OUT_DTTM"), "icu_flag": _c("ICU_DEPT_YN")}),
    "note_line": ("narrative_line", {"kind": lambda r: "note", "mrn": _c("PAT_MRN_ID"), "csn": _c("PAT_ENC_CSN_ID"), "description": lambda r: "CLINICAL NOTE",
                                     "result_time": _c("CREATE_INSTANT_DTTM"), "section": lambda r: None, "line": _c("LINE"), "text": _c("NOTE_TEXT"),
                                     "accession": lambda r: None, "order_id": _c("ID")}),
    "pathology": ("narrative_line", {"kind": lambda r: "pathology", "mrn": _c("PATIENT_MRN", "PAT_MRN_ID"), "csn": lambda r: None, "description": _c("DESCRIPTION"),
                                     "result_time": _c("ORDERING_DATE"), "section": lambda r: "RESULT", "line": _c("LINE_COMMENT"), "text": _c("RESULTS_COMP_CMT"),
                                     "accession": lambda r: None, "order_id": _c("ORDER_PROC_ID")}),
    "proc_ir2": ("procedures", {"mrn": _c("PAT_MRN_ID"), "csn": _c("PERFORMING_CSN_ID"), "proc_time": _c("BEGIN_EXAM_DTTM"), "name": _c("PROC_NAME"),
                                "code": _c("PROC_CODE"), "status": _c("STATUS"), "kind": lambda r: "ir"}),
    "ultrasound_cpt": ("procedures", {"mrn": _c("PATIENT_MRN", "PAT_MRN_ID"), "csn": _c("PAT_ENC_CSN_ID"), "proc_time": _c("SERVICE_DATE"), "name": _c("CPT_CODE_DESC"),
                                      "code": _c("CPT_CODE"), "status": lambda r: None, "kind": lambda r: "ultrasound"}),
}
# anchor columns feed index_candidates (per-anchor duplication is what DISTINCT removes from the canonical tables)
ANCHOR_COLS = ("DOS", "EXAM_STARTED_DATE", "EXAM_COMPLETED_DATE", "DATE_OF_SERVICE")
MRN_COLS = ("MRN", "PATIENT_MRN", "PAT_MRN_ID")


def canonical_rows(sf: SourceFile, rows: list[list[Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table, mapping = CANONICAL[sf.kind]
    idx = {h: i for i, h in enumerate(sf.header)}
    out, anchors = [], []
    mrn_col = next((c for c in MRN_COLS if c in idx), None)
    anchor_col = next((c for c in ANCHOR_COLS if c in idx), None)
    csn_col = next((c for c in ("CSN", "ENCOUNTER_CSN", "PAT_ENC_CSN_ID") if c in idx), None)
    for r in rows:
        d = {h: r[i] if i < len(r) else None for h, i in idx.items()}
        rec = {k: (f(d) if callable(f) else d.get(f)) for k, f in mapping.items()}
        rec = {k: (str(v).strip() if v is not None else None) for k, v in rec.items()}
        rec["cohort"], rec["modality"], rec["path_id"] = sf.cohort, sf.modality, sf.path_id
        out.append(rec)
        if mrn_col and anchor_col and d.get(anchor_col) not in (None, ""):
            anchors.append({"cohort": sf.cohort, "modality": sf.modality, "mrn": str(d[mrn_col]).strip(), "csn": str(d[csn_col]).strip() if csn_col and d.get(csn_col) else None,
                            "dos": str(d[anchor_col]).strip(), "origin": sf.kind})
    return out, anchors


TABLE_COLUMNS: dict[str, list[str]] = {
    "narrative_line": ["cohort", "modality", "kind", "mrn", "csn", "description", "result_time", "section", "line", "text", "accession", "order_id"],
    "report_by_accession": ["cohort", "modality", "accession", "csn", "ordering_date", "text", "line", "description"],
    "component_value": ["cohort", "modality", "mrn", "csn", "procedure_name", "result_time", "component", "line", "value"],
    "labevents": ["cohort", "modality", "mrn", "csn", "result_time", "collection_time", "lab_name", "base_name", "value", "units", "ref_low", "ref_high"],
    "emar": ["cohort", "modality", "mrn", "csn", "date_administered", "medication", "dose", "route"],
    "prescriptions": ["cohort", "modality", "mrn", "csn", "medication", "dose", "ordering_date", "status"],
    "diagnoses": ["cohort", "modality", "mrn", "code", "title", "first_noted", "status"],
    "admissions": ["cohort", "modality", "mrn", "csn", "date_of_service", "admit_time", "los_days", "death_date", "visit_type"],
    "patients": ["cohort", "modality", "mrn", "age", "gender", "race", "ethnicity", "status", "death_date", "bmi"],
    "transfers": ["cohort", "modality", "mrn", "csn", "department", "location", "in_time", "out_time", "icu_flag"],
    "procedures": ["cohort", "modality", "mrn", "csn", "proc_time", "name", "code", "status", "kind"],
    "exam_index": ["cohort", "modality", "mrn", "dos", "accession", "proc_code", "proc_name"],
    "followup": ["cohort", "modality", "mrn", "effective"],
    "weights": ["cohort", "modality", "mrn", "recorded", "kg"],
    "index_candidates": ["cohort", "modality", "mrn", "csn", "dos", "origin"],
}


def parts_dir(enclave: Path = ENCLAVE) -> Path:
    return enclave / "parts"
