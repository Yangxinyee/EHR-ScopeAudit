"""Rule-routed evidence plans (DEV-34, experiments/SCOPEBENCH_PLAN_PROTOCOL.md §3): the no-LLM reference point.

A deterministic parser reads exactly what the model reads — the request text (and, on Tier-B, the structured criteria
the prompt shows) — and the patient's chart, and writes an evidence plan. Executing that plan and submitting its
answer is "rule-routed skill-only". Everything here is built from the dev splits: the Tier-B phrasings come from the
family builders, the Tier-C router's exemplars are the `valid` questions with their templates.

Tier-B requests are generated from fixed phrasings, so the parse is exact by design; Tier-C requests are paraphrased
(FHIR-AgentBench), so the router is a nearest-exemplar classifier plus chart-vocabulary matching, and its errors
measure what a model has to add.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .plans import Plan, PlanError
from .tierc.plans import load_catalog, normalize

# ------------------------------------------------------------------------------------------- Tier-B
_LAB = re.compile(r"\bReturn the three .+ results whose results became available nearest to", re.IGNORECASE)
_CARE = re.compile(r"\bfirst care unit\b", re.IGNORECASE)
_DISP = re.compile(r"\bdischarge disposition\b", re.IGNORECASE)
_COHORT = re.compile(r"\bmeets ALL of the following criteria\b", re.IGNORECASE)


def tier_b_plan(user_payload: str) -> Plan:
    """The plan for a Tier-B request, from the prompt's user payload (JSON: question, contract, criteria, ...)."""
    payload = json.loads(user_payload)
    q = str(payload.get("question") or "")
    if _LAB.search(q):
        return Plan("lab_trend_k3", {})
    if _CARE.search(q):
        return Plan("first_careunit", {})
    if _DISP.search(q):
        return Plan("discharge_disposition", {})
    if _COHORT.search(q):
        return Plan("cohort_criteria", {})
    return Plan(None)


# ------------------------------------------------------------------------------------------- Tier-C
_PID = re.compile(r"\b(\d{8})\b")
_NUMBER_AFTER = re.compile(r"(?:than|below|above|under|over|exceed(?:s|ed)?|of at least|of at most)\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)
OPERATOR_CUES: dict[str, list[tuple[str, str]]] = {
    "comparison": [(r"\b(less|lower|below|under|smaller)\b", "<"), (r"\b(greater|higher|above|over|more|larger|exceed\w*)\b", ">")],
    "aggregate": [(r"\b(average|mean|avg)\b", "avg"), (r"\b(maximum|max|highest|peak|largest)\b", "max"), (r"\b(minimum|min|lowest|smallest)\b", "min")],
    "extreme": [(r"\b(maximum|max|highest|peak|largest)\b", "max"), (r"\b(minimum|min|lowest|smallest)\b", "min")],
    "unit": [(r"\bdays?\b", "days"), (r"\bhours?\b", "hours")],
    "order": [(r"\b(first|earliest|initial)\b", "first"), (r"\b(last|latest|most recent|final|recent)\b", "last")],
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z_]+", text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def masked(question: str, vocabulary: dict[str, list[str]]) -> str:
    """The request's wording with its particulars replaced by their kind: chart names by the placeholder they fill
    (`xlab_name`, `xdrug_name`, ...), numbers by `xnum`. Two paraphrases of one template then differ in wording only."""
    text = question.lower()
    spans = []
    for kind, names in vocabulary.items():
        for name in _mentions(text, names):
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text):
                spans.append((m.start(), m.end(), "x" + kind))
    keep, last = [], -1
    for s0, e0, tag in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if s0 >= last:
            keep.append((s0, e0, tag)); last = e0
    for s0, e0, tag in reversed(keep):
        text = text[:s0] + f" {tag} " + text[e0:]
    return re.sub(r"\d+(?:[./:-]\d+)*", " xnum ", text)


class TierCRouter:
    """Nearest-exemplar template choice (cosine over word uni+bigram counts of the masked request, see `masked`) on
    the dev questions; values from the request and the patient's chart vocabulary; operators from cue words.
    Exemplars are (masked question, template id)."""

    def __init__(self, exemplars: Iterable[tuple[str, str]], catalog: dict[str, dict[str, Any]] | None = None):
        import math
        from collections import Counter

        self.catalog = catalog or load_catalog()
        self._vecs: list[tuple[dict[str, float], str]] = []
        rows = [(Counter(_tokens(q)), t) for q, t in exemplars]
        df = Counter(tok for c, _ in rows for tok in c)
        n = max(1, len(rows))
        self._idf = {tok: math.log((1 + n) / (1 + d)) + 1.0 for tok, d in df.items()}
        for c, t in rows:
            self._vecs.append((self._weigh(c), t))

    def _weigh(self, counts) -> dict[str, float]:
        import math

        v = {tok: k * self._idf.get(tok, 0.0) for tok, k in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {tok: x / norm for tok, x in v.items()}

    def _kinds(self, tid: str) -> frozenset[str]:
        return frozenset(re.sub(r"\d$", "", v) for v in self.catalog[tid]["values"] if v != "patient_id" and not v.endswith("_value"))

    def template(self, masked_question: str) -> str | None:
        """The template of the most similar exemplar among those whose entity kinds match the kinds the request names
        (all exemplars when none does): a template whose placeholders the request does not fill cannot be its template."""
        from collections import Counter

        q = self._weigh(Counter(_tokens(masked_question)))
        named = frozenset(k[1:] for k in re.findall(r"\bx[a-z]+_name\b", masked_question))
        pool = [vt for vt in self._vecs if self._kinds(vt[1]) == named] or self._vecs
        best = max(pool, key=lambda vt: sum(q.get(tok, 0.0) * w for tok, w in vt[0].items()), default=None)
        return best[1] if best else None

    def plan(self, question: str, vocabulary: dict[str, list[str]]) -> Plan:
        """`vocabulary`: placeholder -> candidate names from the patient's chart (see `chart_vocabulary`)."""
        tid = self.template(masked(question, vocabulary))
        if tid is None:
            return Plan(None)
        entry = self.catalog[tid]
        low = question.lower()
        values: dict[str, Any] = {}
        for name in entry["values"]:
            if name == "patient_id":
                m = _PID.search(question)
                values[name] = m.group(1) if m else None
            elif name.endswith("_value"):
                m = _NUMBER_AFTER.search(question)
                values[name] = m.group(1) if m else None
            elif name in ("drug_name1", "drug_name2", "drug_name3"):
                found = _mentions(low, vocabulary.get("drug_name", []))
                values[name] = found[int(name[-1]) - 1] if len(found) >= int(name[-1]) else None
            else:
                found = _mentions(low, vocabulary.get(name, []))
                values[name] = found[0] if found else None
        raw: dict[str, Any] = {"template": tid, "values": values}
        for arg in entry["operators"]:
            raw[arg] = next((v for pat, v in OPERATOR_CUES[arg] if re.search(pat, low)), None)
        try:
            return normalize(raw, self.catalog)
        except PlanError:
            return Plan(None)


def _mentions(text: str, names: list[str]) -> list[str]:
    """Names occurring in the text, longest match first at each position, in order of appearance, non-overlapping."""
    hits = []
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text):
            if all(m.end() <= s or m.start() >= e for s, e, _ in hits):
                hits.append((m.start(), m.end(), name))
    return [n for _, _, n in sorted(hits)]


def chart_vocabulary(resources: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Candidate entity names from a patient's chart, per value placeholder (displays as the benchmark spells them)."""
    from .tierc.skills import VITAL_SYNONYMS, Legal, _coding, _display, _system

    legal = Legal(resources)
    obs = legal.by_type.get("Observation", [])
    out: dict[str, set[str]] = {k: set() for k in ("lab_name", "vital_name", "output_name", "spec_name", "procedure_name", "diagnosis_name", "drug_name")}
    for r in obs:
        sysname, disp = _system(r), _display(r)
        if sysname == "mimic-d-labitems":
            out["lab_name"].add(disp)
        elif sysname == "mimic-chartevents-d-items":
            out["vital_name"].add(disp)
        elif sysname == "mimic-d-items":
            out["output_name"].add(disp)
    out["vital_name"] |= set(VITAL_SYNONYMS)
    for s in legal.by_type.get("Specimen", []):
        out["spec_name"].add(str(_coding(s, "type").get("display") or "").strip().lower())
    for p in legal.by_type.get("Procedure", []):
        out["procedure_name"].add(_display(p))
    for c in legal.by_type.get("Condition", []):
        out["diagnosis_name"].add(_display(c))
    for req in legal.by_type.get("MedicationRequest", []):
        med = legal.ref(req.get("medicationReference"))
        for source in (med, req):
            if not source:
                continue
            code = source.get("code") or source.get("medicationCodeableConcept") or {}
            for cd in (code.get("coding") or []) if isinstance(code, dict) else []:
                if cd.get("display"):
                    out["drug_name"].add(str(cd["display"]).strip().lower())
            if isinstance(code, dict) and code.get("text"):
                out["drug_name"].add(str(code["text"]).strip().lower())
    return {k: sorted(v - {""}) for k, v in out.items()}
