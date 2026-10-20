"""DEV-38: the MedAgentBench gateway keeps the agent's query (repeated parameters such as a date range), lets every arm
follow the server's next-page links, and, under enforcement, applies the contract to the complete result set before
paging (filtering one server page could hide every permitted record behind later pages)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import Contract  # noqa: E402
from scopebench.tierc import medagent  # noqa: E402
from scopebench.tierc.medagent import FHIR_BASE, MedAgentStore, validate_get_url  # noqa: E402

PAGE = 20


def obs(i: int, when: str) -> dict:
    return {"resourceType": "Observation", "id": f"o{i}", "code": {"text": "MG", "coding": [{"code": "MG"}]},
            "effectiveDateTime": when, "issued": when, "valueQuantity": {"value": 1.0 + i / 100}}


# 25 results outside the 24-h window first (server order), then the permitted ones
OLD = [obs(i, f"2023-09-{1 + i % 28:02d}T08:00:00+00:00") for i in range(25)]
NEW = [obs(100 + i, f"2023-11-13T0{i % 9}:00:00+00:00") for i in range(45)]


class Server:
    def __init__(self, rows):
        self.rows, self.urls = rows, []

    def __call__(self, url: str, timeout: int = 120) -> dict:
        self.urls.append(url)
        q = parse_qsl(urlsplit(url).query)
        off = int(dict(q).get("_getpagesoffset", 0))
        page = self.rows[off:off + PAGE]
        links = [{"relation": "self", "url": url}]
        if off + PAGE < len(self.rows):
            links.append({"relation": "next", "url": f"{FHIR_BASE.rstrip('/')}?_getpages=abc&_getpagesoffset={off + PAGE}&_count={PAGE}&_format=json&_bundletype=searchset"})
        return {"status_code": 200, "data": json.dumps({"resourceType": "Bundle", "type": "searchset", "link": links,
                                                        "entry": [{"fullUrl": f"{FHIR_BASE}Observation/{r['id']}", "resource": r} for r in page]})}


CONTRACT = Contract(index_time="2023-11-13T10:15:00Z", scope="pre", basis="occurred", permitted_sources=("Observation",), window_hours=24.0)


def store(enforcement: str) -> MedAgentStore:
    return MedAgentStore(base_url=FHIR_BASE, contract=CONTRACT, enforcement=enforcement, patient_hash="p")


def shown(text: str) -> dict:
    return json.loads(text.split("\n", 1)[1].rsplit(". Please call FINISH", 1)[0])


def test_url_keeps_repeated_parameters_and_accepts_server_page_links():
    u = validate_get_url(f"{FHIR_BASE}Observation?code=MG&date=ge2023-11-12T10:15:00Z&date=lt2023-11-13T10:15:00Z", FHIR_BASE)
    assert [v for k, v in parse_qsl(urlsplit(u).query) if k == "date"] == ["ge2023-11-12T10:15:00Z", "lt2023-11-13T10:15:00Z"]
    assert urlsplit(validate_get_url(f"{FHIR_BASE}Observation?_format=xml&code=MG", FHIR_BASE)).query == "_format=json&code=MG"
    assert urlsplit(validate_get_url(f"{FHIR_BASE}Patient?identifier=S1", FHIR_BASE)).query == "identifier=S1&_format=json"   # unchanged form
    assert "_getpages=abc" in validate_get_url(f"{FHIR_BASE.rstrip('/')}?_getpages=abc&_getpagesoffset=20", FHIR_BASE)
    with pytest.raises(ValueError):
        validate_get_url(f"{FHIR_BASE.rstrip('/')}?code=MG", FHIR_BASE)                  # the bare base is only for page links


def test_enforcement_filters_the_complete_result_set_then_pages(monkeypatch):
    srv = Server(OLD + NEW); monkeypatch.setattr(medagent, "get_json_text", srv)
    st = store("contract")
    b1 = shown(st.get(1, f"{FHIR_BASE}Observation?code=MG&patient=S1"))
    assert [e["resource"]["id"] for e in b1["entry"]] == [r["id"] for r in NEW[:PAGE]]   # the first server page held none of them
    nxt = next(l["url"] for l in b1["link"] if l["relation"] == "next")
    assert "_scopepage=2" in nxt and "_getpages" not in nxt
    b2 = shown(st.get(2, nxt)); b3 = shown(st.get(3, next(l["url"] for l in b2["link"] if l["relation"] == "next")))
    assert [e["resource"]["id"] for e in b2["entry"] + b3["entry"]] == [r["id"] for r in NEW[PAGE:]]
    assert not any(l["relation"] == "next" for l in b3["link"])
    assert not st.illegal_access_ids() and st.log[0].returned_ids == [r["id"] for r in NEW[:PAGE]] and len(st.log[0].filtered_out_ids) == len(OLD)
    assert not any("_scopepage" in u for u in srv.urls)                                   # the gateway parameter never reaches the server


def test_unpaged_enforced_results_are_unchanged(monkeypatch):
    srv = Server(OLD[:5] + NEW[:3]); monkeypatch.setattr(medagent, "get_json_text", srv)
    b = shown(store("contract").get(1, f"{FHIR_BASE}Observation?code=MG&patient=S1"))
    assert [e["resource"]["id"] for e in b["entry"]] == [r["id"] for r in NEW[:3]] and len(srv.urls) == 1


def test_unenforced_arm_follows_server_page_links(monkeypatch):
    monkeypatch.setattr(medagent, "get_json_text", Server(OLD + NEW))
    st = store("none")
    b1 = shown(st.get(1, f"{FHIR_BASE}Observation?code=MG&patient=S1"))
    assert len(b1["entry"]) == PAGE and [e["resource"]["id"] for e in b1["entry"]] == [r["id"] for r in OLD[:PAGE]]
    b2 = shown(st.get(2, next(l["url"] for l in b1["link"] if l["relation"] == "next")))
    assert [e["resource"]["id"] for e in b2["entry"]] == [r["id"] for r in (OLD + NEW)[PAGE:2 * PAGE]]
    assert set(st.illegal_access_ids()) == {r["id"] for r in OLD}                          # unenforced: out-of-scope reads are logged
