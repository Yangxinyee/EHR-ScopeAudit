"""Generic policy engines (paper Supp. S4 baselines): two INDEPENDENT executions of the same contract,
used only to show that filter execution itself is established machinery.

  temporal_sql_view : the contract compiled into a DuckDB SQL view over the evidence table
  abac_pdp          : an attribute-based policy decision point (subject/object/action/environment
                      attributes evaluated against rules derived from the contract)
Both return the legal id set; tests/scopebench/test_policy_engines.py checks them against
scopebench.predicate.legal on random pools. Neither module imports predicate.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from .schema import Contract, EvidenceObject, parse_time


# ------------------------------------------------------------------ temporal SQL view (DuckDB)
def sql_view_legal_ids(pool: Iterable[EvidenceObject], contract: Contract) -> set[str]:
    import duckdb

    rows = [(e.id, e.source, e.occurred, e.available, e.documented) for e in pool]
    con = duckdb.connect()
    con.execute("CREATE TABLE evidence (id VARCHAR, source VARCHAR, occurred VARCHAR, available VARCHAR, documented VARCHAR)")
    if rows:
        con.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?, ?)", rows)
    basis = contract.basis
    index = contract.index.strftime("%Y-%m-%d %H:%M:%S")
    def lit(v: str) -> str:
        return "'" + str(v).replace("'", "''") + "'"

    src_clause = ""
    if contract.permitted_sources is not None:
        src_clause = " AND source IN (" + ",".join(lit(x) for x in contract.permitted_sources) + ")"
    t = f"try_cast(replace(replace({basis}, 'T', ' '), 'Z', '') AS TIMESTAMP)"
    if contract.scope == "full":
        where = "TRUE"
    elif contract.scope == "pre":
        where = f"{t} IS NOT NULL AND {t} < TIMESTAMP '{index}'"
        if contract.window_hours is not None:
            where += f" AND {t} >= TIMESTAMP '{index}' - INTERVAL {int(contract.window_hours * 3600)} SECOND"
    elif contract.scope == "post":
        where = f"{t} IS NOT NULL AND {t} > TIMESTAMP '{index}'"
        if contract.window_hours is not None:
            where += f" AND {t} <= TIMESTAMP '{index}' + INTERVAL {int(contract.window_hours * 3600)} SECOND"
    else:  # at
        where = f"{t} IS NOT NULL AND {t} <= TIMESTAMP '{index}' + INTERVAL {int(contract.at_tolerance_hours * 3600)} SECOND"
        if contract.window_hours is not None:
            where += f" AND {t} >= TIMESTAMP '{index}' - INTERVAL {int(contract.window_hours * 3600)} SECOND"
        if contract.index_evidence_ids:
            where = f"(({where}) OR id IN (" + ",".join(lit(x) for x in contract.index_evidence_ids) + "))"
    con.execute(f"CREATE VIEW legal AS SELECT id FROM evidence WHERE ({where}){src_clause}")
    ids = {r[0] for r in con.execute("SELECT id FROM legal").fetchall()}
    con.close()
    return ids


# ------------------------------------------------------------------ ABAC policy decision point
class AbacPolicy:
    """Rules over attributes: subject = {task scope}, object = {source, time on basis}, action = read,
    environment = {index time, window}. Deny by default; permit when a rule matches."""

    def __init__(self, contract: Contract):
        self.c = contract

    def _attributes(self, e: EvidenceObject) -> dict:
        t = parse_time(getattr(e, self.c.basis))
        return {"id": e.id, "source": e.source, "time": t}

    def decide(self, e: EvidenceObject) -> bool:
        a = self._attributes(e)
        c = self.c
        rules = []
        if c.permitted_sources is not None:
            rules.append(("source-allowlist", a["source"] in set(c.permitted_sources)))
        else:
            rules.append(("source-any", True))
        if c.scope == "full":
            rules.append(("scope-full", True))
        elif c.scope == "at" and a["id"] in set(c.index_evidence_ids):
            rules.append(("index-document", True))
        else:
            if a["time"] is None:
                rules.append(("time-known", False))
            else:
                i, w = c.index, (timedelta(hours=c.window_hours) if c.window_hours is not None else None)
                if c.scope == "pre":
                    rules.append(("before-index", a["time"] < i and (w is None or a["time"] >= i - w)))
                elif c.scope == "post":
                    rules.append(("after-index", a["time"] > i and (w is None or a["time"] <= i + w)))
                else:
                    rules.append(("up-to-index", a["time"] <= i + timedelta(hours=c.at_tolerance_hours) and (w is None or a["time"] >= i - w)))
        return all(ok for _, ok in rules)


def abac_legal_ids(pool: Iterable[EvidenceObject], contract: Contract) -> set[str]:
    pdp = AbacPolicy(contract)
    return {e.id for e in pool if pdp.decide(e)}


ENGINES = {"temporal_sql_view": sql_view_legal_ids, "abac_pdp": abac_legal_ids}
