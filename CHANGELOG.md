# Changelog

## 0.2.0 (2026-10-20, prepared; not yet published)

Code
- Evidence plans: a model declares the operation and evidence it needs; the auditor executes the plan on the
  permitted records without any reference input (`scopebench/plans.py`, `plan_rules.py`, `tierc/plans.py`,
  `tierc/medagent_plans.py`).
- Tier-C runtime store: resources without their own time take the builder's time construction, matching the task
  builder and the typed skills.
- MedAgentBench gateway: under enforcement the contract is applied to the complete result set before paging (the
  gateway pages over the permitted resources); repeated query parameters are kept (`date=ge…&date=lt…` is a range);
  the server's own page links are accepted.
- Analysis: hidden violations of agents that issue their own requests, the MedAgentBench C-L reading for tasks with
  no required evidence, and the panel divergence for rows without an `n_illegal_access` metric.
- Analysis scripts for the evidence-plan, exposure-injection, compute-control, checkpoint-uncertainty and re-run
  analyses; the oracle-gap figure generator.

Results
- Aggregate tables and macros of the current manuscript (four new tables: deployable auditor, compute controls,
  exposure injection, checkpoint uncertainty) and the oracle-gap figure.

Tests
- Tests for evidence plans, the gateway's paging and parameters, the Tier-C store's time construction, backend
  seeds, exposure controls and free-text scoring; tests that need the upstream MedAgentBench source or the Tier-C
  plan catalog skip when those are absent.

## 0.1.1 (2026-10-10)

Renamed to EHR-ScopeAudit; benchmark SPOILER; new paper title.

## 0.1.0 (2026-10-09)

Initial public release.
