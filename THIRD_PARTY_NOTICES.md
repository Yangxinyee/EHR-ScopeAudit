# Third-party attribution and license scope

The root MIT license covers original software and documentation. The
following third-party rights and notices are preserved separately.

## FHIR-AgentBench — CC BY 4.0

Source: https://github.com/glee4810/FHIR-AgentBench

Inspected revision: `bbb42909a5a7eb907d1cd91f72a560729e7037ea`.

Copyright and attribution: the FHIR-AgentBench authors and contributors.
Upstream license: `licenses/FHIR-AgentBench-CC-BY-4.0.txt`.

FHIR-AgentBench-related adaptations in the following files are distributed
under **CC BY 4.0**:

- `scripts/fhir_benchmark_core.py`
- `scripts/run_fhir_agentbench_corrected.py`
- `scripts/scopebench/build_tierc_fhiragentbench.py`
- `scripts/scopebench/validate_tierc_skills.py`
- `scopebench/tasktypes_tierc.py`
- `scopebench/tierc/agent.py`, `contracts.py`, `store.py`, `skills.py`, `plans.py`

Changes include local evaluation adapters, task contracts, logging around
retrieval and tool calls, scope filtering, evidence auditing, typed skill
verification, batching, and model message-format adapters. Upstream benchmarks
and their datasets are obtained separately; they are not vendored in this
release. This attribution does not imply upstream endorsement.

## MedAgentBench — MIT

Source: https://github.com/stanfordmlgroup/MedAgentBench

Inspected revision: `99260117137b09f04837a8c18d18a1107efa55ae`.

Copyright (c) 2025 Stanford Machine Learning Group.
The full upstream notice is retained in `licenses/MedAgentBench-MIT.txt`.

MedAgentBench-related adaptations appear in
`scripts/run_medagentbench_corrected.py`,
`scripts/scopebench/build_tierc_medagentbench.py`,
`scripts/scopebench/run_tierc_medagent.py`, `scripts/scopebench/rescore_medagent_dev37.py`,
`scopebench/tierc/medagent.py`, and `scopebench/tierc/medagent_plans.py`.
Changes include local-model routing, logged FHIR access, evidence contracts,
scope enforcement, repair and verification, message-template compatibility,
and resumable execution. The original task server and task data are obtained
from upstream rather than bundled here.

## MIMIC-IV Clinical Database Demo on FHIR — ODbL 1.0

Data attribution, derivative database terms, and exclusions are documented in
`DATA_LICENSE_NOTICE.md` and the two task data cards. The full ODbL text is in
`licenses/ODbL-1.0.txt`.

## Dependencies

Dependencies are installed separately and retain their upstream licenses.
No model weights, third-party virtual environments, serving binaries, or
third-party benchmark clones are included.
