# Reproducing EHR-ScopeAudit

## What works from a fresh clone

The synthetic demo, CPU unit tests, independent skill/oracle differential
tests, public ScopePair task invariants, and all three measured figure
generators run without credentialed clinical data or model weights.

```bash
python -m pip install -e ".[test,analysis]"
python examples/scope_demo.py
python -m pytest -q
python scripts/audit_release.py --verify
```

Tests that require institutional data, locally reconstructed SPOILER
manifests, complete model trajectories, or upstream FHIR servers are skipped
when those resources are absent. A skipped integration check is not evidence
that a model inference experiment has been rerun.

## Public diagnostic task reconstruction

Obtain the public FHIR demo directly from
https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/ and unpack the `fhir/`
directory. Keep raw FHIR data outside commits. Rebuild into a new directory:

```bash
python scripts/build_scopepair_fhir.py \
  --fhir-dir /path/to/demo/fhir --output-dir local_data/rebuilt-labtrend \
  --n-cases 80 --dev-cases 16 --seed 2027
python scripts/build_scopepair_medadmin.py \
  --fhir-dir /path/to/demo/fhir --output-dir local_data/rebuilt-meddose \
  --n-cases 80 --dev-cases 16 --seed 2027
```

Compare the SHA-256 of the rebuilt `tasks.jsonl` with the released data card
and metadata. LabTrend selects observations using result-availability time;
MedDose selects administration events using occurrence time. Each database
contains 160 tasks: 32 dev and 128 test, with patient-disjoint dev/test cases.
The two families are separate datasets; patient-disjointness is within each
family, not asserted between the families.

## Local model execution

Obtain model weights at the revisions in `configs/model_panel.json`, under
their own provider terms. Install a CUDA-compatible PyTorch build and the
optional local-model dependencies. Local ScopePair runners require CUDA and
reject CPU/disk offload. CPU execution is for tests and analysis.

Example using the released public LabTrend manifest:

```bash
python -m pip install -e ".[local-model]"
python scripts/run_scopepair_fhir.py \
  --model /path/to/model --output-dir runs/labtrend --split test
```

Use `--help` for manifest and decoding options. For the SPOILER harness,
local SGLang HTTP and local Hugging Face backends are provided. The optional
Claude file-queue adapter requires a separately operated responder that writes
completion files. Its input-token counts are character-based estimates. No
remote API credentials or model-serving infrastructure are included.

## Credentialed MIMIC reconstruction

Obtain access to MIMIC-IV 3.1, MIMIC-IV-Note 2.2, and MIMIC-IV-ED 2.2 through
PhysioNet. Set `SCOPEBENCH_PUBLIC_DATA` to a local directory containing their
versioned downloads. This variable's historical name does not mean that all
of its contents are openly redistributable.

```bash
python -m pip install -e ".[ingest,analysis]"
export SCOPEBENCH_PUBLIC_DATA=/path/to/authorized-data
python scripts/scopebench/ingest_mimic_duckdb.py
python scripts/scopebench/build_tierb.py \
  --split dev --per-family 40 --out local_data/tierB-dev
python scripts/scopebench/build_tierb.py \
  --split test --per-family 800 --out local_data/tierB-test \
  --exclude-patients local_data/tierB-dev/patients.json
```

The published Tier-B has 9,600 tasks over six families, two scopes per case.
Reconstruction requires the same source versions and settings. Scoring and
analysis scripts read locally generated predictions, access logs, and stages;
those patient-level files stay out of public commits. The public aggregate
tables alone are insufficient to recompute patient-cluster confidence
intervals or task-level hypothesis tests independently.

## Institutional CCDA

Generic institutional loader and builder code is provided in
`scopebench/tiera/` and the matching scripts. Users supply their own authorized
exports, schema-compatible inputs, and enclave settings through
`SCOPEBENCH_CCDA_ROOT` and `SCOPEBENCH_TIERA_ENCLAVE`. The release defaults put
all CCDA-derived manifests, including stripped copies, inside that enclave.
Original source records, institutional export discovery reports, salts,
patient hashes, and clinician packets are excluded.

The published private-cohort results are available as aggregates. Access to
the original institutional cohort is not conferred by releasing its code.

## Interactive FHIR benchmarks

Fetch upstream code separately at the inspected revisions:

```bash
git clone https://github.com/glee4810/FHIR-AgentBench.git external_benchmarks/FHIR-AgentBench
git -C external_benchmarks/FHIR-AgentBench checkout bbb42909a5a7eb907d1cd91f72a560729e7037ea
git clone https://github.com/stanfordmlgroup/MedAgentBench.git external_benchmarks/MedAgentBench
git -C external_benchmarks/MedAgentBench checkout 99260117137b09f04837a8c18d18a1107efa55ae
```

Follow upstream instructions for their tasks, Docker/HAPI-FHIR services, and
data. The local adapters and SPOILER Tier-C builders are provided here.
Patient maps, reconstructed manifests, native FHIR server records, and raw
task trajectories are generated locally rather than bundled in this release.

## Statistics, operation semantics, and interpretation

`scopebench/SKILL_SPEC.md` specifies the covered operations. Independent
`oracle` and `skills` implementations are checked by differential tests.
Analysis code provides patient-cluster bootstrap intervals, Holm adjustment,
checkpoint aggregation, and the paper's other statistical summaries.

The full system's verified output can include deterministic skill replacement.
It should not be described as unaided model accuracy. The LabTrend all-six
comparison is post-hoc, as in the manuscript. Clinician evaluation remains
future work; its analysis and packet-construction scripts are protocol tooling.
