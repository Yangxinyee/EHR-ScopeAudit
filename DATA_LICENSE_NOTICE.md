# Data sources and release rights

## Public ScopePair task databases: ODbL 1.0

`experiments/scopepair_fhir/tasks.jsonl` and
`experiments/scopepair_medadmin/tasks.jsonl` are derived exclusively from the
**MIMIC-IV Clinical Database Demo on FHIR, version 2.1.0**, by Alex Bennett,
Hannes Ulrich, Joshua Wiedekopf, Piotr Szul, John Grimes, and Alistair Johnson
(PhysioNet, 2025).

Source: https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/

Source license: https://physionet.org/content/mimic-iv-fhir-demo/view-license/2.1.0/

Contains information from the MIMIC-IV Clinical Database Demo on FHIR,
which is made available under the Open Database License (ODbL).

These derivative task databases are distributed under **ODbL 1.0**, including
its attribution and share-alike conditions. A full copy is provided in
`licenses/ODbL-1.0.txt`. Construction scripts provide the transformations;
sanitized metadata retains source-file hashes and exact task-manifest hashes.
Source filenames are relative basenames rather than author workstation paths.

The public demo contains deidentified clinical records. These tasks are
derived from that demo, not synthetic patient observations. Hashed patient
cluster identifiers and FHIR resource identifiers support task construction
and paired evaluation; they do not convert restricted data into open data.
The source FHIR files themselves are obtained from PhysioNet and are not
bundled here.

## Aggregate results

`results/paper/` contains aggregate estimates already represented in the
manuscript's tables and prose. `results/scopebench/tierB_panel.json` retains
checkpoint-level estimates required for the figures. `results/scopepair/`
contains only the overall summary rows for the public diagnostic experiments.
No individual task or patient results from the restricted tiers are included.
ODbL source attribution also applies to produced figures and summaries that
use the public ScopePair database, as stated in the README and data cards.

## Restricted resources: local reconstruction only

Full MIMIC-IV, MIMIC-IV-Note, and MIMIC-IV-ED require the applicable PhysioNet
credentialing and data-use agreements. Their records, derived task databases,
gold evidence sets, and patient-level inference traces are not redistributed
on GitHub. Official guidance on MIMIC-derived resources is available at:

https://www.physionet.org/content/mimiciv/3.1/

Tier-A CCDA records, salts, patient hashes, institutional manifests, and raw
or stripped patient-level outputs are excluded. Generic loader and builder
code is provided for users with their own institutional permissions; including
that code grants no access rights to institutional data.

FHIR-AgentBench and MedAgentBench task datasets and model weights are also
excluded. Obtain them from their upstream providers under their own terms.

## Synthetic example

`examples/scope_demo.py` constructs six invented numerical records in memory.
They have no source patient and are covered by the original software license.
Its oracle-generated outputs are pipeline checks, not measured model results.
