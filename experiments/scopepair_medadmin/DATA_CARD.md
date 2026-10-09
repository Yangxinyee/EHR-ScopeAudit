# MedDose public diagnostic task database

Source: MIMIC-IV Clinical Database Demo on FHIR, version 2.1.0,
https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/

License: **ODbL 1.0**; see `../../DATA_LICENSE_NOTICE.md` and
`../../licenses/ODbL-1.0.txt`. Contains information from that database,
made available under the Open Database License.

## Construction and splits

- 80 patient cases, one selected case per patient within this family.
- Two tasks per case: pre-index and post-index, with six candidate medication administrations.
- 16 development cases / 32 tasks; 64 test cases / 128 tasks.
- Development and test patients are disjoint within the family.
- Temporal selection uses **administration occurrence time**; seed 2027.
- These are deidentified public-demo-derived observations, not invented data.
- Questions, typed gold answers, provenance IDs, and patient-cluster IDs
  support paired evaluation. Raw source FHIR files and model outputs are excluded.

## Integrity

`tasks.jsonl` SHA-256: `764e69058410f1fc8c8fd76fb2ccf49b837eb4ee610aca71de826d1e1bc8c870`.

Task contents are unchanged from the frozen diagnostic manifest. Metadata
retains source hashes and counts; only local source paths are reduced to
basenames, with the public source URL and license fields added. Rebuild
commands are in `../../docs/REPRODUCIBILITY.md`.

## Interpretation

This small diagnostic is separate from the ScopeBench three-tier benchmark.
It supports testing the evidence-selection and structured-computation steps.
It does not establish representative clinical prevalence or clinical utility.
The two families' five-checkpoint comparison is distinct from the six-checkpoint
MIMIC-IV panel. Full-system outputs include deterministic verification/replacement.
