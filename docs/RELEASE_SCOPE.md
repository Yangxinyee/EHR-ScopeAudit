# Public release scope

This repository was prepared from an explicit file allowlist and a fresh,
empty Git repository. The development repository's Git history is not part
of the release.

## Included

- Original scope-contract, retrieval, logging, auditing, scoring, and typed
  operation code; backend interfaces and local execution adapters.
- Three-tier construction and analysis code, with portable defaults and
  institutional outputs directed to the caller's enclave.
- Unit and differential tests, plus integration checks that run only when
  their required local data or services are present.
- Two public FHIR-demo-derived task databases under ODbL 1.0: LabTrend and
  MedDose, 160 tasks each; sanitized provenance metadata and data cards.
- Aggregate paper tables and numeric macros, reduced checkpoint summaries,
  and overall operation summaries. Individual clinical outcomes and model
  trajectories are absent.
- Original figure drawing/generation code, rendered figures, checkpoint
  revision identifiers, documentation, software/data licenses, and citations.

## Excluded

- Full MIMIC-IV/Note/ED records and their derived task/evidence databases.
- Institutional CCDA exports, discovery reports, salts, identifiers, hashed
  patient manifests, and raw or stripped patient-level data.
- Model responses, prompts containing patient records, access logs, stages,
  adjudication packets, ratings, and private/frontier batch completions.
- Third-party benchmark datasets and clones, model weights, environments,
  caches, private service/cluster settings, cloud configuration, and secrets.
- Anonymous conference manuscripts, manuscript revision history, projected
  target specifications, and unpublished reviewer/working notes.

## Release-only adaptations

The frozen task databases and aggregate table values are preserved. The
release replaces workstation paths with local configuration or example
paths, redirects CCDA release copies into the enclave, and makes one
upstream-dependent prompt integration test conditional on the upstream file.
Figure generators read the published aggregate files. Missing Git metadata
in a source-only installation is handled quietly.

The original framework/configuration identities remain stable. The
sanitized ScopePair panel freeze file records historical study provenance;
its old code hashes are not an integrity claim about the ported public code.
`release_manifest.json` provides the public release's file hashes instead.

## Verification

`python scripts/audit_release.py --verify` checks the actual Git publication
payload, approved dataset hashes and attribution, excluded file types,
symlinks, common credential patterns, private workstation paths, and the
release manifest. PDF figure text is inspected through `pdftotext`.
`poppler-utils` is required for this audit command on Linux.

The pattern scan is a supplementary check. Eligibility was established by
source and license review and by excluding entire classes of restricted
material. Future additions require the same source/data-rights review;
regenerating a hash manifest alone does not establish release rights.

The synthetic example demonstrates correct answers and legal citations
alongside illegal access in prompt-only, followed by legal access under
scope enforcement. Model inference, patient-level statistical reruns, and
the future clinician study are not claimed as part of this release test.
