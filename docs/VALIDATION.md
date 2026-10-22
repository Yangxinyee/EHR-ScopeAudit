# Release validation

Release 0.2.0 (2026-10-21) was checked on Linux with Python 3.13.12 in a
new virtual environment, using only the declared `test` and `analysis`
dependencies. The recorded package versions are in
[`requirements/validated-cpu-py313.txt`](../requirements/validated-cpu-py313.txt).
The CI workflow also runs the CPU checks with Python 3.11.

- CPU test suite: **149 passed, 47 skipped**. The skips require external
  clinical data, reconstructed experiment outputs, or benchmark services.
- Synthetic pipeline demo: prompt-only accesses all six invented objects;
  scope-aware and full configurations access the three admissible objects.
  All configurations return the correct answer and admissible citations.
- Frozen artifacts: the two public task manifests, 23 aggregate LaTeX
  tables, and numeric macro values match the manuscript's source artifacts.
- Figure reproduction: the teaser, access/citation gap, operation
  comparison and oracle-gap PDFs regenerated from the released aggregates
  match the manuscript PDFs byte for byte.
- Package build: the wheel includes the operation specification and all
  applicable license notices, and excludes the task databases.
- Publication review: the explicit allowlist, source licenses, dataset
  provenance, credential/path scan, and per-file hash manifest were checked.

These checks validate the released software and saved artifacts. Model
inference experiments and the planned clinician study were not rerun as
part of publication preparation. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for their additional resources and access requirements.
