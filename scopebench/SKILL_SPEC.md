# ScopeBench skill / oracle specification

Version: 1.2 (2026-09-27). Governs two independent implementations
(plan §1.8, §4.3): `scopebench/oracle/` (gold generation, subagent A) and
`scopebench/skills/` (inference-time ChartSkills, subagent B). Neither
implementer sees the other's code. Divergence found by
`tests/scopebench/test_differential.py` is resolved by amending THIS file
(with a version bump and a changelog entry), then both implementations are
re-run. Agreement is a reported artifact.

## 0. Common definitions

- Evidence objects are `scopebench.schema.EvidenceObject` (fields: `id`,
  `source`, `patient`, `occurred`, `available`, `documented`, `occurred_end`,
  `row_ref`, `fields: dict`). Times are ISO-8601 strings; parse with
  `scopebench.schema.parse_time` (naive times are UTC). A missing (`None` /
  `""`) or unparseable timestamp counts as **no time** on that basis.
- A contract is `scopebench.schema.Contract`; legality is ONLY
  `scopebench.predicate.legal(evidence, contract)`. Implementations never
  re-implement the predicate. Inputs arrive already legal for the contract;
  an implementation MAY re-apply `legal` on entry (idempotent).
- Operations are pure functions with no I/O and no randomness.
- Evidence ids are unique within a pool (a manifest invariant enforced by
  `tests/scopebench`); behaviour on duplicate ids is undefined and the
  differential generator never produces them.
- **Total order T** on a basis: key `(0, time, id)` for timed objects and
  `(1, "", id)` for untimed objects, ascending — untimed objects sort after
  every timed object, ties by the string `id` ascending. "Chronological"
  means T ascending on the stated basis. **Min-type picks** (nearest ranking
  order, `first_after`) take the FIRST element of T among candidates; **max-
  type picks** (`last_before`, "largest occurred") take the LAST element of
  T among *timed* candidates — so equal times resolve to the largest id.
- **Nearest** ranking key is `(|t − index| in seconds, id)` ascending over
  timed objects only.
- **Untimed objects** (no time on the operating basis): never selected by
  nearest / boundary / `first_careunit` operations; included by
  `window_count`, `administration_count` and `cohort_criteria`; the
  disposition lookups prefer timed objects and fall back to the smallest id
  among untimed ones only when no timed object of that source exists.
- **Numeric** means `float(fields[key])` succeeds; a missing key or `None` is
  non-numeric; booleans and the strings `"nan"`/`"inf"`/`" 3 "` are numeric.
  Numeric operations skip non-numeric objects.
- Numeric tolerance for equality of two computed numbers: `abs_tol=1e-7,
  rel_tol=1e-7`.
- `params["basis"]`, when the operation takes a basis, defaults to
  `contract.basis` when absent or `None`. §1 and §2 have fixed bases and
  ignore `params["basis"]`.
- `params["which"]` (§4) and `params["agg"]` (§5), when present, must equal
  the registry key's direction/aggregate; otherwise raise `ValueError`.
  Unknown `target`, `kind`, `op` or `agg` values raise `ValueError`.
- Return shape: every operation returns `"status": "ok" | "insufficient"`,
  `"provenance": [...]` (the evidence ids actually used, in the order used),
  plus its operation-specific keys. On `insufficient`, every
  operation-specific key is present with value `None` (scalars, and the §4
  `fields` dict) or `[]` (lists), EXCEPT the label keys `agg` (§5) and
  `target` (§6), which keep their values, and EXCEPT §1, whose insufficient
  shape is exactly the reference's `{"status": "insufficient",
  "provenance": []}`.
- Time strings in outputs are copied verbatim from the object (not
  re-serialized).
- Public entry points: `oracle.OPERATIONS[name](evidence, contract, params)
  -> dict` and `skills.OPERATIONS[name](evidence, contract, params) -> dict`,
  identical signatures; both packages expose `SPEC_VERSION = "1.2"`.

## 1. `lab_trend_k3` — ScopePair-FHIR LabTrend (frozen semantics)

params: `{"k": 3}` (default 3; `k < 1` raises `ValueError`). Basis is fixed to
`available` for both distance and order. Evidence: laboratory observations
of one concept, already legal, with `fields.value`, `fields.unit`.
1. Drop objects whose `value` is non-numeric or whose `available` time is
   missing (no count is reported: the frozen shape has no slot).
2. Select the `k` nearest on `available` (ties by id); fewer than `k` →
   `insufficient`.
3. Order the selected chronologically on `available`.
4. `delta = float(value(last)) − float(value(first))`; `direction =
   "stable"` if `isclose(delta, 0, abs_tol=1e-12)`, else `"increasing"` if
   `delta > 0` else `"decreasing"`.
Return: `{"status","ordered_result_ids":[ids],"delta_last_minus_first":delta,
"direction":..., "provenance":[{"citation":id,"value":fields.value (verbatim),
"unit":fields.unit (verbatim),"issued_time":available}]}`. Must equal
`scopebench.tasktypes.REFERENCE_SKILLS["lab_trend_k3"]` on ScopePair data
(the frozen 2026-07 implementation is the arbiter for this operation only).

## 2. `med_dose_k3` — ScopePair-MedAdmin MedDose (frozen semantics)

params: `{"k": 3}` (default 3; `k < 1` raises). Basis fixed to `occurred`.
Evidence: administrations of one medication/unit, already legal, with
`fields.dose`, `fields.unit`.
1. Drop objects with non-numeric `dose` or missing `occurred`.
2. Select the `k` nearest on `occurred` (ties by id); fewer → `insufficient`.
3. Order chronologically on `occurred`.
4. `total_dose = sum(float(dose))` in chronological order; `mean_dose =
   total_dose / k`.
Return: `{"status","ordered_result_ids","total_dose","mean_dose",
"provenance":[{"citation","dose" (verbatim),"unit" (verbatim),
"effective_time":occurred}]}`.

## 3. `nearest_k` — generic nearest-k retrieval

params: `{"k": int, "basis": ...}`. `k < 0` raises; `k == 0` is `ok` with
empty lists. Over timed legal objects only: select the `k` nearest (ties by
id); fewer than `k` (with `k ≥ 1`) → `insufficient`; output chronological on
the same basis. Return `{"status","ordered_result_ids","provenance":
[{"citation":id,"value":fields.value (verbatim, may be null),"unit":
fields.unit (verbatim),"time":<basis time string>}]}`.

## 4. `first_after` / `last_before` — boundary lookups

params: `{"basis": ..., "which": ...}`. Over timed legal objects:
`first_after` = min-type pick among objects with time strictly greater than
the index; `last_before` = max-type pick among objects with time strictly
smaller than the index. None → `insufficient`. Return
`{"status","citation":id,"time":<basis time string>,"fields":dict(fields),
"provenance":[id]}` (on insufficient: `citation`/`time`/`fields` are `None`).

## 5. `window_count` / `window_sum` / `window_mean`

params: `{"basis": ..., "agg": ...}`. `count` counts all legal objects
(numeric or not, timed or not) and skips nothing; `sum` / `mean` use numeric
objects only. `mean` of zero numeric objects → `insufficient`; `sum` of zero
numeric objects → `ok` with `0.0`; `count` of zero objects → `ok` with `0`.
Sums run in chronological order (T on the basis) with plain addition.
Return `{"status","agg":<name>,"value":number,"n":<objects used>,
"n_skipped_non_numeric":int,"provenance":[ids used, chronological]}`.

## 6. `outcome_lookup` — typed outcome fields

params: `{"target": ..., "field": <fields key>, "basis": ..., "drug": str|null}`.
- `first_careunit`: `first_after` over legal objects with `source ==
  "transfers"`; `value = fields.get(field)`.
- `discharge_disposition` (source `admissions`) / `ed_disposition` (source
  `edstays`): none → `insufficient`; otherwise the max-type pick on
  `occurred` (untimed fallback per §0); `value = fields.get(field)` (`None`
  is a valid value with status `ok`).
- `administration_count`: number of legal objects with `source == "emar"`
  and `str(fields.medication).strip().lower() == drug.strip().lower()`;
  a `None` drug or `None`/missing medication never matches; zero → `ok`
  with `0`, empty provenance, `citation`/`time` `None`.
`time` is the time string on `params["basis"]` of the chosen object for the
three lookup targets (verbatim; `None` if the object has no time on that
basis) and `None` for `administration_count`.
Return `{"status","target","value","citation":id|null,"time":...|null,
"provenance":[ids]}` (for `administration_count`, provenance = matching ids
chronological on the basis).

## 7. `cohort_criteria` — conjunction of structured criteria

params: `{"criteria": [C1, C2, ...]}`; each `C` has `kind` and, optionally,
`basis` (default `contract.basis`). Per kind, the **domain** is:
- `lab_threshold` `{"concept","op","threshold","basis"}`: legal objects with
  `fields.concept == concept` (exact string match); satisfied if ANY object
  in the domain with a numeric value satisfies `value op threshold`
  (`op` ∈ `>=`, `<=`, `>`, `<`).
- `code_present` `{"prefix"}`: legal objects with `source ==
  "diagnoses_icd"`; objects with a `None`/missing `code` are ignored;
  satisfied if ANY `str(code).upper().startswith(prefix.upper())`.
- `exposure` `{"drug"}`: legal `emar` objects; satisfied if ANY medication
  matches as in §6.
- `acuity` `{"max_acuity"}`: legal `triage` objects; satisfied if ANY
  `int(fields.acuity) <= max_acuity` (objects whose acuity does not parse
  with `int()` are ignored).
`satisfied` is `None` (undecidable) when the domain is EMPTY; when the
domain is non-empty but nothing matches, the criterion is decidable and
unsatisfied. Criterion provenance, chronological (T on the criterion basis):
for a satisfied criterion, all satisfying ids; **for an unsatisfied
criterion, all ids of its domain** (the evidence that was examined and
failed — this is what an `exclude` answer must cite); for an undecidable
criterion, `[]`. Decision: `"include"` if every criterion is satisfied
(empty `criteria` → `include`); `"exclude"` if at least one criterion is
decidable and unsatisfied; otherwise `"uncertain"`. Output criteria echo
the input dict plus `satisfied` and `provenance`. Top-level provenance:
for `include`/`uncertain`, the deduplicated union of criterion provenance
ordered by T on `contract.basis`; for `exclude`, the provenance of the
FIRST (list order) unsatisfied criterion.
Return `{"status":"ok","decision":...,"criteria":[...],"provenance":[...]}`.

## 8. Tier-C operations (appended in M5 before the Tier-C build)

`patient_lookup`, `patient_age`, `most_recent_value`, `window_average_24h`,
`threshold_conditional_order` will be added as §8.1–8.5 with the same
conventions; not part of this version.

## 9. Differential test contract

`tests/scopebench/test_differential.py` generates, per operation, ≥ 5,000
random inputs (random pools of 0–40 objects, random timestamps around a
random index with duplicated timestamps, random non-numeric values, random
missing timestamps, random sources) and asserts
`oracle.OPERATIONS[op](...) == skills.OPERATIONS[op](...)` after
normalizing floats with the tolerance above. Any inequality is a spec
defect or an implementation defect; the test prints the first 5 failing
inputs.

## Changelog
- 1.2 — §7: an unsatisfied criterion's provenance is its whole domain (the
  examined-and-failed evidence), so `exclude` decisions are citable; no other
  change.
- 1.1.1 — §0: ids are unique per pool (the last 42 differential disagreements
  were all pools with injected duplicate ids); no implementation change.
- 1.1 — after the first differential run (7/10 operations agreed; 506/764/42
  disagreements in `last_before`/`outcome_lookup`/`cohort_criteria`):
  defined the total order T and min-/max-type picks (equal times in a
  max-type pick resolve to the LARGEST id); defined untimed-object handling
  per operation; `time` in §6 is the `params["basis"]` time for the three
  lookup targets and `None` for `administration_count`; §7 top-level
  provenance is the deduplicated union ordered by T on `contract.basis`,
  and the domain / undecidable rule is stated per kind; insufficient shape
  keeps the label keys `agg`/`target`; §5 keys fixed to `n` and
  `n_skipped_non_numeric`; numeric definition, `k` edge cases, `which`/`agg`
  validation, unparseable timestamps, and non-numeric handling in §1/§2 made
  explicit. All of these were raised in `oracle/NOTES.md` and
  `skills/NOTES.md`.
- 1.0 — initial version.
