---
phase: 02-exact-generalization-validation
plan: 02
subsystem: exact-evidence-boundary
tags: [reciprocal-manifest, immutable-ledger, geff, submission-csv, graph-integrity]
requires:
  - phase: 02-01-authoritative-patched-scorer
    provides: Verified patched organizer and TracksData callables with exact-count regressions
  - phase: 01-competition-control-plane
    provides: Append-only experiment ledger, lazy CLI, and canonical evidence helpers
provides:
  - Deterministic overlap-safe 44b6/6bba reciprocal complete-movie manifests
  - Ledger-authoritative prediction producer resolution before graph or organizer imports
  - Distinct native-float and integer-submission graph integrity gates
  - Hash-pinned CSV projection, fresh-ID GEFF reconstruction, and exact official parity evidence
affects: [02-03-exact-reports, 02-04-cpu-acceptance-and-promotion, 03-model-training]
tech-stack:
  added: []
  patterns: [hostile-sidecar-resolution, membership-hash-binding, pre-import-provenance-gate, authoritative-roundtrip]
key-files:
  created:
    - src/biohub_tracker/manifests.py
    - src/biohub_tracker/evidence.py
    - src/biohub_tracker/graphs.py
    - src/biohub_tracker/submission_io.py
    - tests/test_manifests.py
    - tests/test_graphs.py
    - tests/test_submission_io.py
  modified:
    - src/biohub_tracker/ledger.py
    - src/biohub_tracker/cli.py
    - .gitignore
execution-commits: [6321a55, 2311bc1, e81f5c4]
key-decisions:
  - "A prediction-set sidecar is always hostile input; only an exact immutable REGISTERED plus completed evidence-eligible terminal ledger projection can authorize graph loading."
  - "Native subvoxel graphs remain non-authoritative diagnostics; promotion authority begins only at the integer CSV-rebuilt GEFF inventory."
  - "Manifest discovery reads only exact Zarr v3 and GEFF metadata/content hashes, keeping image chunks and heavy scientific imports outside identity validation."
patterns-established:
  - "Resolve producer, manifest, fold, memberships, inventory, and artifact hashes before importing organizer or TracksData code."
  - "Publish round-trip CSV, rebuilt GEFF, and lineage evidence together only after semantic and official-count parity pass."
requirements-completed: [VAL-01, VAL-02]
coverage:
  - id: D-01
    description: "Mounted paired 44b6/6bba identities produce exactly two deterministic, movie-complete, overlap-safe reciprocal manifests with stable membership hashes."
    requirement: VAL-02
    verification:
      - kind: integration
        ref: "tests/test_manifests.py"
        status: pass
    human_judgment: false
  - id: D-02
    description: "Unknown, nonterminal, ineligible, lineage-mismatched, terminal-mismatched, mixed, incomplete, or stale prediction evidence fails before graph loading."
    requirement: VAL-02
    verification:
      - kind: integration
        ref: "tests/test_graphs.py"
        status: pass
      - kind: integration
        ref: "tests/test_ledger.py#producer evidence compatibility"
        status: pass
    human_judgment: false
  - id: D-03
    description: "Native graphs accept bounded subvoxel coordinates while submission graphs require integral coordinates and reject invalid topology without repair."
    requirement: VAL-02
    verification:
      - kind: unit
        ref: "tests/test_graphs.py#native/submission and integrity cases"
        status: pass
    human_judgment: false
  - id: D-04
    description: "Pinned organizer projection rebuilds fresh-ID integer GEFF graphs with semantic topology and exact per-movie/aggregate official-count parity."
    requirement: VAL-01
    verification:
      - kind: integration
        ref: "tests/test_submission_io.py"
        status: pass
      - kind: e2e
        ref: "tests/test_submission_io.py#test_graph_and_roundtrip_cli_publish_provenance_only_after_all_gates"
        status: pass
    human_judgment: false
duration: 35min
completed: 2026-08-24
status: complete
---

# Phase 2 Plan 2: Exact Generalization Identity Boundary Summary

**Reciprocal embryo manifests, ledger-resolved prediction lineage, strict graph integrity, and authoritative integer CSV/GEFF parity**

## Performance

- **Duration:** 35 min
- **Started:** 2026-08-24T07:10:28Z
- **Completed:** 2026-08-24T07:45:58Z
- **Tasks:** 3
- **Files modified:** 16

## Accomplishments

- Built fail-closed official-data discovery that accepts only paired `44b6_*`/`6bba_*` Zarr/GEFF identities and emits exactly the two reciprocal complete-movie folds with separate train, calibration, and evaluation hashes.
- Made every prediction sidecar subordinate to immutable ledger truth: incomplete legacy records remain readable but cannot authorize evidence, and all provenance/coverage/hash rejection happens before scientific imports.
- Added deterministic native/submission graph policies covering finite bounds, integral identity/time, topology, degrees, cycles, sentinels, fake forks, and complete inventory without mutation or repair.
- Wrapped the hash-pinned organizer converters in process, canonicalized competition CSV rows, rebuilt fresh-ID GEFF graphs, and proved semantic, per-movie official-count, and aggregate parity before immutable publication.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build deterministic reciprocal embryo manifests** - `6321a55` (feat)
2. **Task 2: Bind prediction lineage and enforce graph integrity** - `2311bc1` (feat)
3. **Task 3: Make integer submission-space round trips authoritative** - `e81f5c4` (feat)

## Files Created/Modified

- `src/biohub_tracker/manifests.py` - Exact metadata discovery, reciprocal memberships, overlap audit, semantic self-hash, and immutable verification.
- `src/biohub_tracker/evidence.py` - Frozen prediction claims and reason-coded producer reconciliation against immutable ledger events.
- `src/biohub_tracker/graphs.py` - Complete prediction inventory preflight plus native/submission schema, bounds, and topology validation.
- `src/biohub_tracker/submission_io.py` - Pinned projection, canonical CSV validation, rebuilt GEFF publication, semantic mapping, and official parity evidence.
- `src/biohub_tracker/ledger.py` - Backward-compatible optional registration and completed-producer evidence fields.
- `src/biohub_tracker/cli.py` - Lazy `manifest`, ledger-backed `graph validate`, and authoritative `submission roundtrip` commands.
- `manifests/README.md` - Real mounted-data build/verify procedure without a fabricated official manifest.
- `tests/test_manifests.py`, `tests/test_graphs.py`, `tests/test_submission_io.py`, `tests/test_ledger.py` - Identity, leakage, hostile-evidence, topology, projection, parity, CLI, and compatibility coverage.

## Decisions Made

- Sidecar syntax and self-hashes never confer authority. The local ledger must contain one exact registration and a completed terminal record with explicit `evidence_eligible=true` plus matching graph inventory and artifacts.
- Submission promotion authority is the rebuilt integer GEFF set, not native subvoxel GEFF or its diagnostic score. Rounding is performed once by the hash-pinned organizer projection.
- Coverage is set equality over a reciprocal fold before loading any GEFF. Missing, extra, duplicate, stale, wrong-fold, or mixed-producer artifacts have no intersection-only fallback.
- Existing Phase 1 event bytes and GPU lifecycle/accounting stay unchanged; optional evidence fields are additive, and older complete runs remain evidence-ineligible.

## Deviations from Plan

None - plan executed as specified. The checked-in canonical CSV fixture is deliberately force-tracked despite the generated `submission.csv` ignore rule; runtime CSV and GEFF outputs remain ignored.

## Issues Encountered

- The first synthetic truth pair was byte-identical and correctly triggered the new duplicate-artifact-hash gate. The fixture was made semantically distinct across embryos; production behavior was unchanged.
- A row-order rejection fixture initially failed earlier on its intentionally stale row IDs. The fixture now renumbers throwaway IDs before asserting the dedicated node-after-edge rejection.

## User Setup Required

None. This plan used only local compact fixtures and the existing pinned CPU evaluation environment. It did not access Kaggle, competition data, remote assets, or accelerators.

## Next Phase Readiness

- Plan 02-03 can consume only `PredictionInventory` objects that already resolve immutable producer/fold/membership/artifact evidence and expose authoritative rebuilt graphs.
- VAL-01 and VAL-02 stay globally pending under the shared-requirement gate until their sibling plans finish.
- **Phase-close blocker:** Plan 02-04 must complete and locally reconcile the CPU-only Kaggle official-data acceptance control. Fixture evidence cannot close Phase 2 or fabricate `manifests/reciprocal-embryo-v1.json`.

## Self-Check: PASSED

- Targeted manifest/graph/submission/ledger integration coverage passes, and the full pinned evaluation suite passes: `130 passed`.
- `python -m compileall -q src tests`, `git diff --check`, canonical two-run round-trip equality, CLI graph/round-trip provenance checks, and ignore checks all pass.
- All three task commits exist and contain only Plan 02-02 implementation/tests; no production reciprocal manifest or generated prediction/round-trip artifact is tracked.
- Kaggle GPU and CPU used: `0` hours. Competition downloads, asset writes, submissions, and competition data access: `0`.

---
*Phase: 02-exact-generalization-validation*
*Completed: 2026-08-24*
