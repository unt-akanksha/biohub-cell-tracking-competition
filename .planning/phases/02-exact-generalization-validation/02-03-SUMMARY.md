---
phase: 02-exact-generalization-validation
plan: 03
subsystem: exact-evaluation-and-comparison
tags: [tracking-cellmot, exact-report, diagnostics, pcg64, paired-bootstrap, immutable-ledger]
requires:
  - phase: 01-competition-control-plane
    provides: Immutable experiment events, evidence-eligible producer terminals, and offline-first control-plane patterns
  - phase: 02-01
    provides: Pinned patched organizer scorer, dependency lock, and exact metric regressions
  - phase: 02-02
    provides: Reciprocal manifests, ledger-resolved prediction inventories, and authoritative integer CSV/GEFF round trips
provides:
  - Separate immutable aggregate exact-evaluation lifecycle with canonical core and runtime envelope
  - Complete official pooled, embryo, fold, and movie evidence bound to four reciprocal producer slots
  - Reconciled endpoint, oracle, conditional-link, motion, density, and division diagnostics
  - Strict paired deltas, worst cases, and deterministic embryo-stratified complete-movie bootstrap
affects: [02-04-promotion-and-cpu-control, 03-zebrahub-affinity-gate, exact-model-promotion]
tech-stack:
  added: []
  patterns: [separate-aggregate-event-family, canonical-core-runtime-envelope, diagnostics-never-score, paired-complete-movie-bootstrap]
key-files:
  created:
    - config/evaluation-policy.json
    - reports/exact/README.md
    - src/biohub_tracker/evaluation.py
    - src/biohub_tracker/diagnostics.py
    - src/biohub_tracker/comparison.py
    - tests/test_evaluation.py
    - tests/test_diagnostics.py
    - tests/test_comparison.py
  modified:
    - src/biohub_tracker/ledger.py
    - src/biohub_tracker/evidence.py
    - tests/test_ledger.py
execution-commits: [aa3979c, 93bc7ab, 291a6f5]
key-decisions:
  - "Aggregate exact evaluations use their own event family so Phase 1 experiment transitions and GPU accounting remain byte-compatible."
  - "Comparison is computed from an immutable running aggregate registration, atomically attached as its terminal completion, and accepted only after completed-ledger validation."
  - "Diagnostics must reconcile to official integer counts and carry organizer_input_eligible=false; they can explain but never replace the organizer summary."
  - "Uncertainty uses policy-frozen NumPy PCG64 seed 20260824, 10,000 embryo-stratified complete-movie replicates, and linear percentiles."
patterns-established:
  - "Resolve all four terminal producer event hashes, lineage hashes, inventories, and round-trip evidence before importing or calling the scorer."
  - "Keep timestamps, runtime, peak memory, and optional public presentation metadata outside the deterministic report core."
  - "Pair only canonical sample IDs on one scorer/environment/manifest/policy/member boundary and resample whole movies, never edges."
requirements-completed: [VAL-03, VAL-04, VAL-05]
coverage:
  - id: D-01
    description: "One legal aggregate lifecycle publishes an immutable canonical exact core plus runtime envelope and binds all four reciprocal terminal producers and authoritative round-trip inventories."
    requirement: VAL-03
    verification:
      - kind: integration
        ref: "tests/test_evaluation.py"
        status: pass
      - kind: integration
        ref: "tests/test_ledger.py -k 'exact or producer or evaluation'"
        status: pass
    human_judgment: false
  - id: D-02
    description: "Official results retain complete pooled, embryo, fold, and movie sufficient statistics and fail closed on producer, coverage, graph, round-trip, scorer, or report drift."
    requirement: VAL-03
    verification:
      - kind: e2e
        ref: "tests/test_evaluation.py#test_exact_canonical_complete_movie_report_is_pooled_and_ledger_attached"
        status: pass
    human_judgment: false
  - id: D-03
    description: "Endpoint availability, oracle/link ceilings, conditional association, displacement, density, and division diagnostics reconcile exactly and remain structurally ineligible as organizer inputs."
    requirement: VAL-04
    verification:
      - kind: unit
        ref: "tests/test_diagnostics.py"
        status: pass
      - kind: integration
        ref: "tests/test_evaluation.py#diagnostic reconciliation in exact report"
        status: pass
    human_judgment: false
  - id: D-04
    description: "Candidate-minus-baseline evidence includes pooled, bilateral, fold, movie, count, diagnostic-strata, worst-paired, and lowest-absolute evidence plus deterministic paired uncertainty."
    requirement: VAL-05
    verification:
      - kind: unit
        ref: "tests/test_comparison.py"
        status: pass
      - kind: e2e
        ref: "python -m pytest -q tests/test_comparison.py tests/test_evaluation.py -k 'bootstrap or canonical'"
        status: pass
    human_judgment: false
duration: 54min
completed: 2026-08-24
status: complete
---

# Phase 2 Plan 3: Exact Reports, Diagnostics, and Paired Bootstrap Summary

**Ledger-resolved reciprocal exact reports with reconciled biological diagnostics, worst cases, and deterministic complete-movie uncertainty**

## Performance

- **Duration:** 54 min
- **Started:** 2026-08-24T07:47:00Z
- **Completed:** 2026-08-24T08:41:26Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments

- Added a backward-compatible aggregate exact-evaluation event family that registers exactly four baseline/candidate-by-fold terminal producers, scores only their authoritative integer CSV-rebuilt GEFF inventories, publishes an immutable deterministic core plus runtime envelope, and attaches both hashes exactly once.
- Produced official pooled, per-embryo, per-fold, and per-movie metrics with integer sufficient statistics, node counts/penalty inputs, complete coverage, and immutable scorer/environment/manifest/policy/member/round-trip bindings.
- Added non-authoritative endpoint/oracle/conditional-link, physical displacement, frozen density, and division timing/failure diagnostics that reconcile to official TP/FP/FN and serialize explicit finite no-event states.
- Added canonical candidate-minus-baseline component/count/diagnostic-strata deltas, stable worst and lowest-absolute movies, and 10,000 deterministic PCG64 complete-movie bootstrap replicates stratified within embryo.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build the canonical exact report and aggregate lifecycle** - `aa3979c` (feat)
2. **Task 2: Add reconciled tracking diagnostics** - `93bc7ab` (feat)
3. **Task 3: Add paired deltas, worst cases, and deterministic movie bootstrap** - `291a6f5` (feat)

## Files Created/Modified

- `src/biohub_tracker/evaluation.py` - Cheap provenance preflight, fresh scorer/round-trip execution, official projections, canonical core/envelope publication, and completed-ledger validation.
- `src/biohub_tracker/ledger.py` - Separate exact-evaluation registration/start/completion/failure transitions with immutable four-member and artifact binding.
- `src/biohub_tracker/evidence.py` - Producer registration and terminal event hashes exposed to exact member evidence.
- `src/biohub_tracker/diagnostics.py` - Pure finite endpoint/oracle/link, displacement, density, division, no-event, and aggregate reconciliation logic.
- `src/biohub_tracker/comparison.py` - Strict reciprocal pairing boundary, component/count/strata deltas, stable tails, and lazy NumPy PCG64 movie bootstrap.
- `config/evaluation-policy.json` - Submission/native authority, versioned diagnostic boundaries, and exact bootstrap algorithm/seed/count/percentile policy.
- `reports/exact/README.md` - Compact-report tracking and generated artifact prohibition.
- `tests/test_evaluation.py`, `tests/test_ledger.py`, `tests/test_diagnostics.py`, `tests/test_comparison.py` - Lifecycle, rejection, exact-core, reconciliation, pairing, bootstrap, no-event, and public-score-isolation coverage.

## Decisions Made

- Exact evaluation state is not overloaded onto the Phase 1 experiment state machine. A distinct event family preserves existing experiment behavior and makes reciprocal aggregate ownership explicit.
- The report is built while the aggregate registration is immutably `running`, then its core/envelope and authoritative inventories are attached as the sole completion. Report consumers require that completed terminal and matching hashes, avoiding a circular precondition without weakening lifecycle evidence.
- The official organizer summary remains the only score authority. Diagnostic reproduction must reconcile to those exact counts, carries `organizer_input_eligible=false`, and cannot enter the organizer aggregation callback.
- Bootstrap randomness and percentile interpolation are policy inputs bound into the report: NumPy `Generator(PCG64)`, seed `20260824`, 10,000 complete-movie samples within each embryo, and linear 2.5/50/97.5 percentiles.
- Public leaderboard metadata is allowed only in the non-authoritative envelope; comparison validation rejects public-score fields from the canonical decision object.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- An initial two-workspace determinism check rebuilt producer events with new timestamps. The resulting registration/terminal event hashes correctly changed the core because the immutable lineage inputs were different. Replaying the same frozen event stream into two fresh ledgers produced identical core hash `b65d22bce6338dc0a37afca1cb86f194e511099c5c6ef96491a5d9ac0a9626f1` and distinct runtime envelopes as designed.
- The pinned organizer intentionally warns when a split contains no divisions. Exact output retains `division_jaccard: null`, drops the division term exactly as upstream, and records explicit no-division bootstrap counts.

## User Setup Required

None. This plan used the existing pinned local CPU evaluation environment and compact synthetic fixtures. It did not access Kaggle, competition data, remote assets, GPU/CPU quota, or submission APIs.

## Next Phase Readiness

- Plan 02-04 can consume the completed exact report, bootstrap, and ledger lifecycle to implement deterministic promotion/review/rejection policy and CPU acceptance reconciliation.
- **Phase 2 remains open:** no official training data or real baseline/candidate prediction graphs are mounted locally. Plan 02-04 must execute and locally reconcile the distinct CPU-only official-data control for both reciprocal folds plus union; this synthetic report cannot satisfy that phase-close boundary.
- **Phase 3 entry remains gated:** the official-data truth/self control proves the evaluator, not a learned model. ZebraHub evaluation still requires real evidence-eligible reciprocal model producers and a completed learned-candidate aggregate exact report.

## Self-Check: PASSED

- The required exact-report subset passes: `25 passed`; task-specific diagnostics pass: `4 passed`; canonical/bootstrap selection passes: `3 passed`.
- The full pinned CPU suite passes: `140 passed`; `python -m compileall -q src tests` and `git diff --check` pass.
- Two fresh workspaces with the same frozen producer/aggregate event stream produced byte-identical canonical core hash `b65d22bce6338dc0a37afca1cb86f194e511099c5c6ef96491a5d9ac0a9626f1`; runtime envelope hashes differed.
- No graph binary, prediction CSV, generated exact report, public score, secret, or competition asset is tracked. All three task commits exist and contain only Plan 02-03 implementation/tests.
- Kaggle GPU and CPU used: `0` hours. Competition downloads, writes, submissions, and data access: `0`.

---
*Phase: 02-exact-generalization-validation*
*Completed: 2026-08-24*
