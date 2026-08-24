---
phase: 01-competition-control-plane
plan: 02
subsystem: experiment-tracking
tags: [jsonl, lineage, immutable-ledger, progress, evidence-gates]
requires:
  - 01-01
provides:
  - Append-only experiment identity, lifecycle, amendments, and decisions
  - Deterministic lineage and evidence progress projections
  - Audited seed history for eight prior experiment families
affects: [gpu-guard, exact-validation, model-promotion, submission-selection]
tech-stack:
  added: [python-stdlib]
  patterns: [exclusive-writer-lock, append-only-events, evidence-gated-decisions, deterministic-projections]
key-files:
  created:
    - src/biohub_tracker/ledger.py
    - src/biohub_tracker/progress.py
    - experiments/events.jsonl
    - reports/PROGRESS.md
  modified:
    - src/biohub_tracker/cli.py
    - experiments/README.md
key-decisions:
  - "Historical events remain byte-identical; corrections append amendment events."
  - "Public leaderboard evidence is displayed separately and cannot independently promote a run."
  - "ZebraHub remains visibly incomplete until reciprocal calibration and complete-graph exact scoring pass."
patterns-established:
  - "Every new event validates the complete causal ledger before an fsynced append."
  - "Artifact paths are workspace-bound and their bytes are hashed before evidence is accepted."
requirements-completed: [TRACK-01, TRACK-02, TRACK-03, TRACK-04]
coverage:
  - id: D5
    description: "Registration records stable identity, lineage, hashes, seeds, split, and declared runtime."
    requirement: TRACK-01
    verification:
      - kind: unit
        ref: "tests/test_ledger.py"
        status: pass
    human_judgment: false
  - id: D6
    description: "Terminal lifecycle events preserve metrics, resources, artifacts, and explicit failures."
    requirement: TRACK-02
    verification:
      - kind: unit
        ref: "tests/test_ledger.py"
        status: pass
    human_judgment: false
  - id: D7
    description: "Concurrent, truncated, illegal, and amended ledgers preserve append-only history."
    requirement: TRACK-03
    verification:
      - kind: unit
        ref: "tests/test_ledger.py"
        status: pass
    human_judgment: false
  - id: D8
    description: "Progress reconstructs all prior outcomes and selects the ZebraHub reciprocal graph gate."
    requirement: TRACK-04
    verification:
      - kind: unit
        ref: "tests/test_progress.py"
        status: pass
      - kind: other
        ref: "python -m biohub_tracker progress"
        status: pass
    human_judgment: false
duration: 17min
completed: 2026-08-24
status: complete
---

# Phase 1 Plan 2: Immutable Experiment Ledger Summary

**Append-only experiment lineage, strict evidence-aware lifecycle commands, and a deterministic progress board seeded with all audited prior work**

## Performance

- **Duration:** 17 min
- **Started:** 2026-08-24T03:21:00Z
- **Completed:** 2026-08-24T03:38:16Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Added atomic registration and lifecycle events with exclusive locking, fsync, transition validation, workspace-bound hashing, and explicit truncated-ledger recovery.
- Added terminal metric validation and evidence-gated decisions; a public score alone cannot produce a promotion.
- Seeded eight prior experiment families, kept costly failures visible, and identified ZebraHub reciprocal calibration plus exact complete-graph scoring as the next gate.

## Task Commits

Each task was committed atomically:

1. **Task 1: Append-only experiment registration** - `54952bc` (feat)
2. **Task 2: Lifecycle and evidence gates** - `5a7d0fb` (feat)
3. **Task 3: Deterministic experiment progress board** - `80e4aa8` (feat)

## Files Created/Modified

- `src/biohub_tracker/ledger.py` - Event model, locking, append, repair, reconstruction, transitions, and artifact validation.
- `src/biohub_tracker/progress.py` - Stable JSON/Markdown projections and next-gate selection.
- `src/biohub_tracker/cli.py` - Registration, lifecycle, amendment, repair, decision, and progress commands.
- `experiments/events.jsonl` - Append-only imported audit of prior experiment history.
- `reports/PROGRESS.md` - Current experiment evidence and next action.

## Decisions Made

- Preserve explicit unknowns for unavailable historical hashes/runtime instead of inventing evidence; all newly registered events remain strict.
- Treat lifecycle completion and promotion-gate completion separately so a completed prior run can remain visibly incomplete for the next scientific decision.
- Store numeric metrics as normalized decimal strings to avoid float representation ambiguity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Validated causal structure during full-ledger reconstruction**
- **Found during:** Task 3 (deterministic projection tests)
- **Issue:** Schema-valid events could still be reordered into an illegal lifecycle without a structural check.
- **Fix:** Require one registration, causal start/terminal/decision order, valid amendments, and existing parents during reconstruction.
- **Files modified:** `src/biohub_tracker/ledger.py`, `tests/test_progress.py`
- **Verification:** Reordered legal events project identically; illegal causal structures fail.
- **Committed in:** `80e4aa8`

**2. [Rule 2 - Missing Critical] Represented historical evidence gaps explicitly**
- **Found during:** Task 3 (prior-audit import)
- **Issue:** Several audited historical runs did not retain exact runtime or full schema-era hashes.
- **Fix:** Imported records carry explicit `not_available_imported_audit`/`unknown` values and authorization false; no evidence is fabricated.
- **Files modified:** `experiments/events.jsonl`, `reports/PROGRESS.md`
- **Verification:** All eight families appear with their known evidence and failures intact.
- **Committed in:** `80e4aa8`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical).  
**Impact on plan:** Both changes preserve truthfulness and make the append-only audit stricter without expanding scope.

## Issues Encountered

- Some historical experiment metadata was not recoverable from the prior audit. Those fields remain explicit unknowns and cannot authorize promotion.

## User Setup Required

None.

## Next Phase Readiness

- Launch authorization can now bind a one-time guard decision to a registered run and append start/terminal evidence.
- No GPU work has started; live quota remained 30.00 hours at the last read.

## Self-Check: PASSED

- 18 ledger/progress tests passed.
- The progress command reproduced the seeded history and selected the exact ZebraHub next gate.
- Promotion without non-public evidence, illegal transitions, unsafe paths, duplicate IDs, concurrency corruption, and silent truncation are rejected.

---
*Phase: 01-competition-control-plane*  
*Completed: 2026-08-24*
