---
phase: 01-competition-control-plane
plan: 01
subsystem: competition-intelligence
tags: [kaggle-cli, snapshots, provenance, quota, reports]
requires: []
provides:
  - Read-only fixture and live Kaggle competition snapshots
  - Static notebook provenance classification with source-hash invalidation
  - Deterministic Markdown and JSON competition status projections
affects: [experiment-ledger, gpu-guard, exact-validation, submission-selection]
tech-stack:
  added: [python-stdlib, pytest]
  patterns: [explicit-live-mode, immutable-snapshots, fail-visible-collection, inert-source-audit]
key-files:
  created:
    - src/biohub_tracker/watch.py
    - src/biohub_tracker/provenance.py
    - reports/competition-status.md
  modified:
    - src/biohub_tracker/kaggle.py
    - src/biohub_tracker/cli.py
    - config/competition.json
key-decisions:
  - "Fixture mode is the default; authenticated Kaggle reads require --live."
  - "A no-signature static scan is never represented as a clean or reproduced score."
  - "Current rank is resolved from the downloaded complete leaderboard, not a stale cached value."
patterns-established:
  - "External reads use subprocess argument arrays with shell=False and redacted errors."
  - "Raw snapshots are immutable; compact current reports are atomically replaced."
requirements-completed: [INTEL-01, INTEL-02, INTEL-03, INTEL-04]
coverage:
  - id: D1
    description: "One command captures quota, submissions, complete/top leaderboard, notebooks, and topics into an immutable snapshot."
    requirement: INTEL-01
    verification:
      - kind: integration
        ref: "tests/test_watch.py#test_tracer_snapshot_schema_and_status"
        status: pass
      - kind: other
        ref: "python -m biohub_tracker watch --live --top 20"
        status: pass
    human_judgment: false
  - id: D2
    description: "Notebook provenance excludes explicit metric hacks and invalidates changed reviewed source."
    requirement: INTEL-02
    verification:
      - kind: unit
        ref: "tests/test_provenance.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "Snapshots preserve collection timestamps, content hashes, official policy values, and policy-file fingerprints."
    requirement: INTEL-03
    verification:
      - kind: integration
        ref: "tests/test_watch.py#test_tracer_snapshot_schema_and_status"
        status: pass
    human_judgment: false
  - id: D4
    description: "Current reports show clean score evidence, rank, submissions, quota headroom, active hypothesis, and next gate."
    requirement: INTEL-04
    verification:
      - kind: unit
        ref: "tests/test_watch.py#test_report_contains_required_status_and_quota_math"
        status: pass
      - kind: unit
        ref: "tests/test_watch.py#test_report_is_deterministic_and_missing_rank_is_explicit"
        status: pass
    human_judgment: false
duration: 15min
completed: 2026-08-24
status: complete
---

# Phase 1 Plan 1: Competition Watch and Provenance Summary

**Read-only Kaggle control plane with immutable snapshots, complete-leaderboard rank resolution, static no-hack provenance, and deterministic session status reports**

## Performance

- **Duration:** 15 min
- **Started:** 2026-08-24T03:05:13Z
- **Completed:** 2026-08-24T03:20:00Z
- **Tasks:** 3
- **Files modified:** 22

## Accomplishments

- Added fixture-default and explicit-live competition collection across quota, 59 personal submissions, complete/top leaderboard, 20 top notebooks, and recent topics.
- Added inert source auditing with curated post-patch research candidates, six currently detected explicit metric-hack exclusions, and source-hash invalidation.
- Generated a live status projection showing rank 748, clean public score 0.913, leader score 0.962, and 30.00/22.00 remaining/spendable GPU hours.

## Task Commits

Each task was committed atomically:

1. **Task 1: Read-only competition watch tracer** - `fc19007` (feat)
2. **Task 2: Static notebook provenance audit** - `65551ec` (feat)
3. **Task 3: Deterministic competition status** - `14e6450` (feat)

## Files Created/Modified

- `src/biohub_tracker/watch.py` - Collection normalization, immutable snapshots, status projection, and Markdown rendering.
- `src/biohub_tracker/provenance.py` - Source hashing and conservative provenance precedence.
- `src/biohub_tracker/kaggle.py` - Safe CLI runner, notebook pull, and complete-leaderboard CSV/ZIP reader.
- `policies/notebook_audits.json` - Curated current research candidates, exclusions, and ghost-risk evidence.
- `reports/competition-status.md` - Current compact human-readable live status.

## Decisions Made

- Retained explicit fixture/live separation so tests and ordinary local calls cannot accidentally hit Kaggle.
- Used the complete leaderboard download for rank and the shown leaderboard page for current leader score.
- Kept `automated_no_known_signature` distinct from clean/reproduced evidence.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added complete-leaderboard download parsing**
- **Found during:** Task 3 (current status projection)
- **Issue:** The top-200 JSON view cannot reliably contain the user's rank.
- **Fix:** Added safe temporary ZIP/CSV parsing of the entire leaderboard and retained the top view separately.
- **Files modified:** `src/biohub_tracker/kaggle.py`, `src/biohub_tracker/watch.py`, `tests/test_watch.py`
- **Verification:** Live rank resolved to 748; full-leaderboard parser test passed.
- **Committed in:** `14e6450`

**2. [Rule 1 - Bug] Made notebook tree hashes independent of cache root**
- **Found during:** Task 3 verification
- **Issue:** Absolute cache paths would invalidate unchanged source after relocating the workspace.
- **Fix:** Hash relative source paths and bytes from their common root.
- **Files modified:** `src/biohub_tracker/provenance.py`
- **Verification:** Provenance suite passed.
- **Committed in:** `14e6450`

---

**Total deviations:** 2 auto-fixed (1 missing critical, 1 bug).  
**Impact on plan:** Both changes strengthen the planned freshness and provenance contracts without expanding into mutation or GPU execution.

## Issues Encountered

- The configured Kaggle aliases did not initially match the live team name. The complete leaderboard identified the current team alias as `Indar Karhana`; it is now explicit in competition config.

## User Setup Required

None - existing Kaggle CLI authentication was used without reading credential material.

## Next Phase Readiness

- The immutable experiment ledger can consume live snapshot hashes and the current active hypothesis.
- GPU remains untouched at 30.00 hours; the 8.00-hour reserve policy is ready for enforcement in Plan 01-03.

## Self-Check: PASSED

- 11 watch/provenance tests passed.
- Live read-only watch completed with all collection statuses `ok`.
- Required report sections, quota arithmetic, static source prohibitions, and CLI help checks passed.

---
*Phase: 01-competition-control-plane*  
*Completed: 2026-08-24*
