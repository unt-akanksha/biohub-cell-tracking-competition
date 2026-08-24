---
phase: 01-competition-control-plane
plan: 03
subsystem: gpu-launch-safety
tags: [kaggle-gpu, decimal-quota, preflight, watchdog, single-use-authorization]
requires:
  - 01-01
  - 01-02
provides:
  - Fail-closed live quota and owned-kernel status decisions
  - Content-hashed smoke, dense-memory, and coverage preflight contract
  - Deadline-triggered checkpoint/flush watchdog and single-use push authorization
affects: [exact-validation, zebra-calibration, strong-model-training, offline-submission]
tech-stack:
  added: [python-stdlib, powershell-wrapper]
  patterns: [decimal-boundary-guard, two-read-authorization, hash-bound-evidence, consume-before-push]
key-files:
  created:
    - src/biohub_tracker/guard.py
    - src/biohub_tracker/preflight.py
    - src/biohub_tracker/watchdog.py
    - src/biohub_tracker/launch.py
    - docs/KAGGLE_RUNBOOK.md
  modified:
    - src/biohub_tracker/kaggle.py
    - src/biohub_tracker/ledger.py
    - src/biohub_tracker/cli.py
key-decisions:
  - "Exactly 8.00 projected GPU hours is allowed; anything below is rejected using Decimal."
  - "The registered maximum runtime is authoritative and cannot be shortened at guard time."
  - "Authorization is short-lived, hash-bound, nonce-confirmed, consumed before push, and rechecks live Kaggle state."
patterns-established:
  - "Runs over 1.00 hour require smoke plus dense-memory and complete-coverage evidence."
  - "The in-notebook watchdog invokes every cleanup callback and atomically emits one terminal record."
requirements-completed: [SAFE-01, SAFE-02, SAFE-03, SAFE-04, SAFE-05]
coverage:
  - id: D9
    description: "Launch eligibility requires a registered run, parseable quota, no active kernel, and a legal declared maximum."
    requirement: SAFE-01
    verification:
      - kind: unit
        ref: "tests/test_guard.py"
        status: pass
      - kind: other
        ref: "python -m biohub_tracker guard --run-id phase1-live-guard-readonly-smoke --max-runtime-hours 0.01 --live --json"
        status: pass
    human_judgment: false
  - id: D10
    description: "Decimal guard permits exactly 8.00 remaining and rejects any lower projection."
    requirement: SAFE-02
    verification:
      - kind: unit
        ref: "tests/test_guard.py#test_quota_boundary_uses_decimal_and_two_places"
        status: pass
    human_judgment: false
  - id: D11
    description: "Watchdog checkpoints, flushes, and writes terminal state at the safety margin despite callback errors."
    requirement: SAFE-03
    verification:
      - kind: unit
        ref: "tests/test_watchdog.py"
        status: pass
    human_judgment: false
  - id: D12
    description: "All jobs bind fresh hashed smoke evidence for imports, inputs, batch, model step, checkpoint, and output."
    requirement: SAFE-04
    verification:
      - kind: unit
        ref: "tests/test_preflight.py"
        status: pass
    human_judgment: false
  - id: D13
    description: "Runtimes strictly over 1.00 hour additionally require dense-memory and dataset-coverage evidence."
    requirement: SAFE-05
    verification:
      - kind: unit
        ref: "tests/test_preflight.py#test_one_point_zero_one_requires_dense_and_coverage"
        status: pass
    human_judgment: false
duration: 22min
completed: 2026-08-24
status: complete
---

# Phase 1 Plan 3: Fail-Closed Kaggle GPU Launch Summary

**Decimal quota protection, hashed preflight evidence, resumable budget shutdown, and single-use live-rechecked launch authorization**

## Performance

- **Duration:** 22 min
- **Started:** 2026-08-24T03:38:16Z
- **Completed:** 2026-08-24T04:00:34Z
- **Tasks:** 3
- **Files modified:** 14

## Accomplishments

- Enforced the immutable 8.00-hour reserve, 12.00-hour notebook ceiling, registered-runtime equality, parseable live quota, and no-active-kernel policy with machine reason codes.
- Added versioned, immutable preflight reports whose report and evidence hashes are revalidated at authorization and execution; long runs require dense-memory and full-coverage checks.
- Added a checkpoint/flush watchdog and a short-lived, single-use, source/preflight/state-bound launch flow requiring explicit `--execute` plus the exact nonce.
- Proved the live read path against the account: 30.00 GPU hours before and after, zero active owned competition kernels, and no kernel push.

## Task Commits

Each task was committed atomically:

1. **Task 1: Fail-closed quota and active-run guard** - `cc971e9` (feat)
2. **Task 2: Hashed preflights and runtime watchdog** - `2c0e8d2` (feat)
3. **Task 3: Single-use authorization and guarded launch** - `8503c62` (feat)

Follow-up verification fix: `189accf` (rate-safe sequential live status audit and preserved live decision evidence).

## Files Created/Modified

- `src/biohub_tracker/guard.py` - Decimal quota parsing, active-kernel inspection, reasoned decisions, and state hashing.
- `src/biohub_tracker/preflight.py` - Versioned checks, freshness, run binding, and content/evidence validation.
- `src/biohub_tracker/watchdog.py` - Monotonic deadline, heartbeat, callbacks, terminal persistence, and bounded shutdown exception.
- `src/biohub_tracker/launch.py` - Kernel/path/source validation, immutable authorizations, recheck, consumption, and injected push runner.
- `docs/KAGGLE_RUNBOOK.md` - Register-to-terminal workflow, formulas, watchdog integration, and cloud handoff.

## Decisions Made

- The registered declared maximum—not a launch-time override—is the only runtime used in reserve math.
- Kernel sources, metadata ref/path, quota/status state, preflight report/evidence, run, policy reserve, expiry, and nonce are all authorization inputs.
- A failed push consumes its authorization and appends a launch-failure audit while leaving the registered run available for a newly inspected authorization.
- The active-run check inspects the 20 newest owned competition kernels sequentially: active/queued work sorts newest, this exceeds Kaggle concurrency capacity, and it avoids API 429 ambiguity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Avoided Kaggle status-endpoint rate-limit ambiguity**
- **Found during:** Live Plan 3 verification
- **Issue:** Parallel status reads across the full historical kernel list triggered HTTP 429, which correctly failed closed but made the live guard unusable.
- **Fix:** Sort owned competition kernels by `dateRun` and inspect the newest 20 sequentially, well beyond the platform's possible concurrent run count.
- **Files modified:** `src/biohub_tracker/guard.py`, `experiments/events.jsonl`
- **Verification:** First decision preserved `KERNEL_STATUS_UNAVAILABLE`; retry passed with no active kernels and unchanged 30.00-hour quota.
- **Committed in:** `189accf`

**2. [Rule 2 - Missing Critical] Bound guard runtime to immutable registration**
- **Found during:** Task 3 authorization threat review
- **Issue:** A caller-supplied smaller runtime could otherwise weaken reserve math after registration.
- **Fix:** Added `DECLARED_RUNTIME_MISMATCH` and made authorize/execute use the registered value.
- **Files modified:** `src/biohub_tracker/guard.py`, `src/biohub_tracker/cli.py`, `tests/test_guard.py`
- **Verification:** Mismatch test and the full launch suite pass.
- **Committed in:** `8503c62`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical).  
**Impact on plan:** Both changes close safety gaps while preserving the planned no-launch Phase 1 boundary.

## Issues Encountered

- Kaggle returned HTTP 429 during the first live status attempt. The decision failed closed, GPU use stayed unchanged, and the retry passed after the rate-safe query strategy was applied.

## User Setup Required

None. The existing authenticated Kaggle CLI is sufficient. Cloud credentials are neither needed nor requested until the 8.00-hour handoff boundary.

## Next Phase Readiness

- Phase 2 can build the exact scorer and held-out manifests without GPU use.
- Every future Kaggle GPU candidate now has an auditable register/preflight/authorize/execute/watchdog/terminal path.
- Live quota remains 30.00 hours; 22.00 hours are spendable above the protected reserve.

## Self-Check: PASSED

- 52 tests passed, including all negative launch paths with an injected fake runner.
- The fake happy path received exactly `kaggle kernels push -p <resolved-dir>` and recorded the start event; no test or verification invoked a real push.
- Live before/after quota was 0.00 used and 30.00 remaining.

---
*Phase: 01-competition-control-plane*  
*Completed: 2026-08-24*
