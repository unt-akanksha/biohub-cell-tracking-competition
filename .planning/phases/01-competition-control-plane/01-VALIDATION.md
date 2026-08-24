---
phase: 1
slug: competition-control-plane
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-23
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest compatible with the active Python environment |
| **Config file** | `pyproject.toml` — Plan 01-01 scaffolds it |
| **Quick run command** | `python -m pytest -q tests/test_watch.py tests/test_ledger.py tests/test_guard.py` |
| **Full suite command** | `python -m pytest -q` |
| **Estimated runtime** | Under 30 seconds using fixtures |

## Sampling Rate

- **After every task commit:** Run the task's targeted pytest file.
- **After every plan wave:** Run `python -m pytest -q`.
- **Before `$gsd-verify-work`:** Full suite and CLI help smoke must be green.
- **Max feedback latency:** 30 seconds for fixture tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | INTEL-01 | T-01 | Subprocess uses argument arrays and redacts diagnostics | integration | `python -m pytest -q tests/test_watch.py -k tracer` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | INTEL-02, INTEL-03 | T-02 | Pulled notebooks are read as text and never executed | unit | `python -m pytest -q tests/test_provenance.py` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | INTEL-04 | T-03 | Missing live fields render unavailable, not stale values | integration | `python -m pytest -q tests/test_watch.py -k report` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 2 | TRACK-01, TRACK-03 | T-04 | Ledger is append-only, locked, flushed, and transition-validated | unit | `python -m pytest -q tests/test_ledger.py -k append` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 2 | TRACK-02 | T-05 | Artifact paths and hashes are validated before persistence | unit | `python -m pytest -q tests/test_ledger.py -k finish` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 2 | TRACK-04 | — | Progress is deterministically reconstructed from events | integration | `python -m pytest -q tests/test_progress.py` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 3 | SAFE-01, SAFE-02 | T-06 | Parse/race/active-run uncertainty rejects authorization | unit | `python -m pytest -q tests/test_guard.py -k quota` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 3 | SAFE-03, SAFE-04, SAFE-05 | T-07 | Required preflight and watchdog evidence is hash-bound | unit | `python -m pytest -q tests/test_guard.py -k preflight` | ❌ W0 | ⬜ pending |
| 01-03-03 | 03 | 3 | SAFE-01 through SAFE-05 | T-08 | Default execution path cannot call kernel push | integration | `python -m pytest -q tests/test_cli.py -k launch` | ❌ W0 | ⬜ pending |

## Wave 0 Requirements

- [ ] `pyproject.toml` — package metadata, pytest configuration, and `biohub` console entry point.
- [ ] `tests/conftest.py` — temporary workspace and fake Kaggle runner fixtures.
- [ ] `tests/fixtures/kaggle/` — quota, submissions, leaderboard, topics, kernels, status, and malformed outputs.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Authenticated live read-only watch succeeds | INTEL-01 | Requires the user's Kaggle account and network | Run `biohub watch --live`; confirm a new snapshot, current quota, submissions, rank or explicit unavailable state, notebooks, and topics |
| No GPU launch occurred during Phase 1 | SAFE-01 | Requires checking external account state | Compare `kaggle quota --format json` and active kernel status before/after phase; expected GPU used remains unchanged |

## Validation Sign-Off

- [ ] All tasks have automated verification.
- [ ] Sampling continuity: no three consecutive tasks without automated verification.
- [ ] Wave 0 covers every missing test reference.
- [ ] No watch-mode flags are used.
- [ ] Fixture feedback latency is under 30 seconds.
- [ ] `nyquist_compliant: true` is set after execution evidence is complete.

**Approval:** pending

