---
phase: 01
phase_name: competition-control-plane
status: clean
depth: standard
files_reviewed: 38
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
resolved_during_review:
  critical: 1
  warning: 4
reviewed_at: 2026-08-24T04:12:00Z
review_commit: 3c8c693
---

# Phase 1 Code Review

## Result

Phase 1 is clean at standard depth after fixing one critical concurrency issue and three warnings. The review covered the full phase diff: package source, CLI and PowerShell launch surfaces, policies/config, tests/fixtures, and operating documentation.

The configured GSD reviewer agents are not installed and this session forbids automatic subagent dispatch, so the required review was performed inline using the same phase scope and severity gate.

## Resolved Findings

### CR-01: Single-use marker could be overwritten in a concurrent publish

- **Severity:** Critical
- **Location:** `src/biohub_tracker/io.py`, consumed by `src/biohub_tracker/launch.py`
- **Risk:** `exists()` followed by `os.replace()` was a time-of-check/time-of-use race. Two launchers could both validate an unused authorization, overwrite the same consumed marker, and invoke two pushes.
- **Resolution:** Immutable JSON now publishes with same-directory atomic hard-link creation, which never overwrites. A consumption-race loser returns `AUTHORIZATION_CONSUMED` before the runner.
- **Evidence:** `tests/test_io.py`; `tests/test_cli.py#test_launch_consumption_race_invokes_exactly_one_push`.

### WR-01: Quota could age during the owned-kernel status scan

- **Severity:** Warning
- **Location:** `src/biohub_tracker/guard.py`, `src/biohub_tracker/launch.py`, `src/biohub_tracker/cli.py`
- **Risk:** Reading quota before sequential status calls made the reserve value the oldest external input at push time.
- **Resolution:** Owned-kernel status is checked first and quota is read last at both authorization and execution.

### WR-02: External titles and diagnostics could inject Markdown into the status report

- **Severity:** Warning
- **Location:** `src/biohub_tracker/watch.py`
- **Risk:** Topic titles, notebook evidence, and collection diagnostics were interpolated into tracked Markdown without escaping.
- **Resolution:** Control characters are collapsed and Markdown/HTML metacharacters are escaped before report rendering.
- **Evidence:** `tests/test_watch.py#test_report_escapes_external_markdown`.

### WR-03: Recurring watch did not fingerprint live rules/evaluation content

- **Severity:** Warning
- **Location:** `src/biohub_tracker/watch.py`, `src/biohub_tracker/kaggle.py`
- **Risk:** Local policy hashes and discussion titles could not prove that the current Kaggle rules, evaluation, timeline, data, or code requirements were rechecked each session.
- **Resolution:** Every watch now collects `kaggle competitions pages --content`, stores full content in the immutable snapshot, and emits compact per-page SHA-256 fingerprints in the current report.
- **Evidence:** `tests/fixtures/kaggle/pages.json`; `tests/test_watch.py`.

### WR-04: Negated provenance flags were classified as exploit code

- **Severity:** Warning
- **Location:** `policies/metric_hack_patterns.json`
- **Risk:** The broad source regex matched clean audit declarations such as `metric_hack_used = False` and ordinary submission edge placeholders such as `t = -1`, temporarily excluding all 20 current notebooks.
- **Resolution:** Generic metric-hack language is title-only; source matching now requires an enabled/true flag, and negative-time detection requires nearby fake/sentinel/exploit context.
- **Evidence:** `tests/test_provenance.py`; live top-20 audit resolved to 3 curated research candidates, 8 explicit hacks, and 9 conservative source-review items.

## Security and Behavior Checks

- Subprocesses use argument arrays with `shell=False`; kernel refs and workspace paths are validated.
- Credential files are never opened or serialized; Kaggle diagnostics are bounded and redacted.
- Guard arithmetic is Decimal-based, uses the immutable registered runtime, and has no reserve override.
- Preflight report, evidence, kernel source, quota/status state, run/ref/path, reserve, expiry, and nonce are revalidated before push.
- Authorization consumption occurs atomically before the injected runner; failures remain auditable.
- Ledger writes are exclusive, append-only, fsynced, causally reconstructed, and corrections append amendments.
- Static notebook auditing never executes pulled source and explicit metric-hack matches take precedence.
- No default test or phase verification path invokes `kaggle kernels push`.

## Verification

- `python -m pytest -q`: 57 passed after review fixes.
- `python -m compileall -q src`: passed.
- `git diff --check`: passed.
- Live guard proof before review fixes: zero active owned competition kernels; GPU remained 0.00 used / 30.00 remaining.

## Remaining Assumptions

- Active-job detection queries the 20 newest owned competition kernels by `dateRun`. An active or queued run is necessarily recent, and 20 is far above Kaggle's accelerator concurrency capacity; any status ambiguity still fails closed.
- Phase 2 must pin the exact official scorer source and add metric regression fixtures. Phase 1 now fingerprints changes to the live Evaluation page but intentionally does not implement the scorer.
