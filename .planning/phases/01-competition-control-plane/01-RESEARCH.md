# Phase 1 Research: Competition Control Plane

**Phase:** 1 — Competition Control Plane  
**Researched:** 2026-08-23  
**Question:** What is needed to plan a reliable Kaggle watch, immutable experiment ledger, and fail-closed GPU launch guard?

## Executive Recommendation

Build one dependency-light Python package with a CLI that owns four contracts: read-only Kaggle collection, source-provenance classification, append-only experiment events, and launch authorization. Use the Kaggle CLI's JSON output rather than parsing tables. Keep raw timestamped snapshots ignored from Git, but commit compact policy/config, the append-only experiment ledger, and generated status/progress reports.

The first tracer slice is `biohub watch`: it invokes the authenticated CLI with argument arrays, stores a timestamped snapshot, and renders a compact report. This proves external read, persistent write/read, policy classification, and a user-visible CLI result end to end without consuming GPU.

## Kaggle CLI Contracts

The installed Kaggle CLI exposes machine-readable JSON for the needed read paths:

- `kaggle quota --format json`
- `kaggle competitions submissions biohub-cell-tracking-during-development --format json --page-size 200`
- `kaggle competitions leaderboard biohub-cell-tracking-during-development --show --format json --page-size 200`
- `kaggle competitions topics list biohub-cell-tracking-during-development --format json --sort-by recent`
- `kaggle kernels list --competition biohub-cell-tracking-during-development --format json --sort-by scoreDescending --page-size N`
- `kaggle kernels status owner/kernel`
- `kaggle kernels pull owner/kernel --metadata -p <cache>` for opt-in source auditing

The leaderboard's shown page may not contain the user's rank. The watch should support the full leaderboard download into a temporary directory, unzip the CSV, and match configured team aliases. If that path fails, rank is `unavailable` rather than copied from a prior snapshot.

Kaggle JSON fields are strings for hours and public scores; parsing must be explicit and tested. CLI warnings may precede JSON, so the collector should locate and parse the first valid JSON array/object while preserving stderr separately.

## Notebook Provenance

Score ordering is not a trust signal because pre-patch scores remain visible. Classification precedence should be:

1. `explicit_metric_hack`: high-confidence title/source signature or curated rejection.
2. `reproduced_post_patch`: exact source hash reproduced with the current official metric.
3. `reported_post_patch`: curated source review with a patched-era claim but no local reproduction.
4. `stale_or_ghost_risk`: displayed/title score conflicts with source text or metric-patch date.
5. `automated_no_known_signature`: source was pulled and no known signature matched; this is not equivalent to clean.
6. `unknown`: source was not inspected or changed since review.

The curated registry should seed the current audit: explicit hack notebooks are excluded; Evgen/Yunus/Kunal/Anhad/Yusuke/Pilkwang/Xiaolei-M001/Reyhan clean-source families are allowed as research candidates with unverified score provenance. Source auditing reads text only; it must never execute pulled notebook code.

## Experiment Event Model

Use JSON Lines with one immutable event per line and a small lock file for a single writer. Event types:

- `registered`: identity, hypothesis, parent, config, code/data/model hashes, seeds, split, declared maximum runtime.
- `started`: Kaggle ref, quota before, authorization ID, start time.
- `completed`, `failed`, or `rejected`: actual runtime, quota after, artifact/report hashes, result/failure reason.
- `decision`: promote, retain, retire, or inconclusive with gate evidence.
- `amendment`: references a prior event and supplies corrected fields without changing the original line.

Reconstruction validates unique event IDs, known run IDs, legal state transitions, and amendment targets. Duplicate run registration fails. A malformed/truncated final line is quarantined and reported; it is never silently discarded from the source file.

Seed the ledger with compact records for the prior state guard, pairwise graph, dense topology replication, V-JEPA, SEA-RAFT, ZebraHub selective-SSM, HOCT OOM, and long ranker coverage failure.

## GPU Guard Contract

Authorization inputs are live GPU remaining hours, declared maximum runtime, fixed reserve 8.0 hours, active GPU kernel statuses, and required preflight evidence.

Exact boundary: a job is allowed when projected remaining is exactly 8.0 hours and rejected below 8.0. Decimal arithmetic should use `Decimal`, not binary float. A runtime must be greater than zero and no more than the policy notebook maximum. Jobs above one hour require smoke, dense-memory, and coverage preflight records. All jobs require a watchdog/checkpoint contract.

An authorization is a short-lived JSON artifact containing quota snapshot hash, runtime, projected remaining, experiment ID, preflight hashes, and expiry. The launch wrapper rechecks quota and active jobs immediately before invoking `kaggle kernels push`; authorization alone never launches. Phase 1 tests use a fake runner and must not push a kernel.

## Walking Skeleton

For this CLI project, the user interface is the terminal, the persistent data layer is the snapshot/event store, and the external route is the read-only Kaggle CLI. The skeleton proves:

1. `biohub watch` reads live or fixture Kaggle JSON.
2. It writes a timestamped snapshot atomically.
3. It reads policy and notebook audit records.
4. It renders a session status report.
5. The same command works from a documented local invocation.

No database or web UI is justified; introducing one would contradict the approved offline, dependency-light stack.

## Validation Architecture

### Test Layers

- Unit: hour/score/date parsing, JSON extraction, classification precedence, event transition rules, hash generation, quota boundary arithmetic.
- Contract: fixture outputs for every Kaggle command and stable normalized snapshot schema.
- Integration: fake Kaggle executable/process runner drives watch, ledger, guard, and report without network.
- Live read-only smoke: opt-in `biohub watch --live`; never runs in the default test suite.
- Negative safety: malformed quota, active job, duplicate run, missing preflight, expired authorization, source hash drift, and path escape.

### Sampling

- After each task: relevant pytest file.
- After each plan: full `python -m pytest -q` plus CLI help smoke.
- Before phase verification: fixture integration suite, live `kaggle quota`, and a real read-only watch snapshot.

### Evidence

Tests assert exact exit codes and JSON fields. A rejected launch must contain a machine-readable reason. No test invokes `kaggle kernels push`; the launch runner is injected and records arguments.

## Risks and Mitigations

- **Credential leakage:** never read or serialize Kaggle config/token; redact subprocess diagnostics.
- **Shell injection:** use argument arrays with `shell=False`; validate Kaggle refs and paths.
- **Partial writes:** write snapshots/reports through same-directory temporary files and atomic replace; append ledger under an exclusive lock and flush/fsync.
- **Classifier overclaim:** absence of known signatures is `automated_no_known_signature`, never `clean`.
- **Quota race:** authorize and launch both re-read live quota; active-run check fails closed.
- **Platform drift:** Python core is cross-platform; PowerShell wrapper is a convenience only.

## Sources

- Live command help from Kaggle CLI 2.2.3 on 2026-08-23.
- [Kaggle competition overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview)
- [Kaggle CLI output-format documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/output_format.md)
- [Official baseline and patched metric](https://github.com/royerlab/kaggle-cell-tracking-competition)

