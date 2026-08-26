---
phase: 02-exact-generalization-validation
reviewed: "2026-08-26T00:49:04Z"
depth: standard
diff_base: "79e9670^"
reviewed_head: "49aba966cc27c193cb6eec240dcb2d60b59f4a49"
fix_range: "12df198..49aba96"
re_review_iteration: 2
files_reviewed: 43
files_reviewed_list:
  - ".gitignore"
  - "config/evaluation-policy.json"
  - "config/official-scorer.lock.json"
  - "config/phase2-control.json"
  - "config/promotion-policy.json"
  - "docs/EXACT_VALIDATION_RUNBOOK.md"
  - "experiments/README.md"
  - "kaggle/phase2-cpu-acceptance/kernel-metadata.json"
  - "kaggle/phase2-cpu-acceptance/phase2_acceptance.py"
  - "kaggle/phase2-cpu-acceptance/runtime-dataset-metadata.json"
  - "manifests/README.md"
  - "pyproject.toml"
  - "reports/exact/README.md"
  - "requirements/evaluation-lock.txt"
  - "scripts/get-kaggle-dataset-state.py"
  - "scripts/get-kaggle-kernel-state.py"
  - "scripts/inspect-runtime-bundle.py"
  - "scripts/materialize-official-scorer.ps1"
  - "scripts/run-phase2-cpu-acceptance.ps1"
  - "src/biohub_tracker/acceptance.py"
  - "src/biohub_tracker/cli.py"
  - "src/biohub_tracker/comparison.py"
  - "src/biohub_tracker/diagnostics.py"
  - "src/biohub_tracker/evaluation.py"
  - "src/biohub_tracker/evidence.py"
  - "src/biohub_tracker/graphs.py"
  - "src/biohub_tracker/ledger.py"
  - "src/biohub_tracker/manifests.py"
  - "src/biohub_tracker/progress.py"
  - "src/biohub_tracker/promotion.py"
  - "src/biohub_tracker/scorer.py"
  - "src/biohub_tracker/scorer_lock.py"
  - "src/biohub_tracker/submission_io.py"
  - "tests/test_comparison.py"
  - "tests/test_diagnostics.py"
  - "tests/test_evaluation.py"
  - "tests/test_exact_cli.py"
  - "tests/test_graphs.py"
  - "tests/test_ledger.py"
  - "tests/test_manifests.py"
  - "tests/test_promotion.py"
  - "tests/test_scorer.py"
  - "tests/test_submission_io.py"
findings:
  critical: 5
  warning: 2
  info: 0
  total: 7
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-08-26T00:49:04Z
**Depth:** standard
**Files Reviewed:** 43
**Status:** issues_found

## Narrative Findings (AI reviewer)

### Summary

Phase 2 is still not safe to close at `49aba96`. The fixes genuinely resolve the original symlink escape, red tracked-manifest test, exception-to-gate binding, lineage-based division pairing, pooled diagnostic omissions, global ID reservations, and canonical request validation. Candidate-ledger reconstruction also fixes the original durable-corruption defect. However, the CPU control still authenticates important remote values from the remote values themselves; the new global timestamp rule conflicts with the retry saga; exact-report crash recovery can bless a self-hashed replacement report; stale-lock recovery has a lock-stealing race; and the new metric-exploit gate is fed by an unconditional `"passed"` literal rather than an audit.

Independent verification supplied for this iteration reports **207 passed, 2 skipped** (the skips are Windows symlink-privilege cases), with compile and diff checks passing. I independently reran `git diff --check` for the exact 43-file scope and reproduced the reconciliation timestamp failure described in CR-02. Passing tests do not cover the trust and interleaving failures below.

### Critical Issues

#### CR-01: Local reconciliation still trusts remote division, score, diagnostic, and inventory evidence

**Classification:** BLOCKER
**Files:** `src/biohub_tracker/acceptance.py:1030-1069`, `src/biohub_tracker/acceptance.py:1102-1126`, `src/biohub_tracker/acceptance.py:1187-1224`, `src/biohub_tracker/acceptance.py:1226-1279`, `src/biohub_tracker/evaluation.py:536-583`

**Issue:** `_validate_truth_self_sufficient_row` independently anchors edge totals and node totals to the manifest, but the manifest contains no ground-truth division count. The validator therefore accepts any nonnegative `division_tp` as long as the remote producer copies it into `organizer_row` and diagnostic reconciliation; it also does not independently validate the row's Jaccards, adjusted score, final score, node-count ratio, or the rest of `diagnostic_state`. `_locally_revalidate_control` then rebuilds official summaries, diagnostics, comparison, and bootstrap from those same remote rows. This proves internal consistency, not truth. The four authoritative inventory digests are likewise only schema-checked and compared between identical baseline/candidate slots; they are not derived from a locally trusted graph/CSV artifact. A producer can alter these values, recompute the pending self-hashes and projections, and receive locally branded exact evidence. The current regression test even demonstrates the gap by accepting `division_tp = 1` against a local sample object that has only edge and node counts.

**Fix:** Extend the trusted manifest/source evidence with topology sufficient statistics such as the exact ground-truth division/fork count, and recompute every truth/self scalar from those trusted counts. Independently rebuild diagnostic state rather than copying it from each remote row. Bind authoritative inventory hashes to locally available content-addressed artifacts, or retrieve and verify the remote CSV/GEFF artifacts before completion. Reject reconciliation unless all organizer scalars, division totals, diagnostics, and inventories can be reproduced from independently trusted inputs. Add tamper tests for `division_tp`, per-movie Jaccards/score, a non-reconciliation diagnostic field, and every inventory digest while all remote hashes/projections are recomputed.

#### CR-02: The global timestamp guard makes the deterministic reconciliation saga non-resumable

**Classification:** BLOCKER
**Files:** `src/biohub_tracker/ledger.py:970-984`, `src/biohub_tracker/acceptance.py:1465-1474`, `src/biohub_tracker/acceptance.py:1490-1506`, `src/biohub_tracker/acceptance.py:1535-1555`, `src/biohub_tracker/acceptance.py:1589-1606`

**Issue:** Every reconciliation saga event is deliberately backdated to the CPU start event so retries produce stable hashes. `Ledger.append`, however, rejects an event whose timestamp is earlier than the maximum timestamp of **any** durable event, including unrelated experiment, CPU, or exact lifecycles. Consequently, if any other ledger event is appended while the Kaggle CPU job is running—or between a partial reconciliation and its retry—the first missing saga event fails with `event created_at precedes durable ledger history`. I reproduced this with a CPU registration/start, one unrelated experiment registration one second later, and a valid input-binding event at the start timestamp. This strands the one-use acceptance lifecycle and defeats CR-01's recovery design.

**Fix:** Remove the global cross-lifecycle timestamp comparison. The already-added candidate decode at lines 981-984 is sufficient to reject an event that would make its own lifecycle unreconstructable. If chronology must be enforced, validate only relative to events in the same lifecycle, or add a separate monotonic append ordinal/hash-chain that is not used as semantic event time. Add an integration test that starts a CPU control, appends unrelated experiment/exact activity, then completes and retries reconciliation successfully.

#### CR-03: Crash recovery can promote an arbitrary self-hashed report to durable exact evidence

**Classification:** BLOCKER
**Files:** `src/biohub_tracker/evaluation.py:335-398`, `src/biohub_tracker/evaluation.py:741-810`, `src/biohub_tracker/evaluation.py:857-864`, `src/biohub_tracker/evaluation.py:993-1018`

**Issue:** When the output directory already exists, `_published_report` validates the report's own core/envelope hashes and checks registration identities and members, but it does not rerun scoring, round-trip validation, diagnostics, comparison, or verify that this exact report was durably materialized before the crash. `validate_exact_core` is intentionally shallow for the large `official`, `diagnostics`, and `comparison` trees. While the exact lifecycle is `RUNNING`, recovery then constructs and appends `EXACT_EVALUATION_COMPLETED` from whatever files are present. After a kill between `os.rename` and the completion append, a modified or replacement core with arbitrary scores/comparison can update its envelope reference and be blessed as the canonical completed report. Promotion subsequently accepts the ledger hash because recovery itself created that binding.

**Fix:** Before publishing the directory, append a durable materialization event containing the core hash, envelope hash, and a full artifact inventory, then permit recovery only when every published byte matches that event. Alternatively, recovery must rerun the full scorer/round-trip/diagnostic/comparison pipeline and compare the complete result before appending completion. File existence plus self-hashes must never authorize a terminal event. Add a kill-after-rename test that mutates a score and recomputes the core/envelope hashes; recovery must reject it without completing the lifecycle.

#### CR-04: Stale-lock recovery can steal a newly acquired active lock

**Classification:** BLOCKER
**File:** `src/biohub_tracker/ledger.py:823-877`

**Issue:** `_recover_stale` stats and reads the stale lock, checks its owner, stats it again, and then calls `os.replace(self.path, quarantine)`. The identity checks and rename are not a compare-and-swap. Another contender can remove the stale file and acquire a fresh active lock after line 862 but before line 867; the recovering process then moves that live lock out of the lock path. The byte comparison detects the race only after the active lock has disappeared. Although it tries to restore the file, a third contender can acquire during that gap, leaving the original active writer without visible ownership while another writer enters the ledger critical section. That breaks mutual exclusion and risks interleaved or semantically conflicting ledger writes.

**Fix:** Use an OS-backed advisory/exclusive file lock with process-death release (through a vetted cross-platform locking implementation), so stale recovery does not rename a pathname owned by another process. If pathname locks must remain, implement a platform-specific atomic identity/CAS protocol; otherwise fail closed and require explicit manual stale-lock recovery. Add a multiprocess race test that pauses a recoverer after its final stat, lets another process recover/acquire, then resumes the first recoverer and proves the live lock is never moved.

#### CR-05: The metric-exploit hard gate is still an unconditional pass marker

**Classification:** BLOCKER
**Files:** `src/biohub_tracker/evaluation.py:694-738`, `src/biohub_tracker/promotion.py:306-320`, `config/promotion-policy.json:45-59`

**Issue:** Production report generation always writes `"metric_exploit_audit": "passed"` in `canonical_report_core`; there is no candidate-specific audit function or failing production path for this field. Promotion now correctly rejects a missing or explicit failed value, but every report produced by the evaluator receives the pass literal regardless of graph topology or exploit signature. The scorer-lock and graph/round-trip gates are valuable separate controls, but they do not make this named hard gate an audit result. The fix therefore changes absence into an assertion without establishing the evidence that assertion claims.

**Fix:** Compute a deterministic exploit-audit result from the pinned patched-scorer provenance, frozen exploit regression identity, graph-integrity findings, and candidate-specific known-signature checks. Persist the underlying evidence/hash in the report schema and derive `passed` only when every required check succeeds; otherwise emit `failed` and the reason code. Add a production-shaped candidate fixture containing a known exploit signature and prove the evaluator emits a failed audit that promotion rejects.

### Warnings

#### WR-01: Reconciliation publishes `accepted: true` before any terminal evidence is durable

**Classification:** WARNING
**File:** `src/biohub_tracker/acceptance.py:1613-1668`

**Issue:** The final acceptance document, including `accepted: true` and hashes for five events that may not yet exist, is written at lines 1657-1660. Only afterward are the input binding, CPU completion, exact registration, exact start, and exact completion appended one by one. A process kill during that sequence leaves a stable, apparently accepted report on disk while the ledger remains partial. Retry support may eventually repair it, but until then any consumer that reads the acceptance artifact without separately resolving all hashes against the ledger sees a false terminal claim.

**Fix:** Publish a non-authoritative `.pending`/prepared artifact first and atomically expose the `accepted: true` document only after terminal completion, or add a durable materialized/committed transition and require all consumers to verify it. Add kill-point assertions that no terminal-looking acceptance path is visible before ledger completion.

#### WR-02: Versioned manifest and ledger decoders silently discard unknown fields

**Classification:** WARNING
**Files:** `src/biohub_tracker/manifests.py:164-187`, `src/biohub_tracker/manifests.py:216-236`, `src/biohub_tracker/manifests.py:270-288`, `src/biohub_tracker/ledger.py:165-181`

**Issue:** The manifest root, `evidence`, sample, and fold decoders read required fields but never require an exact key set. `ExperimentEvent.from_dict` similarly checks only that required fields are a subset. Unknown v1 fields are silently dropped before semantic hashes and event hashes are recomputed. This permits two different on-disk documents to validate as the same normalized object, undermines frozen-schema drift detection, and can hide producer/tool disagreement about which fields are authoritative.

**Fix:** Define exact allowed-key sets for each versioned object and reject unknown or missing keys before constructing dataclasses. Require `manifest.evidence` to contain exactly `created_at`; require event root keys to match the v1 schema exactly. If forward-compatible extensions are intended, place them in an explicitly hashed extension object and advance the schema version. Add one unknown-field rejection test at every nesting level.

---

_Reviewed: 2026-08-26T00:49:04Z_
_Reviewer: the agent (gsd-code-reviewer, generic-agent role workaround)_
_Depth: standard_
