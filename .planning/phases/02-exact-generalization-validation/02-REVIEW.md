---
phase: 02-exact-generalization-validation
reviewed: "2026-08-25T23:35:20Z"
depth: standard
diff_base: "79e9670^"
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
  critical: 6
  warning: 7
  info: 0
  total: 13
status: issues_found
---

# Phase 2 Code Review

## Summary

Phase 2 is not ready to close. The exact-evidence boundary has several strong components, but the CPU reconciliation path is neither recoverable nor independently authoritative, the manifest builder can hash data outside the mounted source, the ledger can persist an event that makes itself unreadable, and the production promotion path silently passes a missing metric-exploit audit. The checked-in test suite is also currently red.

Validation performed:

- `.biohub/evaluation-venv/Scripts/python.exe -m pytest -q`: **179 passed, 1 failed**. The failure is `tests/test_manifests.py:165`.
- `git diff --check 79e9670^ -- <all 43 reviewed files>`: passed.
- A focused ledger reproduction appended a backdated terminal event successfully; the immediately following `read_events()` failed with `TransitionError: run r has a terminal event before/after running`.
- Ruff was not available in the evaluation environment (`No module named ruff`), so no Ruff result is claimed.

## Critical Issues

### CR-01 — Reconciliation consumes the one-use acceptance request before validation and cannot resume

**Files/lines:** `src/biohub_tracker/acceptance.py:867-877`, `src/biohub_tracker/acceptance.py:933-979`, `src/biohub_tracker/acceptance.py:995-1079`, `src/biohub_tracker/acceptance.py:1121-1126`

`reconcile_pending_control` only accepts a CPU lifecycle in `RUNNING`. It then appends `CPU_ACCEPTANCE_INPUTS_BOUND`, `CPU_ACCEPTANCE_COMPLETED`, exact registration, exact start, and exact completion as separate durable writes before final report validation and before checking whether the immutable output paths already exist. Any exception or process termination after the first append strands the request in `INPUTS_BOUND`, `COMPLETED`, or a partially completed exact lifecycle. A retry then fails with `ACCEPTANCE_REQUEST_NOT_STARTED` or `REQUEST_ALREADY_CONSUMED`. Even the ordinary case of an existing output file is detected only after both lifecycles have been terminally committed.

**Concrete fix:** Turn reconciliation into a state-aware, idempotent saga. Validate all pending content and output targets before the first transition; on retry, accept and verify `RUNNING`, `INPUTS_BOUND`, and compatible `COMPLETED` states; deterministically recreate and compare each expected event; append only missing events. Publish validated artifacts before the irreversible terminal transition, or add an explicit materialization transition. Prefer a ledger batch append under one lock for transitions that must be atomic. Add crash-injection tests after every append and before each artifact write.

### CR-02 — Remote “untrusted” control evidence is self-authenticated and never independently revalidated

**Files/lines:** `src/biohub_tracker/acceptance.py:170-276`, `src/biohub_tracker/acceptance.py:943-1079`, `src/biohub_tracker/evaluation.py:613-670`

`validate_pending_control` applies a strict schema to the envelope but only checks that `control` is a dictionary and that the overall tree is finite. Its payload, envelope, and inventory hashes are all computed by the same remote producer and have no trusted signature or locally recomputed semantic counterpart. Reconciliation then copies remote `official`, `diagnostics`, `comparison`, inventories, expected sample IDs, artifact hashes, and evaluation-policy hash directly into locally branded exact evidence. `validate_exact_core` checks coverage markers and authority labels but does not reconcile official sufficient statistics, diagnostic totals, comparison deltas/bootstrap values, or inventory hashes. An altered remote result can therefore recompute its own three hashes and be accepted as a locally reconciled official-data control.

**Concrete fix:** Define and enforce an exact `control` schema, bind its policy hash to the locally loaded policy, and locally recompute all official summaries, paired deltas, bootstrap results, diagnostic reconciliations, and inventory/member bindings from independently validated compact sufficient statistics. If those statistics are not sufficient for independent reproduction, retrieve content-addressed graph artifacts and run the pure evaluator locally. Do not append CPU or exact terminal events until this revalidation passes. Add tamper tests that change counts, diagnostics, comparison values, and inventories while recomputing all remote self-hashes.

### CR-03 — Manifest hashing follows child symlinks outside the official data root

**Files/lines:** `src/biohub_tracker/manifests.py:72-95`, `src/biohub_tracker/manifests.py:446-466`

Containment is checked only for the top-level `.geff` path. `_tree_sha256` subsequently enumerates descendants and hashes each `path` without resolving it or checking it against the tree root. A file symlink inside a `.geff` directory can therefore point outside the official mount and have external bytes incorporated into the canonical manifest. This violates the required path-escape rejection and can make source identity depend on unauthorized or mutable files. `graphs.artifact_tree_sha256` already performs the missing per-child resolve-and-containment check.

**Concrete fix:** For every descendant, use `lstat`, reject symlinks/reparse points and non-regular files, resolve the file, and require `resolved.relative_to(directory.resolve(strict=True))` before hashing. Apply the same rule to metadata files. Add Linux/Kaggle tests for file and directory symlink escapes and assert that manifest construction fails with a stable reason code.

### CR-04 — `Ledger.append` can durably write a transition that makes the ledger unreadable

**Files/lines:** `src/biohub_tracker/ledger.py:306-329`, `src/biohub_tracker/ledger.py:332-383`, `src/biohub_tracker/ledger.py:668-705`, `src/biohub_tracker/ledger.py:826-836`

Append validation reconstructs only the existing state. Reconstruction later sorts all events by caller-controlled `created_at`, but `append` never reconstructs the candidate ledger including the new event. A terminal event with a timestamp before its start therefore passes validation against the current `RUNNING` state, is written and fsynced, and makes every later read fail because reconstruction processes the terminal before the start. Clock rollback is enough to trigger this; no malformed JSON is required. This was reproduced during review.

**Concrete fix:** While holding the lock, validate the full candidate sequence before writing, for example by reconstructing `events + [event]` through all lifecycle projections or by decoding `existing_bytes + payload` with quarantine disabled. Also enforce monotonic append metadata (or introduce an append ordinal/hash chain) so semantic order cannot move behind durable history. Add backdated start, terminal, decision, CPU, and exact-event tests that assert bytes remain unchanged on rejection.

### CR-05 — The production metric-exploit hard gate is absent and absence is treated as success

**Files/lines:** `src/biohub_tracker/evaluation.py:597-606`, `src/biohub_tracker/promotion.py:305-307`, `config/promotion-policy.json:33-58`

`canonical_report_core` never emits `metric_exploit_audit`. The promotion policy does not require it, and `_hard_failures` explicitly accepts either `None` or `"passed"`. Consequently every production-generated report reaches the named metric-exploit hard gate with no affirmative audit evidence and silently passes. The tests only exercise the gate by manually injecting a field that production does not produce. This is fail-open behavior at the competition’s most important integrity boundary.

**Concrete fix:** Generate an explicit audit result from the patched-scorer identity, exploit regression fixtures, graph integrity checks, and any submission-specific exploit signatures. Add `metric_exploit_audit: passed` to `required_integrity`, require the exact field in report validation, and change promotion to reject missing as well as failed values. Recompute the frozen policy hash and add a test proving a production-shaped core without the field is rejected.

### CR-06 — The checked-in Phase 2 test suite cannot pass with the tracked accepted manifest

**File/line:** `tests/test_manifests.py:165`

The test asserts that `manifests/reciprocal-embryo-v1.json` does not exist. That accepted official-data manifest is now tracked, so the assertion deterministically fails in the current repository. The full review run ended with 1 failure and 179 passes. A red suite invalidates the phase’s completion claim and masks regressions in later workflows.

**Concrete fix:** Preserve the original intent without asserting global absence: hash/read the tracked manifest before the temporary CLI build, write the fixture manifest only under `tmp_path`, and assert afterward that the tracked manifest is byte-identical. If the intended policy is that no real manifest may be tracked, remove it and update the Phase 2 evidence design consistently; do not leave code, evidence, and test expectations contradictory.

## Warnings

### WR-01 — Review exceptions are not bound to the gates that actually failed

**Files/lines:** `src/biohub_tracker/promotion.py:556-596`, `src/biohub_tracker/ledger.py:1361-1408`, `src/biohub_tracker/ledger.py:1683-1723`

An exception for a `review_required` decision accepts any non-empty `failed_gates` mapping and any non-empty downstream authorization string. Neither recording nor transition validation requires the gate keys to equal the immutable decision’s `reason_codes`, and the submitted values are not verified against the decision inputs. A caller can therefore document `{"unrelated": 0}` while omitting the real node-recall, division, or worst-movie regression.

**Concrete fix:** Derive failed gates server-side. Require an exact key-set match with `reason_codes`, bind canonical observed values and thresholds into the decision event (or load and verify the hashed decision inputs), and restrict downstream authorization to a closed enum. Add omission, substitution, wrong-value, and extra-gate rejection tests.

### WR-02 — Division timing diagnostics pair forks by time instead of organizer correspondence

**File/lines:** `src/biohub_tracker/diagnostics.py:253-293`

Recovered truth divisions and true-positive predicted forks are cross-joined, then greedily paired by the smallest absolute frame offset. The available `prediction_to_truth` correspondence is ignored for this step. With multiple nearby divisions, a predicted fork can be assigned to a different truth division, producing incorrect early/on-time/late diagnostic counts while all TP/FP/FN reconciliation checks still pass.

**Concrete fix:** Use the organizer scorer’s explicit division pairing if available; otherwise derive each TP fork’s truth division through the validated node match/descendant correspondence and reject missing or duplicate assignments. Add a two-division fixture where temporal greedy matching swaps the true pairs.

### WR-03 — Pooled, embryo, and fold diagnostic comparisons silently omit core diagnostic metrics

**File/lines:** `src/biohub_tracker/comparison.py:129-159`, `src/biohub_tracker/comparison.py:202-248`, `src/biohub_tracker/comparison.py:571-574`, `src/biohub_tracker/comparison.py:615-620`

Movie-level diagnostics contain conditional association recall, conditional valid-edge precision/Jaccard, node-count ratio, and oracle gap. `_pooled_diagnostic` retains only endpoint availability and strata, while `_diagnostic_delta` treats all other scalars as optional. The aggregate, by-embryo, and by-fold reports therefore silently drop those diagnostic deltas instead of supplying the complete generalization comparison promised by the evaluation design.

**Concrete fix:** Pool the stored numerators and denominators for all conditional ratios, aggregate node counts rather than averaging ratios, and recompute oracle components/gaps from sufficient statistics. Make required diagnostic fields explicit per comparison level and add tests that assert they appear in pooled, embryo, fold, and movie outputs.

### WR-04 — Exact report publication has an unrecoverable process-crash window

**File/lines:** `src/biohub_tracker/evaluation.py:718-727`, `src/biohub_tracker/evaluation.py:853-878`, `src/biohub_tracker/evaluation.py:892-920`

The output directory is renamed into its immutable final location before the exact completion event is appended. Python exceptions are cleaned up, but a process kill or machine loss between those operations leaves a final output directory plus a `RUNNING` ledger state. A retry refuses the existing directory before it can reconcile or complete the lifecycle.

**Concrete fix:** Make publication resumable: when a matching exact evaluation is running and output exists, verify both report files and their hashes, then append the missing completion event. Alternatively, add a durable materialized event and deterministic recovery protocol. Add kill-point tests immediately before and after the rename and completion append.

### WR-05 — A crashed writer leaves a permanent ledger lock

**File/lines:** `src/biohub_tracker/ledger.py:745-766`

The exclusive lock records a PID but never validates it or its age. If the writer dies before `__exit__`, all future appends time out forever until a human deletes the lock file, which undermines autonomous experiment tracking and recovery.

**Concrete fix:** Record PID, process-start identity, creation time, and a random ownership token. After timeout, recover only a demonstrably stale lock (dead PID/start mismatch and minimum age) through an atomic quarantine/replace operation; never steal an active lock. Test active-lock refusal and stale-lock recovery.

### WR-06 — Run identity uniqueness is asymmetric across CPU and exact lifecycles

**File/lines:** `src/biohub_tracker/ledger.py:1411-1421`, `src/biohub_tracker/ledger.py:1501-1550`

Exact registration rejects collisions with experiments and exact evaluations but does not check CPU acceptance IDs. CPU registration checks its own run ID across all three families, but only checks a proposed `evaluation_run_id` against already-created exact evaluations, not against other pending CPU registrations. This permits CPU/exact ID collisions when CPU is registered first and permits multiple pending controls to reserve the same future exact evaluation ID; the later reconciliation then fails after earlier irreversible events.

**Concrete fix:** Enforce a single global run-ID namespace in every registration direction and reserve proposed evaluation IDs across all nonterminal and terminal CPU registrations. Add symmetric collision tests and two-pending-control reservation tests.

### WR-07 — `cpu-acceptance register --request` cannot parse the request emitted by the canonical registration path

**Files/lines:** `src/biohub_tracker/acceptance.py:750-807`, `src/biohub_tracker/cli.py:632-657`

`issue_acceptance_request` writes source hashes under `source_identities`. The `--request` branch reads `scorer_lock_sha256`, `environment_lock_sha256`, and the other identity hashes from the request root, so the canonical emitted document supplies `None` and fails payload validation. The public CLI option therefore does not consume the project’s own acceptance-request schema.

**Concrete fix:** Parse and strictly validate the request schema, pull hashes from `request["source_identities"]`, independently recompute `acceptance_request_sha256`, and verify `registration_event_sha256` semantics. If importing already-registered requests is not a supported operation, remove the option instead of exposing a broken path. Add a round-trip test from `issue_acceptance_request` output into the CLI branch.

---

_Reviewer: gsd-code-reviewer (generic-agent role workaround), standard depth, read-only source review._
