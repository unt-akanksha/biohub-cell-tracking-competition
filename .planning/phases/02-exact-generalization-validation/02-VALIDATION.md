---
phase: 2
slug: exact-generalization-validation
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-24
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Phase 2 is CPU-only.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `python -m pytest -q tests/test_scorer.py tests/test_manifests.py tests/test_graphs.py tests/test_submission_io.py tests/test_evaluation.py tests/test_ledger.py tests/test_promotion.py tests/test_exact_cli.py` |
| **Full suite command** | `python -m pytest -q` |
| **Estimated runtime** | under 30 seconds on synthetic fixtures |

---

## Sampling Rate

- **After every task commit:** Run the task's targeted pytest command.
- **After every plan wave:** Run `python -m pytest -q` and `python -m compileall -q src tests`.
- **Before phase verification:** Full suite, CLI smoke, source-lock verification, and `git diff --check` must be green.
- **Max feedback latency:** 30 seconds for fixture-only tests; the live official-data CPU control is a separate phase-close gate and may run for hours under its watchdog.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | VAL-01 | T-02-01 | Wrong commit, source hash, dependency, or imported module path fails closed | integration | `python -m pytest -q tests/test_scorer.py -k lock` | ✅ | ✅ green |
| 02-01-02 | 01 | 1 | VAL-01 | T-02-01 | Organizer ordinary-edge, penalty, division, and aggregation semantics remain pinned | unit | `python -m pytest -q tests/test_scorer.py -k 'edge or penalty or division or aggregate'` | ✅ | ✅ green |
| 02-01-03 | 01 | 1 | VAL-01 | T-02-01 | Former exploit and malformed topology cannot appear as clean evidence | regression | `python -m pytest -q tests/test_scorer.py -k 'hack or exploit or malformed'` | ✅ | ✅ green |
| 02-02-01 | 02 | 2 | VAL-02 | T-02-02 | Unknown/conflicting identity and fold overlap are rejected | unit | `python -m pytest -q tests/test_manifests.py -k 'identity or overlap or reciprocal'` | ✅ | ✅ green |
| 02-02-02 | 02 | 2 | VAL-02 | T-02-03 | Unknown/nonterminal/ineligible producer, registered-lineage or terminal-artifact mismatch, wrong fold, coverage, and mode failures block; legacy ledger/GPU behavior is unchanged | integration | `python -m pytest -q tests/test_manifests.py tests/test_graphs.py tests/test_ledger.py` | ✅ | ✅ green |
| 02-02-03 | 02 | 2 | VAL-01, VAL-02 | T-02-03 | Native subvoxel graphs remain diagnostic; submission-space CSV/GEFF projection is integral, complete, lineage-bound, and metric-equivalent | integration | `python -m pytest -q tests/test_submission_io.py` | ✅ | ✅ green |
| 02-03-01 | 03 | 3 | VAL-03, VAL-04 | T-02-04 | A legally started aggregate run with ledger-resolved role/fold producers yields exact counts and a terminal report attachment; lifecycle/member/hash drift rejects | integration | `python -m pytest -q tests/test_evaluation.py tests/test_ledger.py -k 'exact or producer or evaluation'` | ✅ | ✅ green |
| 02-03-02 | 03 | 3 | VAL-04 | T-02-04 | Endpoint/oracle/conditional/strata diagnostics reconcile with organizer counts | unit | `python -m pytest -q tests/test_diagnostics.py` | ✅ | ✅ green |
| 02-03-03 | 03 | 3 | VAL-03, VAL-05 | T-02-04 | Bootstrap and canonical report regeneration are deterministic promotion-stability inputs | regression | `python -m pytest -q tests/test_comparison.py tests/test_evaluation.py -k 'bootstrap or canonical'` | ✅ | ✅ green |
| 02-04-01 | 04 | 4 | VAL-05 | T-02-05 | Producer/aggregate/report integrity failures reject before thresholds and public score is absent from decision inputs | unit | `python -m pytest -q tests/test_promotion.py -k 'reject or public'` | ✅ | ✅ green |
| 02-04-02 | 04 | 4 | VAL-05 | T-02-05 | CPU acceptance and aggregate evaluation lifecycles are legal/backward-compatible; decisions re-resolve immutable producer evidence | integration | `python -m pytest -q tests/test_promotion.py tests/test_ledger.py -k 'promotion or review_required or decision or cpu_acceptance'` | ✅ | ✅ green |
| 02-04-03 | 04 | 4 | VAL-01, VAL-02, VAL-03, VAL-04, VAL-05 | T-02-05 | Pending remote schema cannot validate/promote; request/replay/reconciliation tests and live CPU command legally terminate local CPU/aggregate runs with no GPU/submission | end-to-end + live | `python -m pytest -q tests/test_exact_cli.py` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/fixtures/metric/` — synthetic graph, exploit, aggregation, and CSV round-trip fixtures.
- [x] `tests/test_scorer.py` — scorer lock and organizer-parity regressions.
- [x] `tests/test_manifests.py`, `tests/test_graphs.py`, `tests/test_submission_io.py` — source identity, ledger-authoritative producer rejection, native/submission modes, coverage, integrity, and projection tests.
- [x] `tests/test_evaluation.py`, `tests/test_diagnostics.py`, `tests/test_comparison.py` — aggregate multi-producer lifecycle, exact report, and paired-comparison tests.
- [x] `tests/test_ledger.py`, `tests/test_promotion.py`, `tests/test_exact_cli.py` — old-ledger compatibility, CPU acceptance transitions/accounting isolation, request reconciliation/replay, aggregate attachment, policy, and production CLI tests.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pinned organizer commit/source hashes still match the public repository | VAL-01 | Requires live network provenance | Run `biohub scorer verify --live`; compare commit and every critical file hash to `config/official-scorer.lock.json` |
| Kaggle CPU official-data control acceptance completes | VAL-01, VAL-02, VAL-03, VAL-04 | Requires the live Kaggle-mounted competition dataset and owned external CPU assets | Run `powershell -File scripts/run-phase2-cpu-acceptance.ps1 -Execute`; inspect the one-use local request, CPU registered/started/inputs-bound/completed chain, producer event/hash resolution, aggregate registered/started/completed member/report attachment, production manifest actual count/zero overlap, both folds plus union, and compact reconciliation hashes. Require accelerator `none`, unchanged GPU accounting, and no competition submission. Replay the downloaded envelope and require rejection. This is blocking for Phase 2 completion. |
| Learned model-candidate report completes | VAL-03, VAL-04, VAL-05 | Real ledger-resolved reciprocal predictions are Phase 3 artifacts | Before any Phase 3 promotion, require completed evidence-eligible ZebraHub/model producers with matching registration/terminal hashes, complete an aggregate exact evaluation, and inspect coverage, bilateral, worst-movie, node, edge, division, endpoint, oracle, conditional, and bootstrap evidence. The Phase 2 control or pending remote envelope cannot satisfy this gate. |

---

## Validation Sign-Off

- [x] All tasks have an automated verification command or Wave 0 dependency.
- [x] Sampling continuity: every task has targeted automated verification.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Fixture feedback latency is under 30 seconds.
- [x] Unknown producer, nonterminal producer, registered hash mismatch, terminal artifact mismatch, aggregate member/lifecycle mismatch, and remote-envelope replay have explicit green rejection tests.
- [x] Existing Phase 1 ledgers reconstruct/project unchanged and CPU events leave GPU quota/active-run accounting unchanged.
- [x] The live Kaggle CPU official-data control passed on kernel v13 and dataset v9; its request, CPU producer, aggregate evaluation, production manifest/report, and reconciliation hashes revalidate from the local immutable ledger.
- [x] The official-data control is labeled non-candidate/non-submittable, and learned model-candidate evaluation remains a Phase 3 gate.
- [x] `nyquist_compliant: true` was set only after live execution, reconciliation, replay rejection, and the 180-test full suite completed.

**Approval:** complete — 2026-08-25
