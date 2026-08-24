---
phase: 2
slug: exact-generalization-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| **Quick run command** | `python -m pytest -q tests/test_scorer.py tests/test_manifests.py tests/test_submission_io.py tests/test_evaluation.py tests/test_promotion.py` |
| **Full suite command** | `python -m pytest -q` |
| **Estimated runtime** | under 30 seconds on synthetic fixtures |

---

## Sampling Rate

- **After every task commit:** Run the task's targeted pytest command.
- **After every plan wave:** Run `python -m pytest -q` and `python -m compileall -q src tests`.
- **Before phase verification:** Full suite, CLI smoke, source-lock verification, and `git diff --check` must be green.
- **Max feedback latency:** 30 seconds for fixture-only tests; real complete-movie CPU evaluation is recorded separately.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | VAL-01 | T-02-01 | Wrong commit, source hash, dependency, or imported module path fails closed | integration | `python -m pytest -q tests/test_scorer.py -k lock` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | VAL-01 | Organizer ordinary-edge, penalty, division, and aggregation semantics remain pinned | unit | `python -m pytest -q tests/test_scorer.py -k 'edge or penalty or division or aggregate'` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | VAL-01 | Former exploit and malformed topology cannot appear as clean evidence | regression | `python -m pytest -q tests/test_scorer.py -k 'hack or exploit or malformed'` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | VAL-02 | Unknown/conflicting identity and fold overlap are rejected | unit | `python -m pytest -q tests/test_manifests.py -k 'identity or overlap or reciprocal'` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | VAL-02 | Missing/extra movies, invalid graphs, and stale manifest hashes fail before scoring | integration | `python -m pytest -q tests/test_manifests.py tests/test_graphs.py` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 2 | VAL-01, VAL-02 | Submission-space CSV/GEFF projection is complete and metric-equivalent | integration | `python -m pytest -q tests/test_submission_io.py` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 3 | VAL-03, VAL-04 | Identical manifest pairing yields exact pooled/embryo/fold/movie counts and scores | integration | `python -m pytest -q tests/test_evaluation.py -k exact` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 3 | VAL-03 | Endpoint/oracle/conditional/strata diagnostics reconcile with organizer counts | unit | `python -m pytest -q tests/test_diagnostics.py` | ❌ W0 | ⬜ pending |
| 02-03-03 | 03 | 3 | VAL-04 | Bootstrap and canonical report regeneration are deterministic | regression | `python -m pytest -q tests/test_comparison.py tests/test_evaluation.py -k 'bootstrap or canonical'` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 4 | VAL-05 | Integrity failures always reject and public score is absent from decision inputs | unit | `python -m pytest -q tests/test_promotion.py -k 'reject or public'` | ❌ W0 | ⬜ pending |
| 02-04-02 | 04 | 4 | VAL-05 | Promote/review/retire thresholds are versioned, deterministic, and evidence-bound | integration | `python -m pytest -q tests/test_promotion.py tests/test_ledger.py -k 'promotion or review_required or decision'` | ❌ W0 | ⬜ pending |
| 02-04-03 | 04 | 4 | VAL-03, VAL-04, VAL-05 | CPU tracer produces immutable exact report and ledger projection without GPU | end-to-end | `python -m pytest -q tests/test_exact_cli.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/fixtures/metric/` — synthetic graph, exploit, aggregation, and CSV round-trip fixtures.
- [ ] `tests/test_scorer.py` — scorer lock and organizer-parity regressions.
- [ ] `tests/test_manifests.py`, `tests/test_graphs.py`, `tests/test_submission_io.py` — identity, coverage, integrity, and projection tests.
- [ ] `tests/test_evaluation.py`, `tests/test_diagnostics.py`, `tests/test_comparison.py` — exact report and paired-comparison tests.
- [ ] `tests/test_promotion.py`, `tests/test_exact_cli.py` — policy/ledger and CPU tracer tests.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pinned organizer commit/source hashes still match the public repository | VAL-01 | Requires live network provenance | Run `biohub scorer verify --live`; compare commit and every critical file hash to `config/official-scorer.lock.json` |
| Mounted official dataset yields complete reciprocal folds | VAL-02 | No GEFF/Zarr data is local at planning time | Mount/download official train data, run `biohub manifest build`, and confirm every discovered movie appears exactly once on the reciprocal evaluation sides |
| Real baseline/candidate graph report completes | VAL-03, VAL-04 | Prediction artifacts are not local at planning time | Mount Phase 3 artifacts and run the CPU exact evaluator; inspect coverage, bilateral, worst-movie, and division diagnostics |

---

## Validation Sign-Off

- [ ] All tasks have an automated verification command or Wave 0 dependency.
- [ ] Sampling continuity: every task has targeted automated verification.
- [ ] Wave 0 covers all missing references.
- [ ] No watch-mode flags.
- [ ] Fixture feedback latency is under 30 seconds.
- [ ] Real-data checks remain explicitly incomplete until artifacts are mounted; synthetic passage cannot masquerade as a model promotion.
- [ ] `nyquist_compliant: true` is set only after execution evidence is complete.

**Approval:** pending
