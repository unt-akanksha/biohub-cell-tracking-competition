---
phase: 02-exact-generalization-validation
plan: 04
subsystem: promotion-and-live-cpu-acceptance
tags: [kaggle, cpu-only, exact-evaluation, immutable-ledger, promotion-policy, offline-runtime]
requires:
  - phase: 01-competition-control-plane
    provides: Immutable experiment tracking, read-only Kaggle intelligence, and guarded GPU accounting
  - phase: 02-01
    provides: Pinned patched organizer scorer and dependency/source verification
  - phase: 02-02
    provides: Reciprocal production manifests, producer evidence, and authoritative integer round trips
  - phase: 02-03
    provides: Aggregate exact-evaluation lifecycle, diagnostics, and paired complete-movie comparison
provides:
  - Frozen hard-gate-first promotion policy that excludes public leaderboard metadata
  - Separate CPU acceptance and aggregate exact-evaluation ledger lifecycles
  - Reconciled official-data truth/self control for both reciprocal folds plus union
  - Production reciprocal manifest and accepted exact control report from a CPU-only Kaggle kernel
affects: [03-zebrahub-affinity-promotion-gate, 04-strong-spatiotemporal-model, 05-offline-submission]
tech-stack:
  added: []
  patterns: [request-bound-remote-evidence, local-authority-reconciliation, immutable-offline-runtime, hard-gates-before-scientific-gates]
key-files:
  created:
    - config/promotion-policy.json
    - config/phase2-control.json
    - src/biohub_tracker/promotion.py
    - src/biohub_tracker/acceptance.py
    - scripts/run-phase2-cpu-acceptance.ps1
    - kaggle/phase2-cpu-acceptance/phase2_acceptance.py
    - manifests/reciprocal-embryo-v1.json
    - reports/exact/phase2-control-acceptance.json
  modified:
    - src/biohub_tracker/ledger.py
    - src/biohub_tracker/evaluation.py
    - src/biohub_tracker/cli.py
    - experiments/events.jsonl
    - reports/PROGRESS.md
execution-commits: [81038f3, 1e1cc07, 8b1d918]
key-decisions:
  - "Public leaderboard and notebook scores are structurally excluded from canonical promotion inputs; hard integrity gates always run before scientific thresholds."
  - "Remote Kaggle output remains pending and untrusted until a one-use local request is reconciled into legal CPU and aggregate terminal events."
  - "The official truth/self control proves the evaluation infrastructure only and is permanently non-promotable and non-submittable."
  - "Kaggle CPU work uses one immutable private runtime dataset version per retry and publishes only a bounded final output tree."
patterns-established:
  - "Normalize transport-only CRLF and file-mode differences while preserving commit, ancestry, critical source, dependency, and package-version verification."
  - "Discover Kaggle datasets and competitions only through bounded current and legacy mount namespaces."
  - "Every failed live attempt receives a terminal ledger event; a successful request nonce cannot be replayed."
requirements-completed: [VAL-01, VAL-02, VAL-03, VAL-04, VAL-05]
coverage:
  - id: D-01
    description: "Promotion decisions are deterministic, hard-gate-first, producer/aggregate/report resolved, and invariant to public score metadata."
    requirement: VAL-05
    verification:
      - kind: integration
        ref: "tests/test_promotion.py tests/test_ledger.py"
        status: pass
    human_judgment: false
  - id: D-02
    description: "The live CPU-only Kaggle control produced a production reciprocal manifest and exact truth/self report for both folds plus union."
    requirement: VAL-03
    verification:
      - kind: e2e
        ref: "indarkarhana/biohub-phase-2-cpu-acceptance/13"
        status: pass
      - kind: integration
        ref: "reports/exact/phase2-control-acceptance.json"
        status: pass
    human_judgment: false
  - id: D-03
    description: "Local reconciliation legally completed the CPU producer and aggregate evaluation with a one-use request and immutable event/report hashes."
    requirement: VAL-04
    verification:
      - kind: e2e
        ref: "cpu phase2-cpu-control-20260825T224606Z and aggregate phase2-control-evaluation-20260825T224606Z"
        status: pass
      - kind: manual_procedural
        ref: "replay returned REQUEST_ALREADY_CONSUMED with ledger length unchanged at 75 events"
        status: pass
    human_judgment: false
  - id: D-04
    description: "The control used no accelerator, changed no GPU quota, invoked no submission path, and cannot authorize promotion or submission."
    requirement: VAL-05
    verification:
      - kind: e2e
        ref: "phase2-control-acceptance gpu_quota_before/after: 30.00h remaining, 0.00h used"
        status: pass
      - kind: unit
        ref: "tests/test_exact_cli.py#CPU metadata and submission/GPU tripwires"
        status: pass
    human_judgment: false
duration: 37h32m wall-clock including diagnosed retries; 26m45s successful remote runtime
completed: 2026-08-25
status: complete
---

# Phase 2 Plan 4: Promotion Policy and Official-Data CPU Control Summary

**Hard-gated promotion and a locally reconciled CPU-only official-data control now make complete-movie reciprocal evidence authoritative without trusting remote output, public scores, or metric hacks**

## Performance

- **Duration:** 37h32m wall-clock across autonomous diagnosis/retries; successful Kaggle CPU runtime 1605.24s
- **Started:** 2026-08-24T09:43:59Z
- **Completed:** 2026-08-25T23:15:03Z
- **Tasks:** 3
- **Task 3 files committed:** 27
- **Kaggle GPU:** 0.00h used / 30.00h remaining

## Accomplishments

- Froze promotion policy `911f8e8c12070e29fc9a26119a184347db3f7fcc316093068f5c93adaa8f6d28` with non-overridable integrity rejection, explicit review-required scientific trade-offs, and no public-score input route.
- Added backward-compatible CPU acceptance and aggregate exact lifecycles whose producer registration, input binding, terminal artifact, member, report, and policy hashes re-resolve from the append-only ledger.
- Completed private Kaggle kernel `indarkarhana/biohub-phase-2-cpu-acceptance/13` against immutable dataset `indarkarhana/biohub-phase2-runtime/9`, accelerator `none`, internet disabled, both reciprocal folds plus union, and no competition submission.
- Published production manifest `7d6fd819f63a0a3ac4513c501751d5ccd91900b266d7ab92f5f13bb14198b726` and locally issued exact core `e395282b35ce226ea27188db9654aab15b549fb9311a8d5478ed8e91e1c5125a`.

## Task Commits

1. **Task 1: Freeze hard rejection and public-score-independent policy inputs** — `81038f3`
2. **Task 2: Add CPU/aggregate evidence lifecycles and append exact decisions** — `1e1cc07`
3. **Task 3: Execute and reconcile the CPU-only Kaggle official-data control** — `8b1d918`

## Live Evidence

- CPU run: `phase2-cpu-control-20260825T224606Z`
- Aggregate run: `phase2-control-evaluation-20260825T224606Z`
- CPU event hashes: registered `2cf94342...e162a`, inputs-bound `8a7fd24c...5444b`, completed `7aef03f4...a1ea1`
- Aggregate event hashes: registered `ccffef96...01c04`, started `16c226a6...14d8f`, completed `8148c9a6...49c7`
- Pending payload/envelope: `ef69a095...15e3` / `c3480958...e4a7`
- Local reconciliation: `ea25cf11dd8a995da8b21cf7917eaa613d398c107db28bcaec5bc0618abd56bc`
- Final report core/envelope: `e395282b...125a` / `beac1393...ac3e`
- Runtime bundle/inventory: `235fc0a0...898b` / `94d1c272...83c3`, 8,801 files
- GPU quota evidence: `3cdb6bd7...9005`; before and after both show 30.00h remaining and 0.00h used
- Replay check: rejected with `REQUEST_ALREADY_CONSUMED`; ledger remained 75 events

## Files Created/Modified

- `src/biohub_tracker/acceptance.py` — secure offline bundle build/extract, pending-control validation, request creation, and local reconciliation.
- `kaggle/phase2-cpu-acceptance/phase2_acceptance.py` — bounded CPU-only mount discovery, official truth/self evaluation, and compact output cleanup.
- `scripts/run-phase2-cpu-acceptance.ps1` — owned-asset versioning, CPU launch, SDK version/source verification, guarded polling/retry, quota comparison, and reconciliation.
- `src/biohub_tracker/evaluation.py`, `src/biohub_tracker/ledger.py`, `src/biohub_tracker/cli.py` — pending-control evaluation and legal CPU/aggregate command lifecycles.
- `manifests/reciprocal-embryo-v1.json` — production official-data reciprocal fold manifest.
- `reports/exact/phase2-control-acceptance.json` — accepted non-candidate control with full exact evidence and local hashes.
- `tests/test_exact_cli.py` and Phase 2 regression files — CPU metadata, bundle security, mount layouts, transport normalization, tripwires, and lifecycle coverage.

## Decisions Made

- Casefold image OME-Zarr axis names because current competition images publish uppercase `T,Z,Y,X`; GEFF and all shape/scale/dtype/chunk constraints remain strict.
- Treat CRLF and executable-bit differences as transport normalization only. Git commit, remote, ancestry, critical raw source hashes, locked dependencies, and imported module paths remain fail-closed.
- Give Kaggle CLI `kernels push -t` the watchdog in seconds (`39600`) rather than minutes; `30` had caused server cancellation after 30 seconds.
- Remove extracted runtime and scorer work trees before kernel completion so Kaggle indexes only the compact pending report.

## Deviations from Plan

### Auto-fixed Issues

1. **Current Kaggle mount namespaces** — Added bounded support for `/kaggle/input/datasets/<owner>/<dataset>` and `/kaggle/input/competitions/<competition>`.
2. **Windows archive replacement** — Added validated copy fallback when antivirus/file handles deny directory replacement.
3. **Official uppercase axes** — Normalized image axis case while retaining strict semantic validation.
4. **Cross-platform source transport** — Normalized CRLF/filemode-only Git status differences without weakening provenance checks.
5. **Missing metric fixtures** — Included the five locked official scorer fixtures in the offline runtime allowlist.
6. **Live policy loader name** — Corrected the remote-only call to `load_evaluation_policy` and added source regression coverage.
7. **Kaggle timeout unit** — Changed the launch timeout from 30 seconds to the 660-minute CPU watchdog expressed as seconds.
8. **Output API overload** — Cleaned 1.08GB generated runtime trees before completion and added bounded 429 read retries/cooldown.

**Impact on plan:** Every deviation was required to make the planned fail-closed CPU control work against the current Kaggle runtime while preserving its security, accelerator, and submission boundaries.

## Issues Encountered

- Twelve earlier live lifecycles were preserved as legal failures rather than erased. They exposed mount layout, Windows extraction, metadata case, Git portability, missing fixture, loader naming, timeout-unit, and output-indexing defects.
- Kernel v12 computed successfully but its oversized working tree caused Kaggle output listing to return HTTP 429. Kernel v13 produced the same official-data control after bounded cleanup and reconciled successfully.

## Verification

- Full suite: `180 passed`.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.
- Live scorer provenance: organizer and TracksData verified; pinned fixture output matched.
- Live acceptance: kernel v13 COMPLETE, accepted report, accelerator `none`, internet/GPU/TPU false, quota unchanged, no submission.
- Replay: rejected without a new event or accepted artifact.

## User Setup Required

None. Kaggle authentication was already available and all owned-asset repair, versioning, execution, retrieval, and reconciliation completed autonomously.

## Next Phase Readiness

- Phase 3 can now import and evaluate real ZebraHub `selective_ssm_medium` reciprocal producer artifacts against the production manifest and exact scorer.
- The official-data truth/self control is infrastructure evidence only. It cannot satisfy the learned-candidate gate and is explicitly unauthorized for promotion and submission.
- Kaggle GPU remains fully available: 30.00h remaining, with the protected final 8.00h reserve intact.

---
*Phase: 02-exact-generalization-validation*
*Completed: 2026-08-25*
