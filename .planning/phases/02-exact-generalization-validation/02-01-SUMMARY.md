---
phase: 02-exact-generalization-validation
plan: 01
subsystem: exact-scorer-provenance
tags: [tracking-cellmot, tracksdata, polars, official-metric, regression-fixtures]
requires:
  - phase: 01-competition-control-plane
    provides: Offline-first CLI, immutable evidence patterns, and fail-closed control-plane behavior
provides:
  - Pinned patched organizer and TracksData source/dependency authority
  - Lock-to-fixture-to-evaluate/per_sample_metrics/summarise production tracer
  - Physical matching, penalty, division, aggregation, and exploit regressions with frozen counts
affects: [02-02-manifests-and-roundtrip, 02-03-exact-reports, 02-04-promotion-policy]
tech-stack:
  added: [tracking-cellmot-0.1.0, tracksdata-0.1.0rc7, polars-1.43.2, scipy-1.18.1]
  patterns: [verified-lazy-import, source-and-environment-lock, canonical-exact-counts, test-only-old-source-isolation]
key-files:
  created:
    - config/official-scorer.lock.json
    - requirements/evaluation-lock.txt
    - scripts/materialize-official-scorer.ps1
    - src/biohub_tracker/scorer_lock.py
    - src/biohub_tracker/scorer.py
    - tests/test_scorer.py
  modified:
    - src/biohub_tracker/cli.py
    - pyproject.toml
    - .gitignore
execution-commits: [79e9670, df9e0f3, 523ac3e]
key-decisions:
  - "Only organizer commit 075fc5f and TracksData commit 39dccf3 may supply production scorer callables after source, environment, fixture, and module-path verification."
  - "Official no-division aggregation is represented as a null division Jaccard while every other required non-finite metric fails closed."
  - "The pre-patch scorer is materialized only from an isolated temporary git archive inside a regression test and has no production parser or factory route."
patterns-established:
  - "Scientific dependencies are imported only after the selected scorer command verifies every provenance boundary."
  - "Regression evidence stores integer TP/FP/FN plus canonical decimal summaries rather than rounded leaderboard anecdotes."
requirements-completed: [VAL-01]
coverage:
  - id: D-01
    description: "Exact scoring exposes callables only after fixed organizer, TracksData, dependency, source, fixture, and import-path identities verify."
    requirement: VAL-01
    verification:
      - kind: integration
        ref: "tests/test_scorer.py -k lock"
        status: pass
      - kind: e2e
        ref: "powershell -File scripts/materialize-official-scorer.ps1"
        status: pass
    human_judgment: false
  - id: D-02
    description: "The production tracer executes evaluate, node_recall, per_sample_metrics, and summarise on the perfect linear fixture and emits canonical official evidence."
    requirement: VAL-01
    verification:
      - kind: e2e
        ref: "python -m biohub_tracker scorer verify --lock config/official-scorer.lock.json --checkout .biohub/vendor/kaggle-cell-tracking-competition"
        status: pass
    human_judgment: false
  - id: D-03
    description: "Physical matching, sparse edges, node penalties, valid divisions, fresh-copy behavior, and weighted complete-split aggregation are frozen against the organizer API."
    requirement: VAL-01
    verification:
      - kind: integration
        ref: "tests/test_scorer.py -k 'edge or penalty or division or aggregate'"
        status: pass
    human_judgment: false
  - id: D-04
    description: "The patched hack2 and malformed fork boundary is exact, while isolated old-source behavior is distinguishable and unreachable from production."
    requirement: VAL-01
    verification:
      - kind: integration
        ref: "tests/test_scorer.py -k 'hack or exploit or malformed'"
        status: pass
    human_judgment: false
duration: 42min
completed: 2026-08-24
status: complete
---

# Phase 2 Plan 1: Authoritative Patched Scorer Summary

**A source-locked organizer scorer tracer with exact physical, aggregation, division, and former-exploit regression evidence**

## Performance

- **Duration:** 42 min
- **Started:** 2026-08-24T06:20:55.835Z
- **Completed:** 2026-08-24T07:02:28.984Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments

- Materialized and verified organizer commit 075fc5f5a52d11077f9dc2b074644618f26939e2, exploit patch aa65e90aeb8a774ebb1b549e547787b87ac8a01c, and TracksData commit 39dccf3a243e44274759468cb31b2ad9e7fc1d09 with a 69-package CPU evaluation lock.
- Added the lazy production tracer `lock -> perfect-linear -> evaluate -> per_sample_metrics -> summarise`, producing exact edge counts `2/0/0`, score `1`, scorer-lock hash 1db65dee620059f19bf16633aa54a9f3379eb5d5bdff148a4037b949393b7a9c, verified public-source parity, and explicitly unproven private-container parity.
- Frozen 7-micrometre physical matching, time and anisotropy, sparse-label edges, over/under node penalties, on-time/early/late divisions, macro/micro node recall names, no-division handling, and weighted aggregation.
- Frozen patched `hack2` at edge `5/4/5`, division `0/4/2`, result hash db08e3417e394b526c0e4052a2568466cb9a75b1a3660ee4cfdabf92c14a6538, and false-positive forks `105`, `106`, `122`, and `124`; fork reuse and malformed/cross-component/shared-descendant cases cannot inflate recovery.

## Task Commits

Each task was committed atomically:

1. **Task 1: Prove lock-to-fixture-to-official-summary tracer** - `79e9670` (feat)
2. **Task 2: Freeze ordinary metric and aggregation semantics** - `df9e0f3` (test)
3. **Task 3: Prove patched exploit boundary and reachability ban** - `523ac3e` (test)

## Files Created/Modified

- `config/official-scorer.lock.json` - Organizer/patch/TracksData commits, critical hashes, BSD license, constants, environment, and fixture identities.
- `requirements/evaluation-lock.txt` - Exact CPU scorer dependency resolution.
- `scripts/materialize-official-scorer.ps1` - Safe fixed-remote/fixed-commit checkout and evaluation-environment materializer.
- `src/biohub_tracker/scorer_lock.py` - Frozen lock types and fail-closed checkout, dependency, fixture, and import verification.
- `src/biohub_tracker/scorer.py` - Strict fixture builder, fresh-copy official calls, canonical summaries, and weighted sufficient statistics.
- `src/biohub_tracker/cli.py` - Lazy `biohub scorer verify` route with optional read-only live corroboration.
- `tests/fixtures/metric/` - BSD-attributed compact ordinary, aggregation, division, and exploit cases with official expected counts.
- `tests/test_scorer.py` - Lock, no-eager-import, ordinary metric, aggregation, exploit, and production-reachability coverage.

## Decisions Made

- The verified module must resolve inside a clean fixed-commit checkout; a matching distribution name or Git SHA alone is insufficient.
- The checked-in dependency file and every declared distribution version are verified before scientific imports, keeping Phase 1 CLI commands available in the stdlib-only base environment.
- Organizer `summarise` remains authoritative: adjusted edge score is weighted by edge sufficient-statistic size, division is micro-aggregated and dropped when absent, and organizer node recall stays explicitly macro while project micro recall is separately named.
- Native old scorer code is never installed or selected. Its behavior is observed only through an isolated temporary archive sourced from the organizer Git history in the regression test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Corrected PowerShell project-root initialization**
- **Found during:** Task 1 materializer execution
- **Issue:** `$PSScriptRoot` was empty while evaluating the parameter default in a nested PowerShell process.
- **Fix:** Deferred the default project-root calculation until after parameter binding.
- **Files modified:** `scripts/materialize-official-scorer.ps1`
- **Verification:** The materializer completed twice and replayed the production tracer.
- **Committed in:** `79e9670`

**2. [Rule 1 - Bug] Kept the scorer lock synchronized as expected-count coverage expanded**
- **Found during:** Tasks 2 and 3 fixture expansion
- **Issue:** The lock hashes the complete official-count evidence file, so adding planned cases invalidated the Task 1 hash even though Tasks 2/3 did not list the lock as a modified file.
- **Fix:** Refreshed the expected-file identity in each owning task and added separate perfect-linear, patched-exploit, and hack2 result identities.
- **Files modified:** `config/official-scorer.lock.json`, `src/biohub_tracker/scorer_lock.py`
- **Verification:** Tracer replay and fixture-tamper rejection tests pass after all expansions.
- **Committed in:** `df9e0f3`, `523ac3e`

**3. [Rule 1 - Bug] Preserved small source history for deterministic old-patch isolation**
- **Found during:** Task 3 isolated pre-patch regression
- **Issue:** A blob-filtered checkout could omit the historical metric source and make the safety regression depend on a second live fetch.
- **Fix:** Materialize the small organizer repository without blob filtering; tests archive only the old metric package into a temporary path that never enters production module search paths.
- **Files modified:** `scripts/materialize-official-scorer.ps1`, `tests/test_scorer.py`
- **Verification:** The isolated comparison passes offline from local Git objects, and structural reachability assertions reject an old-version route.
- **Committed in:** `523ac3e`

---

**Total deviations:** 3 auto-fixed (2 correctness bugs, 1 blocking materializer issue).
**Impact on plan:** All fixes were required to keep provenance and regression evidence deterministic; no model, Kaggle job, competition data, or submission scope was added.

## Issues Encountered

- The initially inherited development environment exposed pytest 9 while the scorer lock requires pytest 8.4.2. Reapplying `requirements/evaluation-lock.txt` installed the locked local version and all checks passed.
- Organizer `summarise` intentionally warns when no division occurs. The adapter records a null division Jaccard and a finite official score; the warning is retained rather than hidden.

## User Setup Required

None. The materializer obtains only the two public source repositories and a CPU dependency environment. It does not access Kaggle, download competition data, launch compute, or submit.

## Next Phase Readiness

- Plan 02-02 can require `VerifiedScorer` before manifest, graph, or CSV/GEFF work and carry the final scorer-lock hash into later evidence.
- VAL-01 remains globally pending until Plan 02-02 proves pinned CSV/GEFF round-trip parity; this plan completes the scorer and metric-regression portion only.
- Source parity is verified. Byte parity with the unpublished private Kaggle scorer container remains explicitly unproven.

## Self-Check: PASSED

- `34` scorer tests pass, including all lock, physical-metric, aggregation, exploit, and reachability cases.
- The full repository suite passes: `91 passed`; `python -m compileall -q src tests`, materializer replay, CLI tracer replay, and `git diff --check` also pass.
- Only the orchestrator-owned Phase 2 execution update remained dirty before this summary; ignored vendor checkouts and the evaluation environment are not tracked.
- Kaggle GPU used: `0` hours. Kaggle CPU jobs, competition downloads, submissions, and competition-asset mutations: `0`.

---
*Phase: 02-exact-generalization-validation*
*Completed: 2026-08-24*
