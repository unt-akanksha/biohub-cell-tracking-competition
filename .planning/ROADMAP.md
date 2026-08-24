# Roadmap: Biohub Cell Tracking Competition

## Overview

The roadmap first makes competition work safe and measurable, then establishes leakage-resistant exact validation, promotes or retires the strongest existing affinity lead, trains a stronger spatiotemporal detector/linker within the guarded Kaggle budget, and finishes with an offline reproducible submission and robust final-candidate selection. Each phase is a vertical slice that leaves a usable competition capability.

## Phases

**Phase Numbering:**

- Integer phases are planned milestone work.
- Decimal phases are urgent insertions and are marked `INSERTED`.

- [x] **Phase 1: Competition Control Plane** - Make every session, experiment, and GPU launch observable, reproducible, and quota-safe. (completed 2026-08-23)
- [ ] **Phase 2: Exact Generalization Validation** - Make clean complete-movie, held-out-embryo evidence the authoritative promotion signal.
- [ ] **Phase 3: ZebraHub Affinity Promotion Gate** - Calibrate and exactly evaluate the strongest unfinished prior model as the first end-to-end candidate.
- [ ] **Phase 4: Strong Spatiotemporal Model** - Train reciprocal higher-capacity detector/linker folds within the guarded Kaggle budget.
- [ ] **Phase 5: Offline Submission and Final Selection** - Package, audit, submit, and select reproducible clean final candidates.

## Phase Details

### Phase 1: Competition Control Plane

**Goal:** The competitor can refresh live intelligence, register evidence, and launch only resource-safe Kaggle jobs from one auditable workspace.
**Mode:** mvp
**Depends on:** Nothing (first phase)
**Requirements:** INTEL-01, INTEL-02, INTEL-03, INTEL-04, TRACK-01, TRACK-02, TRACK-03, TRACK-04, SAFE-01, SAFE-02, SAFE-03, SAFE-04, SAFE-05
**Success Criteria** (what must be TRUE):

  1. The competitor can run one command and receive a timestamped status report covering live quota, personal score/rank, submission allowance, notebook provenance, discussion/rule changes, active hypothesis, and next gate.
  2. The competitor can register, finish, fail, reject, amend, and promote immutable experiments while preserving hashes, lineage, metrics, runtime, quota use, and artifact references.
  3. A launch with malformed quota, a conflicting GPU job, no declared maximum, or a worst case that crosses the 8-hour reserve is rejected before Kaggle execution.
  4. Long jobs cannot start until smoke, dense-memory, and coverage preflights pass, and launched notebooks stop and checkpoint before their declared budget.

**Plans:** 3/3 plans complete

Plans:

- [x] 01-01-PLAN.md
- [x] 01-02-PLAN.md
- [x] 01-03-PLAN.md

**Wave 1**

- [x] 01-01: Build competition watch, provenance classifier, snapshot schema, and session status report.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02: Build immutable experiment ledger, lineage/progress projections, and prior-experiment seed records.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03: Build quota/active-run launch guard, notebook watchdog contract, and smoke/dense/coverage preflights.

### Phase 2: Exact Generalization Validation

**Goal:** The competitor can decide whether a model generalizes using pinned official scoring, overlap-safe folds, and diagnostic decomposition.
**Mode:** mvp
**Depends on:** Phase 1
**Requirements:** VAL-01, VAL-02, VAL-03, VAL-04, VAL-05
**Success Criteria** (what must be TRUE):

  1. The patched official scorer is pinned and regression tests detect the old division exploit, coordinate/penalty mistakes, aggregation errors, and CSV/GEFF drift.
  2. Frozen leave-one-embryo-out manifests are hashed into runs and prevent overlapping source material from crossing the training/evaluation boundary.
  3. A candidate receives one exact report with pooled, embryo, fold, worst-movie, node, edge, division, endpoint-availability, oracle-link, and conditional-link evidence.
  4. Promotion is a deterministic policy decision whose inputs and any approved trade-off are persisted in the ledger.

**Plans:** 3 plans

Plans:

- [ ] 02-01: Pin official baseline/metric and build scorer, topology, coordinate, and round-trip regression fixtures.
- [ ] 02-02: Build overlap-aware embryo manifests and complete-movie evaluation runner.
- [ ] 02-03: Add error decomposition, bootstrap/worst-case reporting, and executable promotion policy.

### Phase 3: ZebraHub Affinity Promotion Gate

**Goal:** The existing selective-SSM lead is either promoted with bilateral exact graph evidence or retired with a reusable explanation.
**Mode:** mvp
**Depends on:** Phase 2
**Requirements:** MODEL-01
**Success Criteria** (what must be TRUE):

  1. The three-seed `selective_ssm_medium` artifacts, manifests, hashes, and maturation evidence are imported without using public output as a training target.
  2. Calibration is selected in both embryo directions and evaluated on complete competition graphs with the Phase 2 report.
  3. The candidate receives a reproducible promote/retire decision before any larger affinity pretraining or ensemble is attempted.

**Plans:** 2 plans

Plans:

- [ ] 03-01: Import and verify ZebraHub artifacts, build reciprocal calibration, and pass bounded smoke/preflight gates.
- [ ] 03-02: Run exact complete-graph OOF, attribute changes, and record the promotion or retirement decision.

### Phase 4: Strong Spatiotemporal Model

**Goal:** The competitor has at least one individually promotable learned model that improves detection and association beyond the shared public checkpoint family.
**Mode:** mvp
**Depends on:** Phase 3
**Requirements:** MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, MODEL-07
**Success Criteria** (what must be TRUE):

  1. A baseline and higher-capacity candidate produce comparable examples-per-second, peak-memory, dense-movie, and one-batch learning evidence before fold training.
  2. Training is mixed-precision, resumable, sparse-label-aware, instrumented, and includes predicted candidates, hard negatives, divisions, and documented temporal-artifact augmentation.
  3. Sequential reciprocal embryo folds complete without crossing the 8-hour reserve and each receives exact complete-movie evaluation.
  4. At least one individual model clears promotion, or the phase records a bounded negative result that identifies the next cloud-GPU hypothesis without spending the reserve.
  5. Any ensemble, TTA, or repair is separately attributed and cannot hide a weak individual checkpoint.

**Plans:** 4 plans

Plans:

- [ ] 04-01: Profile the official baseline data/model path and select a capacity candidate under T4 throughput and memory gates.
- [ ] 04-02: Implement the AMP multi-head detector/affinity/division training path, temporal augmentations, and predicted-node association sampling.
- [ ] 04-03: Train and exactly evaluate the two sequential leave-one-embryo-out folds under quota guard.
- [ ] 04-04: Attribute optional calibration, ensemble, TTA, and conservative state-guard additions; promote only clean gains.

### Phase 5: Offline Submission and Final Selection

**Goal:** The competitor can create, submit, reproduce, and choose clean final candidates using both robustness and diversity evidence.
**Mode:** mvp
**Depends on:** Phase 4
**Requirements:** SHIP-01, SHIP-02, SHIP-03, SHIP-04, SHIP-05
**Success Criteria** (what must be TRUE):

  1. A promoted model runs in an Internet-disabled Kaggle notebook under 12 hours and produces a complete, finite, structurally valid `submission.csv`.
  2. Every submitted version is linked to exact local evidence, notebook/weight versions, hashes, runtime, description, public result, and no-hack audit.
  3. Daily and final-slot limits are enforced and exploratory submissions cannot silently become final selections.
  4. Final selection weighs held-out robustness, failure correlation, and model diversity alongside public score.
  5. A clean environment can reproduce training provenance and offline inference from the documented bundle.

**Plans:** 3 plans

Plans:

- [ ] 05-01: Build offline inference notebook, Kaggle weight dataset flow, watchdog, and exhaustive output validation.
- [ ] 05-02: Integrate submission ledger, daily/final-slot policy, and post-submission result capture.
- [ ] 05-03: Produce reproducibility bundle and select up to two complementary final candidates.

## Progress

**Execution Order:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Competition Control Plane | 3/3 | Complete   | 2026-08-23 |
| 2. Exact Generalization Validation | 0/3 | Not started | - |
| 3. ZebraHub Affinity Promotion Gate | 0/2 | Not started | - |
| 4. Strong Spatiotemporal Model | 0/4 | Not started | - |
| 5. Offline Submission and Final Selection | 0/3 | Not started | - |
