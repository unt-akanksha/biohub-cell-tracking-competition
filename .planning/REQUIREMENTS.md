# Requirements: Biohub Cell Tracking Competition

**Defined:** 2026-08-23  
**Core Value:** Reliably improve the clean, private-test-generalizable Biohub tracking score without exploiting the metric or consuming the final 8 hours of Kaggle GPU quota.

## v1 Requirements

### Competition Intelligence

- [x] **INTEL-01**: The competitor can run one command that records a timestamped Kaggle snapshot containing accelerator quota, personal submissions, public leaderboard position, current high-scoring notebooks, and recent discussions.
- [x] **INTEL-02**: The competitor can see notebook score provenance classified as reproduced post-patch, reported post-patch, stale/ghost risk, explicit metric hack, or unknown.
- [x] **INTEL-03**: The project records changes to official rules, metric code, deadlines, data guidance, and organizer announcements without overwriting earlier snapshots.
- [x] **INTEL-04**: Every work session begins from a compact status report containing the best clean score, rank, remaining submission allowance, quota headroom, active hypothesis, and next gate.

### Experiment Tracking

- [ ] **TRACK-01**: The competitor can register an experiment with a stable ID, hypothesis, parent experiment, configuration, code/data/model hashes, seeds, split, and declared maximum runtime before execution.
- [ ] **TRACK-02**: The project appends actual resource use, artifact hashes, status, failure reason, full-movie metrics, per-embryo metrics, fold metrics, and worst-movie delta after execution.
- [ ] **TRACK-03**: The project preserves rejected, failed, incomplete, and superseded experiments and represents corrections as amendments rather than destructive edits.
- [ ] **TRACK-04**: The competitor can render a progress board showing experiment lineage, evidence, GPU-hour cost, promotion state, and the highest-value next experiment.

### Resource Safety

- [ ] **SAFE-01**: A Kaggle GPU launch is rejected unless live quota is parseable, no conflicting GPU job is active, and a positive worst-case runtime is declared.
- [ ] **SAFE-02**: A Kaggle GPU launch is rejected when `remaining GPU hours - declared maximum runtime < 8 hours`.
- [ ] **SAFE-03**: Every GPU notebook stops before its declared wall-clock budget, flushes a resumable checkpoint, and emits a terminal status record.
- [ ] **SAFE-04**: A long run cannot start until a short smoke test has verified imports, inputs, one batch, model forward/backward where applicable, checkpoint round trip, and output location.
- [ ] **SAFE-05**: Dense-movie memory and complete dataset-coverage preflights run before any job with a declared runtime above one hour.

### Exact Validation

- [ ] **VAL-01**: The project pins the official patched scorer and verifies it against regression fixtures covering node matching, adjusted edge penalty, valid divisions, invalid fake forks, micro-averaging, and CSV↔GEFF round trips.
- [ ] **VAL-02**: The project uses frozen leave-one-embryo-out manifests that prevent source-overlap leakage and are hashed into every training and evaluation record.
- [ ] **VAL-03**: Every promotion candidate is scored on complete movies and reports pooled adjusted edge Jaccard, edge Jaccard, division Jaccard with TP/FP/FN, node recall, per-embryo results, per-fold results, and worst-movie delta.
- [ ] **VAL-04**: The project reports detection endpoint availability, oracle-link ceiling, and conditional association accuracy so node errors are separated from linking errors.
- [ ] **VAL-05**: A default promotion gate requires positive pooled score, nonnegative node recall, bilateral embryo evidence, bounded worst-movie regression, and nonnegative division behavior unless an explicit documented trade-off is approved.

### Model Improvement

- [ ] **MODEL-01**: The existing ZebraHub `selective_ssm_medium` artifact is reciprocally calibrated on competition candidate graphs and evaluated with the exact complete-graph metric before it is promoted or retired.
- [ ] **MODEL-02**: The training pipeline uses AMP, resumable checkpoints, throughput and memory telemetry, chunk-aware image loading, and deterministic run manifests on Kaggle T4.
- [ ] **MODEL-03**: A stronger spatiotemporal model learns cell detection, subvoxel localization, temporal affinity or motion, association, and division evidence while respecting sparse/unlabeled cells.
- [ ] **MODEL-04**: Association training includes predicted detection candidates and hard negatives rather than learning only from ground-truth nodes.
- [ ] **MODEL-05**: Training augmentation covers frozen frame pairs, missing-frame gaps, abrupt global translations, acceleration, intensity/contrast changes, and spatial anisotropy.
- [ ] **MODEL-06**: Two sequential embryo-held-out fold models are trained only after the candidate architecture clears T4 examples-per-second, peak-memory, and dense-movie smoke gates.
- [ ] **MODEL-07**: Ensembling, test-time augmentation, and graph repair are attempted only after at least one individually promoted learned model exists, and each addition receives separate attribution.

### Submission and Compliance

- [ ] **SHIP-01**: A promoted inference notebook runs with Internet disabled, stays below Kaggle's 12-hour limit, loads only declared public or team-private artifacts, and writes `submission.csv`.
- [ ] **SHIP-02**: Submission validation rejects missing datasets, duplicate or noncontiguous row IDs, invalid node IDs, dangling edges, backward-time edges, malformed row types, nonfinite coordinates, and CSV↔GEFF mismatches.
- [ ] **SHIP-03**: Every submission record links the notebook version, source hash, weight dataset/version, parent experiment, local OOF evidence, runtime, public score, description, and hack-audit result.
- [ ] **SHIP-04**: The project enforces the five-submissions-per-day limit, tracks the two final-selection slots, and distinguishes exploratory submissions from final candidates.
- [ ] **SHIP-05**: Final candidates include a reproducibility bundle covering environment, data sources, licenses, architecture, preprocessing, loss, seeds, hyperparameters, training commands, inference commands, and artifact hashes.

## Definition of Done

- The live watch, ledger, quota guard, exact validation suite, and progress report work from a clean checkout.
- At least one learned model or calibrated affinity system clears the exact bilateral OOF promotion gate over the clean 0.913-family baseline.
- No Kaggle GPU launch can spend into the 8-hour reserve under its declared worst case.
- At least one clean, reproducible, structurally validated submission is created by the offline notebook and recorded in the ledger.
- Final candidate selection uses OOF robustness and experiment diversity, not public score alone.

## v2 Requirements

### Automation

- **AUTO-01**: The competitor can install an operating-system scheduler that runs the competition watch in the background and alerts only on material changes.
- **AUTO-02**: The offline ledger can optionally synchronize to a hosted ML experiment tracker when Internet access is available.
- **AUTO-03**: Cloud GPU backends can consume the same experiment manifest after Kaggle reaches the 8-hour reserve.

### Modeling

- **MODEL-08**: A large self-supervised video model can be pretrained from scratch on public unlabeled zebrafish microscopy using the cloud GPU backend.
- **MODEL-09**: More than two fold models or multiple architecture families can be trained and stacked after a single model demonstrates reliable bilateral gain.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Metric hacks or exploit discovery for submission | Violates the clean-score objective and is fragile under patches/private evaluation |
| Hand-labeling validation or test cells | Prohibited by competition rules |
| Blind cloning of a top-sorted notebook | Displayed notebook scores can be stale and most clean notebooks share the same public weights |
| Unbounded hyperparameter search | Incompatible with the 22-hour maximum current spendable quota |
| Concurrent Kaggle GPU experiments | Risks quota overspend and ambiguous attribution |
| Hosted-only experiment tracking | Submission notebooks are offline and the project must remain reproducible without an account service |
| Paid/private external datasets unavailable to competitors | Incompatible with external-data accessibility rules |
| More graph post-processing before learned-model evidence | Existing graph-patch experiments show diminishing and sometimes negative returns |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INTEL-01 | Phase 1 | Complete |
| INTEL-02 | Phase 1 | Complete |
| INTEL-03 | Phase 1 | Complete |
| INTEL-04 | Phase 1 | Complete |
| TRACK-01 | Phase 1 | Pending |
| TRACK-02 | Phase 1 | Pending |
| TRACK-03 | Phase 1 | Pending |
| TRACK-04 | Phase 1 | Pending |
| SAFE-01 | Phase 1 | Pending |
| SAFE-02 | Phase 1 | Pending |
| SAFE-03 | Phase 1 | Pending |
| SAFE-04 | Phase 1 | Pending |
| SAFE-05 | Phase 1 | Pending |
| VAL-01 | Phase 2 | Pending |
| VAL-02 | Phase 2 | Pending |
| VAL-03 | Phase 2 | Pending |
| VAL-04 | Phase 2 | Pending |
| VAL-05 | Phase 2 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 4 | Pending |
| MODEL-03 | Phase 4 | Pending |
| MODEL-04 | Phase 4 | Pending |
| MODEL-05 | Phase 4 | Pending |
| MODEL-06 | Phase 4 | Pending |
| MODEL-07 | Phase 4 | Pending |
| SHIP-01 | Phase 5 | Pending |
| SHIP-02 | Phase 5 | Pending |
| SHIP-03 | Phase 5 | Pending |
| SHIP-04 | Phase 5 | Pending |
| SHIP-05 | Phase 5 | Pending |

**Coverage:**

- v1 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0

---
*Requirements defined: 2026-08-23*  
*Last updated: 2026-08-23 after roadmap creation*
