# Biohub Cell Tracking Competition

## What This Is

This is a competition operating system and modeling workspace for the Kaggle Biohub Cell Tracking During Development challenge. It gives one competitor a repeatable way to audit the live competition, run resource-safe experiments, compare them with exact out-of-fold evidence, train stronger generalizable cell-detection and lineage-linking models, and ship reproducible submissions before the September 29, 2026 deadline.

The workspace treats leaderboard pages and post-processing folklore as hypotheses, not evidence. Every promoted change must survive the patched official metric, complete-movie validation, embryo-held-out checks, resource accounting, and a no-metric-hack audit.

## Core Value

Reliably improve the clean, private-test-generalizable Biohub tracking score without exploiting the metric or consuming the final 8 hours of Kaggle GPU quota.

## Requirements

### Validated

- ✓ Kaggle CLI authentication can read competition metadata, submissions, leaderboard state, notebooks, discussions, logs, outputs, and current accelerator quota — live audit on 2026-08-23
- ✓ The official patched scorer and complete-movie OOF evaluation are available in the existing Kaggle experiment lineage — prior experiment outputs
- ✓ The current clean submission baseline is approximately 0.913 public score, with 59 submissions and public rank 751 at the latest audit snapshot — Kaggle CLI snapshot on 2026-08-23
- ✓ A reproducible competition watch refreshes rules, deadlines, discussions, submissions, full leaderboard state, notebook-source provenance, official-page fingerprints, and GPU quota — Phase 1
- ✓ An append-only experiment ledger preserves hashes, lineage, resource declarations, evidence, failures, amendments, and promotion decisions — Phase 1
- ✓ A fail-closed Kaggle launch path requires fresh live quota and active-run evidence, enforces the 8-hour reserve, binds preflights, and consumes single-use launch authorization atomically — Phase 1

### Active

- [ ] Establish leakage-resistant, leave-one-embryo-out validation using the patched official scorer on complete movies
- [ ] Finish reciprocal calibration and exact complete-graph OOF evaluation of the ZebraHub selective-SSM lineage-affinity model
- [ ] Train a mixed-precision spatiotemporal detector and association model that improves node selection, ordinary links, and division topology beyond the public checkpoint family
- [ ] Package inference into a competition-compliant, internet-disabled Kaggle notebook that writes a structurally validated `submission.csv`
- [ ] Promote and submit only clean changes with traceable evidence, then select robust final submissions rather than public-leaderboard-only peaks

### Out of Scope

- Metric exploits, fake nodes, sentinel coordinates, disconnected forks, or score-only topology — prohibited by project policy even if a loophole appears
- Hand-labeling competition validation or test data — disallowed by the competition rules
- Trusting displayed notebook scores without reproducing code under the patched metric — notebook ordering contains stale pre-patch scores
- Re-running failed long jobs without a bounded smoke test and output-coverage preflight — prior jobs lost hours to preventable OOM and coverage failures
- Private code or data exchange outside the Kaggle team — competition rules prohibit it
- Using paid or inaccessible external assets that competitors cannot reasonably obtain — incompatible with competition external-data requirements

## Context

The task tracks nuclei/cells through 3D+time zebrafish microscopy and reconstructs divisions and lineages. The official score is adjusted edge Jaccard plus 0.1 times division Jaccard. Node matching uses physical distance, so detection quality constrains association quality; forum diagnostics summarize this approximately as edge recall scaling with the square of node recall times conditional linking accuracy.

The training corpus contains 199 sparsely annotated movies from two embryos (`44b6` and `6bba`). Movie-level splits can share biological source material, and the public checkpoint was trained across the available movies, so random edge or movie validation can be optimistic. Complete-movie leave-one-embryo-out evaluation is the primary generalization check. Frozen-frame pairs, abrupt global motion, dropped-frame-like jumps, sparse labels, and rare divisions must be modeled explicitly.

The public baseline combines a temporal-attention 3D U-Net detector with a cross-attention node transformer. Its published checkpoint is not trained to convergence. Current clean public notebooks mostly remix the same weights with two-seed blending, harmonic/bidirectional edge fusion, motion fitting, and conservative graph patches. Some high notebook scores are stale pre-patch ghosts; at least one top-sorted notebook explicitly contains the patched fake-division exploit.

The user's recent work is disciplined but has plateaued around post-processing:

- A strict dense-motion topology state guard passed with roughly +0.00037 pooled OOF score and no division regression.
- A broader pairwise graph policy gained roughly +0.00145 pooled score but lost roughly 0.00136 division Jaccard, so it is not promotable.
- Frozen dense topology replication produced no passing policy and was rejected.
- V-JEPA and SEA-RAFT parent-ranking features were asymmetric across embryos and require bilateral evidence before exact graph evaluation.
- HOCT exhausted memory on a dense movie after fallback patching.
- A sparse-PU/ranker control ran for about 11.3 hours before failing because output movie coverage changed.
- ZebraHub `selective_ssm_medium` passed all external-embryo maturation gates across three seeds, with parent top-1 about 0.93245 versus 0.93219 for nearest-parent and mean net recoveries about +51.7. Its next gate is reciprocal competition calibration and complete-graph exact metric evaluation.

The main modeling hypothesis is that a stronger but throughput-conscious spatiotemporal network, trained in mixed precision with embryo-held-out validation and public external pretraining, will generalize better than more post-processing on the shared public weights. Capacity is useful only when paired with leakage-resistant validation, predicted-node training, data-pipeline efficiency, and explicit perturbations for temporal artifacts.

## Constraints

- **GPU reserve**: Always preserve at least 8 hours of Kaggle GPU quota; before launch, reject a job if current remaining hours minus its worst-case declared runtime would fall below 8
- **Current budget**: Audit snapshot shows 30 hours remaining and therefore at most 22 spendable hours; recompute from `kaggle quota` before every launch because quotas refresh and usage changes
- **Notebook runtime**: Kaggle competition submissions permit at most 12 hours on GPU or CPU; internal watchdogs must stop earlier and persist resumable checkpoints
- **Execution**: GPU experiments run sequentially so quota, output ownership, and attribution remain unambiguous
- **Inference**: Competition submission notebooks must run with Internet disabled and produce `submission.csv`
- **Timeline**: Team merger and entry deadline is 2026-09-22 23:59 UTC; final submission deadline is 2026-09-29 23:59 UTC
- **Submission limits**: At most five submissions per day and at most two selected final submissions
- **Data**: External data and pretrained models must be public, free or reasonably accessible, documented, and license-compatible
- **Licensing**: Winning code must be reproducible and released under the required open-source terms; dependencies must permit commercial use
- **Reproducibility**: Training and inference code, environment, hashes, architecture, preprocessing, losses, seeds, and hyperparameters must be recoverable
- **Metric integrity**: The patched official scorer is authoritative; no metric hack may enter training selection, validation, or submission generation
- **Validation**: Promotions require complete-movie micro-averaged scoring, per-embryo reporting, and worst-movie analysis
- **Workspace**: Git tracks project planning and code; generated datasets, model weights, Kaggle downloads, secrets, and large artifacts stay out of Git

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use clean post-patch evidence only | Stale notebook scores and the former division loophole distort apparent progress | ✓ Implemented in Phase 1 provenance policy and watch report |
| Reserve the final 8 Kaggle GPU hours | The user will provide a cloud GPU after this threshold, so Kaggle must fail closed before it | ✓ Implemented in Phase 1 guard and launch authorization |
| Validate by held-out embryo and complete movie | Random edges and overlapping source clips create optimistic validation | — Pending |
| Improve detection and learned affinity before adding more graph patches | Current post-processing gains are small and forum evidence identifies node selection as a major ceiling | — Pending |
| Advance ZebraHub selective-SSM to reciprocal exact-graph calibration | It is the strongest unfinished prior lead with all maturation gates passed | — Pending |
| Train in mixed precision with cached/chunk-aware data loading | Prior Kaggle T4 training and long inference runs were too slow or failed late | — Pending |
| Use vertical MVP phases | Each phase must leave a working, auditable competition capability rather than an isolated technical layer | ✓ Phase 1 delivered and verified as a usable control plane |
| Execute GPU work sequentially | A single quota and strict reserve are easier to protect with one active accelerator job | ✓ Enforced by the Phase 1 active-run guard |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-23 after Phase 1 verification and live Kaggle audit*
