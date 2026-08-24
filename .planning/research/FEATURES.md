# Feature Research

**Domain:** A score-improvement and experiment operating system for the Biohub cell-tracking competition  
**Researched:** 2026-08-23

## Table Stakes

### Competition Watch

- Pull current quota, submissions, leaderboard, notebook metadata, and discussions through the authenticated Kaggle CLI.
- Timestamp snapshots and preserve prior snapshots rather than overwriting history.
- Inspect notebook source for known exploit signatures and label score provenance as reproduced, patched-era reported, stale/ghost, or unknown.
- Surface changes to deadlines, rules, metric code, public data, and organizer announcements.

### Experiment Ledger

- Assign every experiment a stable ID, hypothesis, parent, owner, status, and decision.
- Capture code/data/model/config hashes, seeds, accelerator, maximum runtime, actual runtime, and quota before/after.
- Store complete-movie pooled metrics, embryo metrics, fold metrics, worst-movie delta, bootstrap interval when practical, and node/edge/division diagnostics.
- Preserve failed, rejected, and incomplete runs so expensive dead ends are not repeated.

### Resource Safety

- Read live Kaggle quota immediately before launch.
- Compute `projected_remaining = live_remaining - declared_max_runtime`.
- Fail closed when projected remaining is below 8 hours, quota cannot be parsed, another GPU run is active, or no maximum runtime is declared.
- Add an in-notebook wall-clock watchdog, periodic checkpoints, output flushing, and a final coverage check.

### Validation

- Use the patched official metric and a scorer regression fixture.
- Validate complete movies with leave-one-embryo-out reporting.
- Verify physical coordinates, node IDs, dataset coverage, row types, temporal direction, edge endpoints, and CSV round-trip equivalence.
- Keep local metric, public leaderboard score, and claimed notebook score as separate fields.

### Training and Inference

- Provide short smoke, bounded calibration, fold training, full inference, and resume modes.
- Use AMP and throughput logging.
- Train on temporal artifacts and predicted-node candidate distributions.
- Produce an offline Kaggle notebook and weight dataset with deterministic provenance.

## Differentiators

### Error-Decomposition Dashboard

Break score loss into missing endpoint nodes, incorrect association conditional on detected endpoints, fragmentation/gaps, node-count penalty, and missed/false division forks. Use detection-ceiling and oracle-linker diagnostics to decide where GPU time goes.

### Evidence-Gated Promotion

Treat promotion as code: a run advances only if declared gates pass. A default gate requires positive pooled clean score, acceptable embryo balance, nonnegative node recall, bounded worst-movie regression, and nonnegative division behavior unless an explicit score trade-off is approved.

### Score-Provenance Audit

Persist the source code hash and evaluation date for every borrowed notebook technique. A notebook title or page score never becomes evidence by itself.

### Budget-Aware Experiment Portfolio

Estimate expected information gain per GPU hour. Prioritize reciprocal calibration and two fold-training runs; postpone ensembles and long inference until a single model clears OOF gates.

## Deferred

- Hosted MLflow/W&B synchronization — optional after the offline ledger is stable.
- More than two fold models or multi-architecture ensembles — only after a single stronger model is validated.
- Large self-supervised video pretraining from scratch — move to the cloud-GPU stage unless a short Kaggle probe shows exceptional transfer.
- Automated operating-system scheduling — session-start and pre-launch hooks are sufficient for v1; background polling can be added later.

## Anti-Features

- Metric-hack generation or evaluation.
- Automated copying of the currently top-sorted notebook.
- Submission based only on public leaderboard movement.
- Unbounded hyperparameter search.
- Concurrent Kaggle GPU jobs under one quota.
- Mutation of an experiment record after its decision; corrections must append an amendment.

## Dependencies

`Competition Watch` and `Resource Safety` precede any GPU experiment. `Validation` precedes model promotion. The ledger spans every phase. Strong-model training depends on a leakage-resistant split and throughput benchmark. Submission packaging depends on a promoted checkpoint and complete coverage checks.

## Sources

- [Competition overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview)
- [Competition rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
- [Official metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- [Discussion: full-movie validation and node/link decomposition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101)
- [Discussion: stale notebook scores after metric refresh](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/736937)

