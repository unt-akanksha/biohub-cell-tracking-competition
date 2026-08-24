# Architecture Research

**Domain:** Local control plane plus Kaggle-hosted training and inference  
**Researched:** 2026-08-23

## Component Boundaries

```text
Kaggle CLI / official pages
          |
          v
  competition_watch  ---> snapshots/ + intelligence report
          |
          v
 experiment_registry <--- validation reports <--- official patched scorer
          ^                         ^
          |                         |
   launch_guard ------------> Kaggle notebook jobs
                                    |
                     checkpoints / predictions / submission
                                    |
                                    v
                           artifact + coverage audit
```

### Local Control Plane

- `competition_watch`: read-only Kaggle snapshot and notebook/discussion audit.
- `experiment_registry`: append-only run metadata and promotion decisions.
- `launch_guard`: quota parser, worst-case budget calculation, active-run check, and launch authorization token.
- `artifact_audit`: downloads run outputs and verifies hashes, coverage, schemas, and declared status.
- `reporting`: projects machine-readable records into a concise progress board.

### Kaggle Runtime

- `smoke`: imports, mounts, one batch, one optimizer step, checkpoint round trip, and one small graph score.
- `train`: deterministic split, AMP loop, resumable checkpoints, periodic validation, resource telemetry, and wall-clock exit.
- `calibrate`: reciprocal parent/edge ranking on the opposite embryo and threshold selection without public-score feedback.
- `infer`: offline model load, tiled/chunk-aware prediction, graph construction, full dataset coverage, and CSV export.
- `validate_output`: schema, IDs, graph topology, physical limits, dataset coverage, and CSV↔GEFF round trip.

### Modeling Plane

1. OME-Zarr loader normalizes intensity and samples temporal 3D patches.
2. Spatiotemporal encoder emits dense features, detection heatmap, offsets, uncertainty, and motion/affinity fields.
3. Candidate extraction pools node features at predicted centers.
4. Association transformer scores candidate edges over adjacent frames and explicit gap candidates.
5. Division ranker scores constrained local forks.
6. Decoder constructs a directed graph with physical-distance and temporal constraints.
7. Conservative repair layer applies only independently validated state-guard actions.

## Data Flow and Immutability

Each run consumes a frozen manifest containing split IDs, baseline commit, external asset hashes, configuration, seeds, and parent artifact IDs. Outputs go to a unique experiment directory. The registry receives a new immutable record after completion; a separate decision record promotes or rejects it. Submission generation consumes only a promoted artifact.

Large model/data artifacts live in Kaggle datasets or ignored local storage. Git stores manifests, hashes, source, compact metrics, and planning evidence.

## Build Order

1. Control plane, ignore policy, quota guard, and ledger.
2. Official scorer pin, split manifest, and validation regression suite.
3. ZebraHub reciprocal calibration as a low-cost end-to-end proof of the promotion system.
4. T4 throughput smoke for baseline and one higher-capacity candidate.
5. Two sequential embryo-held-out training runs.
6. Ensemble/calibration only after individual fold promotion.
7. Offline inference notebook, output audit, and submission selection.

This order delivers a usable capability in every phase and prevents expensive jobs from running before their evidence and safety systems exist.

## Sources

- [Official baseline architecture and workflows](https://github.com/royerlab/kaggle-cell-tracking-competition)
- [Official evaluation script](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/scripts/evaluate.py)
- [Official metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)

