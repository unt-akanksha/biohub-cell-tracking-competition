# trackastra-graph-finetune-v4

Status: staged and ready after the active detector experiment; GPU runs remain
strictly sequential.

V4 is the independent association candidate. It keeps the official 27.5M
Trackastra CTC warm start and corrected CC0 graph pretraining from V3. Real
fine-tuning now draws from every available non-validation movie across both
embryo prefixes and targets 7,000 steps within the same hard wall-time. The
validation contract is also stricter:

- one complete movie per embryo prefix selects between pure Trackastra and a
  conservative base-graph hybrid and selects all thresholds;
- the other complete movie per prefix is unread until that configuration is
  frozen;
- the exported-CSV hybrid is validated with a constant base-edge
  pseudo-probability, because test CSVs do not expose the public detector's
  original confidence;
- promotion requires a positive acceptance delta against the exact same-node
  raw base graph and no acceptance worst-movie regression larger than 0.01.

The staged submission reranker preserves detector nodes, replaces learned
associations, validates lineage topology, hash-binds the checkpoint and base
CSV, and refuses both byte-identical and edge-identical replicas. No
submission is produced by this training run.
