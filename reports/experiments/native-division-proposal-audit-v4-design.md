# Source-only proposal coverage audit v4

Previous goal turn made progress: enriched and trained joint-division heads,
rejected their held-out failure, backed up results and safely freed RAM. It was
not a blocker or a wait. The top-five objective remains unproven and active.

Freeze this audit before new image results: all112annotated optimization
division transitions from the existing91source movies; no selection/target/sealed
movies, no fitting, no graph edits, no submission. Reuse pinned image normalization
and archive reader. Classify each failed division as missing parent, daughter,
both, or displacement>20um; separately count one-to-one conflicts. Report nearest
distances and <=7um proximity only as diagnostics, never a changed training label.

Compare exactly two image-only proposal generators: current4xXY mean-pooling,
isotropic DoG sigmas0.7/1.5,3-voxel NMS,threshold0.025; versus2xXY mean-pooling
with physical-scale-preserving sigmas(0.7,1.4,1.4)/(1.5,3,3),NMS(3,5,5),same
threshold0.025. Both retain the4096cap and abort rather than truncate. No GT
coordinates may place proposals. Record proposal counts, physical coverage,
full input hash bindings and latency; higher recall alone is not promotion.

First a fixed four-movie source smoke and synthetic geometry/omission tests,
then the full audit with600second watchdog. GPU jobs sequential and foreign
processes protected. No long training run is authorized by this audit.

Research refresh September14: official baseline still uses temporal3D U-Net
features plus a cross-attention node linker and sparse supervision. Official
division matching allows a one-frame timing offset, so same-transition triplet
recall is not a bound on full official division recall. Our prior independent
parent/persistence repair and v3head failures remain closed, not retries.

Primary sources:
- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

Kaggle discussions735352,737438,732103 were located by search. Direct pages
returned no readable body in this refresh; snippets raise sparse-division
overfitting and joint-assignment questions but establish no new model, weights,
or verified clean score. Do not claim the current public best was surpassed.
