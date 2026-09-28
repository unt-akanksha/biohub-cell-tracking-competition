# Owned learned linker on raw FOCUS nodes: bounded probe

The public raw-FOCUS learned linker previously outperformed the owned physical
flow-only linker on four exposed movies, but its public checkpoint overlap and
inherited settings do not support independent promotion. Test the mechanism
with our own trained checkpoint76f7da6e... instead. No public linker weights or
leaderboard-selected mixture/postprocessing settings are adopted.

Candidate: frozen owned image encoder and association transformer, raw FOCUS
centroids, original cached backward flow. Neural score plus unmodified Gaussian
flow prior, each coefficient1, null-4.5, posterior>.5, maximum one parent and
two children. This is the owned head's original training objective, not a
coefficient sweep. No ILP, pruning, coordinate changes or added detections.

Use official feature-map integer-floor sampling, including boundary feature
extension. Keep precise native coordinates for distance/attention and output;
positional embeddings use fractional downsampled coordinates and relative
window times. Cached flow already uses the validated exact-centroid sampler.
This adapter differs from the older public integer-rounded FOCUS adapter and
needs actual functionality testing; do not claim exact public replay.

GPU probe: first three frames of original training6bba_f1fde7e0 and23af9eeb;
same six cached frames as earlier components. Verify checkpoint/split/cache
hashes, strict state load, unchanged weights, finite neural matrices, identical
repeated head inference. Persist all neural matrices and both graph arrays
before any GT. Host must verify raw-node equality and exact probability/edge
replay. No source-eight or target data access in this probe. One-hour runtime
cap and fresh quota with8h reserve before launch. Use the two-T4 instance;
this tiny probe uses one device, avoiding pointless replication overhead.

Training screen, fixed now: patched-official fresh first-three-frame GT,
unchanged-node control; require positive pooled raw-edge Jaccard gain and no
decrease in ordinary/total correct edges per movie, without decreasing true
division detections. This is feasibility only, not complete-movie validation.
If it fails, do not launch a full eight-movie run of this exact configuration.
If it passes, use unchanged original source and FOCUS-flow gates for full
development comparison; pretraining overlap remains unresolved for FOCUS.
No submission or new target access authorized by a training probe.
