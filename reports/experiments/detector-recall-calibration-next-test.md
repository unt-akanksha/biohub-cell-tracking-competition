# Training-only confidence calibration: proposed bounded test

The completed sparse-control detector increased annotated recall but strongly
over-detected at the inherited confidence cutoff. Its raw edge Jaccard also
regressed. This motivates checking confidence calibration; it does not establish
that a new cutoff will improve independent tracking.

The active PU selection and all current notebooks remain unchanged. No extra
GPU job is authorized by this design artifact or queued for this branch.

## Fixed recipe before collecting calibration data

- Compare frozen parent76f7 with the trained sparse detector0f441c6e, not with a
  public checkpoint. Keep D4, normalization, spatial peak pooling, flow and
  linking unchanged.
- Use the original120 training movies only:96 calibration fitting and24
  calibration-diagnostic movies (`train[::5]` diagnostic), as in the previous
  calibration split. These are NOT neural-model-held-out movies.
- Choose three annotated current-frame times per movie by a deterministic
  hash of movie/time before viewing probabilities. Reconstruct the preceding
  frame context and FP32 normalization exactly as full-movie inference does.
- First run a small four-fitting/two-diagnostic-movie functionality probe.
  Require exact original detector peak extraction, one-to-one official node
  matching, annotation coverage, strict threshold tie tests, and unchanged
  weights before any full collector.
- Record every annotated node, including unmatched nodes with probability0.
  Obtain confidence only from its one-to-one matched image peak. Unannotated
  predictions are not treated as negative labels.
- On fitting annotations, choose the highest representable FP32 probability
  cutoff retaining at least the parent's matched count minus floor(0.005*N).
  The cutoff may only rise from the existing cutoff. If the candidate cannot
  meet this requirement at baseline, no fitted cutoff is returned.
- Evaluate the frozen cutoff on calibration-diagnostic movies. Report pooled
  and every-movie recall, retained annotations, and exact threshold comparisons.
  Diagnostic failure means no full source-selection rerun; no cutoff sweep.
- Any surviving recipe still needs the same complete eight-movie patched
  official comparison against the frozen D4 baseline, not only the weak sparse
  control, then a separately declared embryo audit. Nothing here authorizes
  submission or opening the remaining65 target movies.

The helper in `research/detector_recall_calibration.py` is implemented and unit
tested. The six-movie GPU collection entry point now exists in
`scripts/collect-detector-calibration-probe.py` and has now completed as
`biohub-owned-detector-calibration-probe-v1/1` in94.121s total,37.521s collection.
It uses the original FP32 inference loader, temporal context and peak extractor,
with parent and sparse candidate pinned to separate GPUs. It stores all peak
coordinates/confidences and all annotated nodes, allowing exact rematching
after a future fitted cutoff rather than relying only on censored matched peaks.
The frame-selection/matching helper passed5 tests in the CPU graph environment;
the initial integer-time schema failure was caught and fixed locally.
All18 frame artifacts, checkpoint/source hashes and original training labels
were verified by `scripts/summarize-detector-calibration-probe.py`. Exact
rematching at the probe-only fitted0.9942006469 cutoff retained140/149 fitting
annotations, equal to the parent, and46/46 diagnostic annotations. One fitting
movie lost one annotation while another gained one. The probe passes its
diagnostic recall check; it does not authorize deploying this small-sample
cutoff. Full96-fitting/24-diagnostic collection should preserve this frozen
recipe and verify overlapping input records against the small probe.
Full calibration and deployment remain unimplemented. This is empirical
training recall preservation, not a statistical guarantee of recall on new
embryos, a complete candidate, or evidence of a score improvement.
