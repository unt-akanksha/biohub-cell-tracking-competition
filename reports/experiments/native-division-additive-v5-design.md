# Additive division supervision v5: frozen paired data experiment

Previous goal turn made progress: source-only image and matching audits found
21additional optimization divisions eligible under an isolated additive match
policy, with all44existing events preserved. It was not blocked. No stronger
submission or top-five result has been established.

Keep all v3training and selection packets byte-identical. Extract only the21
additional **positive** parent/two-daughter triplets (3from44b6,18from6bba) using
the original image-only DoG proposals, original three physical patch scales and
normalization. No negative labels are inferred from new or unannotated cells;
the existing2,305optimization negatives remain unchanged. Extra positive-event
sampling is the sole scientific change. Each event contains three distinct
image detections, known adjacent-frame GT parentage and <=20um image displacement.

Candidate matcher is pinned to
`a4283493608f0aa3867d6ff79d400c6a52c7cc1151f4a2e0ecd7106c6d0a6fee`.
Strict3.25um matches are retained. Added matches require an unused nearest
image peak within7um, at least3.25um separation from the next peak, and no
competing new annotation. This still carries annotation uncertainty; it is a
testable supervision hypothesis, not a biological-identity guarantee.

Hash-bind the source-only audit, per-frame raw-image hashes, cached proposal
identities and generated recipes. Smoke the first new event from each embryo
before full extraction. Verify every proposed patch center against fresh
image-only proposals. Extraction300seconds maximum, no persistent raw-image
cache, foreign GPU/RAM guards, no shared environment changes.

Reuse the exact v3frozen source-specific encoders and feature construction. Old
feature arrays remain byte-identical and are appended with21new optimization
rows only; original selection row identity and feature values must compare
exactly. The source44and6new heads keep the original v3recipe: final step2000,
zero initialization, full-batch Adam0.01, balanced BCE, weight L2sum0.01,
source-optimization standardization floor0.1/clamp10, independent CNN-origin and
transformer-origin linear heads plus a same-data geometry control.

Use the same source-calibration and source/opposite-embryo gates as v3, without
relaxation. Also report the paired comparison against the old frozen head on
identical selection examples. Only source-passing heads may open the opposite
diagnostic; only two individually passing heads can form the fixed half/half
ensemble. Old v3failures stay rejected. No target graph edits or submission
until complete-movie patched scoring and runtime gates pass.

The original seven selection positives remain seven: do not add validation
examples or change thresholds in response to the previous missed held-out cell.
No model architecture/seed/regularization search is authorized by this run.
