# trackastra-raw-confidence-acceptance-v1

Status: staged behind the active V5 training run; GPU execution remains
strictly sequential.

This is a clean posthoc evaluation of the independently fine-tuned 27.5M
Trackastra checkpoint. It does not retrain, create a submission, read the public
leaderboard, or reuse V5's training-time acceptance decision. Its purpose is to
test the extra signal retained in the owned baseline's raw GEFF output:
per-edge learned confidence before graph postprocessing.

The test-output audit found stable node IDs and 92.85% to 97.63% confidence
coverage on final CSV edges. The new linker can lock very-high-confidence base
edges, replace low-confidence edges when the independent Trackastra model has a
stronger alternative, and apply a fixed 0.80 fallback only to postprocess-only
edges. This is materially different from copying all public associations or
assigning them one constant confidence.

The configuration grid is intentionally small. One complete movie from each
embryo selects the method and thresholds; the other complete movie from each
embryo is scored only after the choice is frozen. Acceptance requires a
positive proxy delta against the same-node raw graph and no more than a 0.01
worst-movie regression. If it fails, neither this lane nor its candidate kernel
can create a submission.

The later candidate is hash-bound to the accepted model, the posthoc terminal,
the exact owned 0.927 CSV, and the exact raw test-graph tree. It refuses both a
byte-identical submission and an edge-identical result.
