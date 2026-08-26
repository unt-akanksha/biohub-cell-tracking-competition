# LSM-FM public-node refinement design — 2026-08-26

## Objective

Test whether an independently trained 35,072,515-parameter LSM-FM feature-36
heatmap can improve the frozen public graph's node-to-ground-truth matching by
moving coordinates only. Every node ID, edge, frame assignment, and node count
is preserved. This creates a materially different graph only when independent
image evidence supports a coordinate displacement.

The run is a held-out validation experiment, not a competition submission.

## Why this lane is next

The independent feature-36 detector's global radius-2 centroid improved all
eight selection movies but missed the immutable worst-movie gate by one match.
Its acceptance ceiling is still uncertain, while the frozen public graph has
already measured `0.96902842596521` annotated-node recall. Refining that graph
therefore has a higher probability of improving the exact edge metric than
replacing all nodes with the lower-recall independent detector.

## Frozen candidates

The unmodified public graph is the control. Eight candidates use a radius-1 or
radius-2 probability centroid with probability power 2, then blend 25%, 50%,
75%, or 100% from the original public coordinate toward that independent
centroid. One global strategy is selected for every movie.

Selection requires:

- a strict pooled matched-node gain over the public control;
- no selection movie recall regression worse than `0.002`;
- no per-movie or embryo-specific strategy choice;
- no leaderboard evidence.

The four acceptance movies remain unopened unless selection passes. Acceptance
promotion requires no pooled matched-node loss and no movie regression worse
than `0.002`. Candidate acceptance GEFFs are retained for a later exact official
graph-metric gate without another GPU inference pass.

## Provenance and operational controls

- LSM-FM checkpoint and learned Biohub head hashes are verified before loading.
- Public graphs provide topology and starting coordinates only.
- LSM-FM probabilities provide every displacement.
- Refined graphs preserve node IDs, node count, edges, and frame coverage.
- The kernel has no competition-submit command or test-inference path.
- Selection and acceptance use complete movies from both embryo prefixes.
- A later production submission, if ever authorized, must require exactly two
  CUDA devices and shard whole movies across them.
