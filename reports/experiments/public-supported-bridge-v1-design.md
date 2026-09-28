# Supported detector bridges v1

Frozen September 13, 2026, before candidate generation or scoring.

The September 10 D4, localization projection, and isolated short-track recovery
experiments all failed their unchanged movie/embryo gates. They remain rejected.
The recovery experiment added three annotated true edges but also added isolated
components without measured benefit in the other movies. Do not tune that budget
or select settings by these four movie scores.

This distinct, bounded experiment connects existing track ends to existing track
starts through one or two **actual detector candidates** discarded by graph
selection. It never restores an isolated component or invents a detection.

Fixed algorithm, applied identically to both original and D4 parents:

- Use saved raw candidate coordinates and learned association probabilities.
  Verify all pre-ILP node IDs against this array before accepting its provenance.
- Retain only reciprocal, unique maximum-probability raw edges with probability
  at least 0.88 (the existing public strong-track threshold). No threshold sweep.
- Follow those edges from an existing terminal node through one or two omitted
  raw nodes to an existing parentless node. Every step is one frame, at most
  6 microns (the public tight-motion radius), including final endpoint positions.
- Reject an omitted node within 3 microns of any retained same-frame node
  (the public detector pooling radius); do not duplicate an existing detection.
- Preserve every existing node, coordinate, edge, and division. Added nodes keep
  their detector positions. Add only complete paths, with existing public caps
  of 120 nodes and 1.2% of the parent node count. Rank by minimum then mean learned
  probability with deterministic node-ID ties. No ground truth is an input.

CPU synthetic functionality tests precede all eight complete-movie candidate
graphs. Persist and hash every graph before opening truth. Evaluate with the
same patched official scorer, same full GEFF inventories, same strict pooled
combined/raw-edge improvement and every-movie/every-embryo nonregression gates.
Also require an actual true-edge increase and nonregression against its own
parent. No GPU is needed for this test. If it adds no paths, terminate as a
negative result without a GPU run or redundant official rescoring.

These four exposed movies overlap public-model training and contain no annotated
positive divisions. Even a diagnostic pass is not independent validation and
cannot by itself authorize submission. Do not combine failed candidates or
route by embryo/movie. No public outputs, hidden labels, arbitrary node filling,
metric modifications, leaderboard tuning, or Kaggle submission in this run.
