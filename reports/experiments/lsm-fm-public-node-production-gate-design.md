# Public-node production-alignment gate

The first exact diagnostic compares the frozen raw public graph with the same
raw topology after independent LSM-FM coordinate correction. That delta is
useful but is not yet a production claim because the public notebook applies
image-centroid refinement and topology postprocessing after raw inference.

If the predeclared v2 candidate passes its node and integer-rounding gates,
`materialize_public_node_candidate.py` will run a stricter follow-up:

1. Verify exact node-ID, frame-assignment, and edge equality between every raw
   public control and refined graph.
2. Replace only the raw z/y/x values with LSM-FM coordinates.
3. Run the frozen public postprocessor and hash-pinned DeepCenter checkpoint,
   skipping the public intensity-centroid pass that the independent coordinate
   estimator replaces.
4. Round coordinates exactly as the official submission converter does.
5. Score the complete processed candidate against the processed control with
   the pinned organizer metric.

The gate has no competition submission path. A positive production-aligned
delta is evidence for building a future two-GPU whole-movie candidate, not
authorization to submit it.
