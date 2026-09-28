# Bounded label-free inference functionality, not deployment

The full72D correction screen is still running, with its source/optimizer/gates
unchanged. Its supervised batches contain few known targets; inference must
also handle every originally unknown target. Dense all-target descriptors and
transformed feature matrices can exceed the existing allocation guards.

Implement a new inference adapter, not a change to training. Process32target
nodes per block (validated optional1..64), retaining all source candidates and
the fixed null candidate. Use the frozen candidate, descriptor and transform
functions and identical coefficients. No label reads, target filtering, source
pruning, coordinate refinement, graph fabrication or new geometry. Preserve
all original global node IDs; return real-parent IDs or-1 for null. Exact ties
choose the first source before the final null, matching the original scorer.

Use a conservative1GiB active-array estimate, with only bounded logits retained;
do not assemble all descriptors at once. Test dense equivalence at several
block sizes, empty-source/target frames, invalid model/identity rejection,
unknown-target coverage and nonmutation. Then reuse only the already audited
57b7cc1e/frame31 training packet and the two fixed CPU smoke models. Compare
all label-free outputs against bounded reference chunks and every known-target
decision/posterior against the original CPU smoke. Record timing and working
array bounds; no new diagnostics, source, target, GPU or cloud access.

The smoke models are deliberately not deployable candidates. No CSV, graph or
submission is exported. A real inference integration remains conditional on
full correction gates, host verification, diagnostic/full-movie/embryo checks,
offline environment and end-to-end runtime admission. A small packet timing
must not be extrapolated into a claim that full hidden inference meets12hours.
