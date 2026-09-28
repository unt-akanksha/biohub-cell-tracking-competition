# Frozen CPU candidate: detector-anchored localization projection

September10,2026. This design is written before generating/scoring either
projection candidate. Existing D4 experiment and failed gates remain unchanged.

Hypothesis: uniform public five-frame line fitting can move a retained, high-
confidence genuine detection too far from its observed center when links or
local motion are unreliable. The previous stage audit shows perfect annotated-
cell raw recall for corrected D4 on four training diagnostics, followed by losses
in graph selection and post-processing. It did not isolate smoothing from pruning.

Fixed change: preserve the public final graph's nodes, edges, times and IDs.
For each original detection retained after all processing, compare its final
integer coordinates with its pre-repair detector coordinates. Project excess
displacement along that ray into a ball of radius **1.625 micrometers**, exactly
one voxel on the detector's isotropic downsampled grid. Integer ray projection
uses48bisection steps; no tunable radius search. Points already within the bound
are unchanged. Image-guided inserted gap nodes have no genuine detector origin
and remain unchanged. No model or image data changes, no count manipulation,
no orphan-node reinsertion, no edge changes, and no embryo routing.

Two predeclared arms: apply the same projection to the pinned LF-DCTTA original
and to its corrected-D4 research variant. The unmodified original is the public
reference. Each candidate must improve pooled combined/raw edge-Jaccard versus
the original, without a negative combined change on either embryo or any movie.
The corrected-D4 projection must additionally not regress versus its own parent
on pooled and per-movie combined scores. Do not change gates after results.

All four complete movies and all candidates are persisted with hashes before
the scoring phase. Node-membership/edge equality and unchanged time coverage
are checked; count-adjustment terms must stay equal to the relevant parent.
Use the same patched official scorer and complete verified sparse annotations.
Report every arm and failure, not just a winner. All data are previously exposed
training diagnostics; a pass is not independent CV or submission authorization.
No GPU, public-leaderboard tuning, new annotations, or external writes required.

Separate CPU attribution compares retained nodes at detector vs final positions
to distinguish deletion losses from coordinate losses. This is explanatory
analysis, not permission to hand-correct labels or choose a different radius.
