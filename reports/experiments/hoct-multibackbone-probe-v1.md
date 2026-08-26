# hoct-multibackbone-probe-v1

Status: completed and rejected at the clean association gate on 2026-08-26.
No submission was created.

The official `general_v1` and CTC-specialized `ctc_v0` checkpoints are not
replica-like seeds. A label-free CPU audit on the exact owned test topology
found probability correlation of `0.650` and top-parent agreement of `0.601`
on a 44b6 movie, and `0.787` / `0.805` on a 6bba movie. This diversity supports
testing them together on Biohub's clean held-out topology.

V2 freezes both 6,252,593-parameter backbones and fits an independent
289-parameter probe for each on all 195 non-validation ground-truth movies.
Every movie is still fully scanned for candidate-recall evidence, but feature
examples are now capped deterministically and evenly per movie before global
concatenation. This hard-bounds each backbone at 600,000 examples (about 691 MB
of float32 features), prevents dense movies from dominating the probe, and
removes the prior risk of accumulating an unbounded multi-gigabyte list before
the cap was applied.
The selection grid contains only twelve association variants: four individual
backbone/head choices, three support-aware log-odds weights for each of the two
head types, and one minimum-consensus variant per head. The latter dampens an
edge to the weaker model probability when both evaluated it, giving selection
a conservative disagreement-aware option. Exact zeros outside one model's
candidate neighborhood do not veto evidence from the other model.

The final linker and thresholds use the same small V1 grid. Two complete movies
select the entire configuration by maximizing the worse per-movie delta versus
the base across the two embryo prefixes; pooled score is only a tie-breaker.
The bounded hybrid family now includes a `0.80` base-division keep threshold,
matching the predeclared fallback confidence for postprocess-created edges, and
one correction-only arm that strongly favors the base unless HOCT supplies
better evidence.
Only then are the two disjoint acceptance movies inferred, using only the
backbone or backbones required by the frozen winner. Their ground-truth graphs,
including base-comparator scores, are not loaded until after the entire choice
is frozen. Acceptance requires a positive proxy delta and no more than a 0.01
worst-movie regression. No submission or leaderboard read occurs.

The notebook now attaches the preceding Trackastra acceptance kernel solely as
a topology cache. It reuses `processed_validation.csv` even if Trackastra's
later model gate rejects, but only after verifying the producer terminal, the
exact public preset/config/postprocessor hashes, the DeepCenter epoch-2 hash,
all four expected node counts, the CSV hash, and the absence of a submission.
This artifact contains no ground-truth labels. If any check fails, HOCT runs the
same exact materializer itself. The reuse avoids repeating the expensive
DeepCenter pass and reserves more of the four-hour GPU window for both HOCT
backbones.

The runtime and private GPU notebook are staged with a four-hour hard cap. This
lane now runs directly instead of first spending another four-hour budget on
the general-only probe: it contains that exact backbone/head option as well as
the complementary CTC option, so the smaller run would duplicate GPU work
without adding a stronger scientific gate.

## Terminal result

The kernel completed in 4,296.166 seconds (1.1934 GPU hours), leaving 26.95
Kaggle GPU hours. Both probes learned the training task: the general probe BCE
fell from 0.1327 to 0.0215 and the CTC probe BCE from 0.6073 to 0.2257. Candidate
generation retained 126,582 of 126,590 eligible consecutive ground-truth edges
(0.99994 recall) before balanced sampling. The failure is therefore clean
generalization evidence, not a broken training run.

The frozen selection winner was
`general_ctc_blend_w0.75:biohub_probe` with the HOCT-only linker. It regressed
the base proxy by 0.05384 pooled on the two selection movies, including a
0.07930 worst-movie regression. On the two untouched acceptance movies its
adjusted edge Jaccard was 0.81754 versus 0.82188 for the frozen base, a
0.004340 delta. It also introduced seven additional false-positive divisions
(10 versus 3) without recovering a true division. The predeclared acceptance
gate consequently failed.

Decision: retire all HOCT-only and HOCT-dominant submission paths. Preserve the
probe checkpoint only as negative evidence and as a possible diagnostic
feature source. The next association experiment must be correction-only: keep
the accepted base edges locked except where multiple parents are locally
ambiguous, then test frozen SpatialDINO appearance evidence with a strictly
bounded adjustment. This directly addresses the missing morphology/intensity
signal in the point-only HOCT adapter.

Bound outputs:

- launcher terminal SHA-256: `dba40bccca19fab0c337ac9ad390e93a7dce47daed59095cd93223e394754e31`
- training terminal SHA-256: `8ddd87ca0dd748b668e8c4f61409930d071ef84f9dd82d96fd750e8b98943f69`
- complete validation SHA-256: `ea42d2976f4530684c38dc172b3fca5a70f1072a09dc52bdbe686eddafc53599`
- probe checkpoint SHA-256: `7f0d2f64d3103880edf9913956e3ab51f0be6d731ba68fd2f355c04e4b2907c2`
