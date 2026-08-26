# hoct-multibackbone-probe-v1

Status: staged as the next sequential independent experiment. It is not
launched concurrently with the active Trackastra gate.

The official `general_v1` and CTC-specialized `ctc_v0` checkpoints are not
replica-like seeds. A label-free CPU audit on the exact owned test topology
found probability correlation of `0.650` and top-parent agreement of `0.601`
on a 44b6 movie, and `0.787` / `0.805` on a 6bba movie. This diversity supports
testing them together on Biohub's clean held-out topology.

V2 freezes both 6,252,593-parameter backbones and fits an independent
289-parameter probe for each on all 195 non-validation ground-truth movies.
The selection grid contains only ten association variants: four individual
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

The runtime and private GPU notebook are staged with a four-hour hard cap. This
lane now runs directly instead of first spending another four-hour budget on
the general-only probe: it contains that exact backbone/head option as well as
the complementary CTC option, so the smaller run would duplicate GPU work
without adding a stronger scientific gate.
