# hoct-multibackbone-probe-v1

Status: staged conditional escalation behind the cheaper single-backbone HOCT
probe. It is not registered, pushed, or launched yet.

The official `general_v1` and CTC-specialized `ctc_v0` checkpoints are not
replica-like seeds. A label-free CPU audit on the exact owned test topology
found probability correlation of `0.650` and top-parent agreement of `0.601`
on a 44b6 movie, and `0.787` / `0.805` on a 6bba movie. This diversity supports
testing them together if V1 establishes that HOCT features transfer to Biohub.

V2 freezes both 6,252,593-parameter backbones and fits an independent
289-parameter probe for each on all 195 non-validation ground-truth movies.
The selection grid contains only ten association variants: four individual
backbone/head choices and three support-aware log-odds weights for each of the
two head types. Exact zeros outside one model's candidate neighborhood no
longer veto evidence from the other model.

The final linker and thresholds use the same small V1 grid. Two complete movies
select the entire configuration. Only then are the two disjoint acceptance
movies inferred, using only the backbone or backbones required by the frozen
winner. Acceptance requires a positive proxy delta and no more than a 0.01
worst-movie regression. No submission or leaderboard read occurs.

The runtime and private GPU notebook are locally staged with a four-hour hard
cap. They remain conditional because running two backbones before seeing V1
would spend GPU without an evidence gate.
