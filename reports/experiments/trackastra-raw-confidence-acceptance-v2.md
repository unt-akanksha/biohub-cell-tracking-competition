# trackastra-raw-confidence-acceptance-v2

Status: failed before Trackastra inference; no submission was created and the
lane is not being retried ahead of HOCT.

V1 did not reach Trackastra inference. Its public-topology materializer omitted
the audited notebook's frozen preset cell, so defaults admitted DeepCenter's
`checkpoint_last.pt` at epoch 500. The first expected-node-count assertion then
failed. This was an infrastructure failure rather than negative model evidence.

V2 executes the exact preset cell before the exact config and postprocess
sources. The notebook source is hash-pinned, the DeepCenter loader must resolve
the epoch-2 `best.pt`, and the materializer additionally verifies its SHA256
before reading any validation labels. Each processed validation movie must
match the node count recorded by the public notebook's downloaded
`validator_results.csv`.

Once topology is verified, the scientific protocol is unchanged: two complete
movies select among a small frozen association grid, and the two disjoint
movies are read once for acceptance. The public leaderboard is not queried or
used, no test submission is produced, and a failed gate retires Trackastra.

## Terminal result

V2 loaded the required epoch-2 DeepCenter `best.pt` checkpoint and verified its
SHA-256, then reproduced 44,139 nodes for `44b6_12dfb391`. For
`44b6_267148e4` it deterministically produced 21,843 nodes, 75 more than the
21,768 recorded by the earlier public validator artifact (0.34% drift). The
exact-count guard aborted at 631.972 seconds before model inference. Quota fell
from 28.33 to 28.15 hours and `submission_created` remained false.

This is comparator-integrity evidence, not negative Trackastra model evidence.
The downstream HOCT materializer remains pinned to the public source hashes and
DeepCenter hash but now permits at most 0.5% node-count drift from the reference
artifact. Trackastra is not retried because HOCT is the higher-priority
independent association experiment.
