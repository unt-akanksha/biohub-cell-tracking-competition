# trackastra-raw-confidence-acceptance-v2

Status: staged as the single bounded infrastructure retry of v1; no submission
is created.

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
