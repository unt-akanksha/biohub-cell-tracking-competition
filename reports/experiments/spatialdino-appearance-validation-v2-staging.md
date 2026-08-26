# SpatialDINO appearance validation v2 staging

Status: registered and preflight-passed; intentionally not launched while
`spatialdino-pu-adaptation-v1` is running.

The v1 experiment failed before model inference because Kaggle did not expose
the attached HOCT kernel output. That failure was environmental and provided no
evidence for or against the appearance-correction hypothesis.

V2 changes only input transport. The exact completed HOCT artifacts were copied
into the private dataset `indarkarhana/biohub-hoct-processed-validation-v1`.
The dataset was downloaded back from Kaggle and all six files matched their
local SHA-256 values, including `processed_validation.csv` at
`6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b`.
Its manifest states that the processed CSV contains no ground-truth labels.

The scientific protocol is unchanged: a frozen 21.5M-parameter SpatialDINO
ViT-S/8 may only make degree-preserving two-edge swaps on the fixed topology.
Two complete movies select one of 18 configurations; two disjoint complete
movies remain unopened until selection passes. Public leaderboard evidence is
not read and the notebook refuses to create a submission.

The focused suite passed 20 tests. All eight preflight checks passed under
internal report hash
`1085f39235014bbade6a509f990126c5c86e22670ac39a40c89424a47791ca47`.
The notebook has a two-hour declaration and a 6,900-second hard stop. It remains
a single-T4 validation job; the mandatory two-GPU movie sharding policy applies
to later submission notebooks.
