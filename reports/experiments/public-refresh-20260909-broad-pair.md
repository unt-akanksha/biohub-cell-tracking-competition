# Public refresh during the broad paired fit

The live Kaggle CLI was queried by creation date on September 9. No
score-sorted list, known older exploit notebook, public prediction file, or
public checkpoint was downloaded. Current experiment settings remain frozen.

One new source was inspected without execution:
[backtracking/biohub-focus3d-silver-hedge-v1](https://www.kaggle.com/code/backtracking/biohub-focus3d-silver-hedge-v1).
Downloaded notebook SHA-256:
`459cedbe2068fff6b4dfc57e20a9103a14bbc059baa4612c9f7693433a618de4`.
Cache: `.biohub/research/public-refresh-20260909-broad-pair/focus3d-silver`.

It uses the same public FOCUS-3D runtime family already investigated here,
followed by geometric relinking, gap repair, short-track removal, coordinate
smoothing and division recovery. Its full preset removes nodes and imposes
fixed per-frame/global division caps. This is not sufficient evidence of a
metric exploit, but it is also not validated evidence for importing those
heuristics. All downloaded code cells have no execution count. The local
evaluation cell calculates/accumulates its per-sample metric outside the movie
loop, so the displayed summary would cover only the final movie. It imports
the support-pack scorer rather than our pinned patched scorer. Decision:
do not execute, copy, promote, or allocate GPU to this reproduction.

Search also resurfaced the
[synthetic division dataset discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103).
This is not new to the project: Synthetic256 and larger real/synthetic models
were already tried. In particular the recovered 66.98M-parameter Capacity-PU
V3 completed 2,000 updates but failed its real-domain localization gate
(recall 0.584416 against 0.85), despite strong synthetic performance. Those
results are in `research/EXPERIMENT_LOG.md`; do not repeat capacity-only or
synthetic-only training under the guise of a new discovery.

The present broad experiment instead uses the frozen real-only source-embryo
training list and tests association supervision with a learned detector. No
new claim of superiority over the clean public baseline is supported yet.
