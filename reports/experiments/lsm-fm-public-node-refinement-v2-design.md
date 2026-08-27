# LSM-FM public-node refinement v2

This is not a public replica. It inherits the public graph topology explicitly,
but replaces its node coordinates with a bounded contribution from the
independently trained 35.1M-parameter LSM-FM feature-36 detector.

The failed v1 run requested eight public control graphs that the attached graph
runtime never contained. V2 does not tune a strategy on the four available
movies. It freezes `public_lsm_r2_p2_b025` before label access: a radius-2,
probability-power-2 local centroid with only a 0.25 blend. Radius 2 and power 2
come from the prior independent detector localization lane; the conservative
blend follows the observed generalization advantage of partial rather than full
coordinate correction.

All four available public-control movies form the acceptance set. The first
gate requires non-regressed matched-node recall and a per-movie regression no
worse than 0.002. If that passes, the pinned organizer scorer compares the
unchanged public topology and refined topology in both native-float space and
the integer-rounded space emitted by the official GEFF-to-CSV converter.

The archived public notebook proxy (`0.929443`) is legacy evidence only. It
used an older approximate division implementation and scored an in-memory
postprocessed graph. The pinned organizer checkout at commit
`075fc5f5a52d11077f9dc2b074644618f26939e2` includes the later anti-exploit
edge and division patches; its candidate-minus-control integer-space delta is
the authoritative diagnostic for this experiment.

The notebook has no competition submission command. Any eventual submission
remains separately gated and must use exactly two GPUs with whole-movie
sharding.
