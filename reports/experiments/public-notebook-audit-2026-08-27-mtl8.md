# Public notebook audit: `arnav170/biohub-mtl8`

Date: 2026-08-27

Decision: reject as a source candidate and do not copy its code, weights,
predictions, or tuned constants.

## Evidence

The authenticated Kaggle CLI pulled the public notebook and metadata into the
read-only local cache. Its metadata declares one `NvidiaTeslaT4`, internet off,
the Biohub competition input, and three public datasets owned by `pilkwang`.
The notebook text advertises leaderboard score `0.923` and labels the method as
dual-seed harmonic bidirectional fusion.

This is not an independently clean solution:

- its own experiment receipt says `leaderboard_feedback_used_for_configuration:
  True`;
- comments compare several parameter values using the real leaderboard and
  revert values after leaderboard regressions;
- it identifies a parent diagnostic kernel and attributes the fusion rule to
  another public CC0 notebook;
- it uses a single T4, so it also does not meet this project's mandatory
  exactly-two-GPU final-inference policy.

Those facts make it unsuitable for either direct reuse or evidence-based model
selection in this project. Its reported score cannot establish generalization
because the configuration was selected with public leaderboard feedback.

## Non-copying research takeaway

Forward/reverse association agreement is a general tracking hypothesis, but
this notebook does not provide clean evidence for its tuned `0.30` reverse
weight. The current candidate therefore remains frozen without this mechanism.
If bidirectional Trackastra scoring is explored later, it must be independently
implemented, include forward-only as an exact control, predeclare its fusion
grid, select only on unopened clean movies, and pass the same one-shot pinned
processed gate. No constant or implementation from this notebook is admitted
to the runtime.

## Candidate comparison

The staged candidate is materially independent: reciprocal scratch fine-tuned
Trackastra models plus two 19.2M-parameter physical-scale temporal 3D appearance
encoders, trained on corrected synthetic and disjoint real folds. It uses no
public predictions or public model weights, forbids leaderboard selection, and
requires exactly two GPUs with whole-movie sharding and a two-hour Kaggle
notebook finalization reserve.
