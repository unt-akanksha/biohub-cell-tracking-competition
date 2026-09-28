# Next distinct test: fixed owned-detector ensemble

Implemented, small GPU probe verified, and full source validation **REJECTED**.
Do not rerun the failed ensemble or sparse-calibration recipes unchanged.

All8 complete source movies regressed. Official score0.5039128669 versus
retained parentD4 baseline0.6298395328; raw edge Jaccard0.6183928773 versus
0.6798300425. The ensemble recovered47 extra true edges but added871 false
edges and350 false divisions;366940 nodes versus219373. Recall rose to0.998077,
but this did not translate into stronger tracking. No weight/threshold sweep,
new target access or submission follows this failed gate. Retain parentD4.
Report: `owned-detector-ensemble-selection-v1-result.json`.

Motivation: the original detector and PU detector have different per-movie
errors. PU improved the source score but failed robust target transfer. A fixed
ensemble is a different candidate, not promotion of PU alone. This remains a
hypothesis; no claim that ensembling must help or that either model beats public
bests. Neither prior gate failure is erased.

Freeze parent76f7 and PU b07f39 checkpoints, equal0.5 probability weights,
eight-view detector D4 separately for each model, parent native features and
the unchanged standalone flow/link policy. Keep the original detector cutoff.
No coefficient fitting, threshold sweep, movie-specific choice or graph pruning.

Do not directly average saturated sigmoid tensors before spatial peak finding.
Compute the ensemble logit stably as:

    logaddexp(-softplus(-a), -softplus(-b))
      - logaddexp(-softplus(a), -softplus(b))

This is the log odds of the equal-probability mixture, retaining FP32 logit
ordering for highly confident peaks. Test numerical equivalence in the ordinary
range, stability in extremes, shape/device handling and identical-input behavior.

First run the existing three-training-frame functionality comparison, including
strict checkpoint receipts, nonempty graphs, finite coordinates, no node
truncation and unchanged frozen modules. No source or target labels in this
probe. Only a successful bounded probe can justify complete-eight-source
validation against the retained frozen baseline using the existing score,
raw-edge, recall and worst-movie safeguards. Remaining65 target movies stay
closed. The two components are owned checkpoints, not public notebook replicas.

## Verified smoke

`biohub-owned-detector-ensemble-probe-v1/1` completed in73.116s. All17 bundled
Torch tests passed before the real three-frame comparison. Parent D4 produced
269nodes/160edges (14TP/5FP/2FN); mixture produced303nodes/185edges
(13TP/10FP/3FN). This tiny training sample is worse on edge counts, not evidence
of accuracy improvement. Functionality, strict reload, frozen tensor hashes,
finite coordinates and GEFF round trips passed. Peak counts remained below
the2048 abort guard. No held-out movie was opened.

Verified report: `owned-detector-ensemble-probe-v1-result.json`.
Frozen launch SHA:7196f42ad6a11af897388893fac305bca414420499f3be13915f41941bc23666.
The complete-eight-source test must still beat retained parent D4 on score and
raw edge Jaccard, retain mean recall within0.005, improve at least5/8 movies,
limit each adjusted-edge loss to0.02 and worst-movie loss to0.01. No fitting on
these results, no new target access automatically authorized.
