# Public notebook and discussion audit — 2026-08-26

This audit is a point-in-time research input, not a leaderboard selection loop.
The Kaggle CLI listed the competition notebooks by votes and the current source
of eight representative high-vote or high-score notebooks was pulled for local
inspection. Explicit metric-hack notebooks were excluded from modeling ideas.

## Main conclusion

The visible 0.91–0.927 frontier is largely one public model lineage, not a set of
independent heavy models. The recurring backbone is Pilkwang's temporal 3D U-Net
detector, an 8.3M edge predictor, a second detector seed, ILP/link repair, and
small threshold, division, or temporal-consistency changes. Copying another
notebook in this family is unlikely to produce a durable gain.

Representative examples:

- `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` is the shared
  dual-seed detector/linker lineage.
- `evgendvorkin/biohub-0-927-lb`, `yunusgmsoy/kimi-notebook-v17`, and
  `rockerritesh/0-926-biohub-divsub` retain the same assets and most of the same
  pipeline, changing division geometry, reverse-time association weights, or
  checkpoint selection.
- `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` adds a bounded
  three-frame acceleration look-ahead to the same detector/edge backbone.
- `yusuketogashi/clean-approach-lightweight-local-cv-no-hack` rescues selected
  short tracks from the same public baseline family.
- `xiaoleilian/biohub-ct-mix-divaug` is more independent: two base-24 3D U-Nets,
  augmentation diversity, peak fusion, and geometric linking. It is useful
  diversity evidence, but it is not a substantially larger representation.
- `jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans` combines DoG
  detections with the generic pretrained Trackastra CTC model. Its proxy uses
  only two train movies and the model is not Biohub-fine-tuned, so our
  Trackastra lane remains materially different.

The current submission with public-output SHA-256 `33c179...` is therefore
retained only as an attributed benchmark. It is not an owned candidate and
should never be promoted as evidence of model improvement.

## Same-day delta audit

A second source pull covered six notebooks visible near the current high-vote
or high-score frontier: `evgendvorkin/biohub-0-927-lb`,
`yunusgmsoy/kimi-notebook-v17`,
`anhadmahajan06/biohub-track-your-cells-development`,
`flexonafft/biohub-harmonic-fusion`,
`rockerritesh/0-926-biohub-divsub`, and
`salemali7/biohub-cell-tracking-92-6`.

After trimming whitespace and deduplicating code lines, the first three have
pairwise Jaccard similarities of 0.944–0.965. The latter three are even closer
at 0.983–0.996. Cross-cluster similarities remain 0.722–0.757 because both
clusters share most of the same public dual-seed TemporalUNet, DeepCenter, and
graph-repair implementation. Every current source explicitly records
`metric_hack_used` as false, but none supplies a materially independent heavy
model. This strengthens the decision to spend GPU only on owned SpatialDINO,
temporal-contrastive, or other independently learned candidates.

## Discussion evidence that changes our plan

Current competition discussions independently reinforce the same direction:

- Retraining is recommended because the public U-Net/transformer checkpoint has
  hit a post-processing wall.
- Detector misses are squared in the approximate edge-recall ceiling; detector
  recall, count calibration, and conditional linking should be measured
  separately.
- The public weights' split manifest reportedly places all 199 annotated videos
  in training. A validation score against those checkpoints is therefore not a
  clean estimate of their generalization and can invert post-processing
  conclusions.
- External or synthetic data is repeatedly suggested for division robustness.
- Ambiguous links benefit from multiple temporal/model views rather than another
  global average—consistent with our bounded hybrid Trackastra linker.

Accordingly, the work order is:

1. Independently test the official 35.5M Spotiflow 3D detectors on a disjoint
   selection/acceptance design.
2. If viable, train with fully labeled physical synthetic volumes and then
   positive-unlabeled Biohub adaptation, keeping the public detector fixed only
   as a conservative reference.
3. Fine-tune the 27.5M Trackastra graph transformer with true split exclusion,
   then use it only on ambiguous associations in a bounded hybrid.
4. Do not spend GPU on more public-threshold or division-radius replicas.

## Validation caveat

Our four complete acceptance movies are excluded from our own candidate
selection/training. However, the frozen public detector used as a comparator—and
as Trackastra's current node source—was reportedly trained on all 199 annotated
movies. Comparisons against it are conservative but must be described as
`candidate-disjoint`, not as fully clean for every component. A fully owned
detector trained without the four acceptance movies is required for the final
clean model comparison.

Sources inspected:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/code>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730924>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352>
