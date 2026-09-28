# Public source refresh: September 14, 2026

## Additional source-only inspection, approximately 09:00 UTC

`binasalama/biohub-learned-unet-transformer-ilp-gap-recovery` was pulled as
source and metadata only; no cells executed or attached predictions downloaded.
Notebook SHA-256:
`bcb1bd98df91a23bdf6afc749a160479a51e8e1bdaab681b4bea5c1e8447c89a`.
Its metadata attaches the existing primary support pack and notebook figures.
Code references `weights/{METHOD}/split_0/edge_predictor_best.pth`, detection
threshold .985, ILP and gap/division heuristics. This inspection identifies an
existing checkpoint family, not a new independently trained ensemble component.
It is not a complete integrity certification or verified leaderboard-score claim.

## Earlier refresh

Authenticated Kaggle source inventory refreshed using dateRun, page size 20.
This is a recent-source search, not proof of the global best clean notebook.
Existing submission 56219125 remains PENDING at this turn's authenticated check;
there is still no returned score to attribute to the new trajectory candidate.

Two downloaded source-only notebooks were reviewed; neither was executed:

- `hengck23/cell-point-detector`: dense-supervision StrongUNet3D demo, dataset
  `hengck23/hengck23-cell-point-detector-demo`. Notebook SHA-256
  `ac42811a80c0b3ee78d023e19d780663132cb4b6289cd17703cce7397ea973a2`.
  Dataset metadata SHA-256
  `e53394796b6eda5a519d63a22be70ae15e1b471259835fdc8d6768279ba4d3bd`
  declares license `unknown`. The 69,455,847-byte model checkpoint was NOT
  downloaded or executed. The small model source was downloaded for inspection
  only; it is not in our implementation or runtime bundle.
- `karl0106/biohub-all199-submission-kernel`: the 913-byte script only locates and
  copies an attached precomputed `submission.csv`. Source SHA-256
  `897d4fdeeb71453e2591330a210ea9f2a895cd6d1e579ab4efedd49c6c4ddcaa`.
  Not usable for this task; no attached predictions were downloaded. This audit
  does not establish how its author generated the CSV and does not accuse it
  of a metric hack.

Known metric-hack sources were not pulled. The new `.947 runnable` listing was
already identified in prior work as the same public weight family, not a new
independently strong ensemble member.

The [FOCUS-3D discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217)
contains participant reports of dense augmented-pair training and detector
distillation, alongside unvalidated hypotheses. It also contains suggestions
about node-count targeting and modifying GT positions; those are NOT adopted.
Our new experiment uses only exact known image transformations on optimization
images, with the existing licensed public encoder frozen. No hidden-data
adaptation, pretrained FOCUS weights, public predictions or unknown-license
detector enters it. See [the frozen experiment](dense-warp-correspondence-v1-design.md).
