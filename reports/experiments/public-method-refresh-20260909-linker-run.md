# Public-method refresh during the raw-linker run

The live Kaggle CLI was queried by notebook creation date, not score, and
discussions by newest date. No public prediction files were downloaded.

Recent notebook titles include `biohub-hf-sister18`, `biohub-hf-veto035`,
`biohub-lineage-forge-fork`, and `biohub-what-the-metric-can-see`. Titles alone
do not establish architecture, integrity, or generalization; none was promoted
or copied. Metric-oriented material remains excluded from model selection.

The September 8–9 comments in [magic or overfitting?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145)
suggest better localization for dense training examples and a more expensive
association refinement stage focused on difficult candidate edges. The author
also points to public Ultrack detectors, while explicitly leaving their
alignment with competition annotations unverified. These are hypotheses,
not independent complete-movie validation evidence. No thresholds, checkpoints,
or coordinate rules in the active experiment were changed in response.

The current raw FOCUS plus learned-linker run tests a related but distinct
question: whether an external segmentation detector can support the frozen
association model while preserving every detected point. Its results must be
scored before deciding whether localization/association adaptation is useful.
Even a diagnostic gain requires independent held-out training/validation
before production promotion; the current association weights overlap the
four diagnostic movies.
