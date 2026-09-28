# Live guidelines and discussion refresh, September10 19:23UTC

Read all eight live official competition pages via the authenticated CLI. Web
page extraction returned empty bodies, so it was not used as proof of review.
The36,507-character rules and835-character Code Requirements are byte-identical
to the August27 snapshot. Rules SHA25614a61abea978cda14209163cd047ac8167b3d8f5dc3d450d2e21412f026686bb;
Code Requirements556ed778342f2f9ad5265d9e21aeb6e607281a836cab196412b0dc23a4682191.
All raw pages, text and topic inventory are cached under
`.biohub/cache/guidelines-audit-20260910-1925` (actual receipt19:23UTC).

## Binding candidate requirements reconfirmed

- Notebook-only, CPU/GPU <=12hours, Internet disabled, exact`submission.csv`.
- At most5 daily submissions,2 selected finals,5 team members. The rules also
  contain generic single-submission Hackathon clauses; this challenge operates
  as the scored notebook/leaderboard competition, not that generic format.
- Entry/team mergerSeptember22; finalSeptember29, all23:59UTC.
- Every test dataset represented; integer voxel node coordinates and graph
  references valid; exact schema and consecutive index.
- Embryo-disjoint train/test; shown test is a training copy, reruns replace it
  with hidden data approximately training-set size. Local test execution alone
  therefore does not establish hidden runtime or generalization.
- External data/models must meet public/reasonable-access requirements;
  submission code dependencies must meet OSI/commercial-use requirements.
  Winner's own solution licensing is MIT, with the stated external-resource
  exceptions. Preserve full training/inference/environment documentation.
- No validation/test hand-labeling or human prediction; no private sharing
  outside the official team; no deceptive graph/metric manipulation. Keep
  competition data private to authorized participants despite CC0 data terms.
- Official combined edge/division metric remains authoritative. Scores over1
  can occur by definition; score magnitude alone is not proof of a metric hack.
  Our explicit fabricated-node/fork exclusions remain in force.

Sources: [rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules),
[code requirements](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/code-requirements),
[timeline](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/timeline),
[evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation).

## Targeted discussion text review, not empirical reproduction

Read complete returned text for topics740573,739686,740145. Embedded figures
were not inspected, and participant numerical results were not reproduced.

- [Division steps/base rates](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573):
  discussion warns that long continuation steps outnumber divisions, sparse
  annotations bias event samples, and detector localization errors alter
  measured displacement. New proposed cues include multi-frame parent/daughter
  appearance and intensity changes. Treat as hypotheses, not reliable
  thresholds or permission to hand-label evaluation data.
- [Localization and association](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145):
  discusses localizing candidates more accurately to improve training links,
  inspecting ranking ceilings and multi-frame refinement. Compatible with our
  measured ranking bottleneck, but not evidence that a particular public model
  generalizes. No participant's cutoff or leaderboard configuration adopted.
- [Metric discrepancy discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739686):
  participant reports about unchanged scores after deletions are unresolved;
  no authoritative scoring change was established. Do not use deletion folklore
  or score matching to select our candidate.

The new external Ultrack acquisition/model links need independent licensing,
provenance and train/test-overlap checks before any use. No external volumes or
weights were downloaded. No new image-based label changes made.

Public notebook dateRun refresh found only the already excluded dhiaalhemdani
slug updated at19:12:55; skipped before pull. Other latest entries were already
known. No new notebook downloaded/executed, no clean public-best claim, no
leaderboard-based selection. This refresh causes no change to the running
frozen pair-appearance comparison.
