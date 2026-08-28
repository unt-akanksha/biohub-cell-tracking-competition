# Biohub public frontier refresh — 2026-08-27

Status: read-only Kaggle API and public-source audit completed. No public
prediction, checkpoint, code fragment, leaderboard score, or configuration was
imported into an experiment. No GPU or submission was used.

## Current notebook field

The authenticated competition kernel listing was read twice: once by votes and
once by displayed score. Displayed-score ordering is not admissible evidence.
The list is headed by notebooks explicitly named `metric hack`, and discussion
736937 reports that old inflated pre-fix notebook scores remain visible after
the metric was corrected. The public notebook UI therefore cannot be used to
rank clean candidates.

Five recent or highly visible notebooks were downloaded only for provenance
and structural inspection:

| Public reference | Notebook SHA-256 | Normalized code SHA-256 | Disposition |
|---|---|---|---|
| `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` | `36a03710951747f59d72c5b62f4224d6258a5937fe1d2cda92607ae03e6a8fd9` | `f539653af4d038b721cde9c54135ad79662c8cc0f5d38d1aed9ccfc845bdee9b` | Read-only hypothesis evidence |
| `flexonafft/biohub-harmonic-fusion` | `880fce7de12cbb441521836bd2c94b755d4ad7e5e67c9ab0d92bf216e4c4fdbf` | `c5337008a6d60e693e8e28bef5c71c4c7fb93be392e7e6c35e9997f5c0d1bdd3` | Public-baseline fork; do not reproduce |
| `rockerritesh/0-926-biohub-divsub` | `a0c44e988ca4eadfddb6022f933c128042eaf51b9f74c12e5aa4e6c62f0c18cf` | `28e6e9ce9a7980f85b3c4d49aa3fbd7f908e41cd23097ad34047fc70daf795c9` | Public-baseline fork; do not reproduce |
| `evgendvorkin/biohub-0-927-lb` | `07c3fdf449bbf0f4a1d1d1ea7f8cdaae1aad88cb4d7a92c5ff081bc81ef37e4c` | `2e9c79f6fa58576a25fee8dd83cd96808709869fb069d562ba00eaa63816bc1a` | Leaderboard-tested configuration; exclude |
| `anhadmahajan06/biohub-track-your-cells-development` | `20a257d6315ce3b087ac59288623c0f7fb550f9169f3831d2fee5356d295fe25` | `23b48f36a09b43014d00a31fc3ba3e6744e4def868d16acb5b771cde9dca02ef` | Public-baseline fork; do not reproduce |

The apparent 0.926–0.927 group is not evidence for a distinct heavy model.
After removing comments, blank lines, and formatting, the harmonic and division
notebooks have `0.9925` line-set Jaccard overlap. The 0.927 and “track your
cells” notebooks have `0.9700` overlap. Their shared structure is the familiar
public dual-seed 3D U-Net/node-transformer/ILP stack plus closely related graph
rules. One 0.927 notebook explicitly comments that parameter values were tested
on the real leaderboard; none of those values can enter our selection process.

## Discussion evidence

The authenticated discussion API returned the following current signals:

- Discussion 737543 describes public notebooks as essentially the same learned
  U-Net → node transformer → ILP lineage clustered around 0.92–0.933, while the
  top group is described as structurally different. A reply recommends working
  detection → linking → division in that order.
- Discussion 737101 recommends separating missing-endpoint failures from wrong
  associations and validating complete movies with the official scorer and
  movie-level OOF splits. Edge-level random CV is called misleading.
- Discussion 737438 reports crowded-frame divisions as a structural failure of
  post-hoc geometric rules and asks whether division should live inside the
  learned assignment. This supports our joint division head and candidate-set
  contextual objective; it does not justify copying a public rule.
- Discussion 737577 reports that changing a scalar division weight can admit
  thousands of false positives for one true division. Weight-only tuning is not
  a credible replacement for learned division evidence.
- Discussion 730160 documents leakage from public checkpoints trained on all
  annotated movies and a real ablation whose local sign reversed on the
  leaderboard. It independently supports reciprocal leave-one-embryo-out
  training and untouched acceptance.
- Discussion 734330 contains organizer confirmation that ZebraHub imaging and
  resources are allowed and do not overlap the competition test set.

Direct discussion links:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737577>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330>

## Effect on our lane

1. Keep contextual v3 first. It is project-authored, trains outgoing child and
   reciprocal parent ranking jointly, uses label-free transition context, and
   is selected on disjoint full-movie evidence rather than public score.
2. Keep multiscale v4 as the capacity experiment. At 46,386,607 parameters per
   fold it is structurally distinct from the sampled public notebooks, starts
   prediction-preserving from accepted v3, and uses no public code or weights.
3. Do not port harmonic/divergence constants or public graph code. Forward and
   reverse support is already learned by v3/v4. A future t+2 division-context
   feature is admissible only as an independent implementation whose thresholds
   and promotion are frozen before untouched acceptance.
4. Do not restart another detector replacement before v3 evidence. Prior clean
   detector lanes consistently failed the dense-movie recall gate, while the
   frozen public detector already measures about 0.969 annotated-node recall on
   the four opened controls. The unresolved near-term question is therefore
   whether learned contextual association improves edges without changing the
   high-recall node set.
5. Continue to ignore displayed notebook rank and public leaderboard deltas.
   Promotion requires full-movie reciprocal CV, both-fold improvement, exact
   processed acceptance, unchanged node recall, and per-movie regression floors.

This refresh changes no frozen recipe. It strengthens the evidence that the
current v3 → high-capacity v4 sequence targets a genuine structural gap rather
than reproducing the public tuning cluster.
