# Biohub public frontier refresh — 2026-08-28, pass 2

Status: authenticated, read-only Kaggle source and discussion audit completed.
No public code, prediction, checkpoint, threshold, or leaderboard result was
used to configure an experiment. No GPU session or submission was started.

## Notebook audit

Recent and visible notebooks were listed by recency and votes, not selected by
their displayed competition score. Copies were downloaded into the ignored
audit cache only so their provenance and source lineage could be checked.

| Public reference | Source SHA-256 | Finding | Disposition |
|---|---|---|---|
| `rishabhr0y/biohub-938-sdw85` | `6d0f4314afdb1f90021898edb3559841152e21c4e3b197dcae460d368a52aedd` | `0.99930265` normalized line-set Jaccard with `evgendvorkin/biohub-0-927-lb`; the only unique executable line changes `BIOHUB_SECONDARY_DETECTION_WEIGHT` from `0.475` to `0.85` | Exact public lineage plus a leaderboard-facing scalar; exclude |
| `ericwang03/biohub-daily-probe-lane-5` | `f540c713b357d0f402d2f438d61d924e73276e6f0f6d9830394ff439061379aa` | Shared lineage with the public harmonic/0.927 family and explicitly documents leaderboard feedback as configuration evidence | Exclude |
| `tatyanagrankina/biohub-cell-tracking-during-development` | Read-only audit copy retained | Same public-field family; no independently trained high-capacity model exposed | Do not import |
| `legtarrr/biohub-rules-v14-v3` | Read-only audit copy retained | Rule-based public candidate | Do not import |
| `legtarrr/biohub-baseline-nn-v2` | Read-only audit copy retained | Public baseline lineage | Do not import |

The notebook title or displayed score is not admissible scientific evidence.
In particular, the apparent 0.938 candidate does not establish a new model
family or a reproducible generalization improvement. It is a near-exact public
replica whose single changed scalar cannot enter our training, selection, or
submission chain.

## Discussion refresh

Discussion 737543 now contains a useful error decomposition from a participant:
detection recall `0.9945`, adjusted edge Jaccard `0.9212`, and division Jaccard
`0.1176`. A reply reports that division localization is far more sensitive near
roughly 3 µm than the official 7 µm node-matching radius, and the participant
reports 39% of division nodes outside that tighter neighborhood while remaining
inside 7 µm. These values are hypothesis evidence only. They are not copied as
thresholds or gates.

Discussion 737896 shows an example of a very dim missed node but provides no
new model, controlled ablation, or full-movie evidence. Discussion 737101 still
supports separating missing endpoints from incorrect associations and using
full-movie out-of-fold validation. Discussion 737577 still argues against
scalar division-weight tuning as a substitute for learned division evidence.

Links:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737577>

## Decision

The active v3 → v4 chain remains unchanged. V3 is the first clean submission
candidate and v4 is the project-authored high-capacity experiment. Neither uses
the public notebook implementations or their tuned constants.

The next bounded research lane is division-localization refinement. It will:

1. preserve node identifiers, times, counts, and every edge;
2. permit coordinate changes only for predicted division parents and their two
   immediate daughters;
3. measure its intervention size without loading ground truth or scoring;
4. require a newly frozen, leakage-safe full-movie protocol before any quality
   decision; and
5. remain outside the submission chain until it independently passes that
   protocol and exact processed acceptance.

The four already opened public-node acceptance movies cannot be reused for
threshold, blend, model, or scope selection. They may only support a label-free
mechanical audit that reads the two prediction artifacts and verifies the
bounded transformation.

That production-aligned audit is now complete. The processed control contains
97,971 nodes and 218 predicted division parents. A coordinate donor supports
217 complete parent/daughter triplets, selecting 651 nodes (`0.6645%`) while
leaving one incomplete event exact. All 651 supported nodes have a nonzero
coordinate change; median displacement is `0.9051 µm`, p90 is `2.2262 µm`, and
the maximum is `4.4276 µm`. No truth path was supplied and no metric was run.
This establishes mechanical scope only, not candidate quality.
