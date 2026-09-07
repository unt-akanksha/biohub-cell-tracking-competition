# Biohub public frontier refresh — 2026-09-07 r2

Status: authenticated Kaggle source/output/status audit plus current discussion
refresh completed at approximately 06:30 UTC. Public predictions remain
controls only and are not eligible candidate artifacts.

## Current competitive signal

- The current discussion reports a public-leaderboard leader near `0.962` and
  the top ten at roughly `0.945+`. This is forum evidence, not clean validation
  evidence, but it confirms that the public `0.92–0.943` notebook lineage is no
  longer the competitive frontier.
- A current third-place competitor recommends the development order
  `detection -> linking -> division`; their stated reason is that better
  detection makes the later two layers easier. This agrees with our first
  peak-rank checkpoint, which already reaches `0.978156` synthetic selection
  AP but only `0.675325` positive recall on sparse real crops.
- The organizer's patched division scorer remains authoritative. No
  pre-patch hub/fork construction, off-volume node, public prediction, or
  leaderboard-selected threshold is admitted to any project gate.

## Newly audited public artifacts

| Artifact | Source/status evidence | Decision |
|---|---|---|
| `mjcho2023/one-faint-cell-costs-two-errors` | Complete CPU EDA; source SHA-256 `87d6efa51226473d317a617b1b749d024e62a3f11790cf90c4fa8fdc2c0fc3e8` | Useful one-movie diagnostic only. On `6bba_05db0fb1`, transiently faint annotated cells move the nearest detection from a reported `1.68` to `4.08` µm and account for 67/1,216 matched-node failures, commonly charging one FN and one FP. This motivates generic temporal fading augmentation, not reuse of predictions or movie-specific rules. |
| `rishabhr0y/biohub-greenfield-seed-a-dev32-score-v1` | Source SHA-256 `acd9b7ef8a6143e97b7becaf87957e28139f23bc8acec1ca8aa3448fe9498f29`; Kaggle status `CANCEL_ACKNOWLEDGED`; no output files | The source describes a promising independent contract—13 epochs, 124 training movies, cross-fitted 32-movie development scoring, and 39 unopened holdout movies—but the exact checkpoint/package is not publicly attached and no score receipt exists. Research signal only; it cannot authorize a model or submission. |
| `rishabhr0y/biohub-sam4celltracking-submission` | Source SHA-256 `1aaaa6f80dd91a7b5d0150beb758ade6cf93b7e699f0d49857c4b771ac562398`; status `CANCEL_ACKNOWLEDGED`; no `submission.csv` | Uses SAM2.1 Hiera-L memory features for association, but explicitly consumes the public `0.943` detector-node CSV. The cancelled execution stopped partway through the last movie. The memory-linking idea may be studied independently; this artifact is neither complete nor a non-replica detector candidate. |
| `sushanthtiruvaipati/biohub-xiaoleilian-divaug-fork-v1` | Source SHA-256 `69334ca4579545c6ea3e3459b82a2b56f3de866ac03b66190399111d754d6df0`; all 314 normalized code lines occur in `mdmahfujulkarim/biohub-cell-tracking-graph-pipeline` (314/345 union, Jaccard `0.910145`) | A direct subset/fork of the same two small public U-Nets. The executed output predates the current rounding cell and records public checkpoint validation recall only `0.5731/0.6589`. Excluded as a replica and weaker detector family. |

## Resulting experiment order

1. Let the already-running independent peak-rank member finish all frozen
   checkpoints; do not promote its step-1,000 failure.
2. Run the queued conservative positive-unlabeled/depth-attenuated member and
   the 67.0M-parameter capacity member.
3. Screen the independently licensed NucVerse3D generalized checkpoint on
   optimization crops, advancing automatically to disjoint selection only if
   the optimization receipt passes.
4. Add a project-authored temporal faint-cell augmentation member. It may use
   only generic local fading on training examples and must pass the same
   disjoint real, complete-movie, and patched-official gates.

No artifact in this refresh creates or authorizes a competition submission.

