# Completed recovery experiments and September 13 refresh

The numerical experiments below completed September 10, 2026. Their terminal
artifacts were recovered/verified on September 13; they did not run for three days.

## Results: diagnostic, not leaderboard scores

All four complete movies, patched official scorer, micro aggregation:

| Candidate | Combined score | Edge TP / FP / FN | Promotion screen |
| --- | ---: | --- | --- |
| Pinned public reference, our inference | 0.9448387313 | 1325 / 44 / 38 | Reference only |
| Eight-unique-view D4 correction | 0.9476193515 | 1324 / 39 / 39 | Fail |
| D4 + detector-anchored coordinate projection | 0.9482513251 | 1323 / 37 / 40 | Fail |
| D4 + strong pruned-track recovery | 0.9493103203 | 1327 / 39 / 36 | Fail |

The original-reference projection scored 0.9406876281. Original-reference
pruned-track recovery scored 0.9443976450 with no true-edge gain. Both fail too.
These are separate frozen experiments, not options selected or mixed by movie.

The best pooled diagnostic still regresses by 0.0061730266 on 6bba_23af9eeb and
0.0004952928 across embryo 6bba relative to the reference. Do not promote it or
report 0.94931 as a Kaggle score. These public checkpoints overlap training;
the four movies contain no annotated positive divisions. They do not establish
new-embryo generalization, division recovery, or meeting the user's 0.945 target.

## What the experiments taught us

Retention attribution distinguishes dropped detections from moved coordinates.
On the worst D4 movie, 473 annotated cells match after ILP, 462 remain matched
after node deletion at detector positions, and coordinate changes recover two
to reach 464. Thus 11 annotation matches were lost with deletion; smoothing did
not cause all 11. Original: 475 -> 468 -> 467. On 6bba_23af9eeb, original
smoothing gains two matches; D4 smoothing gains none. These sparse annotation
counts are not exhaustive cell recall or precision.

Strong short-track recovery restores four annotation matches and three true
edges in the worst D4 movie. But it adds 104 predicted nodes there, plus 119,
13 and 5 in the other movies with no measured edge benefit. Association
confidence alone is not sufficient evidence for restoring an isolated track.
This motivates endpoint-supported recovery rather than tuning node budgets.

CPU times: coordinate projection 18.766 s; strong-track recovery 104.687 s.
Combined existing D4/scorer, projection and recovery tests: 33 passed in 7.99 s.
No new GPU use was needed for these completed experiments.

## Immutable receipts

- Retention attribution: `public-d4-retention-v1-result.json`,
  SHA256 `5c6e64142fac0a11c9a114e2a4496017a890ac00193b22b8eb156961c448cf39`.
- Projection: `public-localization-projection-v1-result.json`,
  SHA256 `390f215802a87f97a2e183ad02f7ed066477299f5c4ad224af95cbe90c402f10`.
- Strong-track recovery: `public-pruned-track-recovery-v1-result.json`,
  SHA256 `724c9d51e22ee34d5875262d12e93a5983597ecbb16ec785b6ceb6f75175ade3`.
- Original/D4 parent: `public-d4-full-movie-v1-result.json`,
  SHA256 `04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b`.

## Fresh external state, September 13

Authenticated AWS inspection: the same Antelume instance is running at a new IP.
SSH succeeds using the previously pinned host key; no disabling host verification.
GPU compute-process query is empty. Root filesystem has 4.8 GiB free. The old
owned `/tmp/biohub-d4-preflight-v1.o4fsjO` directory is absent; local exact weights,
source, all eight outputs and logs remain available. Raw movies can be downloaded
again from competition storage. No RSNA files/processes/environment were changed.
No cloud GPU experiment was launched September 13. The instance was not stopped;
an idle instance still bills. It is available for the user's other projects.

Authenticated Kaggle history still returns August 26 submission 55784044 as the
latest; its score field is empty (unknown, not zero). No September 13 submission
has been made. The five-good-submission request remains unmet. No Kaggle GPU
launch, quota expenditure, or change to the 8-hour reserve policy in this work.

Official rules and code pages were fetched again at 19:18 UTC. Their complete
content hashes are unchanged from the thoroughly reviewed September 10 snapshot:
rules `14a61abea978cda14209163cd047ac8167b3d8f5dc3d450d2e21412f026686bb`,
code `556ed778342f2f9ad5265d9e21aeb6e607281a836cab196412b0dc23a4682191`.
All eight current pages are cached in `.biohub/cache/guidelines-audit-20260913-1921`.

## Public research refresh

Current notebook and discussion inventories were queried, excluding previously
identified fabricated-time/node and title-declared metric-hack sources before
pulling. Three new notebooks were downloaded as source only; none was executed.
This limited review does not certify a global clean leaderboard best.

- [Learned UNet Transformer ILP Gap Recovery](https://www.kaggle.com/code/binasalama/biohub-learned-unet-transformer-ilp-gap-recovery),
  current version 5: inspected pipeline/configuration is the familiar TemporalUNet
  plus node-transformer/ILP family using the 50-epoch support package. It is not
  evidence of a new, independently strong network.
- [Champion v5](https://www.kaggle.com/code/caassicca/biohub-champion-v5-thr099b),
  version 1: source/configuration inspection shows the same dual-seed family,
  a 0.99 detector threshold and geometric division checks. It also retains a
  proxy validator/sweep. Its title and comments are not accepted score evidence;
  no proxy sweep, dependency installer, notebook cell or predictions were run.
- [A dividing nucleus gets smaller, not dimmer](https://www.kaggle.com/code/zhincez/a-dividing-nucleus-gets-smaller-not-dimmer):
  fully read source and narrative. It separates half-max volume from peak
  brightness, avoiding fixed-patch mean-intensity size confounding. Useful
  division-feature hypothesis, not a ready trained submission model. Its
  division-rich movie selection, anisotropic box measurement, independently
  resampled controls despite pairing, and multiple per-lag intervals limit
  statistical interpretation. No values were independently reproduced here.
- Full textual comments of discussions [740573](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573)
  and [740145](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145)
  were refreshed. They suggest multi-frame appearance and better localization;
  participant claims and unviewed figure attachments are not validation evidence.
- The user's [LDA paper](https://arxiv.org/abs/2604.03928) supports an inexpensive
  frozen-feature projection comparison, with a fine-grained-task warning. That
  comparison has already been performed here: lower held-out correction NLL but
  84 missing-parent decisions versus the required 115. It remains unpromoted;
  do not restart the same experiment under a new name.
