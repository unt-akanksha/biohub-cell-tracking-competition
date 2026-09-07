# Public notebook and discussion audit — 2026-09-07

Status: authenticated, read-only Kaggle source refresh complete. Public
leaderboard claims are contextual evidence only. They cannot select a model,
checkpoint, threshold, ensemble weight, or submission.

## Decision

Do not copy the newly advertised `0.942`–`0.948` notebooks. The high-scoring
notebooks sampled from the September 5–7 frontier remain one shared public
TemporalUNet3D, node-transformer, ILP, DeepCenter, and graph-repair lineage.
Their changes are mainly TTA, detector-fusion weights, and post-processing
constants. They are useful controls but do not satisfy the non-replica goal.

Two structurally distinct learned sources are retained only as research leads:

- `muhanqiu/biohub-final-submission-our-weights` replaces the public edge
  checkpoint with author-finetuned weights and averages association logits over
  eight XY D4 views. It still embeds much of the public inference stack, and no
  score or clean validation result was reproduced here.
- `hengck23/cell-point-detector` demonstrates a three-level 3D U-Net with
  channels `(64, 128, 256)` and local-maximum peak extraction. It is a one-movie
  detector demonstration, not an end-to-end submission or independently
  verified strong ensemble member.

The active project-authored graph-context lane is therefore unchanged. Its
eight 74.73M-parameter members learn division decisions from temporal,
multiscale, and permutation-invariant graph context, and promotion remains
conditional on frozen selection plus independent complete-movie development
evaluation.

## Reproducible lineage measurements

`research/public_notebook_lineage.py` trims and deduplicates nonempty code
lines and reports pairwise Jaccard overlap. Raw notebook hashes are:

| Notebook | Raw notebook SHA-256 | Unique code lines |
|---|---|---:|
| `busyaprime/biohub-0-942-lb-one-knob-past-the-public-line` | `b63c1aae2cb2fd10ef74d045af41eafc026a957c10f9ee0d8bc36fac93b0c22a` | 2,890 |
| `flexonafft/biohub-harmonic-fusion` | `f74d700b5af2dd0ca9b7afddf3bac65d5c1a5bd4935455a5779ae54695baab95` | 2,782 |
| `redoctopusk/biohub-948tta` | `ff8fd7db0ec0553dab196340835b3958e884f9deccb83739f527021685eda933` | 2,847 |
| `rishabhr0y/biohub-detfusion-sdw80-adaptive` | `4211a5fbcbf5556b1d1b7cb5d7fb53063955648ee274ee77b27b116ad6ff1259` | 2,983 |
| `rishabhr0y/biohub-detfusion-sdw30-exact0943` | `ea5abc2b05b4952d16a5960cb417012325e537c66c190ef9cc0c77211d981682` | 2,998 |
| `skomuro/biohub-gpu-rerun-20260905` | `86f39f9de638a3037338b04bc1482e66cc24c56a27e4f04bf803fa6d8f7eca50` | 2,883 |
| `muhanqiu/biohub-final-submission-our-weights` | `1be2bb5fa6bf76f93ac25151c4ef61a5b9e1d5589afb08aa4a99cf054265b83e` | 1,736 |
| `hengck23/cell-point-detector` | `ac42811a80c0b3ee78d023e19d780663132cb4b6289cd17703cce7397ea973a2` | 413 |

The advertised `0.948` TTA notebook overlaps the `0.942` one-knob notebook at
`0.969447`, Flexon's notebook at `0.959276`, and the two Rishabh detector-fusion
variants at `0.934949` and `0.931593`. The Rishabh variants overlap each other
at `0.988364`. Skomuro's rerun also overlaps this group at `0.858093`–`0.902922`.
This is one public-lineage cluster, not independent ensemble evidence.

Muhan's notebook is more modified (`0.501904`–`0.535690` overlap with the
cluster) because it installs shared-point D4 edge-feature TTA and a private
author-finetuned checkpoint. Hengck's detector is genuinely separate
(`0.008873`–`0.013201` overlap), but it does not include a full tracking result.

Static policy scanning found no known metric-exploit signature in these eight
sources. This means only that no registered signature matched; it is not a
clean-score claim or a reproduction. Explicit metric-hack notebooks and the
patched far-away-fork exploit remain excluded.

## Late September 6 refresh

Five additional notebooks near the top of the live hotness inventory were
downloaded after the initial audit. None adds an independent model family:

| Notebook | Raw SHA-256 | Unique code lines | Highest overlap with audited family |
|---|---|---:|---:|
| `amanatar/biohub-cell-tracking-v` | `93639fc9126698dd6e00521640a52b2e43629ea8148fb63cfa0c45738f4df225` | 2,774 | `0.990681` |
| `analyticaobscura/biohub-lb-942` | `e59373ea569332e4ce34a94d8579507970345a62d38b99fe900c2c438b4220fc` | 2,908 | `0.921549` |
| `anhadmahajan06/biohub-track-your-cells-development` | `310807cf4471b5eaef3db7ec51e3c487472cc4deb700ad3cd3f2603e7abc80da` | 2,775 | `0.987478` |
| `kunaldesale2408/biohub-cell-tracking` | `6dc327ced8fa9ce74106b34054f52da4a434809243cc87a169021febdaf67568` | 2,778 | `0.981462` |
| `nusrati/0-940` | `75a1f4ccf3b2b5a18783e0c034ebcd061be262624eaea43332d07c8907a8dd9a` | 2,773 | `0.989247` |

Their closest matches are the same Flexon, RedOctopusk, Busyaprime, and
Rishabh notebooks already excluded above. Each retains the public
TemporalUNet/Trackastra/ILP stack and nearly identical post-processing. The
only exploit-related text found was an explicit `metric_hack_used: False`
field; isolated-node pruning is present as ordinary graph cleanup. Static
inspection does not prove a clean score, but the very high code overlap is
enough to exclude all five as independent ensemble members.

## Current discussion findings

A September 7 discussion from a competitor stuck at `0.933` observes that the
public field has converged on nearly identical UNet, transformer, and ILP
stacks, while the top ten are reported above `0.945`. A current third-place
competitor answered that gains come from all three layers and recommended the
order detection, then linking, then division. This supports separately gated
learned components rather than another public configuration sweep.

The strongest new detector lead uses local peak learning to cope with sparse
labels: an annotated peak is trained to outrank its neighborhood while
unannotated pixels are omitted from the loss. The same discussion recommends
dense pseudo-labels from agreement among open-source trackers and short-track
supervision, but neither idea is admitted without embryo-disjoint validation.

The official competition page still requires notebook submissions, no internet,
at most 12 hours for CPU or GPU notebooks, a `submission.csv` output, and only
freely/publicly available external data and pretrained models. The final
submission deadline is September 29, 2026. The host's patched metric remains
the governing evaluation; no exploit behavior is used.

## September 7 incremental refresh

`redoctopusk/biohub-948tta2` is the only newly created frontier notebook since
the earlier refresh. Its source SHA-256 is
`3395f8df72c6d63d243fdb4fede1f1febdd36bfc086b2f0663fec3ccc9dbb189` and its
normalized-code Jaccard overlap with `redoctopusk/biohub-948tta` is `0.994411`.
It is therefore another public-lineage variant, not an independent model or
ensemble member. The 16 added unique lines collect secondary-detector feature
maps from the D4 passes that the notebook already performs and average those
features for association. They introduce no additional detector forward pass.
No new metric-exploit signature or test-label access appears in the diff.

Because this change is association-side and compute-neutral, it is retained as
an explicitly attributed backbone update for the project-authored peak-ranking
detector. The public notebook's predictions and advertised score remain
excluded, and the combined candidate still needs the frozen clean-validation
and complete-movie promotion gates.

## September 7 localization refresh

`mjcho2023/a-z-axis-voxel-floor-on-cell-localisation` is a genuinely distinct
EDA source (raw SHA-256
`799b501b42aef6456650c4842d576c8977727e9e690a336a0e0493540e133e62`;
normalized-code overlap `0.000326` with `948tta2`). It uses one deliberately
hard training movie and sparse labels, so its reported residuals are not a
promotion result. Its reproducible geometric observation is nevertheless
relevant: the z pitch is four times the x/y pitch, integer z localization can
be comparable to a complete one-frame displacement, and post-link smoothing
cannot repair an edge decision made before coordinates move.

The active detector already predicts bounded continuous offsets and restores
them before graph finishing, and the public association model already treats a
pooled-grid z step as one quarter of a pooled-grid x/y step in its relative
coordinate branch. A source trace found one remaining loss: the edge model was
given rounded detector coordinates, so the learned offsets affected later
finishing but not the initial association decision. Before the first detector
selection checkpoint, the candidate was therefore frozen to provide continuous
pooled-grid coordinates to positional and pairwise edge scoring while retaining
nearest-grid feature sampling for compatibility with the frozen linker's
training contract. The worker manifest records this mode as
`association_coordinate_mode=subvoxel`; it remains ineligible unless the full
candidate passes the pinned patched official scorer on all complete validation
movies.

Two other late kernels do not change the decision. `anvithpothula/biohub-x69`
contains only a five-line supersession note and no runnable candidate.
`flexonafft/biohab-lineage-forge-adaptive-tracking` has `0.951591` normalized
overlap with `948tta2` and is another configuration of the shared public
lineage, not an independent ensemble member.

`qiweiyin/focus3d-nuclei-physical-pp-submit` is structurally independent and
therefore more interesting than the public lineage. Its raw notebook SHA-256
is `459cedbe2068fff6b4dfc57e20a9103a14bbc059baa4612c9f7693433a618de4`.
It replaces detection with the newly released roughly 1.1B-parameter FOCUS-3D
nuclei instance model, converts masks to centroids, and then uses
coordinate-only physical relinking. The notebook reports 98.5% sparse-node
recall and 95.2% annotated-edge recall on one movie, but no complete-movie
patched-official score or independent leaderboard result is supplied.

FOCUS-3D source code is BSD-3-Clause, but the exact checkpoint is not yet
eligible for this workspace: the official Hugging Face model card has no
declared weight license and is auto-gated, while the 4.5 GB Kaggle runtime
dataset declares only `other`. The Antelume root also has only 2.1 GB free, so
copying this runtime there would displace the active reproducible queue. No
weight, prediction, or GPU budget is assigned until the checkpoint license is
made explicit and a complete-movie clean evaluation can be staged safely.

The executed outputs of
`rishabhr0y/biohub-detfusion-sdw30-exact0943` were subsequently downloaded to
audit its claimed validation lineage. The run log states
`BIOHUB_VALIDATOR_ENABLE=0`, reports zero held-out samples, and explicitly
skips scoring. Its emitted `dual_seed_frame_retention_guard_report.json` is
also stale relative to the executed configuration: the report names secondary
detection weight `0.475`, detector threshold `0.96875`, and gap distance `5.8`,
whereas the live run declares `0.30`, `0.965`, and `5.0`. The report marks
leaderboard feedback as used and quality as `candidate_unverified`. The
confidence-dominance rule and claimed `0.943` parent therefore supply no clean
validation evidence and are excluded from the frozen first candidate. They may
be reconsidered only as a separately pre-registered experiment after an
individually promoted detector exists.

References:

- <https://www.kaggle.com/code/redoctopusk/biohub-948tta>
- <https://www.kaggle.com/code/redoctopusk/biohub-948tta2>
- <https://www.kaggle.com/code/muhanqiu/biohub-final-submission-our-weights>
- <https://www.kaggle.com/code/hengck23/cell-point-detector>
- <https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development>
- <https://www.kaggle.com/code/amanatar/biohub-cell-tracking-v>
- <https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking>
- <https://www.kaggle.com/code/analyticaobscura/biohub-lb-942>
- <https://www.kaggle.com/code/nusrati/0-940>
- <https://www.kaggle.com/code/mjcho2023/a-z-axis-voxel-floor-on-cell-localisation>
- <https://www.kaggle.com/code/anvithpothula/biohub-x69>
- <https://www.kaggle.com/code/flexonafft/biohab-lineage-forge-adaptive-tracking>
- <https://www.kaggle.com/code/qiweiyin/focus3d-nuclei-physical-pp-submit>
- <https://github.com/yu-lab-vt/FOCUS-3D>
- <https://www.biorxiv.org/content/10.64898/2026.08.25.746907v1>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation>

## September 7 executed-output refresh 2

The newest runnable sources still do not provide a clean candidate to copy.
`busyaprime/biohub-0-942-lb-one-knob-past-the-public-line` explicitly reports
choosing detector threshold `0.96` from public-leaderboard submissions, so the
constant is excluded from training and validation selection. Its normalized
source overlaps `redoctopusk/biohub-948tta2` at `0.964152`; both remain members
of the same public TemporalUNet/transformer/ILP lineage. The downloadable
`948tta2` four-movie validator rows aggregate to edge Jaccard `0.918273`,
division Jaccard `0.166667`, and unadjusted total `0.934940`. A notebook title
claim is therefore not independent evidence of a `0.948` model.

Two low-overlap sources were checked through both source and executed output:

- `rishabhr0y/biohub-greenfield-seed-a-dev32-score-v1` describes a legitimate
  13-epoch, 124-training-movie detector with a 32-movie cross-fitted
  development role and 39 unopened holdout movies. The public run was
  cancelled after only the first 15-movie direction. It produced no complete
  development receipt; its completed threshold sweep peaked at adjusted edge
  Jaccard `0.842158` before per-movie evaluation. It is not a candidate.
- `rishabhr0y/biohub-sam4celltracking-submission` adapts SAM2.1 Hiera-L memory
  features for linking, but takes its nodes from a public `.943` detector
  dataset and its run was cancelled during the fourth hidden movie with no
  submission or validation output. The architectural idea is diverse, but the
  release supplies neither an individually strong clean member nor a finished
  runtime demonstration.

`muhanqiu/biohub-final-submission-our-weights` remains a potentially useful
association-TTA research lead, not a drop-in member. It uses author-finetuned
weights and eight-view shared-node association evidence, but retains roughly
half of the audited public stack and publishes no comparable complete-movie
patched-score receipt.

The new `mjcho2023/one-faint-cell-costs-two-errors` EDA strengthens the current
failure hypothesis without selecting a parameter: on one deliberately hard
movie, 67 of 1,216 matched nodes were localized onto a neighboring cell, which
turned one detection error into an edge false negative plus false positive.
This is consistent with prioritizing faint-cell localization and using
continuous detector coordinates during association. It does not justify a
movie-specific repair.

The authenticated submission inventory is unchanged: the last project
submission is the August 26 clean public-family reproduction and the best
scored project submissions remain `0.913`. No September public output is
submitted or promoted from this audit.

Additional references:

- <https://www.kaggle.com/code/busyaprime/biohub-0-942-lb-one-knob-past-the-public-line>
- <https://www.kaggle.com/code/rishabhr0y/biohub-greenfield-seed-a-dev32-score-v1>
- <https://www.kaggle.com/code/rishabhr0y/biohub-sam4celltracking-submission>
- <https://www.kaggle.com/code/mjcho2023/one-faint-cell-costs-two-errors>

## September 7 public-detector teacher screen

Two CC0 Pilkwang detector checkpoints were downloaded by exact dataset file,
hashed, and evaluated only as possible training-time teachers. The 400-epoch
`unet_transformer_alltrain_seed314159_v1` checkpoint has SHA-256
`ee6c123717c9f99945888b502c6301c5d769bf9647bc0b96f0016037df42d8c`;
the independently initialized epoch-402 `unet_transformer_5090_50ep_v1`
checkpoint has SHA-256
`8294faafd646274e4f81e5a96d407a295d7ef2c7b47e2c50ac42931a543aec60`.
Their model-state cosine similarity is only `0.408907`, so the screen did not
mistake identical snapshots for diversity.

Both checkpoints were run at the replay cubes' existing isotropic physical
resolution, using the published two-frame sliding contract and averaging the
two center-frame logits. On the frozen 17-crop, 154-point real selection role,
top-64 local-max recall within 2.5 voxels was only `0.564935` and `0.623377`;
the probability average fell to `0.584416`. Mean nearest-peak distances were
`4.5612`, `4.1111`, and `4.6878` voxels respectively, with p90 distances above
`10` voxels. Although mean probabilities at rounded true centers exceeded
`0.98`, the BCE heads emitted many saturated competing maxima. Their top-64
peak agreement was `0.637868`.

Public-teacher distillation is therefore rejected before any AWS GPU time. No
teacher weight, logit, prediction, threshold, or leaderboard result enters a
project model. A fresh authenticated kernel listing after the prior audit also
found only `muelsyse111/biohub-zarr-metadata-and-memory-planner` newer than the
last snapshot; source inspection confirms it is metadata-only memory planning
with no model, prediction, metric, or submission candidate.

Additional references:

- <https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1>
- <https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1>
- <https://www.kaggle.com/code/muelsyse111/biohub-zarr-metadata-and-memory-planner>

## September 7 public-source refresh 3

An authenticated `dateRun` listing at `2026-09-07T08:32:35Z` found no new
competition notebook after the already audited metadata-only memory planner at
`07:28:42Z`. The project submission inventory is also unchanged, so no new
leaderboard observation enters model or threshold selection.

A newly indexed external competition repository was inspected at commit
`446589b772f4d98adecc3f6ba15434f0e0d57069`. It is methodologically useful but
not a submission candidate: the retained detector uses fixed Difference-of-
Gaussians proposals plus a small positive-unlabeled 3D appearance CNN and
reports sparse-label recall `0.894906`, but mean local edge Jaccard only
`0.674186`. Its division models fail every frozen policy guard, no released
checkpoint is present, and the repository contains no license file. No source,
weight, prediction, parameter, or submission artifact is copied. The generic
DoG observation is consistent with the independently measured train-role blob
diagnostic and does not alter any competition-selected threshold.

Additional reference:

- <https://github.com/matt-ceran/biohub-cell-tracking>

## September 7 authenticated refresh 4

An authenticated `dateRun` listing at approximately `2026-09-07T09:13Z`
remained unchanged. The newest competition notebook was still
`muelsyse111/biohub-zarr-metadata-and-memory-planner`, last run at `07:28:42Z`;
all model-bearing entries in the first page were already source/output-audited
above. No newly executed clean model, released checkpoint, complete-movie
patched-score receipt, or distinct submission appeared. Consequently no public
source, weight, prediction, parameter, leaderboard-selected constant, or score
entered the project pipeline from this refresh.

The next detector change instead comes from a project-authored diagnostic on
the hash-pinned competition-train optimization role. It compared fixed generic
scale-space responses without reading selection, sealed-audit, test, or
leaderboard data. This is recorded in the experiment log and is not treated as
public leaderboard evidence.
