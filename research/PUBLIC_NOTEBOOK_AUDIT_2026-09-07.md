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

References:

- <https://www.kaggle.com/code/redoctopusk/biohub-948tta>
- <https://www.kaggle.com/code/muhanqiu/biohub-final-submission-our-weights>
- <https://www.kaggle.com/code/hengck23/cell-point-detector>
- <https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development>
- <https://www.kaggle.com/code/amanatar/biohub-cell-tracking-v>
- <https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking>
- <https://www.kaggle.com/code/analyticaobscura/biohub-lb-942>
- <https://www.kaggle.com/code/nusrati/0-940>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation>
