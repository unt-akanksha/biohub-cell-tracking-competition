# Public notebook and discussion audit — 2026-08-31

Status: authenticated, read-only Kaggle source and discussion refresh complete.
This is research evidence, not a public-leaderboard selection loop. Public
predictions, displayed scores, and metric edge cases cannot authorize a model,
threshold, ensemble weight, or submission.

## Decision

Do not copy C33, C34, or C35. All three are close variants of the existing
public dual-seed TemporalUNet, node-transformer, and graph-repair family rather
than independent heavy models. They remain useful attributed controls, but
they do not satisfy the project's non-replica requirement.

The active project-authored lane remains the four-member 71.25M-parameter
temporal localizer trained on Synthetic256 plus competition-train-only real
replay. It preserves the public control's node count and topology while
learning coordinate corrections from image evidence. Promotion remains
conditional on the frozen synthetic, real, and complete-movie gates.

## Reproducible lineage measurements

`research/public_notebook_lineage.py` trims and deduplicates nonempty code
lines, then reports pairwise Jaccard overlap. The refreshed source hashes and
key comparisons are:

| Notebook | Raw notebook SHA-256 | Overlap with public 0.940 EMA |
|---|---|---:|
| `shienzhang/biohub-c33-ema-velocity-094plus` | `a0a3a77735ac69e35ab58379ab40ce494b5aea9ced2f91489d5a719f8c4a02ea` | `0.935484` |
| `tangai1/biohub-c34-linking-cleanroom-20260831` | `35c739e40568373505b2d7b275c3f74bd492315a4b9185afe2e7b1726d36c9ab` | `0.928000` |
| `tangai1/biohub-c35-fallback-detection-cleanroom-20260831` | `678cfbae3c03cddb53621326f09f8b65d1fb5d4149ce2dd08a2e92487b229878` | `0.926716` |
| `notoverkil/biohub-base-0937` | `eb01b88a7128ff167cf73f3b0d1d8fc77f38618a0c9a94de1e7b0d886acf0a30` | `0.924426` |

C34 and C35 overlap at `0.995869`. Their normalized source differs by only 12
lines across four cells: notebook/preset labels, the bidirectional edge weight,
the secondary detector weight, and guard constants. No new detector or linker
architecture is introduced. C33 overlaps C34/C35 at `0.966621`/`0.965283`; its
main recognizable addition is a per-track EMA velocity state within the same
public family.

The independent public 3D-detector notebook remains structurally distinct
(`0.0047`-`0.0051` overlap with this family), but it depends on a private fold
checkpoint and has no clean result reproduced under this project. It stays a
deprioritized research lead, not an ensemble member.

## Late top-50 delta

An authenticated second top-50 pull after the initial audit inspected five
additional late or high-vote sources. Static source scanning found no known
metric-exploit signature in any of them; this is not evidence that their score
claims are clean or reproduced. Lineage comparison against the project's
attributed public `0.927` control found:

| Notebook | Raw notebook SHA-256 | Code-line overlap with control | Decision |
|---|---|---:|---|
| `evgendvorkin/biohub-0-933-lb` | `635b4c8f3f900a9a9080c3072108d6537fe94ad6c7681983f9cf0ee8aa0536cb` | `0.940597` | Public-lineage replica; exclude |
| `flexonafft/biohub-harmonic-fusion` | `f72ca1a24450fa6e8e165cc6ab1f8cd4a716de9d1fcb52fed5cdf34e9c8afb15` | `0.916611` | Public-lineage configuration variant; exclude |
| `rishabhr0y/933-biohub-bidir30` | `c7cdda0acf9dc704865165feae06d933fc85748dd4d8e9454734b91a65f0eb10` | `0.916917` | Public-lineage configuration variant; exclude |
| `xiaoleilian/biohub-m001-ens3-sm6-sim2` | `732a0695d8503c669e68847debea5256d8af77eb48c4faa66c0f458c54f9fb8a` | `0.003255` | Independent but weaker research control |
| `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` | `36a03710951747f59d72c5b62f4224d6258a5937fe1d2cda92607ae03e6a8fd9` | `0.474286` | Public assets plus bounded ranker/look-ahead; do not copy |

The Flexon and Rishabh sources overlap each other at `0.998238`, confirming
that the two displayed `0.933` candidates are effectively the same public
notebook family rather than independent ensemble evidence. Yusuke's additional
local association ranker and one-frame acceleration look-ahead are legitimate
non-exploit ideas, but its detector and support assets remain public and the
owned graph-context model already learns a broader temporal context. Xiaolei's
source is genuinely independent and uses three small 3D U-Net branches with
four Y/X-flip views, but it reports only `0.8623` on its own 24-movie validation;
it does not meet the requirement that every expensive ensemble component be
individually strong. None changes the current heavy-model work order.

## Admissible findings from the metric analysis

The measured architecture notebook is retained as analysis only. Its useful,
non-exploit observations are that annotations are sparse, duplicate detections
are costly under one-to-one matching, localization precision has a sharp
micron-scale effect, and embryo-disjoint evaluation matters. These facts
support an image-conditioned, node-count-preserving localization correction
and movie/embryo-aware validation.

The notebook also analyzes metric edge cases and count behavior. Those parts
are excluded: no fabricated nodes, far-away divisions, duplicate manipulation,
count tuning, or metric-specific construction may enter the candidate. The
patched fake-division technique discussed publicly remains classified as an
explicit metric hack.

## Discussion refresh

The recent high-score discussion reports that public solutions have converged
on roughly the same U-Net, node-transformer, and ILP stack. A top competitor's
advice is to improve detection, linking, and division as separate layers,
starting with detection quality. A separate division thread reports that missed
divisions concentrate in crowded frames and that standalone hand rules fail
multiple checks. Together, these findings argue for learned image/context
features and frozen component-level validation—not another threshold patch.

Work order after the current localizer finishes:

1. Admit only localizer members that pass every precommitted selection and
   sealed-audit gate.
2. Evaluate the fixed equal-member consensus against the exact 0.940 control
   on frozen complete movies.
3. Build and submit only if the candidate is distinct, improves the frozen
   proxy, preserves graph integrity, and finishes within the notebook limit.
4. If localization is rejected, use the independently structured 3D detector
   only as an architecture/research lead; do not import its private checkpoint
   or copy C33/C34/C35.

References:

- <https://www.kaggle.com/code/shienzhang/biohub-c33-ema-velocity-094plus>
- <https://www.kaggle.com/code/tangai1/biohub-c34-linking-cleanroom-20260831>
- <https://www.kaggle.com/code/tangai1/biohub-c35-fallback-detection-cleanroom-20260831>
- <https://www.kaggle.com/code/sleepymegacat/the-metric-decides-your-architecture-8-measured>
- <https://www.kaggle.com/code/evgendvorkin/biohub-0-933-lb>
- <https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion>
- <https://www.kaggle.com/code/rishabhr0y/933-biohub-bidir30>
- <https://www.kaggle.com/code/xiaoleilian/biohub-m001-ens3-sm6-sim2>
- <https://www.kaggle.com/code/yusuketogashi/no-hack-biohub-cell-another-approch-3rd>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737438>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154>
