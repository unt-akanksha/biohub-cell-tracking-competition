# Public frontier refresh and a concrete enhancement

September 10, 2026, approximately 21:32-22:00 UTC. Current user priority is an
enhanced public baseline, not another standalone FOCUS-head experiment.

## Public source and score evidence

Source-only downloads, before any execution:

| Notebook | SHA256 | Kaggle-reported best public score |
| --- | --- | ---: |
| [sjlee101/biohub-lf-dctta](https://www.kaggle.com/code/sjlee101/biohub-lf-dctta) | `95f08bb82388e9206f45c86daf926bb3d7a7fff9889c89b5fa9596f8e13c3bd4` | 0.947 |
| [sjlee101/biohub-lf-dctta-v020](https://www.kaggle.com/code/sjlee101/biohub-lf-dctta-v020) | `7129370737acb810fc8aa1dc0b515e782571ad61f9cc0b689d777f2c257fb012` | 0.947 |
| [flexonafft/lineage-forge](https://www.kaggle.com/code/flexonafft/biohub-lineage-forge-precision-tracking) | `f1c7d777e5c41a937ce4b679f65f2e0d5795aec411e1cccda99487a86b06dbb2` | 0.946 |

Scores were read from Kaggle's public `GetKernel` metadata, not inferred from
notebook titles or proxy outputs. LF-DCTTA's public page data additionally
reports version 1, run 348619543, COMPLETE, linked score 0.947 and notebook
runtime 7,126.602 seconds. That is the author's visible run, not our hidden
submission runtime or score. This establishes a strong public reference,
not a global certification of the highest clean score. No predictions or
submission CSV were downloaded or reused. Direct browser notebook rendering
failed; the public page's read-only data API provided the metadata.

LF-DCTTA and v020 differ only in DeepCenter safe-division threshold (0.25 vs
0.20). Lineage Forge has that same 0.20 threshold, different introductory text,
and removed comments; its substantive inference remains the same. Do not
treat these as three independent model families. Their public-score tuning
comments are not our selection evidence. We hold the LF-DCTTA reference fixed.

Known negative-time/fabricated-node sources were skipped before pulling.
Title-declared metric-hack variants from amanatar, ravi123a321at and outwrest
were also skipped, not inspected or executed. This is a limited source audit,
not blanket certification of every transitive dependency or notebook.

## Verified gap: the eighth view is a duplicate

In all three notebooks, the purported anti-diagonal view is `R90` followed by
XY transpose. Algebraically this is the horizontal flip already in the list.
The ensemble therefore contains seven unique views, with horizontal reflection
double-weighted and anti-diagonal reflection missing. It affects both public
detectors, both association-feature averages, and the DeepCenter gate.

Our isolated change is R180 plus transpose, with the corresponding inverse.
Exactly six rotation arguments change in the fully materialized predictor and
two in DeepCenter. Models, thresholds, fusion weights, postprocessing and the
eight model calls per stage remain unchanged. No new training or metric trick.

The independent NumPy proof checks square, rectangular and 6D feature layouts:
all inverses are exact; corrected unique views = 8 rather than 7. For a fixed
asymmetric synthetic predictor the corrected average's symmetry error is at
most 5.68e-14 versus 8.875-21.25 for the original. This is geometry evidence,
NOT a measured real-model or tracking-score improvement.

Ten source/geometry/preflight extraction tests pass. The exact dynamic public
patch chain is reconstructed by AST literal reads, without executing notebook
cells, dependency installers, graph postprocessing or its proxy scorer. Source
audit receipt SHA256:
`d54cb97b4819eed9563f85d272fc59c3fe2155cb7f1c0be7ea499e63243d08b0`.
The generated source-only research notebook is not approved for direct launch;
its inherited proxy sweep is not admitted as our selection procedure.

## Discussion findings and prior experiments

Read the full textual messages of these current discussions, not their images:

- [Division steps are not long steps](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573):
  sparse-label base rates, localization-induced apparent motion, and suggested
  multi-frame division appearance cues. Small intensity studies and suggested
  t-10:t+10 crops are hypotheses, not a validated division classifier.
- [Magic or overfitting?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145):
  candidate recall followed by appearance/multi-frame ranking and localization
  refinement. Public Ultrack weights are linked, but not yet licensed/overlap
  audited or downloaded for this experiment.
- [Post-Processing Plateau?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740103):
  participant plateau claim, no new validated method in its single post.
- [Does CV match LB?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160):
  participant report of a local improvement reversing on the leaderboard after
  removing postprocessing from checkpoints trained on evaluation movies.

Our own provenance audit already established secondary training on all 199
movies; primary best-checkpoint membership remains unverified. Public-control
diagnostics cannot be called independent CV. Existing additive owned-flow
repair added zero edges and failed; unanimous graph salvage also failed.
The completed tree-head screen fails absent-parent recovery (83 vs required
115). None enters this candidate. Our owned-model D4 code already used eight
unique views; its earlier embryo failure is not fixed by this public-code bug.

## Exact GPU preflight staged, not launched

The 65,672,738-byte bundle contains exact public best-state weights, DeepCenter,
reviewed model/encoder definitions, and two existing full training frames from
44b6_24264f12 (46,47). No annotations were read. All three public model datasets
currently declare CC0-1.0 through authenticated Kaggle metadata. Kaggle's own
[Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code)
describes public notebooks as Apache-2.0 code; final submission must still carry
source attribution and complete dependency/notebook-license documentation.

The initial staging gate correctly rejected our cached `checkpoint_last.pth`
resume files: the public notebook uses different `edge_predictor_best.pth`
state dictionaries. Exact best files were downloaded, verified and staged;
no hash requirement was relaxed and old files were preserved.

Bundle: `.biohub/cache/public-d4-preflight-v1`.
Manifest SHA256:
`879906e3da6164493aa9a2a83bf27012f58c780557fa03e8702a0e7d4b10a1be`.
Preflight uses one freed GPU, two CPU threads, 45% allocator memory limit and
a 1,200-second hard watchdog. It refuses existing GPU compute processes and
never stops other projects. It checks actual executed view fingerprints,
strict model loading, finite outputs, paired output differences, runtime,
VRAM and unchanged input hashes. No training, labels, Kaggle call or submission.
Full movies and official-scoring/runtime gates remain afterward; a passing
preflight alone does not authorize submission or prove beating 0.947.

Fresh AWS read-only check near 22:00 UTC: Antelume i-0d12195df0d3558f3 STOPPED,
g5.xlarge, no public IP. User was asked to free one GPU for this 20-minute test.
No AWS start, RSNA change or Kaggle GPU launch. Kaggle quota was not re-queried
because no launch is proposed; any future launch requires a fresh reserve gate.
Today remains 0/5 qualified submissions. Do not claim .945 or an improvement.
