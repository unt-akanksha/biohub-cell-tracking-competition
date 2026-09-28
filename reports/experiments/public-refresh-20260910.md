# Public-source refresh after the rejected longer fit

## Latest screen during paired detector training

The latest-run CLI listing was refreshed again. New
[Hank Lin one-knob fork](https://www.kaggle.com/code/hanklin2004/biohub-0942-baseline-one-knob-fork)
was downloaded as source only, never executed. Notebook SHA256:
`b487e51b88b7aa5415146f9e851e865b6106c8862065ad0c819bed18a1b7bdfc`.
It retains the support50/seed314159/DeepCenter family, detector threshold0.96,
secondary edge weight0.15, short-track pruning and division fraction caps.
Its markdown reports0.942 and warns that notebook titles and four-movie proxy
scores can disagree with submissions. These are author-reported figures, not
our independent reproduction. No new checkpoint or validated architecture
was established. Do not import its public-score-selected settings as evidence
of a generalization gain. No known excluded exploit source was pulled.

The freshly indexed [drissou-milad repository](https://github.com/drissou-milad/biohub-cell-tracking)
describes a small Gaussian-peak/CNN/Hungarian demo and explicitly reports no
competition submission or multi-sample training yet. It is not a stronger
model source. Discussions732103 and737896 returned no readable body through
the browser; their latest contents were not verified.

Sources were discovered by latest run date, not leaderboard score. Only source
and metadata were pulled; no public predictions or new public weights were
downloaded or executed. Known older metric-exploit notebooks were skipped.

- [Latest baseline script](https://www.kaggle.com/code/mohit98765/biohub-cell-tracking-v1-baseline):
  SHA `019a89b3a124fa68c2b4a7f7f02e23feda66564503374a32eb6602ae148658df`.
  Screening found its stated division strategy targets AOGM-style FN/FP costs,
  not the pinned Biohub scorer. It widens daughter searches to100/110 and
  removes angle constraints. Offline metadata attaches no model artifacts,
  while the script attempts pip/model downloads and falls back to LoG.
  No reason to spend GPU on this as a stronger validated base. This is a
  screening rejection, not a claim that every line was audited for exploits.
- [Secondary edge-feature TTA](https://www.kaggle.com/code/sjlee101/biohub-lf-sectta-w1):
  notebook SHA `ecdee5465fdbe63cf20e52fc9bddf49b4c6dcf9f6027934c38debf85fd3a1046`.
  Same support50/seed314159/DeepCenter datasets and existing public configuration
  family; the seed314159 manifest already records199 training movies. No new
  independent checkpoint provenance. New hypothesis worth isolating: inverse-
  aligned spatial encoder-feature averaging for the linker. Do not copy its
  public-score-selected settings, graph pruning, or repair rules as evidence.

The new local experiment uses our hash-bound jointc502 checkpoint, eight unique
XY dihedral transforms with tested inverses, and preserves the native detection
logits exactly. It is source-inspired, not a run of the public notebook. A CPU
gate passed four tests in3.52s; a real three-training-frame probe is prepared,
not a submission or full-movie accuracy claim.

[From Detection to Identity](https://github.com/luisrosa/from-detection-to-identity)
provides aggregate analysis and methodological examples, not released production
weights or an end-to-end tracking system. Its production architecture and
inference remain private. It is not a checkpoint source.

The [competition discussion on improvement order](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543)
suggests detection before linking and divisions. Treat this as participant
advice, not verified comparative evidence; our own longer-fit regression is the
reason not to promote a division-only change as a complete repair.

## Refresh while integrated image-motion linker trains

Latest-run listing was refreshed after the integrated 1,000-step fit was
accepted as `biohub-image-motion-linker-v1/2`. Source-only screening:

- [Adaptive kernel](https://www.kaggle.com/code/binasalama/biohub-v2-adaptive-kernel),
  SHA `56662f271f3a492308d3ce8dc0df266abcfee619f44e968168d5a69e47035c8c`:
  percentile thresholding, connected components, Hungarian distance matching,
  gap-2 edges and removal of components shorter than three nodes. No trained
  checkpoint or complete-movie comparative evidence in the source. Its direct
  gap-2 edges also differ from our next-frame graph contract. Do not adopt as
  a stronger base. This screening is not an accusation of intentional hacking.
- [DeepCenter TTA v015](https://www.kaggle.com/code/sjlee101/biohub-lf-dctta-v015),
  SHA `b47475d7beecc1276d6a43e21418782be67625ac9a2e5eebae7eb568ecb82543`:
  adds inverse-aligned spatial averaging to the DeepCenter veto stage, retains
  the support50/seed314159 family, short-track filtering and fraction caps.
  No newly established independent checkpoint provenance. Its embedded comments
  cite public-score changes from threshold variations; those are not our model
  selection evidence. TTA is an idea, not proof of improved generalization;
  our separately tested linker feature-TTA experiment already regressed.

The official repository's local and remote HEAD both resolve to
`075fc5f5a52d11077f9dc2b074644618f26939e2`. Its README explicitly attributes the
author baseline checkpoint to the documented three-epoch command. However,
that command uses `data/train/dataset_splits.json` if present and only otherwise
falls back to a seed-0 split. The published artifact listing still does not
include the original split manifest. This improves command provenance, not
proof of exact held-out movie membership; the prior independence block stands.

A fresh web read of discussion 737543 failed, so no claim is made that its
latest comments were verified. No known older metric-exploit notebook was
pulled, and none of these new sources was executed or used for predictions.

## Further source-only screen during detector-TTA validation

- Newly discovered `sushanthtiruvaipati/biohub-xiaoleilian-divaug-fork-v1`
  was screened before execution. Despite its title, its final cell explicitly
  calls itself a metric hack and adds negative-time nodes. **Rejected and
  marked DO_NOT_EXECUTE** in the ignored research cache. Add it to future
  exclusions. SHA `ac8cb4f44529b5711284f6841a42ff3ad1e0ef509de84ac202cd82e0ef18d4ee`.
- [Detector threshold variant](https://www.kaggle.com/code/sjlee101/biohub-lf-det095)
  sets `BIOHUB_DET_THRESHOLD=0.95` while retaining the public seed314159 model
  family. No new independent checkpoint evidence established. SHA
  `b650040477d153e1a533c9369c3768b1e6b84b726ce17fd90c3deb247779207c`.
- [Another competitor's project](https://github.com/JunhaoLiXD/Biohub_Cell_Tracking)
  reports a reproduced public0.941 model and a motion-EMA proxy gain awaiting
  reproduction. This is their report, not our verified clean/held-out result;
  the README does not establish exact pretrained train/validation disjointness.
  No code or weights adopted. Our causal-motion branch already tests motion
  history and remains below the learned image-flow reference.

Fresh discussion page retrieval returned no readable body. Do not claim the
latest comments were reviewed. No screened notebook was executed.

## Further date-run refresh during the adaptation cache

CLI metadata now lists evgendvorkin's revision at12:17:56UTC, tharunkumar369's
lineage notebook at12:17:07UTC and a new
[Hank Lin combined-knobs revision](https://www.kaggle.com/code/hanklin2004/biohub-0942-v2-combined-knobs)
at12:01:43UTC. Pulled only the latter's source/metadata into ignored cache
`public-audit-20260910/hank-combined-1201`. SHA256:
`d2fa7b0ca9bd503851c0cfae6bd40e5dba0601250e699819091da833eb850c97`.

Its notes explicitly describe leaderboard-guided parameter changes. Source
retains the shared public detector/linker/DeepCenter family and changes
configuration weights/thresholds, not a newly trained architecture. No execution,
weights, predictions or settings adopted. This source-only inspection is not
a full hack-free certification or verification of its claimed leaderboard score.
The newer tharunkumar revision was not pulled; the previous project audit had
already identified all199-movie training overlap in that lineage's ranker.

Known fabricated-node/negative-time sources remain excluded. Browser access
to the notebook failed; discussion737543 again returned no readable body, so
latest discussion comments were not verified. No leaderboard-selected values
entered our adaptation pipeline.

## 11:29 UTC date-run refresh and source-only inspection

Updated notebook:
[evgendvorkin/biohub-0-942-lb-proxy-score-0-9417](https://www.kaggle.com/code/evgendvorkin/biohub-0-942-lb-proxy-score-0-9417),
listing lastRun11:22:28.957UTC. Pulled source and metadata only through Kaggle
CLI to ignored cache `public-audit-20260910/evgen-1122`. Notebook SHA-256:
`7611ed5b54dded0b4a1111a8d4458820413b2d1e21d5e1a44ff1907ec5e9710f`.

Markdown explicitly compares proxy/leaderboard outcomes across threshold,
secondary-model weight and division-geometry revisions. Current configuration
includes secondary detector weight0.80, edge weight0.20, bidirectional0.15,
short-track filtering and several fraction caps. These leaderboard-selected
settings are not independent evidence for our model selection. No weights,
predictions or settings adopted; source not executed. This limited inspection
does not certify the entire notebook hack-free or identify the public best.

Also listed newer revisions of crystalbaby/biohub-lf-dctta-v020 and
sjlee101/biohub-lf-dctta020-sectta1; these revisions were not pulled here.
Known fabricated-node/negative-time exploit sources remain excluded.
Browser notebook retrieval failed and the discussion page returned no readable
body. Therefore no claim of verified latest discussion content is made.

## 13:11 UTC source-only refresh

Latest-run listing showed a12:31:26UTC revision of
[tharunkumar369/biohub-cell-tracking-lineage-submission](https://www.kaggle.com/code/tharunkumar369/biohub-cell-tracking-lineage-submission).
Source and metadata only downloaded to `public-audit-20260910/tharun-1231`.
Notebook SHA256 `aeb5b94dba28c7c93855772c560935962a1200763b5fa3b502c39d4ed3cdeadd`.
This revision differs from the earlier22-feature ranker snapshot: current code
uses350-epoch primary override dfb848aa..., dual temporal models, eight-view
edge-feature augmentation, secondary feature blend0.75, harmonic association
and DeepCenter veto. Its own receipt sets
`leaderboard_feedback_used_for_configuration=True`; a seven-option
post-processing sweep and score-based combination are present. Source still
pins the old packaged metric31baf45b.../divisiond1cf1e0a..., not our authoritative
patched scorer. Its claimed public/proxy scores are not verified evidence.

No weights or predictions downloaded, no source executed, no parameters
adopted. This limited source review is not a whole-notebook metric-integrity
certification. Keep the longer-trained models/consensus ideas as hypotheses,
not evidence supporting our promotion. Known fabricated-node/negative-time
exploit notebooks were not pulled. No new readable discussion body obtained.
