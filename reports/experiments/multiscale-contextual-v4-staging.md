# Multiscale contextual v4 staging

Date: 2026-08-27

Status: project-authored model, ZebraHub pretraining wrapper, reciprocal Biohub
transfer wrapper, metadata contract, downstream model loader, and portable
runtime implemented and locally tested. Not registered, remotely published,
launched, calibrated, scored, or submitted. The contextual v3 recipe and
evidence remain unchanged.

## Why this is the next distinct capacity ablation

The current competition discussion reinforces two decisions already present in
our pipeline. First, public detector weights were trained on all annotated
movies, so validation on those movies is contaminated and can reverse an
ablation's sign on the leaderboard. Our external-pretraining plus reciprocal
leave-one-embryo-out gates avoid that trap. Second, adding training and feature
diversity is the clean route beyond the public two-seed plateau, not copying a
public prediction or selecting on the public leaderboard.

ASCENT (ICCV 2025) reports a complementary representation idea for 3D
fluorescence tracking: compress volumetric information into an efficient 2D
representation and apply a deep encoder. The v4 implementation retains that
high-level research hypothesis but uses no ASCENT source, weights, constants,
predictions, or evaluation results. It is a project-authored residual branch
tailored to our physical temporal patches and contextual edge objective.

Primary references:

- Kaggle CV/leakage discussion:
  <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160>
- ASCENT open-access paper:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html>
- CELLECT (Nature Methods, 2025), whose adjacent-frame 3D embeddings, division
  head, physical-scale concerns, and recommended motion/global-optimization
  extensions independently support the combined v3/v4 design:
  <https://www.nature.com/articles/s41592-025-02886-x>
- HOCT (July 2026), whose edge-centric geometric context supports comparing
  candidate links; its public checkpoints remain excluded after their clean
  Biohub acceptance regression:
  <https://arxiv.org/abs/2607.11754>

## Architecture

- Family: `temporal_multiscale_contextual_pair_fusion_v4`
- Parent/control: `temporal_contextual_pair_fusion_v3`
- Parameters: `46,386,607` per fold (`2.2357x` v3's `20,747,761`)
- Native path: unchanged physical temporal 3D residual encoder
- Projection inputs per temporal channel: axial mean, max, stable standard
  deviation, center plane, and learned axial-attention projection
- Projection branch: 96-channel 2D stem followed by seven residual blocks up
  to 768 channels
- Fusion: residual appearance and division adapters on top of the complete v3
  outputs
- Edge representation: unchanged contextual candidate token and
  outgoing/incoming/transition edge-set pooling
- Objective: unchanged outgoing daughter ranking, 0.35-weight reciprocal
  parent ranking, embedding auxiliary loss, and division loss

The warm-start loader strict-loads every v3 3D encoder, embedding projection,
division head, logit-scale, edge-token, and edge-head tensor. Only new axial
and residual-adapter keys may be absent. It then zeroes the final residual
layers, making the initial v4 predictions numerically equivalent to the exact
accepted v3 checkpoint. Missing or shape-changed shared tensors fail closed.

## Frozen experimental boundary

V4 is conditional on v3 passing its ZSNS005 selection/audit and untouched
ZSNS001 gates. Its pretraining wrapper then starts from the accepted v3 fold
and uses the same:

- 64 ZSNS004 optimization shards;
- eight ZSNS005 selection and eight separated one-shot audit shards;
- two isolated GPU workers with distinct seeds;
- microscopy-safe augmentation and unaugmented validation;
- at-least-0.01 composite, positive top-1/MRR, and nonnegative division-top-2
  gain gates; and
- prohibition on competition data, public predictions, leaderboard selection,
  and submission construction.

The reciprocal transfer wrapper uses exactly two GPUs and inherits the v3
transition adapter, 20,000-step low-rate Biohub transfer recipe, real/synthetic
replay, both-fold improvement requirement, synthetic-retention floor, unopened
calibration movies, and absence of a submit command. It accepts only a strict,
hash-bound v4 ZebraHub terminal that records the accepted v3 warm start.

No v4 GPU run is authorized while the current v3 evidence lane is active. A
future launch must pass the live quota guard and leave at least eight Kaggle GPU
hours. If v3 already yields an accepted competition candidate, v4 is an
independent improvement experiment, not a reason to delay the authorized v3
submission.

## Local evidence

- Default parameter inventory exactly matches `46,386,607`.
- The model is more than twice v3's capacity.
- Axial attention begins as a stable uniform projection.
- V3 warm-start rejects missing or shape-changed shared tensors.
- Warm-started v4 initially reproduces v3 embeddings/division logits within
  floating-point normalization tolerance.
- A complete CPU bfloat16 path through the 3D branch, projected 2D branch,
  contextual candidate scatter, bidirectional loss, and backward pass is
  finite and populates gradients in both branches and the edge head.
- The pretraining and transfer wrappers remain distinct run families and retain
  exactly-two-GPU/no-submission contracts.
- The local portable runtime contains the v4 model and both wrappers alongside
  the complete hash-bound v3 training, calibration, processed-acceptance, and
  whole-movie inference stack. Its private remote dataset remains unpublished.
