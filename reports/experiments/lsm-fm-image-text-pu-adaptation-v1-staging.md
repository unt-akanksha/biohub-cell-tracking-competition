# LSM-FM image-text PU adaptation v1 staging

## Decision

Launch a new private training-and-validation experiment using the official
feature-36 LSM-FM image-text student. This is a search candidate rather than a
paired ablation: it intentionally increases backbone capacity, pretraining
scope, per-movie diversity, optimizer coverage, and adaptation depth after the
feature-24 run improved the limiting movie but missed the immutable `0.65`
worst-movie gate.

No competition submission is created or authorized by this run.

## Primary-source and license audit

- Paper: <https://arxiv.org/abs/2605.26026>
- Official repository: <https://github.com/AdinaScheinfeld/lsm_fm_public_repo>
- Audited repository commit:
  `cb772551652c3ee762048936e396743fe9a65e10`
- Official weights: <https://doi.org/10.5281/zenodo.20146516>
- Zenodo file: `swinunetr_image_text_best.ckpt`
- Published size: `1,040,329,520` bytes
- Published and verified MD5: `c33bbaad11934ffe0b0036ecbe144b78`
- Verified SHA-256:
  `f08122ee471fae0c555f3b71df37087d6dd78ffda7bf000f0510aa262c1c5d15`
- Weight license: CC-BY-4.0
- Runtime implementation: MONAI 1.5.1, Apache-2.0

The pinned official repository has no license file despite its README badge,
so no official finetuning implementation was copied. The runtime contains only
the licensed `student_encoder.*` tensors, our detector/trainer/evaluator code,
MONAI, attribution, and license text. It does not include the BERT text encoder.

## Exact checkpoint transfer

The official checkpoint was loaded with PyTorch's restricted weights-only
unpickler after allowlisting only its explicit MONAI metadata and NumPy dtype
classes. The extracted student has:

- 159 state tensors;
- 36,032,614 state elements, including persistent buffers;
- feature size 36;
- one image channel and a 512-channel pretext output;
- masked reconstruction, teacher distillation, image-text alignment, and
  image-text contrastive pretraining.

All non-output tensors strict-load into the feature-36 MONAI SwinUNETR. Biohub
replaces only the 512-channel pretext head with a newly initialized one-channel
heatmap head. The resulting detector has 35,072,515 parameters, compared with
15,702,979 in the previous feature-24 candidate.

## Training candidate

- training movies: 187;
- deterministic seed: `20260827`;
- pairs per movie: 2;
- target optimizer steps: 3,072;
- warm-up/unfreeze step: 768;
- unfrozen Swin stages after warm-up: 4;
- decoder learning rate: `1e-4`;
- encoder learning rate: `1e-6`;
- weak and strong activation graphs: serialized for a 16 GB T4;
- frozen pseudo-label teachers: the same two independently seeded TemporalUNets;
- validation exclusion: all 12 clean validation movies;
- selection: 8 movies with pooled recall >= `0.80` and worst recall >= `0.65`;
- acceptance: 4 movies, read only if both selection gates pass.

Public predictions, public Kaggle code, leaderboard selection, and metric hacks
are absent. Teacher outputs are used only to construct conservative PU targets.

## Runtime and preflight evidence

- private runtime dataset:
  `indarkarhana/biohub-lsm-fm-image-text-pu-runtime-v1@version1`;
- staged/downloaded manifest SHA-256:
  `5b73f47d4695492ebad504539ee1be9538faa259115a848f1ecd348c74154721`;
- stripped checkpoint SHA-256:
  `aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7`;
- downloaded MONAI inventory: 428 files, all hash-bound;
- exact 64-cubed CPU forward/backward/AdamW step: finite;
- focused tests: 35 passed;
- nine-check preflight logical SHA-256:
  `966a3a36985f98a14ec469af7d674fde5f21bba5a5735c84b7847505f4b7530a`.

The notebook is private, uses one T4 for training, has internet and TPU disabled,
and is guarded by a 6,900-second hard stop. Any later submission remains a
separate action requiring two CUDA devices and whole-movie sharding.
