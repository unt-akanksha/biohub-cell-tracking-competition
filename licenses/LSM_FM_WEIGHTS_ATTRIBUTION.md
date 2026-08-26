# Light Sheet Microscopy Foundation Model weights

- Work: Light Sheet Microscopy Foundation Models
- Creator: Adina Scheinfeld
- Paper: https://arxiv.org/abs/2605.26026
- Model record: https://doi.org/10.5281/zenodo.20146516
- Official repository: https://github.com/AdinaScheinfeld/lsm_fm_public_repo
- Audited repository commit: `cb772551652c3ee762048936e396743fe9a65e10`
- Source checkpoints:
  - `swinunetr_image_only_best.ckpt`
    - SHA-256: `ef0f3d100f9a9aaa5d9a48bb9b07e7f1b0e0d1cc690cdcd308b1e0446bd01634`
  - `swinunetr_image_text_best.ckpt`
    - SHA-256: `f08122ee471fae0c555f3b71df37087d6dd78ffda7bf000f0510aa262c1c5d15`
- License recorded by Zenodo: CC-BY-4.0

Each Biohub runtime contains only a mechanically stripped copy of the relevant
checkpoint's `student_encoder.*` tensors. The source checkpoints are otherwise
unmodified. Biohub replaces the 512-channel pretext output layer with its own
one-channel detection head during training. The feature-36 image-text model was
pretrained with masked reconstruction, teacher distillation, image-text
alignment, and contrastive objectives; Biohub does not redistribute its text
encoder or copy the official repository's finetuning code.
