# BiomedParse v2 transfer audit — 2026-08-26

## Decision

Do not use BiomedParse v2 weights in this prize competition without explicit
license permission. Retain its 2.5D small-object architecture as a research
reference only; it is not the next Biohub experiment.

## Primary evidence

- Official repository: <https://github.com/microsoft/BiomedParse>
- Audited commit: `e02096c03af0d79c6994ffc2d60a49eeb0361e1f`
- Repository code license: Apache-2.0
- Official model record: <https://huggingface.co/microsoft/BiomedParse>
- Model record revision: `e473e5b2b1a3f44649734afd3dc7cf1770aaa9e2`
- Model access: public metadata but gated checkpoint access
- Model license: CC-BY-NC-SA-4.0

The model card and repository state that v2 covers 3D microscopy, including
light-sheet brain activity, plaques, nuclei, and vessels. Its 3D path is not a
native volumetric transformer: it uses fractal volumetric encoding to combine
neighboring slices into RGB images and performs slice-wise inference. It is a
text-prompted segmentation system built around BoltzFormer, not a point-localization
or temporal-association model.

## Relevance and blockers

The small-object BoltzFormer decoder and light-sheet exposure make the model
scientifically relevant. However:

1. the checkpoint's non-commercial, share-alike license is not a safe default
   for a prize competition or distributable Kaggle candidate;
2. access is gated, preventing a fully unattended, immutable checkpoint audit;
3. Biohub's single-channel 4:1 anisotropic movies would need a new fractal-RGB
   preprocessing and point-extraction path;
4. the pretrained targets are semantic masks/prompts rather than cell-center
   heatmaps or temporal links.

No weights were downloaded, no source was copied into a runtime, and no GPU
experiment is scheduled. Reconsider only if the checkpoint owner grants
competition-compatible permission and a clean selection-only prototype shows
that its nuclei prompt improves the limiting `6bba_57b7cc1e` localization.
