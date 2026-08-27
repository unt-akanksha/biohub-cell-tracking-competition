# Temporal contextual processed acceptance v3 staging

Status: local one-shot two-GPU materialization kernel staged and tested; not
pushed, launched, exact-scored, or submitted.

This is the first stage allowed to touch the four processed acceptance movie
images. It can run only after both contextual transfer folds and both clean
calibration folds improve. The selected checkpoints and blend weights are
immutable at that point, so this stage cannot tune or redirect the candidate.

## Frozen boundary

- Candidate family: `trackastra_contextual_pair_fusion_blend`
- GPUs: exactly `2`, one embryo prefix and two whole movies per GPU
- Processed movies: `44b6_12dfb391`, `44b6_267148e4`, `6bba_062c8d37`,
  `6bba_07e24132`
- Frozen comparator CSV SHA-256:
  `6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b`
- Frozen raw-confidence graph tree SHA-256:
  `559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e`
- Comparator role: topology/control only
- Processed ground truth: unread
- Hyperparameter selection: forbidden
- Exact scoring: separate CPU stage
- Competition test images: unread
- Kaggle submission command: absent

The public comparator is not treated as our model. It supplies only the fixed
detector topology and raw confidence control. The contextual candidate is
rejected during materialization if it does not change association edges or if
its CSV hash equals the comparator, preventing an exact public replica from
advancing.

## Package

- Runtime: `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`, only
  version `4`; version `3` is excluded because its sparse contextual-logit
  scatter was not autocast dtype-safe
- Runtime manifest SHA-256:
  `cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d`
- Kernel:
  `indarkarhana/biohub-temporal-contextual-processed-acceptance-v3`
- Notebook SHA-256:
  `973f30cd25833d60b0552f4358566c7e6c535ce6717789f264cf2b5a185294a7`
- Metadata SHA-256:
  `0fd5798a6ab4b2c005c93b4991070abf4025f8ef7dcb454f20081c5468a6a3c6`
- Notebook watchdog: `21,600` seconds
- Materializer hard stop: `19,800` seconds
- Internet: disabled

After materialization, the candidate must be downloaded and scored once with
the pinned official CPU scorer. Promotion requires positive pooled exact gain,
identical node recall, no movie regression worse than `0.002`, changed edge
sets, and exact contextual-v3 architecture/hash evidence. Only accepted
evidence may unlock later two-GPU whole-movie test inference; uploading remains
a separate user-authorized action.
