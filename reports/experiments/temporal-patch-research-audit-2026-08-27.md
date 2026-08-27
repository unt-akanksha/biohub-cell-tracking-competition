# Temporal patch research audit — 2026-08-27

## Decision

Run the already verified `temporal-patch-dual-fold-v1` without changing its
architecture after seeing results. If its
clean two-fold gate fails, the predeclared next family is
`temporal-patch-pair-fusion-v2`: retain the physical 3D encoder and replace the
fixed cosine-only appearance evidence with a learned, candidate-limited pair
head. Do not respond by merely widening the scratch encoder or retuning against
the four opened processed-acceptance movies.

No Kaggle GPU, public prediction, public notebook code, leaderboard feedback,
or competition submission was used for this audit.

## Primary research evidence

- Trackastra demonstrates that contextual attention, positional information,
  shape descriptors, and a division-aware parental objective matter for cell
  association; its direct projection-head ablation is materially weaker. The
  project already retains Trackastra's verified pretrained initialization as
  the geometry control rather than another failed domain fine-tune.
  <https://arxiv.org/abs/2405.15700>
- CELLECT learns a 64-channel 3D cell embedding from adjacent frames and uses a
  lightweight inter-frame MLP over the nearest spatial candidates to classify
  identity and division. This directly supports learning candidate-pair
  evidence instead of assuming cosine similarity is already calibrated.
  <https://www.nature.com/articles/s41592-025-02886-x>
- LSM-FM provides a pretrained multimodal 3D light-sheet backbone and reports
  improved transfer across downstream volumetric tasks.
  <https://arxiv.org/abs/2605.26026>

## Project-specific evidence

- The verified feature-36 LSM-FM Biohub run has 35,072,515 parameters and
  improved both pooled localization and the hardest selection movie over the
  feature-24 predecessor. It still failed the immutable per-movie gate, so its
  detector output is not admitted as a candidate.
- The LSM-FM backbone expects 64-cubed input. Applying it independently at
  every submitted node, potentially twice for reciprocal ensembling, creates a
  poor final-inference cost profile relative to the current 17-cubed physical
  encoder. It is therefore reserved as a possible training-time teacher or
  tiled feature source, not silently inserted into v1.
- CELLECT's repository is GPL-licensed and its released weights target a
  different acquisition domain. No source or weights are copied. Only the
  paper's general learned-pair-scoring principle is used for the v2 design.
- The current v1 division head is trained but is not part of checkpoint
  selection. This is safe because calibration includes exact zero division
  weight, but it makes a learned joint pair/division head a higher-value v2
  change than simply adding encoder depth.

## Frozen v2 experiment contract

The following choices are fixed before any v1 cloud or processed-acceptance
result is observed:

1. Keep the three-channel `t-1,t,t+1`, 17-cubed, physical-scale 3D encoder,
   EMA checkpointing, global disjoint embryo splits, all-daughter supervised
   contrastive loss, and two-GPU reciprocal folds.
2. Score only geometrically eligible candidates. For each source-target pair,
   concatenate normalized source and target embeddings, their absolute
   difference and product, normalized physical displacement, distance, and the
   source division logit.
3. Use one project-authored MLP with widths `1024 -> 512 -> 128 -> 1` after the
   pair feature projection. Train it with the same all-positive row objective;
   keep the embedding contrastive loss as an auxiliary term.
4. Calibrate the learned pair probability jointly with the frozen Trackastra
   geometry control on the same 12 reserved movies per fold. The grid must
   retain exact zero-weight control and the existing pooled-gain/per-movie
   regression gates.
5. Permit LSM-FM only as a separately hash-bound training-time teacher after a
   measured runtime/memory smoke. The production candidate must remain within
   the fixed 36,000-second two-T4 inference hard stop and 7,200-second notebook
   reserve.
6. Open the four processed-acceptance labels once only after both folds and
   both calibration results pass. Never tune v2 from that result or from the
   public leaderboard.

## Execution order

1. Execute v1 under the guarded two-GPU Kaggle budget, then use the
   user-provided cloud host only after the protected reserve handoff.
2. If both appearance folds and both clean calibration folds improve, run the
   one-shot processed gate; v2 remains unused.
3. If v1 fails before the processed gate, implement and run the frozen v2 pair
   head. Do not spend another experiment on a wider cosine-only encoder.
4. A competition submission remains a separate user-authorized action and
   must use exactly two GPUs with whole-movie sharding.

## Implementation and verification status

The predeclared v2 fallback is now implemented without changing the frozen
scientific contract. Its project-authored pair head has 20,869,325 total
parameters per fold and is integrated through reciprocal training, reserved
movie calibration, two-GPU processed materialization, the pinned CPU exact
gate, and exactly-two-GPU whole-movie final inference. The final sharder adds
1,024 cost units per adjacent-frame pair for v2 so pair-head work influences
load balance as well as the existing 12,288 units per encoded node.

Verification completed before packaging:

- full repository suite after the exact 71/128 inventory repair:
  552 passed, 2 skipped only for unavailable Windows symlink privilege;
- focused v1/v2 integration and runtime-builder checks: 51 passed;
- focused pinned pair-fusion exact-gate checks: 6 passed.

The independently extracted runtime package is
`.biohub/staging/biohub-temporal-pair-fusion-runtime-v2-heavy-temporal3-candidatepair-ema-t4x2-controlsource-coherent-outputverified-final-20260827.zip`.
It is 121,107 bytes with SHA-256
`3ffcef716e8ce7e964279b4d4d75621967f87238a1d693c2719db2bc607f9eab`.
Its manifest SHA-256 is
`ed589b751d4be7454a73db5bbbc04a3f8b2aa552488c7035e536fad4f6c83f71`;
all 30 declared source files verified after fresh extraction. The package has
no submission command. No v2 model has been trained yet, no Kaggle GPU was
used, and no competition artifact was submitted.

The exact execution commands and stop conditions are frozen in
`reports/experiments/temporal-patch-pair-fusion-cloud-handoff-v2.md`.

The first v1 Kaggle launch failed closed after 11.964 seconds because Kaggle
mounted one uploaded dataset directory directly. The mount-aware retry then
passed both runtime verifiers but rejected the exact 71/128 competition movie
inventory before dense probing. The exact-inventory repair binds effective
96/45 reciprocal training counts without changing model science. Run
`temporal-patch-dual-fold-v1-inventory-repair` was launched once under the live
two-T4 guard with 21.13 hours remaining and an 11-hour ceiling, projecting
10.13 hours after the ceiling versus the protected 8-hour reserve.
