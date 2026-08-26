# spotiflow-synthetic-finetune-v1

Status: training completed; frozen real-movie detector acceptance is next.

The official pretrained models failed as direct Biohub replacements, so this
run tests the higher-value hypothesis rather than promoting them: dense domain
adaptation on 1,200 corrected CC0 volumes. `smfish_3d` is the warm start because
it won the disjoint eight-movie model/normalization selection, not because its
drop-in acceptance passed.

Training uses the full 35,489,892-parameter model, 128 deterministic synthetic
validation volumes, four epochs of 768 sampled crops, AdamW at `3e-5`, and
streaming lazy inputs. A CPU smoke completed forward, backward, and AdamW step
with 125 finite-gradient parameter tensors at the exact `32x64x64` training
shape. The Linux peak extensions were separately exercised by the completed
Kaggle inference run.

Resource envelope: one T4, Internet/TPU off, 6,900-second hard stop, maximum two
GPU hours, no competition input, and no submission code. The resulting model
must subsequently pass the same clean disjoint detector acceptance.

The guarded T4 run completed all 3,072 updates in 384.078 launcher seconds
(300.888 trainer seconds) and left 29.21 GPU hours. The checkpoint changed from
its warm start and finished with synthetic validation F1 0.88338; its best
observed synthetic validation F1 was 0.89243 at epoch 2. These synthetic metrics
are diagnostic only—the model is not promoted until the frozen real-movie
acceptance passes.
