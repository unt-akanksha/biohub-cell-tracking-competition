# spotiflow-synthetic-finetune-v1

Status: staged first learned detector candidate.

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
