# Temporal contextual autocast runtime propagation

Date: 2026-08-27

Status: repaired private runtimes remotely verified; every downstream notebook
rebound and tested; no downstream kernel launched and no submission created.

## Finding

The first two-GPU ZebraHub contextual pretraining attempt failed before
optimizer step 1 because CUDA autocast produced FP16 contextual logits while
the sparse destination tensor was FP32. PyTorch index assignment requires the
source and destination dtypes to match. The repair promotes the logits to the
destination dtype at the scatter boundary, preserving gradients and the FP32
loss path. It changes no architecture, data split, seed, objective, training
schedule, checkpoint gate, calibration grid, or submission policy.

The repaired path passed the local CPU bfloat16 autocast regression, including
a combined forward/backward through the temporal patch encoder, contextual
candidate logits, pair loss, reciprocal-parent loss, embedding loss, and
division loss. All outputs, losses, and populated gradients were finite.

## Published private runtimes

- Pretraining runtime:
  `indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3`, version `9`
- Pretraining manifest:
  `aff4e21f675848f94cbfd42e4d1a43ebc5b470fbc21db3934f85797066fc65cb`
- Acceptance runtime:
  `indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1`, version `3`
- Acceptance manifest:
  `6aefc98b953a15b853731bc970a9734ddcab08a51938bcc9769f29fa6bba1f29`
- Transfer/downstream runtime:
  `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`, version `4`
- Transfer manifest:
  `cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d`

Each listed runtime was downloaded again through the Kaggle CLI and matched
its local manifest and per-file hashes. Pretraining runtime versions `1`-`8`,
acceptance runtime versions `1`-`2`, and transfer runtime versions `1`-`3` are
excluded from their respective repaired executions.

## Rebound notebook evidence

| Stage | Admissible runtime | Notebook SHA-256 | Amendment event |
|---|---:|---|---|
| Frozen ZebraHub acceptance | `3` | `41835ade5603bb7f64c3703d2eb5568198b3d4ce26e43bef13c708b68a3bc874` | `evt-f2a4fe22c2ba4e4bb69b430f1912294b` |
| Competition transfer | `4` | `7ca20f9097d14dba745a3eb0aededc561e691e904cb85274d91219a282c4ee61` | `evt-8317c749fe0b4fda9e897e2c5d53c325` |
| Clean calibration | `4` | `4596cd9fd7211c27f6b437268c8e719847f0e7cce91438cc86195e79aba497e1` | `evt-b16268b01cbc48ab888afbf589341769` |
| One-shot processed gate | `4` | `973f30cd25833d60b0552f4358566c7e6c535ce6717789f264cf2b5a185294a7` | `evt-e7d6ff404968417281296d40887dde04` |
| Final candidate | `4` | `e21119833e11fe514caf355dc46e57e8448f0fcd4a1fb4b47f7607e25959813c` | `evt-c462d6dd56ad4702b4ec495cab55ade7` |

The original registrations remain intact. The listed append-only amendments
record both old and new versions, manifests, and notebook hashes, with
`scientific_recipe_changed: false`.

## Verification and execution boundary

The five focused notebook-builder modules passed `10/10`, including byte
determinism, exact two-GPU assertions, disabled internet, whole-movie sharding
for the processed and final stages, absence of any Kaggle submit command, and
the new fail-closed runtime-manifest bindings.

The repaired pretraining retry was launched only after a live quota guard
projected `10.76` Kaggle GPU-hours remaining after its declared window, above
the required eight-hour reserve. It uses exactly two T4 GPUs, no Biohub
competition input, and no submission command. Downstream stages remain gated
and unlaunched.
