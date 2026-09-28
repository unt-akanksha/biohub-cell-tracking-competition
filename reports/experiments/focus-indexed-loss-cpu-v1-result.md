# Integer-label training adapter: CPU functionality passed

Private/offline CPU script `indarkarhana/biohub-focus-indexed-loss-cpu-v1/1`
completed12.709s, worker6.821s. No competition inputs, model inputs or GPU.
Exact remote version1 and disabled GPU/TPU/internet metadata verified.

The new adapter converts audited integer labels (-1 unknown, source row or
source-count for known null) to the EXISTING sparse-parent/missing-null loss.
It introduces no new objective, metric modification or graph nodes. Its
synthetic result matches explicit cross-entropy over the same classes.

Verified with actual Torch execution:

- Unknown logit columns receive zero direct gradient; this does not claim
  unknown input features cannot influence supervised outputs through attention.
- A known missing-parent target pushes real-parent logits down.
- Two daughter columns can learn the same real parent.
- Invalid/fractional/out-of-range labels are rejected.
- Diagnostic, embargo and missing-role samples are refused by the optimizer guard.
- Unknown-only columns and empty source sets have the expected zero loss.
- A synthetic linear head completes40 AdamW steps: NLL1.8762802->0.2537946.
- Restricted checkpoint load reproduces predictions exactly.

Two local builder/role-guard tests passed before launch. Host then verified
every downloaded runtime file against the frozen script's embedded sources,
the saved source manifests, terminal, result flags and actual checkpoint SHA.
The same checked source must be used by any future adaptation experiment.

Frozen wrapperSHA:
01c2e21b0fe5252d74fa2ceeb6dcae3cb594a9011157a78db9a6f9d4108936de.
Actual resultSHA:
cd945f1bcd4aaafc077c5db0faa8cccdde5f68cf8f56ef0739c54a0d429aa54a.
CheckpointSHA:
b25806285f1e4c21238f849bd6157d12479e20b2da957d465da9d0bfb6d1e8f0.
AdapterSHA:
804d096dd49224cb2360676e45c4818184cab0e53293428134c04d4d1066f48d.
Artifacts:`.biohub/cache/kernel-outputs/focus-indexed-loss-cpu-v1`.

This is functionality evidence only. No biological training, score gain,
source-selection/target access or submission. It does not bypass the pending
eight-movie raw-cache and label-coverage audit. Detector cache/1 remains the
sole GPU job; no automatic adaptation training is queued.
