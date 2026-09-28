# Parent-dropout training: verified diagnostic failure

Completed 800 fixed alternating original/augmented fitting updates. Final
unweighted diagnostic NLL0.2183131454, correctparents2588/2645, correctabsent11/27.
The original neural control is0.2190088986,2585parents,12absent; the physical
control detects18absent. The unchanged missing-parent gate therefore FAILS.
No source-tracking promotion, new target opening or submission followed.

The full training run started the original owned model, not the smoke checkpoint.
All1121 fitting augmentations exactly replayed the audited labels/identities.
Only the association head trained; frozen encoder/detector/flow hash unchanged.
All nine planned checkpoints (4,100,...800) were recovered and hash-verified;
each worker save strictly reloaded and exactly replayed a real cached-pair logit
matrix. Host verification reconstructed both shuffled fitting queues and checked
every one of the800 update identities, augmentation decisions and finite gradients.
Original complete diagnostic controls and the final gate independently replayed.

Worker116.633748seconds; launcher169.451439seconds (2.82minutes), cap900seconds.
Peak allocatedGPU163964928bytes. No Biohub GPU job remains active. Fresh shared
quota after completion:8.88h, preserving8h. AWS credential file remains last
modified14:18:52UTC; no new cloud query or claim of current instance state here.

Final checkpointSHA:
`90b3de729895214a85661c977e6a9ba89c2d58228b1092cc1a8f69baf676ccbe`.
Worker resultSHA:
`aaebf0646eaf0f42ad8b3da14a7109b423072a22b3e9b5c5d2093a415dd043b3`.
Host receiptSHA:
`854d43703018ff135cfe464c5597903df484fa9ac52d4fc3fc6eaa71f4c1c623`.
Artifacts: `.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1`.

Interpretation: adding conservative candidate-dropout supervision did not fix
real missing-parent discrimination in this frozen-encoder head configuration.
Do not extend it unchanged, adjust the exposed diagnostic threshold, or present
the small NLL gain as tracking-score evidence. This does not prove all missing-
parent augmentation or architectures ineffective. Qualified submissions today0/5.
