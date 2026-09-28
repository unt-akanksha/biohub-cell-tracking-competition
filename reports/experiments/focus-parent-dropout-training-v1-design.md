# Fixed 800-step original/counterfactual head training

Requires host-verified successful parent-dropout smoke-v3/1 before staging.
Start from the original integrated model, not the four-step smoke checkpoint.
Same audited 12 fitting movies and unchanged four diagnostic movies. Reconstruct
every augmentation and require exact audited identities before optimizer.

800 updates, alternating original on odd steps and augmented on even steps.
Each stream uses its own shuffled full 1,121-pair queue; both consume the original
Python RNG seeded244691. No diagnostic input, sampling by loss, threshold sweep,
or post-result checkpoint choice. Use the smoke-tested square-root weight from
20,293 parent/1,458 null fitting labels, AdamW1e-4,weightdecay1e-4,clip1,FP32.
Freeze encoder/detector/flow; only association head trains.

Replay initial complete physical/neural diagnostic controls before optimizer.
Save optimizer/model/RNG/both queues atstep4 and every100; each saved checkpoint
must strict-reload and reproduce real cached-pair logits exactly. Evaluate only
the fixed final800step model with the ORIGINAL unweighted complete diagnostic.
Gate unchanged: NLL below both controls, parentcorrect>=2585/2645 and
absentcorrect>=18/27. A pass only permits separate full-source tracking validation.

Use same private offline two-T4 environment and900second cap as successful smoke,
780second internal deadline/840second watchdog; checkpoints preserve interrupted
work. An interrupted run is incomplete, never a successful quality result.
Freshquota must preserve8h after0.25h worstcase. Sequential jobs; no cloud/RSNA
mutations, new target movies, inference threshold changes or submission.
