# Frozen temporal-encoder experiment v6

Data extraction finished133.848s,739packets/3,108unchanged triplets. The full
temporal model functionality smoke passed:9,669,985parameters,8,288,608encoder
parameters receiving nonzero gradients, exact reload, exact daughter symmetry
and boundary mask equivalence. Median batch8step time0.02780s. Its mixed-source
smoke weights are FORBIDDEN as training initialization and will not be reused.

Before any model-selection result, freeze eight sequential fits: two source
embryos x two source-specific native encoder origins x context on/off. The same
seed, initial architecture, original source encoder, class-balanced sampler and
augmentation schedule are paired between on/off arms. Model is a trainable
3Dresidual encoder, chronological before/after/difference/product fusion, and a
parent/two-daughter symmetric nonlinear event head. The old parent/linker heads
are discarded, including the old transformer context.

Each fit gets1,500successful updates:200head-only warmup,1,300full-encoder
updates. Batch8(4positive+4negative sampled with replacement); AdamW encoder
LR2e-5, head LR2e-4, weight decay0.01, constant LR, AMPfp16, clip norm1.
Anchor encoder to its source initialization with1e-4times SUM squared parameter
differences after warmup. Same physical jitter<=0.8125um, common XYrotation and
gain for original/context images; no temporal reversal or independent crop
misalignment. Use the final checkpoint, no selection-based early stopping.

Source-only gates: probability threshold=max source-negative probability+1e-4
(conservative numerical margin), zero FP, >=50%recall and lower balanced NLL
than the fixed same-data v5geometry control. For each encoder origin, select
the source-passing arm with lowest source balanced NLL (tie favors context off).
Freeze that choice before opening opposite-embryo predictions. Do not evaluate
the losing arm on the opposite embryo. Apply the same opposite gate unchanged.
Only two individually passing chosen experts may form a fixed half/half
probability ensemble with source-only threshold calibration and the same gates.

No extra validation examples, target-pilot labels, threshold relaxation or seed
search. The seven selection positives remain a serious evidence limitation.
Even a screening pass still needs all eight complete-movie official scores,
per-embryo/worst-movie analysis and offline two-T4 runtime before submission.

Estimated fit time from smoke roughly6-12minutes plus evaluation; global hard
cap1,800seconds. Sequential GPU ownership checks each25updates, explicit
checkpoint/RNG/optimizer persistence every250updates. Save output in owned
/dev/shm to avoid filling the shared root disk. No foreign process interruption,
shared environment changes, instance shutdown or Kaggle GPU training.

R1stopped after428successful updates with a runtime error, before any selection
result. Step250checkpoint weights and optimizer are finite at AMPscale65536.
R2adds exact RNG/optimizer recovery and a bounded AMPoverflow diagnostic. A
nonfinite unscaled gradient must skip the optimizer update, halve the scale and
retry with a new balanced batch; it must not count as a successful update. At
most50skips/arm and no scale below1. This corrects numerical execution only;
the architecture, labels, learning rates,1500successful updates and gates stay
unchanged. The diagnostic weights cannot be used for model selection. Resume
the paired run from the original step250checkpoint, not from diagnostic weights.
