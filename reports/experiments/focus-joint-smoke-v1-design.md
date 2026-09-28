# Joint encoder/head adaptation: four-step resource and gradient gate

Start from original owned checkpoint76f7da6e..., not a failed head checkpoint.
Use immutable12fit/4diagnostic feature/label contracts for provenance, but open
images/optimize only4 fitting pairs in this smoke. Deterministically selectfirst,
largest source-times-target matrix, most verified-null targets, most known
parents; deduplicate then fill with earliest fitting pairs. No truncation.

Before updating, recompute original FP32 encoder-indexed features and full
neural-plus-physical logits on all4 pairs and require exact cached-feature
and logit replay. Original quantile normalization, native-coordinate indexing,
positions, original2movie physicalGaussian and null-4.5 remain unchanged.

Enable gradients in temporal3DUNet and associationtransformer. Keep encoder
eval mode so runningBatchNorm statistics remain fixed; trainable weights still
receive gradients. Detectorhead and embeddedflow weights remain frozen. The
actual detector is externalFOCUS; no new detector predictions are made here.
AdamWlrencoder1e-5/head1e-4,weightdecay1e-4,globalclip1,seed244691. AMPFP16 with
GradScaler, FP32 weightedparent/null loss using the same fitting-count-derived
weight8.1728227. Foursteps only, finite nonzero gradients in both trained groups.

Save optimizer/scaler/RNG states and full model atstep4. Require exact FP32
real-image checkpoint reload, changed encoder/head tensors, unchanged detector
and flow tensors, finite loss/norm, and report peakallocated/reserved GPU memory.
No diagnostic scoring or model-quality selection here. Passing permits planning
a larger fixed training run, not submission or quality promotion. Larger run
still needs the unchanged diagnostic/full-movie gates and its own budget check.

Private offline Kaggle twoT4 environment; declared1h maximum with freshquota
preserving8h reserve. Small imageLRU bounds hostmemory. No cloud restart, shared
environment change, RSNA mutation, source/target images or competition submission.
