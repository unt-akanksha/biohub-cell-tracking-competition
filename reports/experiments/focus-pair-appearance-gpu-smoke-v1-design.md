# Bounded FP64 GPU compatibility test for pair appearance

The complete descriptor/real CPU-smoke receipt and independent verifier passed.
Measured eleven-movie objective passes take7.7–10.2 seconds on CPU. Scaling the
smoke evaluation counts across12folds suggests roughly11CPU hours for both
arms, an uncertain estimate rather than a runtime guarantee. Test a mathematically
equivalent GPU objective before any full comparison, without changing features,
weights, regularization, optimizer settings or acceptance thresholds.

Use only the already selected 57b7cc1e/frame31 fitting packet, the exact frozen
CPU projection/models and source hashes. Preserve every original source
candidate and known target. Also test one fixed synthetic variable-size group
layout including an empty-source/null-only group, since the full dataset has
variable candidate counts. No further fitting/diagnostic/source/target data.

Implement analytic FP64 grouped-softmax loss and gradient on one CUDA device,
with length-based segment max/sum and matrix-vector products. Compare CPU/GPU
objectives and gradients at zero and each saved CPU smoke coefficient vector:
rtol2e-7,atol2e-6. Compare CPU fitting counts/metrics on reconstructed descriptors
against the verified host controls. Fit each of the two smoke heads from zero
using unchanged SciPy L-BFGS-B max500,gtol1e-8,ftol1e-12 and physical coefficient
bounds. Require final objective within1e-5 of its CPU counterpart, exact discrete
parent/absent decisions, and exact saved-model prediction reload. These are
functionality/equivalence checks, not held-out quality or a submission.

Notebook private, GPU enabled, TPU/Internet disabled, competition input attached;
all required training packet/source/model bytes embedded with checksums. No
dependency downloads, public checkpoint, cloud start/stop or shared RSNA changes.
Declared whole-notebook cap300seconds, worker allowance240seconds less elapsed
setup, emergency timer285seconds. Push with Kaggle's300second runtime limit.
Refresh quota before launch and reject if remaining-300/3600<8. No concurrent
Biohub GPU experiment. Persist input/runtime/model hashes, numerical comparisons,
GPU/worker/launcher timing and peak memory. A successful smoke still requires
full-size throughput/memory admission before the complete quality comparison.

API reference: [PyTorch segment reduction](https://docs.pytorch.org/docs/2.14/generated/torch.segment_reduce.html).
