# Frozen owned-feature extraction on raw FOCUS proposals

Prepared September 10, 2026. Not a training or submission job. Stage only after
the complete eight-movie detector cache and its sparse-label audit pass.

Purpose: reuse the existing frozen encoder and flow once, enabling inexpensive
association-head adaptation to FOCUS proposals without repeatedly encoding images.
The prior owned model already trained on predictions from its own detector;
FOCUS-specific proposal-domain adaptation is the new hypothesis, not generic
predicted-node training.

Scope is exactly the detector-cache contract: two previous three-frame replay
clips first, then four fitting and four diagnostic original-training movies,
all 100 frames and all 99 adjacent pairs. No source-selection or target movies.
Diagnostic movies were in the original encoder/linker training, so this is
not an independent evaluation. Unknown labels are ignored, never negatives.
At least 100 known-parent targets and one known-absent target per role are
required merely to allow feature extraction; these are not promotion gates.

Frozen checkpoint SHA256:
`76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144`.
Original quantile normalization, FP32 encoder/head, native subvoxel coordinates,
official feature indexing and positional embeddings. Flow remains the original
frozen embedded model under its FP16-quantized input/AMP contract on GPU 1;
encoder/head use GPU 0. Exact raw node order and geometry are retained.
The existing trailing-border flow adapter extends field values, not coordinates.

Small functionality gate before any new movie: reproduce both prior neural
probe matrices and normalized-image hashes exactly; resampled flow must match
the earlier cached physical flow within absolute 1e-5 microns, zero relative
tolerance. Every saved feature packet must reload exactly and reproduce the
same head logits. Any failure aborts, with no automatic retry or larger run.
The tolerance applies only to float32 physical flow, not node identity or logits.

Per-movie manifests persist packet hashes and counts. Terminal success requires
all ten records, unchanged model/flow tensor hashes, exact raw hashes, and no
optimizer. The host must verify actual outputs before any adaptation starts.

Execution: sequential Kaggle two-T4, private, Internet/TPU disabled, competition
training input and exact kernel versions. One-hour declared maximum; inherited
3480-second worker watchdog and earlier launcher watchdog. Fresh quota must
leave at least eight hours after the declared maximum. Antelume and other
projects are untouched. No GPU launch is implied by this prepared design.
