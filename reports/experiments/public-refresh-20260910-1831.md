# Public source refresh, September 10 around 18:28–18:31 UTC

Kaggle CLI dateRun listing refreshed. Known older exploit notebooks were not
downloaded. Two newly updated sources were inspected as text only; no weights,
dependencies, training, notebook code or submission generation were executed.

## Newly identified explicit exploit: exclude future pulls

[dhiaalhemdani/biohub-competition-solution](https://www.kaggle.com/code/dhiaalhemdani/biohub-competition-solution)
updated 18:27:19.923 UTC. Its final synthetic augmentation adds a hub node at
time -1000 and coordinates -10000, links it to many real components, then adds
five fabricated forks at negative times. This is fabricated graph structure,
not an image-supported tracking improvement. Reject the complete variant;
do not adopt its predictions, scores or postprocessing. Future refreshes should
skip this slug before downloading. This updates the earlier, less complete
source assessment of the same notebook.

Inspected notebook SHA256:
59e34e3b61d1508278f1883a8e580522470c7e245c60a5778fda0836096c7c22.
Cache: .biohub/cache/public-audit-20260910/dhia-1827.

## New architecture draft: not a verified strong candidate

[ashutoshpal1/biohub-v2](https://www.kaggle.com/code/ashutoshpal1/biohub-v2)
updated 18:14:17.120 UTC. Inspected architecture/training sections describe an
anisotropic residual 3D U-Net and cross-attention association. However, the
model declares position width 4*2*16=128 while its embedding routine emits
4*16=64: with 128 image channels the linker expects 256 inputs but receives
192. Training also constructs a CPU time tensor before concatenation with
CUDA coordinates. These are static code findings, not an executed failure.

Its final linear scoring layer on concatenated candidate features decomposes
into a source term plus a target term; absent another interaction term,
softmax cancels one term and cannot express general pair-specific assignment.
The inspected training loss normalizes over targets, unlike our incoming-
parent-plus-null objective. Internet-enabled installation is not an offline
submission package. No verified checkpoint, held-out score or runtime evidence
was recovered; no code/configuration adopted. This was a targeted source
screen, not a complete transitive clean-code or licensing certification.

Notebook SHA256:
ea213c0bad5f70e303f49918390089f4f721763804b8d3478cffaf4708845538.
Cache: .biohub/cache/public-audit-20260910/ashutosh-1814.

No fresh readable competition discussion/rules body was obtained in this
refresh. Do not claim a full discussion/rules review or a new clean public best.
