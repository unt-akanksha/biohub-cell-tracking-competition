# Assessment: arXiv 2604.03928v2 for Biohub

## Execution update, September10 20:15UTC

Both predeclared appearance arms have now completed12 correction-held-out folds
and independent replay. LDA pooledNLL0.342701,parent9880,absent84; full72D
NLL0.349453,parent9854,absent84. Both fail the unchanged115absence requirement.
Thus the paper inspired a useful low-cost representation comparison, not a
qualified submission or evidence of0.945. See
[complete comparison](focus-pair-appearance-comparison-v1-result.md).
The original assessment below predates these experiments.

## Original prospective assessment

Decision: worth a bounded CPU representation probe, not a justified backbone
replacement or evidence of reaching 0.945. No LDA experiment has run here yet.
The concurrent balanced candidate-ranker experiment is a separate method.

## Evidence from the paper

The study inserts LDA before logistic regression on frozen image features.
It reports gains in 11/12 coarse-grained configurations but losses in all six
fine-grained ones. Projections and scaling use training data only. Binary LDA
has at most one discriminant direction. See the
[v2 paper, methods and results](https://arxiv.org/html/2604.03928v2).

## Fit to our actual problem — our inference, not the paper's claim

Our candidate ranker currently summarizes the owned encoder's 32-dimensional
node features with one cosine similarity. That may discard useful pairwise
appearance information before ranking. A supervised projection of richer pair
descriptors could offer a cheap additional signal using already-cached features.

There is no natural shared class vocabulary of cell identities across embryos.
We must not turn individual training track IDs into supposedly generalizable
image classes. A plausible label is whether an adjacent candidate pair is the
known parent link; only already verified known-target groups can supply it.
Unannotated or ambiguous targets must remain unknown, not negative examples.

The task requires separating visually similar cells and detecting missing
parents. Collapsing all appearance and motion to a binary one-dimensional
projection could remove precisely the cues needed for these decisions. Our
features are also much smaller than typical global image embeddings, and our
runtime includes volumetric detection and graph work: a smaller classifier
does not imply comparable end-to-end GPU savings.

## Proposed test, not an executed/promoted model

1. Use only the twelve existing fitting movies. Build pair descriptors from
   frozen source/target features, retaining complete candidate coverage.
2. Compare the current motion/cosine representation with richer uncompressed
   pair features and an LDA-based pair score alongside the existing features.
   The uncompressed comparison is necessary: otherwise extra information and
   projection benefit are confounded. Treat this as a Biohub adaptation, not
   a reproduction of the paper's classification experiment.
3. Fit every supervised projection, scaler and classifier within each training
   fold. Never fit LDA globally before leave-one-movie-out evaluation. Account
   for repeated frames and class imbalance; do not tune against exposed source
   or diagnostic movies, leaderboard feedback or the remaining target pool.
4. Stream pair descriptors to avoid a multi-gigabyte dense allocation. Start
   with numerical, role/label and actual cached-packet smoke tests; measure the
   CPU cost before declaring the complete-run cap. No fresh GPU extraction is
   needed for the first representation probe.
5. Keep the current parent/absent/NLL screening requirements, then the existing
   diagnostic, complete-movie patched metric, embryo and runtime gates.

Repeated deterministic seeds do not establish new-embryo generalization; our
acceptance remains based on the appropriate held-out unit. No paper accuracy
gain can be converted directly into an expected Biohub tracking-score gain.

## Code/provenance

The [author repository](https://github.com/IndarKarhana/lda-image-classification)
has an [MIT license](https://github.com/IndarKarhana/lda-image-classification/blob/main/LICENSE),
verified directly. Observed repository tree SHA:
e4e7e4463270dca91d612c37fd5163ac26df8025.
The README-advertised reduction/lda.py raw URL returned 404; actual entry points
need checking before reuse. No repository code was executed, installed or
adopted, and the full published benchmark was not independently reproduced.
