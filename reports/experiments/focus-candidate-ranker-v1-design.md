# Direct candidate-supervised motion/appearance ranking

Freeze before fitting. Previous presence-only corrections cannot repair parent
order, and improved conditional residual density worsened full-source graphs.
This experiment optimizes the actual candidate-versus-null softmax likelihood.

Data: exact12 existing fitting movies, all99 adjacent transitions, all known
parent/absent labels. Every source detection remains a candidate for each known
target. No sampled negatives, top-K restriction, synthetic labels, node pruning,
coordinate movement or new diagnostic/source/target reads. Original unknown
targets are ignored by the supervised loss only. The original encoder trained
on these movies; LOMO therefore tests the new ranking correction, not whole-
system or embryo-held-out independence.

Start from the original two-movie motion Gaussian and null logit-4.5. For each
real candidate use8features: real-class intercept,3 standardized signed motion
residuals,3 squared residuals, and cosine agreement of the frozen32-channel
source/target image features. Null has eight zero features. Fit additive logits
by unweighted grouped cross-entropy, ridge1 on slopes, unpenalized intercept,
zero initialization, L-BFGS-B max500,gtol1e-8,ftol1e-12. Squared-residual
corrections have upper bound0.5 so total squared-distance coefficients cannot
become positive. No search of features, weights, thresholds or hyperparameters.

First verify finite-difference gradients and optimizer/parameter roundtrip on
synthetic inputs and the largest fitting packet by source-count times known-
target-count, selected before fitting. This smoke proves functionality only.
Then12 movie-held-out fits, each on the other11movies. Controls: physical-only
posterior and the unchanged original neural+physical summary on the same labels.
Require pooled NLL below both, correct-parent and correct-absent counts at least
as high as both,8/12movie NLL wins versus neural, and no movie NLL regression
over0.02 versus neural. Failure stops without full fit/diagnostic/source scoring.

Use CPU with2numeric threads. Feature cache and packed training arrays capped
at2GiB each; audit choice counts before a large allocation. Extraction+real
smoke cap300seconds; a subsequent full LOMO run has a separate900second cap.
Persist packet hashes, group/label identities, all candidate rows, model files,
fold results and timing. No GPU/cloud changes. A passing screen is not a
submission: unchanged diagnostic and complete-movie gates still apply.
