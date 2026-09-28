# Fitting-only nonlinear uncertainty screen

Freeze before running. Hypothesis: a quadratic offset-logistic correction can
model interactions between motion uncertainty, appearance agreement and parent
ambiguity that the previously tested linear correction cannot express. This is
not another GPU head fit, a threshold sweep, or a claim of stronger tracking.

Use only the existing 12 fitting-movie summaries, with actual hashes, complete
label coverage and original replay checks. Never load the four diagnostic
summaries, source-selection movies or target movies. No public weights or
leaderboard feedback enter fitting or selection. Leave one fitting movie out
at a time; all normalization statistics and coefficients come from the other
11 movies. Original encoder training included these movies, so this tests
correction generalization only, not independent end-to-end generalization.

Retain the seven frozen presence features, the original log-sum-exp offset and
unchanged real-parent rankings. Standardize raw features on each fitting fold;
append all 28 degree-two monomials, then standardize the resulting 35 features
on that fold. Fit ordinary unweighted offset logistic likelihood plus fixed
ridge1 on slopes, unpenalized intercept, L-BFGS-B max1000 iterations. No
hyperparameter search, class weighting or decision-threshold tuning. Report
joint-class decisions using the unchanged conditional maximum term.

Controls: original uncorrected posterior and the existing fixed linear method,
refitted on exactly the same 11 movies. Require pooled NLL below both controls,
pooled correct-parent and correct-absent counts at least as high as both,
NLL better than the linear control on at least 8/12 movies, and no movie NLL
regression exceeding0.02 against the linear control. These screening gates
are fixed before results. Failure means no final fit or diagnostic evaluation.
A pass only permits a separately persisted full-fitting model and the unchanged
diagnostic screen; it is not full-movie tracking promotion or submission.

CPU only, two numeric threads, 300-second process cap. Save all fold identities,
model coefficients, metrics, source hashes and timing. Do not overwrite a result.
