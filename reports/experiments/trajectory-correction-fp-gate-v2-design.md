# Fixed FP-label correction experiment

Frozen before complete36-case metric audit and before either new gate fit,
September14 2026. This is independent of the live fork16fits/evaluations.

Hypothesis: the prior gate ignores beneficial wrong-link removals because its
labels measure only changes in correct links. The four-movie diagnostic proved
four instances, but the full source scan found only36fully-known candidates.
This is a controlled label ablation, not a proposed larger fingerprint router.

1. Audit ALL36fully-known partial-neutral unequal-edge-count components from
   the existing60optimization movies with fresh complete-movie official scoring.
   Do not include the12mixed-unknown cases. Freeze selection before rescoring;
   replay each baseline against the pinned prior result. No oracle graph export.
2. Replace only those36previously ignored labels by the sign of official
   score delta: positive1, negative0, within1e-12 remains ignored. Retain negative
   outcomes if present; no cherry-picking positive audits. Other old partial
   labels retain their limitations and are unchanged. This does not create
   complete-metric supervision for the whole dataset.
3. Keep exactly the old38prediction-only component features, standardization,
   logistic fitter, L2=0.1 and threshold0.5. No IDs, movie fingerprint expansion,
   architecture menu, threshold search, held-fold calibration or epoch selection.
   Replay the old fits first to verify control coefficients/normalization.
4. Fit47source6movies to predict13source44movies and vice versa. Freeze all60
   revised-gate graphs before full official scoring. Existing backbone/anchor
   exposure remains disclosed; this is not pristine project-level OOF.
5. Compare against BOTH ungated fixed8anchor and the old rejected gate, with
   complete-movie pooled, per-embryo, per-movie/worst-movie and division counts.
   Promotion requires pooled improvement over ungated and old gate, neither
   embryo nor any movie worse than ungated, raw-edge nonregression and no
   division loss/extra false divisions. A failed gate is not refit on all data
   or added to the pending submission. No automatic submission is authorized.

First functionality gate: prior12-counterfactual full diagnostic succeeded;
sampling unit tests reject unknown/ambiguous cases. Then the all36audit must
complete, control fits replay, and two small complete-movie graph masks must
validate before remaining graphs. CPU only, no new image/model download.
