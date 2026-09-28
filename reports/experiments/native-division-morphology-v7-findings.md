# Morphology v7: source screen rejected, September14,2026

The new low-dimensional nuclear-profile hypothesis completed on the unchanged
3,108image-derived triplets. No source model passed its predeclared comparison
against BOTH geometry and legacy crop-statistics controls. Opposite-embryo model
predictions were not opened, and no candidate/full-movie/submission is authorized.

| Source | Features | TP / positives | FP | Balanced NLL |
| --- | --- | ---: | ---: | ---: |
|44b6|Geometry6|0/1|0|0.69480145|
|44b6|Legacy crop statistics30|0/1|0|0.35072618|
|44b6|New morphology18|0/1|0|0.29403080|
|6bba|Geometry6|3/6|0|0.17208299|
|6bba|Legacy crop statistics30|5/6|0|0.08512595|
|6bba|New morphology18|5/6|0|0.12735045|

Zero source FP is imposed by calibration, not an independent precision estimate.
Source44 trained on244rows/9positives; source6on2,126rows/56positives. All six
paired CPU fits converged in19to36L-BFGS iterations with exact saved-model reload.
Every feature row passed daughter-order symmetry. No threshold/seed/feature
search was performed after these results; do not relax source gates to admit
the five-of-six result or target the single missed movie.

Interpretation: morphology gives better probability estimates than geometry on
both sources, but adds no recovered source-selection event over the existing
crop-statistics control. Its6bba probability estimates are worse than that paired
control. The44b6NLL gain does not compensate for zero recall. These very small
positive samples do not establish broad generalization; selection was already
exposed in earlier research, not a new independent holdout.

The [primary symmetry-based mitosis paper](https://pubmed.ncbi.nlm.nih.gov/30590471/)
supports investigating daughter similarity, not assuming every Biohub division
is symmetric or every fluorescence-volume proxy is conserved. We did not use
that paper's code, pretrained weights or predictions. Our descriptor is not a
reproduction of its method and normalized nuclear intensity is not cell dry mass.

Functionality:4unit tests passed in19.29seconds. The two-real-packet optimization
smoke passed on16triplets in0.141seconds, including exact serialization. Full
extraction and all paired fits completed30.469seconds,session64849 terminal0.
No GPU, image download, live-fit modification or existing-candidate change.

Artifacts: `.biohub/cache/native-division-morphology-v7-full` contains frozen
features and six source-only model archives; JSON report contains complete
per-source/per-movie source counts, thresholds and hashes. Training-designSHA
`1a4b7a2df7165f5bc0bcb73711b0cd45d1df04b8deddb823456a0d52eabf3425`;
featureSHA`62ce49da978c48529289af7745a079d594d75644daf3043b7d158ec4cbcad8b9`.
Do not promote/refit/ensemble these weights based on this failed screen.
