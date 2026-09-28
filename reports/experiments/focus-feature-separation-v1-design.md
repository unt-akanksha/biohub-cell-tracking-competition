# Fitting-only image-feature separation diagnostic

Before another GPU intervention, assess whether simple metrics on frozen
image features distinguish each uniquely annotated real parent from the
closest incorrect parent under the existing calibrated physical prior.
Use all eligible known-present targets from the four fitting movies only.
Unknown and known-absent labels do not define positive pairs. All source
candidates are retained, with no prediction mutation or learned threshold.

Report per-movie and pooled raw cosine, pair-centered cosine, L2 and physical
pairwise win rates, ties half-credit; true/wrong raw cosine means and margin
quantiles. Frame-pair centering uses all feature vectors without labels and
is observational only. The negative parent is selected by physical cost,
not image-feature similarity. No optimizer, GPU or diagnostic/source/target
representation comparison; verify existing complete feature artifacts first.

This is not a tracking score or a candidate promotion gate. Similar cosine
values alone cannot establish collapse. Even poor cosine discrimination does
not prove the learned transformer cannot use those features. No model change
or centering intervention is authorized for promotion by this audit alone.
