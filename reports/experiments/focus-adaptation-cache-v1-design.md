# Broaden training-only FOCUS proposal coverage

The two available full training movies provide708 positive fitting targets
but only3 safely known missing-parent targets, with none in the temporal
diagnostic. They cannot establish reliable missing-parent behavior. Do not
convert unknown targets into null labels or promote a positive-only fit.

Cache eight additional complete original training movies: four from the fixed
96-movie fitting pool and four from the fixed24-movie diagnostic pool. Within
each pool choose the first four in the existing frozen ordering, excluding
the two replay movies. No labels, scores, density estimates or leaderboard
information select these movies. Exact membership is embedded in the notebook
contract. Diagnostic movies are reserved from the proposed FOCUS-specific
adaptation, not from the already-completed encoder/linker training.

Reuse the exact completed FOCUS source-cache implementation and author weight
identity, changing only movie scope and run identifiers. The successful real
six-frame detector smoke is reverified on the host; the new job also reproduces
those same six frames. Preserve all raw centroids, no public tracking edges,
node pruning, metric hooks or postprocessing. Source-selection and target
movies remain closed during this job. FOCUS pretraining overlap remains unknown.

The matching previous eight-movie job took34.8min; expect roughly35-50min,
with a hard one-hour cap and saved per-movie outputs. Fresh Kaggle quota must
be at least9h before launch to preserve8h even at the declared worst case.
No other Biohub GPU job may overlap. This is data preparation, not a submitted
candidate or a model-quality result. Do not automatically launch training:
first verify outputs and label coverage under the conservative sparse-label
policy. If missing-parent coverage is still inadequate, report that limitation.
