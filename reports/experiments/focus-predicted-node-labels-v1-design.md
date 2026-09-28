# FOCUS-specific sparse training-label inventory

Correction to the preceding hypothesis: the frozen original training code
already calls detect_and_match and indexes features at its OWN DETECTOR's
predicted positions. It does not train on GT positions alone. The guarded
matcher in run_association preserves that detector output. Known-parent-absence
supervision was already included. A generic switch to predicted-node training
would repeat existing work. The distinct possible intervention is FOCUS-specific
proposal-domain adaptation, not introducing predicted nodes for the first time.

Before GPU work, audit supervision on two available complete raw FOCUS training
movies,6bba_f1fde7e0 and6bba_23af9eeb. Both are original source training members;
none of the eight source-selection or target movies are included. Use exact
previously verified raw-node/flow caches, official7um node matches and current
patched scorer. Replay full training controls before consuming their match map.

Only a uniquely matched annotated target with one adjacent annotated parent
receives a label. A unique source match gives its source-column class. Otherwise
the null class is permitted only if EVERY source detection is strictly farther
than7um from the known parent's annotated position. Unmatched or ambiguously
matched targets, annotated births, unmatched nearby parents and non-adjacent
links remain unsupervised. Division daughters may each learn the same parent;
no synthetic graph edges/nodes/divisions are introduced.

For later adaptation feasibility, predeclare fitting frames0..69, diagnostic
frames80..99, and a temporal gap. Fitting transitions0..68; diagnostic80..98;
all others embargoed. No frame is shared across roles. These are within-movie
adaptation diagnostics, NOT independent validation of the existing checkpoint.
Both full movies were already used by its original training. Do not claim new
holdout status. Report all positive/null/unknown counts; no run is automatically
authorized just because a cache was built.

This CPU label inventory changes no predictions, runs no optimizer and makes
no submission. The next decision is whether the available supervised examples
support a small FOCUS-specific head adaptation, or require more training-only
data. Preserve the8h GPU reserve and the existing source promotion gates.
