# Additive past-motion features v1

September 13, 2026, before new source selection predictions. Previous goal turn
was progress: the source-only annotation audit and completed external transfer
screen changed the next action. The external recipe remains rejected and closed.

The optimization-only motion diagnostic has a mixed result. At the existing
geometry>=3 reference cutoff, motion compensation changes 44b6 positive coverage
5->5 and negative coverage3->7; in6bba positives23->24 and negatives80->57
(some rows lack history). It is NOT safe to replace the geometric gate by this
heuristic. No candidate mask, distance bound or numerical quality gate changes.

Test one controlled feature addition instead: append the 14 fixed, daughter-
symmetric descriptors from `comoving_division_features.py` to the existing
1,346-dimensional frozen-image-head-v1 feature bank. Use the ORIGINAL Biohub
three-frame patches and external-only encoder warm starts, not the later
two-frame ZebraHub transfer inputs or its fitted heads. Past motion comes only
from a unique adjacent predecessor and optionally its predecessor. No outgoing
daughter edges, neighborhood labels, future continuation or target labels enter
the feature function. Training uses supervised past tracks; deployment would
use predicted past tracks, so actual predicted-candidate validation is essential.

First predecessor missing/ambiguous -> explicit all-zero unavailable vector.
Normalize on source optimization only; std floor1e-3 and clip+/-8. Preserve old
feature blocks1283/18/45 and add a fourth14-feature block, each scaled by sqrt
width. Fit one L2=0.01 logistic head per source, max300 L-BFGS-B iterations,
analytic gradient, same class/eligibility quality weighting. For attribution,
fit the same fixed-L2 source-only control on the cached original1,346 features.
The control is not an alternate submission route and cannot reopen old failures.
No penalty, seed, source-role, feature-subset or threshold search.

Use the original optimization/selection movie roles; original audit and final
probe remain excluded. Each head's fitting, normalization and source threshold
exclude its opposite Biohub embryo. Both source heads must pass eligibleAP>=.55
and at least2TP strictly above all negatives before any target-embryo score is
opened. Freeze both first. Transfer gate remains AP>=.55 and>=1TP each,>=3pooledTP,
<=1pooledFP at unchanged source thresholds. Failure closes this recipe.

No GPU is needed for source feature preparation/fitting. Only if both heads
pass is fresh cross-embryo encoder extraction justified on an initially idle
Antelume GPU, after a small same-code smoke, sequential and runtime-bounded.
Any patch-level pass still needs actual predicted candidates, division-positive
complete-movie patched official scoring, worst-movie and offline two-GPU runtime
evidence. Historical project exposure is acknowledged; no pristine audit claim.
No Kaggle GPU use, RSNA mutation or live submission is authorized by this design.
