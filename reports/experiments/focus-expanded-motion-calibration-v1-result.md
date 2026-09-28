# Fourteen-movie motion calibration: rejected

Completed CPU-only in118.172seconds. Nine tests passed. Original two-movie
residual fit replayed exactly;14 training movies supplied11,661 unique ordinary
adjacent links. Diagnostic/source/target movies were excluded from fitting.
Fitted parameters and all8 source predictions were saved before source scoring.
Actual raw-node identities and original full-source controls replayed exactly.

| Complete-eight source metric | Previous2movie fit | New14movie fit |
|---|---:|---:|
| Combined score |0.7805420622|0.7804171043|
| Raw edge Jaccard |0.7987813135|0.7993806382|
| True divisions |3|2|
| False divisions |154|172|

Mean residual ZYXum:[0.315566838,-0.098898824,0.246624441]. Varianceum2:
[4.672996533,1.925570183,2.084388984]. No floor activated or threshold changed.
FOCUS-flow baseline combined0.7753325939,rawJ0.7937315832; new fit improves both
over that control but loses a true division. Movie67ebd073 remains0.036827074
below the original parent, violating the fixed0.02 maximum loss. Seven of8
movies improve over parent; six of8 improve over FOCUS-flow. These partial
gains do not pass the unchanged promotion conditions.

Decision: no promotion, target expansion or submission. Global Gaussian data
expansion did not resolve robust tracking quality; do not extend this arm
unchanged or select another covariance from these source results. A later
conditional/non-Gaussian model would need a new training-only protocol and
training-movie-held-out evidence before source evaluation. No such fit has
been performed or selected here. Source movies remain exposed development,
not an independent score or leaderboard estimate.

ResultSHA520a1e100107ddec46b1e4e25cf33ea3a82e09a1a8b043183804ac8583f3c5b4.
FitSHA379044ecc55dbba9bf59a216f46c7999858b805eca352b18873df2944c6e9ade.
Prelabel manifestSHA3ba9512b51ce6da33487d54e5221c130e995a33d35a094f446789568ef3a1d70.
CPU88667terminal; no GPU run or remote mutation. Five requested submissions0/5.
