# Full covariance failed the held-out screen

All 14 original conditional-mean folds replayed exactly from verified saved
fitting arrays. Replacing diagonal uncertainty with full covariance worsened
pooled proper NLL from 5.741789848 to 5.750390939. Only 7/14 movies improved;
the frozen gate required at least 8 and pooled improvement. MSE stayed exactly
8.516682181 um², as expected from identical mean predictions. Worst-movie NLL
did not worsen, but that partial pass does not qualify the model.

No final covariance model was saved. No source/target was scored or opened.
GPU use: zero. CPU experiment: 0.797 seconds. Seven numerical/guard tests passed.
Result SHA-256: `6e712e7ca3fd19e5de48f2a695a941887c7f064a202441b36348f0f359fec5e1`.
Keep the prior diagonal conditional-motion component; do not tune covariance
against exposed source movies after this failed fitting-only screen.
