# Full-movie motion-only validation

Frozen candidate: parent76f7 detector D4, embedded flow3006 averaged over eight
XY views with inverse vector/grid transforms, FP32 mean, exact zero-neural
execution shortcut. Original detector threshold and neural0/spatial1/null-4.5
linking constants remain fixed. No graph pruning, new training or public models.

Real three-arm smoke v2 passed29 tests and exact native/shortcut graph parity.
Flow averaging alone added one false edge on the three-frame training sample;
there is no measured full-movie gain yet. That sample is functionality evidence,
not an accuracy selection gate.

Full validation opens only the same eight previously consulted source movies.
For every movie, actual predicted coordinates must exactly match the retained
parentD4 graph; its complete reference manifest is pinned to
f77b9eff4b965afcd9e4f571e4e32f09f23c894e962bb712ed66b83e75b35679.
Any coordinate mismatch or >2048 nodes per frame aborts without truncation.
Neural and flow tensors must remain unchanged. All792 image pairs must execute
flowD4 and the zero-weight shortcut; CPU scoring verifies these receipts.

Comparison uses the same complete-movie patched official metric and retained
parentD4 baseline. Required: positive score and raw-edge Jaccard deltas, mean
recall loss <=0.005, >=5/8 improved adjusted-edge scores, every movie loss <=0.02,
and worst-movie loss <=0.01. No threshold/coefficient sweep follows a failed gate.
No new target movie or submission is authorized by this experiment alone.

GPU inference is capped at one hour and must pass fresh quota reserve checks.
The separate offline CPU scorer uses short title/slug names, avoiding the prior
51-character SaveKernel rejection.43 local inference/scorer/reference tests pass.
