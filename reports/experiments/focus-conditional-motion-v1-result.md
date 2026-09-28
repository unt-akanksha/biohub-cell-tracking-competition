# Conditional motion: held-out correction screen passed

CPU training/extraction39.610seconds, GPU0. Fourteen fitting movies supplied
11,661 ordinary unique consecutive links. Every residual exactly reproduced
the prior extractor; raw/flow provenance and all saved example arrays verified.
No source/target or unchanged diagnostic movies entered this fit/evaluation.

Each held-out movie uses a model and standardization fitted only to the other
thirteen movies, compared with the globalGaussian fitted on those same thirteen.
Ridge1 and six features were fixed prospectively, not selected from this result.

| Held-out measure | Global mean | Conditional mean |
|---|---:|---:|
| Pooled proper Gaussian NLL |5.804118857|5.741789848|
| Pooled squared3D error,um2 |8.816051723|8.516682181|
| Worst-movie squared3D error,um2 |20.154004121|18.654468714|

Ten of14 movies improveNLL; four worsen. All four declared feasibility gates
pass. The complete per-movie results remain in the JSON, including regressions.
The model was then fitted once to all14 training movies and saved as portable
JSON; exact prediction replay passed. A separate verifier recomputed all14
fold fits/metrics and the final fit from savedarrays: exact replay throughout.

ResultSHAad168df2c52620470e659436815b4fd8dd004095b385eee9e21a7b857c6cf226.
ModelSHA508d0589d7ed961cb8d325c574719c918ec8005f15274343106ab34f1377968f.
Examples manifestSHAc3394bd5230117303fcd1d69986f033b299538c561fb3210e1414bd34003a243.
VerificationSHAef6c9f50d6c0cc138ad041a0c5a2de6e1d46e2c72d3bdda1aa6da0db6a2020bc.

This supports testing graph inference, not submission promotion. Calibration-
component held-out evaluation does not establish independence of the pretrained
flow/detector. No new tracking score or leaderboard claim follows yet. Separate
source protocol keeps null/posterior/degree/rawnodes fixed and existing gates.
