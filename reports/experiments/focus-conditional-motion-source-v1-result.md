# Conditional motion: stronger source component, promotion gate failed

CPU-only complete-source evaluation finished131.438seconds. All8candidate
arrays were saved before sourceGT; raw geometry and original parent/FOCUS-flow
controls replayed exactly under the patched official scorer. No fitting took
place during this source comparison; the14movie-held-out-verified model was fixed.

| Complete-eight source metric | Previous calibrated motion | Conditional motion |
|---|---:|---:|
| Combined score |0.7805420622|0.7838611993|
| Raw edge Jaccard |0.7987813135|0.8012639505|
| True / false / missed divisions |3 /154 /8|5 /178 /6|
| Mean node recall |0.9643973729|0.9643973729|

Candidate edges:5959TP,627FP,851FN. Six of8movies improve over FOCUS-flow,
seven improve over the original parent. All additional gains/division-preservation
conditions pass. However67ebd073 adjusted edge score0.549636330 is0.043314438
below the parent, violating the unchanged0.02 maximum per-movie loss. The
worst-movie-minimum guard itself passes; it is the individual-movie loss guard
that fails. Do not conflate these two different robustness checks.

Decision: retain as stronger component evidence, not a promoted/submission-ready
candidate. No target expansion or submission authorized. No source-specific
fallback or retrospective gate change. These8movies are exposed development;
0.78386 is not an independent or leaderboard prediction and does not meet the
user's0.945target. FOCUS pretraining overlap remains unresolved.

ModelSHA508d0589d7ed961cb8d325c574719c918ec8005f15274343106ab34f1377968f.
ResultSHA998e43b168cd4402a081349d48b0cea9a0843783e72f22000c5e7c7c4ce8b400.
Prelabel manifestSHA729c96afd7204f573974546696e2027773c0891007924d77e1bee80926f01091.
Sixteen focused tests pass4.22s. CPU65605andtest10166terminal; noGPUrun/spend.
Five requested good submissions remain0/5.
