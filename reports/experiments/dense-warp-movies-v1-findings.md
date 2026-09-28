# Dense-warp complete-movie result: both linkers rejected

The fixed 12-arm Antelume run completed in 2,017.274 seconds (33.62 minutes).
All four original controls reproduced prior graphs; raw detector candidates and
non-linker tensors remained exact. All 114,380,387 output bytes are verified and
backed up locally. Labels were opened only after every output was frozen.

| Four-movie pooled metric | Original | Source44 linker | Source6 linker |
|---|---:|---:|---:|
| Patched combined score | .9383918673 | .9250084728 | .9185392048 |
| Raw edge Jaccard | .9163179916 | .9061851391 | .9031055901 |
| Edge TP / FP / FN | 2190 / 97 / 103 | 2183 / 116 / 110 | 2181 / 122 / 112 |
| Division TP / FP / FN | 1 / 0 / 4 | 1 / 0 / 4 | 1 / 1 / 4 |
| Emitted nodes | 102234 | 106775 | 106690 |

Both embryo aggregates regress. Source44 improves one movie but loses on three;
source6 loses on all four. Worst movie remains 44b6_267148e4: original .79453,
source44 .73401, source6 .73698. These are exposed diagnostic movies, not fresh
CV. These four-movie numbers are not the eight-movie validation score or the
submitted candidate's public .946.

Synthetic training sharpened conditional parent confidence but did not improve
the real graph. Extra retained detections did not compensate for fewer correct
links and more false links. The frozen detector was unchanged: this is downstream
retention/association behavior. No metric-count targeting or GT modification.

Reject both final weights and preserve the partial-screen failures. Do not tune,
route, ensemble or submit them. No remaining-four confirmation is justified for
these failed arms. Future learned models need a different training/data rationale;
lower conditional CE alone is insufficient.

Evidence:

- `dense-warp-movies-v1-score.json`, SHA-256
  `74e42215696319936b5d74226bb93df77263e8f7e8186155c258f5a6c09ec61a`.
- `dense-warp-movie-v1-full-result.json`, SHA-256
  `b2e581d751f9b9266c6deba4916155480902644845663d977ac2ff8c6fb70b14`.
- `dense-warp-movie-v1-full-harvest.json`: verified backup, remote copies retained.

The .946 submission remains unchanged. A separate exact-output runtime speedup
test was launched only after this GPU run terminated; it is not a new quality
submission or an ensemble of these failed weights.
