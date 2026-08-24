# Pitfalls Research

**Domain:** Sparse 3D+time cell tracking under Kaggle resource and metric constraints  
**Researched:** 2026-08-23

| Pitfall | Warning signs | Prevention | Phase |
|---------|---------------|------------|-------|
| Notebook score ghosts | Title/page score exceeds what the source or current patched metric reproduces | Pull source; search exploit signatures; label score provenance; reproduce before adoption | Intelligence |
| Metric exploitation | Out-of-volume nodes, negative time, sentinel hubs, fake forks, or topology with no biological evidence | Static no-hack audit, graph invariants, explicit ban in promotion policy | Every phase |
| Validation leakage | Random-edge CV is strong while held-out embryo or leaderboard transfer is weak; clips share source cells | Freeze two leave-one-embryo-out folds and report complete-movie scores | Validation |
| Sparse-label misinterpretation | Treating every unlabeled location as background suppresses true cells | Positive-unlabeled or masked objectives; separate detection-count calibration from annotated recall | Training |
| Detection ceiling hidden as linking error | Link model changes do not help; oracle linker remains limited by missing endpoints | Track node recall, endpoint availability, oracle-link score, and conditional association accuracy | Validation/training |
| Rare-division instability | Edge score rises while division Jaccard falls; large variance by movie | Stratified division sampling, local topology training, pooled TP/FP/FN, nonnegative division gate | Calibration |
| Physical-coordinate mistakes | Good voxel-distance results but poor official matching | Centralize `(z,y,x)` scale `(1.625,0.40625,0.40625)` microns and test it | Foundation |
| Temporal artifacts | Failures cluster on frozen pairs, abrupt jumps, or gaps | Augment freezes, missing frames, global translations, acceleration, and intensity changes | Training |
| T4 input starvation | Low GPU utilization and multi-hour epochs | Profile Zarr reads; cache normalized chunks/candidates; persistent workers; pinned transfers | Training |
| Late notebook failure | Hours complete before output coverage, IDs, or schema fail | Preflight one movie; progressive coverage manifests; validation before final export; watchdog checkpoints | Foundation/inference |
| OOM after scaling | Patch fallback still fails on dense movies | Estimate activation/candidate memory, cap candidates, gradient accumulation, chunked inference, adversarial dense smoke | Training/inference |
| Public-LB overfitting | Many submissions differ by tiny post-processing edits and no OOF evidence | Submission gate and daily budget; public score is secondary evidence | Submission |
| Quota reserve violation | A nominally short job can run to Kaggle's 12-hour limit | Use declared worst case, live quota, fail-closed parser, in-notebook wall-clock exit | Foundation |
| Unreproducible winner package | Missing external asset hashes, training manifest, or license provenance | Record everything at run creation and test clean offline inference | Every phase |
| Giant-model overfit | Training loss improves but opposite embryo worsens | Increase capacity only after throughput smoke; regularize, pretrain publicly, and require reciprocal fold gains | Training |

## Competition-Specific Lessons Already Observed

- HOCT failed on a dense movie despite patch fallback: dense-movie memory smoke must be an entry gate, not an afterthought.
- The ranker control failed after about 11.3 hours because movie coverage changed: coverage invariants must run before and during expensive inference.
- Frozen dense topology policies slightly worsened the exact pooled metric: do not infer value from action-level intuition.
- V-JEPA and SEA-RAFT improved one direction but degraded the other: bilateral embryo evidence is mandatory.
- The strict state guard's small positive result is useful as a conservative fallback, but is not a substitute for learning a stronger detector and linker.

## Sources

- [Official metric details](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- [Discussion: validation overlap and public-model leakage](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160)
- [Discussion: frozen frames and global jumps](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283)
- [Discussion: architecture and node-recall ceiling](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604)
- [Organizer metric-patch announcement](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154)

