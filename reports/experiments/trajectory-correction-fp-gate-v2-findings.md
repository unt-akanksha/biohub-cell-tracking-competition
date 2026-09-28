# FP-label correction gate v2: rejected

The fixed, preregistered label-only ablation completed all60source movies in
205.985seconds, session62324 TERMINAL success. All36audited newly labeled
examples were positive (4source44,32source6); no new feature or threshold was
introduced. Both old control fits replayed within1e-10, both small complete-movie
smokes passed, and all60graphs were frozen before official scoring.

| Arm | Complete-movie pooled score |
| --- | ---: |
| Baseline | 0.918039753 |
| Ungated fixed8anchor | 0.931693079 |
| Old rejected gate | 0.928197249 |
| Revised FP-label gate | 0.927361469 |

Held44: anchor0.945631832 -> revised0.946592536, better also than old0.946382643.
Held6: anchor0.929638450 -> revised0.924803382, worse also than old0.925738418.
Against anchor,9movies improve,35regress,16are neutral. Worst regression
6bba_67ebd073 -0.028568117; next6bba_7f87b3d8 -0.026648420. All division counts
remain14TP/39FP/91FN. Finite60movie coverage and division preservation pass;
pooled, raw-edge, both-embryo and every-movie promotion checks fail.

The target omission was real but fixing it did NOT solve cross-embryo routing.
Do not keep only the successful held44fold as an identity-based deployment
rule, sweep thresholds on these outcomes, refit all60source movies, promote
the gate, or apply it to the pending Kaggle submission. Prior backbone/anchor
exposure and remaining partial-supervision limitations remain disclosed.

ReportSHA`2d3c9d47679cbeed82c77677c2381c340679c3660541954548592ac9d72cebea`.
RunnerSHA`0ee646777dc228e749208e71c4e704007ae241931645ec1c37612bd3100f3bf9`;
designSHA`658eb16ee01f64d7fa1c45991eea1323caef276b1b4e43f630d40671bc441870`.
Held44modelSHA`7046789017591280927f409c35dd00721f831fde18e05696e7fd53c298555b9b`;
held6modelSHA`2621bb23d3d380d894ebaf3dbd586983cbd1f37fc43b9c7bd004332bffec1b00`.
Models and complete graphs remain in .biohub/cache/trajectory-correction-fp-gate-v2
for reproducibility, explicitly unauthorized for submission.

Paired with the whole-movie perfect-choice bound (+0.000353565maximum source
gain), these results favor completing the stronger learned event models over
another ad-hoc gate retry. Existing fork16fits/evaluations remain unchanged.
