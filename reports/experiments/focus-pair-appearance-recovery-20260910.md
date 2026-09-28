# Full-feature numerical recovery and bounded inference

September10 around19:49UTC. Previous goal turn made progress: verified LDA
screening and launched the full-feature arm. This turn completed label-free
inference functionality, observed the full arm's actual terminal failure,
diagnosed its fitting curvature and launched an evidence-gated numerical
recovery. No candidate has been promoted and the full goal remains unfinished.

## The original full arm is terminal, not still running

Session2266 completed. LDA's verified gate failure is unchanged. The full arm
reached500iterations/561.219fit seconds in its first training fold, then raised
`STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`. Fold total582.297seconds;
complete sequential run1015.391seconds. No held-out full-feature score, model
export, diagnostic/source/target use or submission occurred. Original source,
limits, prepared arrays and failure receipts are preserved.

Overall result SHA256dadaa3293fbb0b99252bae246186a7e336f2259a154b05010b3d60c995919bbf.
Full failure SHA2560b3476a20f2487193b90c0d42d2ff4f2352ae908ddfb2fedc3916db0dd14a75a.
The original implementation did not retain in-progress optimizer coefficients;
the recovery explicitly writes25iteration checkpoints and terminal coefficients.

## Numerical evidence and new run

On exactly the same first fold's11training movies, the zero-point Hessian
condition number is30274.668. An invertible bound-preserving coordinate change
reduces this to1.84974; computing curvature took15.984CPU seconds. This supports
ill-conditioning as a contributor, not a guarantee that recovery will converge.
The original model, features, labels, physical offsets, original-theta ridge,
weights, zero initialization, max500iterations and quality gates remain fixed.

Original20target real smokes passed: new full objective19.35268135484 versus
19.35268135515; LDA27.17465716225 versus27.17465716217. Every target decision
matches, with full123iterations/148evaluations and LDA34/43. No LDA quality
rerun is authorized or launched. Hessian-vector and chain-rule tests pass.

Curvature profilec1b4af86d8a21cd6a84512dc05e76d8860e883e3731db3976c5e819f83d795c9;
matrix5bdceda7c105400823a1afda99fe1ad1b41b428023d52068c86d0f8051b95607.
All4,460,090 first-fold choices and10,403groups replayed exactly from original
prepared arrays. Only exactly zero floating-point probability terms were
omitted in arithmetic; no candidate was dropped from loss or prediction.

`focus-pair-appearance-conditioned-lomo-v1` now runs in session5342 after29
targeted tests passed3.04s.12full-feature folds,2CPUthreads,9000second cap;
each fold computes its own training-only optimizer coordinates and validates
the full Hessian-vector derivative before optimization. Models are saved in
original coordinates before scoring. No retuning or fallback on failure.
Actual artifacts:`.biohub/cache/focus-pair-appearance-conditioned-lomo-v1`.
No complete recovery fold or quality decision existed at this initial update.

## Inference functionality completed without promoting a model

New label-free scorer batches32target cells and retains every source plus the
fixed null candidate. It preserves native coordinates and global IDs, processes
unknown targets, handles empty frames and matches original tie-breaking. It
does not generate a graph/CSV or grant deployment permission.

On the original57b7cc1e/frame31 packet, both fixed CPU smoke models processed
1160source nodes,1187target nodes,1,378,107choices, including1167originally
unknown targets. All decisions matched an alternate split-matrix reference with
different17target chunks. Maximum posterior difference3.67e-15; every original
known-target decision and serialized prediction replayed. Input arrays unchanged.
CPU forward times1.797s full and2.765s LDA; conservative working-array estimate
154,577,920bytes (not measured process RSS). Whole smoke10.766seconds,0GPU.
This is one packet's runtime, not proof that hidden inference finishes in12hours.

Inference receipt5de3a814abf5635fbc131ffdd5f7fe710c64fa75dd687fb70d00cf12bad7f086.
Thirty-six inference/base tests passed17s;20conditioning/inference tests1.12s.
All smoke/profile handles are terminal. Only recovery5342 remains active.

## Resources and outcome

No new GPU launch, cloud start/stop, RSNA or shared-environment changes. Last
read-only cloud observation remains stopped at19:17UTC; quota8.22h then is
historical and must be refreshed before any GPU launch. Public/rules audit
completed19:23UTC in the preceding turn; no metric-hack/LB selection change.
No diagnostic/source/new-target data opened. No new qualified submission:0/5.
