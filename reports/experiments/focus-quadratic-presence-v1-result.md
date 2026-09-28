# Nonlinear presence correction rejected; parent ranking is the larger limitation

September10,17:30UTC. Completed the fixed 12-movie leave-one-out CPU screen in
5.687seconds. Independently replayed all saved fold models, fitting-only
normalization, original/linear/quadratic metrics and screening decisions.
Twenty-one relevant tests passed in3.81seconds, including a guard proving the
loader opened only the12 approved fitting summary arrays.

| Pooled fitting-domain LOMO diagnostic | Original | Linear correction | Quadratic correction |
| --- | ---: | ---: | ---: |
| Unweighted joint NLL, lower better | 0.667407 | 0.633682 | 0.656424 |
| Correct parent /10,754 | 9,835 | 9,894 | 9,889 |
| Correct absent /161 | 72 | 33 | 34 |

All five frozen screening gates fail. Quadratic NLL beats linear on only4/12
movies; worst regression is +0.148704 on6bba_4f99ce20. No final full-fitting
model, diagnostic evaluation, source-selection evaluation, target access or
submission followed this failure. The data were used by the original encoder,
so these folds test the correction only, not independent end-to-end tracking.

## Action-changing error budget

The verified summary decomposition separates presence likelihood from
conditional parent-ranking likelihood. The original conditional parent term is
6,317.875 of total7,284.748 NLL:86.727%. A perfect presence model with the same
conditional parent distribution has joint-NLL lower bound0.578825. Its maximum
possible correct-parent count is9,909/10,754;845 parent-ranking errors cannot
be corrected by changing presence alone. The existing linear correction is
already only15 correct parents below that ranking ceiling.

This retires presence-only calibration as the next improvement lane for this
fixed representation. A future experiment must improve candidate ranking or
the detector/representation; do not launch another polynomial/threshold/loss
sweep simply because GPU access is unavailable. This does not establish that
more model capacity or more GPU time will succeed. No bound here is a tracking
score or a forecast of leaderboard performance.

Result SHA256:2f5ccbef14fd2571fef6242473946b791f20904e3c8f6c300c844ecf44f53250.
Verification SHA256:c2c1a40c1d74128a89c1797927f5bb91e6e6753c23f1c14e23a9098fa65e1fc1.
Fold model/result files: `.biohub/cache/focus-quadratic-presence-v1`.

## Resource and access state

No GPU used, no live process left. Last actual Kaggle quota remains8.80h at17:18;
refresh before any launch. AWS profile inventory contains onlydefault and
148971207977_InventoryOptimization-EC2-Access. Named profile STS returns
ExpiredToken; default returnsNoCredentials. No AWS environment-variable fallback
was present. Credential file last-write time remains14:18:52UTC. No secrets
were printed or modified, and no RSNA/shared-instance changes were made.

The preceding goal turn was progress: a completed, verified GPU experiment.
This turn is also progress: a completed CPU experiment and a verified ranking
error budget that changes the next research action. Goal remains unfinished;
0/5 new qualified submissions today. Existing public checkpoint provenance
remains insufficient for independent validation (all199 training movies in
secondary; unknown primary split), not a reason to treat training scores as
generalization evidence.
