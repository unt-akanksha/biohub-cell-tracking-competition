# Temporal contextual processed autolaunch v1

Status: armed locally behind independent transfer and calibration evidence gates;
no processed run or submission exists at this record point.

The host verifier consumes the actual Kaggle calibration artifact layout. It
binds the launcher to the aggregate terminal, each aggregate fold to its worker
result, both appearance and Trackastra terminals to their checkpoint bytes, and
the 12 predeclared calibration movies for each reciprocal fold to the transfer
configuration. It reconstructs pooled metrics from confusion counts, requires
the complete `2 x 5 x 4 = 40` frozen grid per fold, recomputes every zero-control
delta and eligibility decision, and independently selects the deterministic
winner. A recorded success flag or selected row cannot authorize the next stage
when any recomputed evidence differs.

Only two improved folds with clean-selection flags can authorize processed
materialization. Submission remains explicitly unauthorized. The watcher then
requires enough quota for the six-hour ceiling, with no reserved quota after
the user's 2026-08-28 override, validates the frozen two-T4 processed notebook
and metadata hashes, and pushes it once. Remote status checks occur only after
the preceding local launch terminal exists and then at five-minute intervals.
The watcher contains no competition submission command and does not inspect the
leaderboard.

- Verifier SHA-256:
  `4621b0c3ce272393df92b0b6a5b19da83edbd85368f0faa14b46a80bd2b8fd74`
- Watcher SHA-256:
  `f34ac3baa41d380134f68d387406abad149682c5a8b14ea103c6ab1c322f8409`
- Verifier test SHA-256:
  `8aacb8e41c46e1b1cf7c2dc0277aa26f59b7019628319680fa5505c7b7c7c8e1`
- Watcher test SHA-256:
  `a7b292ae9d1b7c2c9268e377beab2532d586806dd79652846505e390730eeba2`
- Focused tests: `11 passed`
- Processed notebook SHA-256:
  `08c0316da23940e21490d56909d3bb88c690b9103dfb0d2cf57c40de96e8665b`
- Processed metadata SHA-256:
  `624a14a202cde15a1bfc4cf942241df1f80801094e60e8886182c8dffc826125`
- Watcher PID at arm time: `38008`
- Local launch terminal:
  `.biohub/automation/temporal-contextual-processed-launch.json`
- Local strict-verification evidence:
  `.biohub/automation/temporal-contextual-calibration-verification.json`
