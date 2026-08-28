# Temporal contextual calibration autolaunch v1

Status: armed locally behind the transfer evidence gate; no calibration run or
submission exists at this record point.

The watcher waits only on the local transfer-launch terminal until the frozen
transfer starts. It then checks the remote transfer status at five-minute
intervals, downloads a completed output, binds the launcher to the aggregate
training terminal, and runs the version-4 runtime's strict checkpoint verifier.
Calibration is permitted only when both folds verify, the contextual-v3 family
and exact two-GPU evidence match, competition artifacts are absent, and the
verifier explicitly authorizes calibration while refusing submission.

After that evidence passes, the watcher requires at least `6.0` Kaggle GPU
hours, verifies the frozen calibration notebook/metadata hashes and
`NvidiaTeslaT4` contract, and pushes the private calibration kernel. Capacity
errors may retry; any scientific, artifact, source, or metadata error stops the
chain. The watcher contains no competition submission command and does not
read leaderboard results.

- Watcher SHA-256:
  `5095862f4b3a758848e3e44173b7a3867a7852e2780466da156ebb4bc75750d9`
- Watcher test SHA-256:
  `5a0b94195525ae1fa22612cad36bd51ae8bb6c991f79ebd97a3b1024b1d69681`
- Focused tests: `6 passed`
- Runtime manifest SHA-256:
  `cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d`
- Calibration notebook SHA-256:
  `4596cd9fd7211c27f6b437268c8e719847f0e7cce91438cc86195e79aba497e1`
- Calibration metadata SHA-256:
  `46f0b73396f50d1e3bebb40ce2f811b20d791347dbf51b0b512e2e71e492b955`
- Local terminal:
  `.biohub/automation/temporal-contextual-calibration-launch.json`
- Local strict-verification evidence:
  `.biohub/automation/temporal-contextual-transfer-verification.json`
