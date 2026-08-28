# Temporal contextual transfer autolaunch v1

Status: armed locally after the accepted ZSNS001 gate; no transfer run or
submission exists at this record point.

The live quota after the two-T4 acceptance was `0.85` hours, below the frozen
transfer notebook's `11.0`-hour ceiling. Starting immediately could strand a
checkpoint before its final validation and terminal evidence. Kaggle reports
the next refresh at `2026-08-29T00:00:00Z`.

`scripts/wait-launch-temporal-contextual-transfer.ps1` is a bounded,
hash-locked launcher. It verifies the completed acceptance kernel, the exact
transfer notebook and metadata hashes, the `NvidiaTeslaT4` machine shape,
exactly-two-GPU contract, disabled TPU/internet settings, and the repaired
acceptance kernel-source slug. It sleeps locally until two minutes after the
reported refresh rather than polling Kaggle, then requires at least `11.0`
hours and pushes once. Capacity errors may retry at one-minute intervals; any
other error stops fail-closed. It never submits to the competition or reads a
leaderboard.

- Launcher SHA-256:
  `8c007d32286ad8e2f5306637258fae7b6fe853d1f578fb38a17348dee1d92efb`
- Transfer notebook SHA-256:
  `17f0f25562c4a0878d9070ea0b238867bf190b1b9b754899bf0d395dee86f353`
- Transfer metadata SHA-256:
  `ad275e1f816bd629b6b6bd677d1d9cb86029605abd1f04e44d5874ffc037461e`
- Accepted source:
  `indarkarhana/biohub-zsns001-contextual-gate-v1`, version `1`
- Transfer target:
  `indarkarhana/biohub-temporal-contextual-transfer-v3`
- Local terminal:
  `.biohub/automation/temporal-contextual-transfer-launch.json`
- Local log:
  `.biohub/automation/temporal-contextual-transfer-launch.log`
