# ZebraHub contextual acceptance host verifier

Date: 2026-08-27

Status: implemented and verified locally; no Kaggle launch, GPU use, model
selection, or submission occurred.

The frozen acceptance notebook already validates its own two worker terminals.
This independent host verifier closes the post-download trust boundary before
the acceptance output can become a transfer input. It:

- binds the unique launcher terminal to the aggregate terminal by SHA-256;
- binds both aggregate fold records to their exact worker terminals;
- independently verifies and strict-loads both original v3 checkpoints;
- reconstructs both seeded random baselines and checks their state hashes;
- validates all metric ranges and inventory counts;
- recomputes the broad composite/top-1/MRR/division-top-2 gate for each fold;
- verifies the frozen ZSNS001 manifest and inventory hashes;
- verifies the two-GPU, no-competition-data, no-public-prediction,
  no-leaderboard-selection, and no-submission declarations; and
- rejects any CSV, checkpoint, NPZ, or other model artifact in the acceptance
  output.

Real checkpoint verification completed against pretraining terminal
`bbe504907186af058d14f1287492b64e9b83b8c890cb620519ce1536265a618f`.
Both 20,747,761-parameter checkpoints strict-loaded. The independently
reconstructed baseline state hashes are:

- `target_44b6`:
  `8ec53c0236cc2523ec80dfa6a84d836b5e4351ef0105e526c24675ac668ee834`
- `target_6bba`:
  `8364502dd8723a1f9a0dad5280ab3cba4a2e4f48dc05233b560abd2faf22c38a`

Source SHA-256:
`2c9f45effaa8bb5eb23a6d86041cf2811755c1fd8c97ee98223bb5e7d9d2ba96`.
Test SHA-256:
`030f1c543609e94bf73d0b6bcdd29c73b69eb5b890e30b5322fb5515a3476870`.
The focused acceptance suite passed 13 tests, including deliberate aggregate
divergence and prohibited-submission-artifact rejection.

This host-only addition does not change the remote acceptance runtime,
notebook, metadata, scientific recipe, existing preflight, or downstream model
selection policy.
