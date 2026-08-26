# Positive-unlabeled detector adaptation — next high-upside lane

Status: designed, not scheduled ahead of the active HOCT gate.

## Why this lane exists

The clean public detector reaches 0.969 annotated-node recall on the four
candidate-disjoint validation movies. The official 35.5M Spotiflow model reaches
0.778, and the synthetic-only fine-tune reaches 0.092. Replacing public nodes
with either tested checkpoint would erase any association gain. At the same
time, retaining public nodes limits the current HOCT candidate to an
association-only improvement.

Biohub's 133,318 annotated training nodes represent only about 2.82% of the
organizer-estimated cells. Conventional dense supervision is therefore invalid:
it labels most real cells as background. The next detector experiment must be
positive-unlabeled rather than another synthetic-only or fully supervised run.

## Frozen design

- Initialize from the original official `smfish_3d` Spotiflow checkpoint, not
  the failed synthetic checkpoint.
- Exclude all four validation stems from every teacher/student training read.
- Use the public two-seed detector only as a teacher. The primary checkpoint is
  `pilkwang/biohub-tracking-support-pack-50ep-v1`'s
  `checkpoint_last.pth`; the independent secondary is
  `pilkwang/biohub-temporal-unet3d-seed314159-v1` with weight SHA-256
  `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`.
  Intersections of the two seed peak sets are high-confidence
  pseudo-positives; their union is an uncertainty band, not background.
- Force every annotated Biohub node to be a positive even when the teacher
  misses it.
- Apply a background loss only outside both seeds' low-probability support and
  weight it at most 0.02. Unknown voxels contribute no negative target.
- Train weak/strong transformed views with coordinate-consistent peak targets;
  begin with the detection head/last decoder block before considering deeper
  unfreezing.
- Calibrate thresholds using image responses and organizer node-count metadata
  only. Read selection labels after the training schedule is fixed, then read
  the two acceptance movies once after threshold selection.

## Promotion gate

The student must match or improve the public detector's annotated recall in
both embryo prefixes, avoid more than a 0.01 worst-movie recall regression, and
keep projected node counts within the predeclared density envelope. A larger
network or changed checkpoint hash alone is not evidence. No leaderboard score
may promote the lane.

This experiment is intentionally deferred until HOCT returns clean association
evidence. If HOCT passes, the detector student can be composed with the accepted
linker; if it fails, the detector can still be evaluated independently without
reusing acceptance labels.
