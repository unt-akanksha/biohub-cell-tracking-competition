# spotiflow-detector-acceptance-v1

Status: staged; waiting for the active Trackastra experiment to release the GPU slot.

This is our first non-public detector candidate. It evaluates the two official
Spotiflow 3D checkpoints (`synth_3d` and `smfish_3d`), each with 35,489,892
parameters, against Biohub images. No competition submission is created.

The experiment is deliberately an acceptance screen before fine-tuning. Eight
movies (four per embryo) select the checkpoint and normalization mode from 12
uniform frames each. Four different complete movies form the frozen acceptance
set. There is no selection/acceptance overlap and no leaderboard feedback.

For each movie, the threshold is selected from detector probabilities and the
organizer-provided `estimated_number_of_nodes` metadata. Ground truth is not
read until after threshold selection. The acceptance report compares annotated
node recall, physical localization error, node-count ratio, and temporal count
roughness against the frozen public raw graphs.

Biohub supervision is only 2.82% dense by the locked manifest (133,318 labeled
nodes versus 4,725,117 organizer-estimated nodes). Therefore this run does not
fine-tune Spotiflow with its conventional fully supervised background loss. If
the pretrained detector passes, the next experiment will use positive-unlabeled
training: high-confidence pseudo-positives, forced GT anchors, and a low-weight
unlabeled term.

Verified upstream:

- Spotiflow repository commit: `b4f464552afc3424944a9c0fa60ac024e4013f1f`
- License: BSD-3-Clause
- Official `synth_3d.zip` MD5: `a031f1284590886fbae37dc583c0270d`
- Official `smfish_3d.zip` MD5: `c5ab30ba3b9ccb07b4c34442d1b5b615`
- Runtime manifest SHA-256:
  `2b55f535ce37c3c8cd153f1dcc5a1fc546abf7b2f2717e850e22c471e504d55a`

Resource envelope: one T4, Internet off, hard stop at 6,900 seconds, declared
maximum two GPU hours. It must run sequentially after Trackastra.
