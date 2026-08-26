# spotiflow-finetuned-acceptance-v1

Status: staged clean acceptance for the completed learned detector.

The 35.5M checkpoint is hash-bound to the completed synthetic training output.
For each of four complete real movies, normalization is fixed to Spotiflow auto
and a detection threshold is calibrated from twelve uniform image frames plus
the metadata node-count estimate. Ground-truth coordinates are read only after
that movie's threshold is frozen.

Promotion has two levels. Improvement over the pretrained Spotiflow recall
retains dense adaptation as useful transfer learning. Replacing the current
public detector requires beating its same-movie raw graph and avoiding a
material per-movie regression. The public leaderboard is not read and the run
cannot create a submission.
