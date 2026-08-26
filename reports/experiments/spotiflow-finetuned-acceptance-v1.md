# spotiflow-finetuned-acceptance-v1

Status: completed; learned detector retired as a drop-in candidate.

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

The guarded run completed in 162.778 seconds. On four complete movies the
fine-tuned model recalled only 218 of 2,357 annotated nodes (0.09249), versus
0.77768 for the pretrained Spotiflow control and 0.96903 for the frozen public
detector. Three movies produced no detections at the threshold selected by the
fixed image-only calibration rule. The synthetic checkpoint is therefore
retired for both drop-in use and further detector adaptation; no submission was
created. The detector remains frozen while work moves to density-matched learned
association reranking.
