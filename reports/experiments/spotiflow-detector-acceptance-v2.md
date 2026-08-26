# spotiflow-detector-acceptance-v2

Status: staged after repairing the offline dependency installation; no submission is produced.

This is the first independent detector acceptance run. It compares the two
official 35,489,892-parameter Spotiflow 3D checkpoints on eight disjoint
selection movies, then evaluates the selected arm on four complete held-out
movies against the frozen public detector graphs.

Thresholds use only image predictions and organizer-provided node-count
metadata before ground truth is scored. The run reports annotated-node recall,
physical localization error, count ratio, and temporal count roughness. It does
not use leaderboard feedback.

The retry installs exact graph-IO wheels with `--no-deps`, keeps Kaggle NumPy
2.0.2 unchanged, then smoke-tests SciPy assignment and loads a complete GEFF
graph before any model inference. This directly repairs the dependency resolver
failure observed in `trackastra-graph-finetune-v2`.

Resource envelope: one T4, Internet off, hard stop at 6,900 seconds, declared
maximum two GPU hours. Promotion is only to positive-unlabeled Biohub
fine-tuning; this run cannot create a competition submission.
