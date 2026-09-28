# Deployment fit and additional division-positive validation

Frozen after the cross-source trajectory model passes the exposed four-movie
diagnostic (0.9448387313 ->0.9462748346, +2edgeTP, unchanged44edgeFP and node
counts). This is NOT a leaderboard score or independent validation of the public
backbones, which trained on overlapping competition data. Source motion model
fits are independently verified by an alternate weighted least-squares solver.

Hidden test embryos are disjoint from training. Therefore deployment must not
look up a known embryo prefix to select a model. Standard final refit: concatenate
the two frozen source optimization trajectory banks, fit the exact same AR2
recipe with equal movie weight, and calibrate once on the concatenated original
source selection banks. No new data roles, thresholds, hyperparameters or
selection among deployment variants. Both audit/final probe sets and the four
diagnostic movies remain excluded. This single model applies identically to
every movie, including previously unseen embryo identifiers.

Freeze the final model before testing. Recheck its exact four complete-movie
graphs against original public baseline with the unchanged gates; no promotion
of the deployment refit if it loses the cross-source evidence. Then run the
unchanged original public image/model pipeline on four100-frame movies:
44b6_12dfb391,44b6_267148e4,6bba_062c8d37,6bba_07e24132. These contain positive
divisions and were excluded from the motion fit; they have historical project
exposure and are not a pristine test set. Freeze original and repaired predictions
for all four before scoring complete GEFF annotations with patched official
counts, per-embryo and worst-movie reporting. Require no per-movie/per-embryo
combined regression and no division-count regression; joint eight-movie raw and
combined improvement with genuine edgeTP gain. No retuning if it fails.

Actual same-code GPU smoke: one8-frame movie,600s maximum; full original model
inference once per movie,4x100frames,3600s maximum on idle Antelume A10G only.
Apply the tiny fixed repair on CPU to that same base output, no second expensive
model pass. Hash inputs/output, no truth in cloud bundle, no installs,70%CUDA
allocator limit,2CPU threads, sequential jobs, do not touch RSNA. Full output
and model backups precede any owned scratch cleanup. Shared instance not stopped.

Offline submission packaging still needs genuine two-worker/runtime testing,
full hidden-size runtime budget, license/provenance packet and fresh Kaggle quota
with8h reserve. These research checks do not certify a0.945+ hidden/LB score.
