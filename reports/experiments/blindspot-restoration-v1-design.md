# Source-trained blind-spot restoration before FOCUS detection

Hypothesis: improve image signal before the stronger FOCUS detector rather
than tuning tracking counts. The completed fixed-division association test
did not recover its weak movies. This is not the public notebook's noisy-input
autoencoder, peak-logit modification, percentile filter or retention guard.

[Noise2Self](https://proceedings.mlr.press/v97/batson19a.html) provides the
self-supervised denoising principle using conditional independence of noise.
Our implementation is an original small3D CNN with masked dilation1 followed
by dilation2/4 and a1x1 head. It has no normalization layers or identity skip.
Odd/even receptive-field arithmetic excludes the input center. This property
is conditional on fixed weights and normalization. Actual Biohub noise
independence and denoising benefit remain unverified; the paper does not
establish either for these images or this implementation.

## Completed synthetic functionality check

14 local geometry/builder tests passed. Private offline CPU-only Kaggle
`biohub-blindspot-cpu-probe-v1/1` COMPLETE, no inputs or GPU. Source wrapperSHA
`a38e66a7c852b9706ce0add7443d25b06382924c9bbb42ca89e09ebd012c78bf`.
Exact executed files match the staged sources. Raw receiptSHA
`6b0d19df5aae896f00650b336980904b0a71ba88a98165616f89275099d245fc`.
Torch2.10.0+cpu; five interior/boundary probes show zero center derivative
and zero output change after perturbing that center, with nonzero neighboring
dependence. Full/tiled max error0.0.32 synthetic optimizer steps lower MSE
0.4663170->0.0204683; central derivative remains0 and restricted checkpoint
reload is exact.11.305s launcher,6.343s worker. These are functional checks,
not microscope-image or detector-validation results. Width16 has14,321
parameters; this is an auxiliary restorer, not a replacement for FOCUS.

## Real-image probe protocol

Use the original120 training-pool movies, split deterministically into96
fitting and24 diagnostic movies (`fold.train[::5]` held aside). Exclude all8
tracking-selection movies and every44b6 target movie. Read only images at
frames0,49,99. Per-movie/frame seeded native32x64x64 patches; no annotations.
Fit global affine normalization using fitting pixels only. No per-test-movie
normalization fitting, input noise synthesis, signal/count-based pruning,
identity blending or threshold sweep. Train width16 from scratch for100
steps, Adam1e-3, batch4, AMP, seed20260910. Retain only the declared final
step; no diagnostic-selected checkpoint.

Diagnostic proxy: held-out-pool interior-voxel MSE against noisy observations,
compared with a fixed26-neighbour mean excluding the center. Require finite
training/output, retained blind spot and serialization, lower pooled proxy
MSE than the neighbour mean and gains on at least18/24 diagnostic movies.
Noise assumptions limit this proxy: passing permits a detector smoke test,
not tracking promotion. No automatic1000-step extension or source detector
rerun. Any real-image GPU launch needs fresh quota>=9h for its1h cap and
must follow the user's small-test-first requirement.

Only after a real-image proxy and detector smoke succeed should a complete-
source comparison be declared. It must retain the existing source gate and
show gains over FOCUS-flow0.7753326 without unacceptable per-movie regression;
target-embryo checks and offline two-GPU submission runtime remain required.

## Execution update

Latest result: v2 ERROR after completing100 steps,119.523s launcher, but all
weights/diagnostics are now preserved and host-verified. The quality proxy
FAILS decisively: model MSE0.0007877016 vs neighbour0.0000138540,56.86x higher,
zero of24 movies improved. Do not run FOCUS on this checkpoint's output.
The strict GPU derivative audit reports center gradients up to1.026e-10
despite exactly zero output change under center perturbations and nonzero
full-input response. Numerical backward artifacts are a plausible explanation,
not confirmed by a CPU replay of these weights. No such audit or extended
training is scheduled: it would not overturn this checkpoint's proxy failure.
No quality threshold was relaxed. All360 input patches and five logged losses
replayed v1; actual executed runtime hashes verified. Final checkpointSHA
e4b073079e07a96e5966c79a1cbfd55137332cdcca52072212a759719a684431.
Resumable steps20/40/60/80/100 and final weights are saved locally and in the
Kaggle output. `blindspot-real-probe-v2-result.json` records source/artifact
identities. No active GPU job, detector rerun, new target access or submission.

Historical failure/repair launch:

Superseding status: v1 ERROR after100 training steps,82.018s launcher. Its
single-point gradient assertion can reject a legitimate inactive ReLU region;
the error did not distinguish zero neighbor sensitivity from center leakage.
No final checkpoint/diagnostic score was saved before that assertion. Host
harvest recovered the identity and360-patch inventory only. See failureJSON.

Repair `biohub-blindspot-real-probe-v2/1` is accepted after22 focused tests.
Exact same architecture, optimizer,100-step schedule, seed, image split and
proxy thresholds. It separately checks zero center influence at15 locations
and actual output response to changing the full input across3 predetermined
training samples. Every20-step snapshot includes model/optimizer/scaler/RNG;
final weights and proxy diagnostics are saved BEFORE the functional audit.
Fresh quota11.28h,declared1h,reserve8h. Frozen notebookSHA
`1f49aed74d7ab47223afa6b04684747b3c7443a67fb9cdc1bf9079031da7967f`.
No larger training run or detector inference queued; repair result pending.

Historical v1 launch:

`biohub-blindspot-real-probe-v1/1` accepted and confirmed RUNNING at11:01UTC.
Fresh quota11.31h; declared1h cap leaves10.31h worst-case, preserving8h.
18 local geometry/contract/builder checks passed before launch, after the
completed CPU tensor check. Two T4s, offline, no pretrained model input.
Frozen notebookSHA:
`4aab44d86d15688621f2cc983e39604e609de28b31751522fc109754412f3063`.
Native training-only patch load and the100-step check are running; no real
denoising quality result yet. No automatic extension, detector rerun, target
access or submission. Do not modify staged notebook or model source bytes.
