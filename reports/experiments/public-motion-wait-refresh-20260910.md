# Source-only refresh while motion validation runs

## Execution evidence at 08:34 UTC

Kaggle `indarkarhana/biohub-flow-spatial-tta-selection-v1/1` was
authoritatively RUNNING. Live receipts covered six of eight complete source
movies, each with coordinates identical to the frozen parent D4 graph. This
is progress, not a completed accuracy result. CPU controller PID 35396 remains
live; do not duplicate its scoring launch. No new submission was made.

A fresh read-only EC2 DescribeInstances request for Antelume returned
`RequestExpired` using the named project profile. The default profile returned
`NoCredentials`. Antelume utilization is unknown; no process, instance, file,
or other-project workload was changed. Local UTC and the provided clock tool
agree to within seconds; this alone does not diagnose the AWS authentication
failure.

## New DAE notebook: hypothesis, not an adopted candidate

Source: https://www.kaggle.com/code/ghazarosghazaros/biohub-dae-self-distill-repeat

Downloaded source only; never executed. Notebook SHA-256:
`19c4084688f91a6ee1a0741dee36055ccdb12dbf5fcd2a560f9577ad0b37edd7`.
It retains the public support50/seed314159/DeepCenter artifacts. Its distinct
code trains a small per-video denoising autoencoder for 30 updates on eight
frames with artificial Gaussian corruption, then blends its reconstruction
into the detector input. A second change repeatedly blends rasterized,
blurred high-confidence peaks into detection logits. That is inference-time
logit modification, not a newly trained large detector checkpoint.

The peak operation includes a 40th-percentile confidence cutoff and rejects
changes if predicted-node retention falls below 90%. Other public graph and
division filtering remains. These mechanisms do not enter our candidate:
they lack independent complete-movie evidence here, and the node-count guard
is outside our fixed evidence policy. This is not a finding that every such
filter is an intentional metric exploit. This was a targeted source screen,
not a comprehensive clean-code certification.

The source's 0.946 parent claim is the author's statement, not a verified
leaderboard observation or our score. Neither that number nor its coefficients
select our models. A future denoising hypothesis would require a separately
declared training/validation protocol and unchanged graph-validity rules.

## FOCUS-3D: correct the earlier license conclusion

**Subsequent history reconciliation:** this refresh initially consulted the
September 7 audit without the later execution records. The repository already
contains Apache-2.0 attribution and completed FOCUS detector experiments.
This is not a newly cleared model and must not trigger a repeat of those runs.
See `focus3d-frozen-validation-v1-result.json` (retired physical linker;
historical packaged-scorer result, not current-metric promotion evidence),
`focus-raw-detections-v1-verification.json` (400 frames, 65,091 raw centroids),
`focus-raw-learned-linker-v2-result.json` (current-metric diagnostic score
delta -0.0637988), and `focus-edge-consensus-v1-result.json` (diagnostic
gain +0.0007173, no independent promotion). The statements below about no
download/run describe this refresh only, not the project's full history.

The currently readable official model card explicitly states Apache-2.0 for
the weights. Thus the older September 7 claim that the weight license is
undeclared is no longer supported by the evidence available to us. Do not
rewrite that historical audit or infer when the text changed: today's public
API still gives the same repository revision recorded on September 7.

- Model card: https://huggingface.co/Qinghua-thu/FOCUS-3D
- Source/code license: https://github.com/yu-lab-vt/FOCUS-3D/blob/main/LICENSE
- Public metadata: https://huggingface.co/api/models/Qinghua-thu/FOCUS-3D?blobs=true
- Paper: https://www.biorxiv.org/content/10.64898/2026.08.25.746907v1

Observed repository revision: `115258efcc9ee44e69db3902bce2511d0ae24e2f`.
The API still reports automatic access gating. Exact nuclei checkpoint:
`model_final_nuclei.pth`, 4,468,804,784 bytes, public LFS SHA-256
`b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a`.
The README blob identifier is `e3e2398f0d356fdaa11d6d120753011ddf9508e8`.
No gated files were fetched, no terms accepted, no personal information shared,
and no checkpoint downloaded or run. A mirror must not be assumed identical
without matching the official checkpoint hash and distribution terms.

FOCUS-3D is already a known independent architecture in our September 7 audit,
not a newly discovered model family. License visibility is a meaningful
correction, but exact training-data/embryo overlap, legitimate artifact access,
dependency compatibility, mask-to-center accuracy, and full-movie runtime
remain unresolved. Public sparse recall and paper segmentation benchmarks
are not Biohub patched-official tracking scores. The full-paper retrieval
failed this turn, so no new provenance clearance is claimed.

## Discussion evidence limits

Indexed excerpts from discussion 737896 raise dim-node/annotation ambiguity;
comments are participant hypotheses, not host-authorized annotation removal.
Discussion 732103 cautions that sparse-label recall can favor clear cells and
describes synthetic pretraining plus real fine-tuning. Discussion 739685
questions whether local linker improvements transfer to the leaderboard.
These are ideas, not independent evidence that overrides our recorded tests.
Direct discussion page bodies were not readable; latest full-thread coverage
is not claimed. Known negative-time/fabricated-division notebooks remain
excluded from pulls and execution.

- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739685

No active or completed frozen notebook was rebuilt. No new target movie was
opened and no prediction, threshold, or ensemble weight was changed by this
refresh.
