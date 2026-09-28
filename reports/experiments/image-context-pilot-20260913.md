# Image-context model: execution record

September 13, 2026. Prior goal turn was progress: supported bridging completed
and was rejected as exactly neutral. That branch remains closed.

## New evidence

The graph-context archive builder reads sparse GT node arrays; the submission
builder reads dense predicted nodes. This is not a metric exploit, but the
old `context_labels_used=false` field must not be interpreted as image-derived
neighborhood provenance. Source exclusion of edge arrays does not erase the
fact that neighbor coordinates came from annotations.

The optimization-only domain diagnostic completed in 27.125 CPU seconds:

| Input | Training mean valid tokens | Inference mean valid tokens |
| --- | ---: | ---: |
| 44b6 | 12.3104 | 42.3384 on 328 candidate triplets |
| 6bba | 18.3090 | 41.6953 on 548 candidate triplets |

In 44b6, 99.39% of the saved inference candidates exceed the optimization
99th-percentile token count. In 6bba the training p99 already reaches 43, so
that statistic is zero despite a large mean/median shift. Actual inference
feature counts were checked against the faster count calculation. No division
targets, selection/audit shards, model logits, or score were opened by this
diagnostic. These counts do not prove causal model reliance or a score gain.

## Candidate and split

Image-derived context replaces annotation-neighbor tokens with peaks, processed
half-max component size and relative contrast from the existing raw-image crop
triplets. This uses the existing 74,732,308-parameter CNN/transformer architecture,
initialized only from exact external ZebraHub checkpoints. It does not reuse
the failed Biohub-trained graph models or change public baseline weights.

The first data build completed but found an infeasible source-only selection
gate: v3-role 44b6 selection has one eligible positive, while two are required.
Its immutable v1 data was not trained. V2 restores the pre-existing original
optimization/selection roles, retaining two eligible 44b6 selection positives
without reducing any numerical gate or searching model scores. Original audit
shards remain excluded; some historically opened v3-audit source-only movies
now belong to original optimization. No pristine project-wide audit claim.

V2 feature preparation completed in 61.656 seconds. Four packets contain 2,480
examples across 121 movies, with all input hashes, shapes, finite values, masks,
binary labels, positive weights, movie separation, and embryo exclusion checked.
The pure feature/calibration tests pass 8/8, including tied-score threshold safety.

Two models are planned sequentially: source 44b6 -> target 6bba and source 6bba
-> target 44b6. Each uses only its source embryo for optimization, selection and
thresholds. Twenty real optimizer steps and checkpoint reload must pass before
the 2x2,000-step, 40-minute maximum pilot. No model has yet been trained at the
time of this entry; transfer to an isolated Antelume directory is in progress.

## Frozen artifacts

- V2 data manifest SHA256:
  `39032899ecea16e87bfb53d45909f06993d39ff0cfbb0bbb16a7b875f9dcecce`.
- Cloud bundle manifest SHA256:
  `467076cacc2587444267f6e14f4edb93dd578c27fb1eff76ffbc06cbb290bdb6`.
- Compressed cloud bundle: 513,605,772 bytes, SHA256
  `ee0bf3e7a3a73bc4ec68dfb511a5bacb5a921d9a098a5b5700e231edf73f9ed1`.
- Isolated destination: `/tmp/biohub-image-context-v2.ScdSdY` on the existing
  Antelume instance. No RSNA deletion, package installation, instance shutdown,
  Kaggle GPU use, or submission.

Model training source: `scripts/run-image-context-pilot-v2.py`. Exact external
warm starts and all copied runtime/data files are bound by the cloud manifest.
The bootstrap package initializers are intentionally inert to avoid importing
the historical unrelated Torch modules. Both models, source thresholds and
held-out embryo results must pass before complete-movie evaluation is justified.
This is not a submission candidate yet.

## Completed execution, 20:28 UTC

The actual GPU smoke passed in 16.569 seconds, including exact checkpoint
prediction replay. The full sequential two-model pilot completed 595.126
seconds (9.92 minutes), terminal `rejected_source_selection`. Source 44b6 best
step 750: eligible AP 0.833333, one TP before the first FP, below the unchanged
two-TP gate. Source 6bba best step 1,250: AP 0.824296, seven TPs before any FP,
source threshold 1.609375. The joint gate FAILS; opposite-embryo scores were
not opened. No model is promoted, re-seeded, extended or submitted.

Full-pilot peak CUDA allocation 2,084,172,800 bytes. Including the smoke,
611.695 seconds of these two job walltimes, not billable instance uptime.
The GPU process query was empty after completion; the shared instance remains
running. Root free storage is now 2,748,997,632 bytes; keep all other projects
untouched. Source 44b6 weights/history/terminal were independently size/hash
verified locally while source 6bba trained. Full terminal/resume backup is now
in progress. Pure feature, threshold and inference-adapter tests pass 10/10.

Exact source-selected checkpoint hashes:
- 44b6: `18802e460f736be710f0cec47b081e2cddc1848a3796591cda801c874b7ebae1`
- 6bba: `2f4dc62a1c501cea8f1fa1bef109d71fc4238ebd964baaa31696e63b582721bd`

A distinct frozen-encoder/regularized-head screen is specified separately in
`frozen-image-head-v1-design.md`; it does not reuse these Biohub-trained weights.
