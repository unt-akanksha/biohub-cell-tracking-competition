# Proposed null-balanced association-head training

Prepared after both twelve-movie calibration-only models failed the unchanged
missing-parent gate. This changes the training objective of the association
head itself, not merely a fitted presence offset or a decision threshold.
No trained checkpoint or GPU launch yet; real Torch/gradient checks remain
required before staging the GPU experiment.

Start from the original owned model41e82...,not a failed adapted checkpoint.
Use the already verified twelve fitting feature caches and unchanged four
diagnostic caches. Freeze encoder/detector/flow; train only the association
transformer. Keep original800steps,AdamWlr1e-4/wd1e-4,gradient clip1,seed244691,
uniform supervised fitting-pair queue, neural/physical weight1, null logit-4.5,
and original physical calibration. No checkpoint choice based on diagnostics.

For every supervised target, use the ordinary multiclass parent/null negative
log-likelihood, but weight a verified-null target by sqrt(10754/161), derived
only from fitting counts. Real-parent weight1; normalize the batch loss by the
sum of active target weights. Unknown labels(-1) remain completely excluded.
Both division daughters may select the same parent. This is a fixed moderate
frequency correction, not a sweep or weights chosen from diagnostic outcomes.

Before real training: test weight1 against the original indexed-parent loss,
exact zero gradients for unknown targets, finite loss/gradients, correct known-
null gradient direction, integer-label guards and same-parent division labels.
Then a four-step real fitting smoke must verify gradients, frozen trunk hashes
and strict saved-checkpoint replay before continuing to the declared800steps.
Diagnostic evaluation remains unweighted and unchanged: NLL below both fixed
controls, parent correctness>=2585, absent correctness>=18 on the same2645/27
examples. No inference threshold, node inventory or graph rule changes.

A pass only permits complete-source tracking validation, not a submission.
Budget/ownership checks and small environment replay precede any GPU launch.
Antelume's RSNA job and other account activity must remain undisturbed; no GPU
run is authorized by this design alone when resource checks fail.
