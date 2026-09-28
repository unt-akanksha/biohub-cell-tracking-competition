# Fixed checkpoint-average hypothesis, September 14, 2026

The final source models and their equal-weight ensemble failed full evaluation.
Do not replace their results or call a different checkpoint the original model.
Source-only checkpoint inspection found large final-epoch fluctuations in the
six division-geometry coefficients. Source6norm0.790811559357at13400updates
collapsed to0.000000599363at13671. Simulating271Adamupdates with only the
unchanged regularization gradient reproduces all six final values exactly.
This is consistent with absent effective rare-event data gradients in that
tail, not proof that a different checkpoint generalizes better. Source44's
same control over269updates does NOT reproduce its final geometry weights.

Test ONE averaging rule: uniform average of every original regular50-update
checkpoint strictly after epoch2 and through epoch3. Exclude irregular extra
epoch-end checkpoints to avoid their extra weight. This yields25snapshots for
source44(2450..3650) and91for source6(9150..13650). No best-checkpoint selection,
score weights, start-window sweep, late stopping, or refitting. Keep every
original final model and failed result unchanged.

This is a new post-hoc optimization hypothesis, not a predeclared part of the
original training. Both source validation results and the ten-movie selection
results are already exposed. First assess source training losses/rare-event
coverage, without opening new selection/validation annotations. No averaged
checkpoint is submission-authorized from construction or source fitting alone.
If source diagnostics support it, a separately frozen complete-movie evaluation
is still required with all previous exposure disclosed. Averaging is not a
permission to tune repeatedly on the failed ten-movie comparison.

The general motivation comes from [weight-averaging research](https://arxiv.org/abs/1803.05407),
which tested deep-network SGD. Our30coefficient latent structured Adam problem
is different; a benefit here is a hypothesis, not a consequence of that paper.
No new image-network training or GPU expenditure is needed to construct these
diagnostic averages. The separate existing-division-scope audit is not used to
select snapshots or create prediction-time ID/annotation rules.
