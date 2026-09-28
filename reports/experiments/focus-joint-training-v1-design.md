# Bounded800step joint encoder/association training

Requires host-verified successful backoffsmoke before staging. Start original
owned76f7... checkpoint, not the4step smokecheckpoint or failedhead. Use all
1121 supervised fitting pairs from12fixed movies;377diagnostic pairs from4
other training movies remain excluded from optimizer. Same null-balanced loss,
8.1728227weight, fixed original physicalGaussian/null/posterior semantics.

Only change versus head-only training: learn the temporal encoder jointly with
associationhead using real normalized image pairs. EncoderLR1e-5/head1e-4,
AdamWweightdecay1e-4,clip1,seed244691,AMPFP16 and tested bounded backoff. Keep
encoderBN runningstats eval and detectorhead/flow unchanged. Uniform shuffled
supervised fitting-pair queue;800successful updates, no checkpoint choice.

Repeat four fitting image/logit replays and complete original physical/neural
diagnostic controls before optimizer. Require the unchanged final diagnostic
gate: unweightedNLLbelowbothcontrols,parentcorrect>=2585,nullcorrect>=18 on
same2645parent/27null labels. A pass only permits full-source tracking evaluation.
No source/target image/label access, inference threshold changes or submission.

Save actual optimizer/scaler/RNG/queue checkpoints atstep4 and every100steps.
Each save must strict-reload and exactly replay real-image FP32 logits. Record
finite jointgradients and overflow skips; stop if bound exceeded. Soft training
stop1500worker seconds persists currentcheckpoint and marks incomplete; no
quality promotion. GPU notebook hardcap1800s, launcher deadline1680s to leave
shutdown margin. Freshquota beforelaunch must leave>=8h after0.5h worstcase.
Only launch after the priorBiohubjob is terminal. No RSNA/cloud mutations.
