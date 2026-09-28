# Source disagreement corpus: completed, classifier not justified

Eight source optimization movies completed all 100 frames each on Antelume in
1,366.58 seconds (22.78 minutes). All 84,600,205 output bytes were copied locally
and checked against the remote file set, sizes and SHA-256 hashes. Session 19235
is terminal. The GPU process exited; no GPU process was listed after completion.

Label-free features were frozen before opening these eight source annotations.
The 28-feature implementation checks original/raw node identity and excludes
synthetic/nonshared endpoints. Seven feature/supervision/attribution tests pass.
Complete source GEFF files, not the old 16-transition subset, supply the labels.

There are 3,357 eligible observed neural-versus-motion disagreements, but only
12 unambiguous supervised examples: 11 favour the neural parent and one favours
the motion parent. The other cases are unannotated, unmatched, lack a parent
annotation or have ambiguous alternatives. None is silently made a negative.
This is not evidence for always choosing the neural parent, a trained arbiter,
or a submission gain. In particular, the observed 11:1 ratio is too sparse and
source-specific to justify a deployment rule. No model was fitted.

Next check: broaden the *source training distribution* to all observed parent
candidate groups, including existing agreements and nearby image-derived
alternatives. Freeze candidate geometry before attaching labels; measure how
many known parent errors are actually repairable. This uses downstream trajectory
context absent from the prior native-image parent heads, but it still needs
separate source-selection, embryo-transfer and complete-movie evidence before
promotion. It is not permission to tune to exposed validation errors or repeat
the rejected division/warp policies. No additional GPU run is currently queued.

The submitted public score remains 0.946; top-five is not achieved. GPU instance
billing continues while CPU analysis runs. Source images/runtime remain available
in the owned RAM directory for a justified follow-up. RSNA and shared packages
were not changed. Separately, 192,774,084 bytes of completed dense-warp outputs
were removed from RAM only after exact local and remote backup verification;
all are locally recoverable.

Receipts: trajectory-disagreement-source-v1-full-harvest.json,
trajectory-disagreement-source-v1-supervision.json,
biohub-completed-memory-cleanup-20260914.json.
