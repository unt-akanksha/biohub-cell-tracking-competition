# trackastra-graph-finetune-v6

Status: completed; checkpoint retained only for the predeclared exact
processed-topology posthoc gate. No submission was created.

V5 established that all 195 real graphs, four validation graphs, 384 corrected
synthetic graphs, the offline numerical stack, and the official 27.5M
Trackastra checkpoint load successfully on Kaggle. It then failed before the
first optimizer step because PyTorch deliberately rejects probability-space
binary cross entropy inside CUDA autocast.

V6 makes one implementation repair and no scientific parameter change. The
Trackastra forward pass remains mixed precision, while its normalized
probabilities and BCE are promoted to float32 inside a nested autocast-disabled
region. A regression test instruments the BCE call and proves that it executes
outside its parent autocast context; the focused model/control suite and trainer
self-test pass.

The scientific candidate is unchanged: 1,200 corrected synthetic steps, 7,000
real steps, a 3:1 dense-distractor ratio on real windows, all 195 non-validation
movies, and the 27,456,880-parameter official CTC model. Training-time graph
validation is provisional. Promotion still requires the separate V1 posthoc
gate, which materializes the exact public final-CSV validation topology and
uses disjoint selection/acceptance movies without leaderboard feedback.

## Result

The run completed all 1,200 synthetic and 7,000 real steps in 2,022.8 seconds
of notebook wall time and produced a hash-bound 27,456,880-parameter model.
Its provisional raw-graph validation did not generalize: acceptance proxy fell
from `0.8457157` to `0.7799176` (`-0.0657981`), and the worst movie fell from
`0.8431698` to `0.7349798`. The model is therefore not a submission candidate.

The exact processed-topology posthoc gate remains justified once because V6's
contract explicitly declared the training-time topology provisional. That gate
is the only remaining use of this checkpoint and cannot create a submission.
Failure retires the Trackastra lane immediately. Public leaderboard feedback
was not used.
