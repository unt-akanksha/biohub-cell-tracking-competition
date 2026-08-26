# trackastra-graph-finetune-v6

Status: staged corrective rerun of V5; no concurrent GPU job and no submission
code.

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
