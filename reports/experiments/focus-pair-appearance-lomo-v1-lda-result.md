# LDA pair representation: completed and verified, missing-parent gate fails

September10 19:31UTC. All12 LDA correction-held-out folds completed in421.0CPU
seconds. Host replay completed in145.125s without optimizer refitting. No GPU,
all-fit final model, diagnostic/source/target evaluation or submission for this
arm. The separately predeclared full-feature arm continues in live session2266.

| Same12movies | Physical | Original neural | Prior weighted ranker | LDA addition |
| --- | ---: | ---: | ---: | ---: |
| Pooled NLL, lower better |0.802871|0.667407|0.362457|0.342701|
| Correct parents /10,754 |9514|9835|9891|9880|
| Correct missing parents /161 |115|72|73|84|

LDA lowers NLL versus the original neural control on all12movies; smallest gain
0.00522684 on4f99ce20. Four unchanged gates pass, but84 missing-parent decisions
is below115 required. No promotion or threshold/weight adjustment. Relative to
the prior weighted model, it exchanges11 correct parent links for11 improved
missing-parent decisions and lower NLL. This supports further comparison of the
representations, not a claim that LDA solves tracking or reaches0.945.

Every fold used only the other11movies for class moments, scaling, Fisher
projection, class weight and head fitting. All4,691,320 choices and10,915 groups
were retained. The verifier refit each fold-only projection exactly, replayed
the saved model through the original streamed training objective, and computed
held-out log losses/argmax decisions independently one complete group at a time.
It replayed all original/weighted controls and the unchanged acceptance gate.
The encoder saw these fitting movies: this remains a correction-only screen,
not new-embryo validation or a Kaggle score.

Actual result SHA2569d3713fae680f67ef149444f32c176c3705cfaed74b7a9122efba6a5ea59a779.
Host verification SHA256c3045123571fa1daccf23f3cd784d751ca240f9bf384b942a1a5562337f704f4.
Fold artifacts:`.biohub/cache/focus-pair-appearance-lomo-v1/lda`.
Verification:`focus-pair-appearance-lomo-v1-lda-verification.json`.

The full72D arm is still fitting its first movie-held-out model, last live
output100iterations/121.125fit seconds. Do not restart it: session2266 is live,
with the9000s whole-run watchdog. Verifier39992 is terminal. All GPU smoke,
download and CPU profile handles are terminal. No new Biohub GPU job active;
no cloud or RSNA mutation. Last quota observation8.22h at19:17 is historical,
not a current account balance. No further GPU launch needed for this comparison.

The full goal and request for five qualified submissions remain unfinished0/5.
