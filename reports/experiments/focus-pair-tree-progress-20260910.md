# Nonlinear candidate ranking: resource recovery complete, quality screen running

September10,20:32UTC. No new qualified submission yet. LDA and full linear
appearance corrections remain rejected by the unchanged missing-parent gate.
Their complete verified comparison is in focus-pair-appearance-comparison-v1-result.md.

## Completed first-fold resource recovery

Recovery36363 is terminal,244.203s elapsed, actual numerical workerPID confirmed.
Peak5,241,425,920B (4.88GiB), below the unchanged6GiB cap. It preserved all
4,460,090candidate/null rows and10,403known groups from11fitting movies. Every
stored feature/margin/group replayed exactly from source before resuming.
The excluded movie6479435d was not loaded or scored during this recovery.

The original stopped75tree checkpoint was retained and only25remaining rounds
were fitted. The separate real smoke75+25 reproduced the original100trees and
scores exactly. Full-size native/portable/reloaded prediction error0; weighted
training loss4276.409665 ->4138.679778 at75trees ->4091.381915 at100trees. These
are fitting losses, not held-out quality or tracking scores. Redundant source
maps were released before the unchanged DMatrix construction; no parameters,
candidate counts, objective, histogram algorithm or budget were changed.

ReceiptSHA256fefb07cf2f38aafa90eaab53d6f672a02d81b34e46869f4e326faa5c867400cb.
WorkerSHA2562fa2ea0531ea3a1746f1abe84d55002fe21247ae50029e68e8e73e7c22174fd5.
PortablemodelSHA2565d23292090fc7b4572b1a0f9b296161c9287c72c03b1a98bced1e0910f2c5874.
The initial stopped profile's4.5MB launcher reading remains invalid as a worker
measurement. Only the identified owned profile child was stopped; no other
project was modified. Guard regression tests cover actualPID, isolated imports,
real memory allocation and owned-child termination.21 combined tests passed;
the final13 focused tree/recovery/LOMO tests passed1.20s before launch.

## Live complete comparison

Session21863 runs scripts/fit-focus-pair-tree-lomo.py, launched20:32UTC.
Twelve sequential correction-held-out folds; existing completed first model
reused after verifying its native75tree prefix. Remaining folds train the exact
fixed100tree model on their other11movies and save the complete composite before
held-out scoring. Each original fold-only full linear baseline is hash-bound;
scalers/projections/counts and training objective replay before residual fits.

First fold completed1.406s with no refitting. CandidateNLL0.548993 versus full
linear0.554247; parents417/492 versus420, absent11/20 unchanged. This one fold
is neither an aggregate gain nor a promotion. No response-dependent retuning.
Only after all12 may the original five gates be evaluated. Passing still needs
independent host replay, all-fit export and diagnostic/complete-movie/embryo/
offline runtime checks. No final all-fit model or submission in this run.

Two localCPU threads,900s per fold,7200s whole-run cap,6GiB actual-worker guard,
at least7GiB free memory checked before each launch. Expected60-90minutes with
two-hour hard cap, not a guaranteed finish time. Fresh workerPID identities
prevent the Windows launcher-accounting defect. All prior train, verifier,
install, smoke, failed-profile, recovery and test sessions are terminal.
No GPU/cloud/RSNA mutation. Historical Kaggle8.22h at19:17 is not a fresh quota;
refresh before futureGPU launches and preserve8hours. Goalunfinished;0/5new
qualified submissions.
