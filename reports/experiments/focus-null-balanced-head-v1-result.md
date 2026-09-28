# Null-balanced head: diagnostic failure

Kaggle biohub-focus-null-balanced-head-v1/1 completed800steps after the real
four-step smoke passed. This is our trained association head, not copied public
predictions. The unchanged diagnostic feasibility gate failed; not promoted,
not submitted, and no new source/target evaluation authorized for this arm.

| Diagnostic | Original neural | Final weighted head | Requirement |
|---|---:|---:|---:|
| Unweighted NLL |0.219008899|0.216273710|Below both controls|
| Correct parent /2645 |2585|2588|At least2585|
| Correct absent parent /27 |12|11|At least18|

Physical control NLL0.333914953,parent2500,absent18. Initial controls replayed
exactly before optimizer. Fixed weight8.172822710416556 from fitting10754/161,
1121 supervised fitting pairs,377 diagnostic pairs. No diagnostics entered
optimizer; original frozen encoder hash remaineda9a07f33....

Worker115.906899s. Smoke checkpointSHAebdb30fb9c67f4849a5ee6bf62a608035b20928c73e6914eb1a1cd2492c4c221,
final checkpointSHA03663b9367e158e41719ba32bb2953b6bca209d2480a84ae7e5b86ce65336ea4.
Final model tensorSHA093fb33b566039d7c81b78ab9811f672321337ce5d9c270b9940af69a17d7552.
Both checkpoints passed exact real-logit reload in the worker. Host verification
receipt is recorded separately once downloaded artifacts have been checked.

Host verification completed successfully: actual source bytes, upstream feature
evidence, saved checkpoint hashes and unchanged gate agree. Receipt:
focus-null-balanced-head-v1-result.json; worker resultSHA
8707c59d9e7232afb788f45c9f364956d2ffe9526bcc0d297f2a6b4c180fbe08.
Launcher169.913475s. Verification confirms experimental integrity, not quality.

Conclusion: this fixed800step, frozen-encoder, frequency-weighted head variant
did not improve missing-parent recognition. This does not prove all reweighting
or larger training fails. Do not increase weight or choose checkpoints based on
these27 diagnostic examples. Do not extend this failed arm unchanged to source.
No independent tracking score or leaderboard claim follows from these numbers.
