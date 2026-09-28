# Joint encoder/linker training: verified failure

September 10, 2026, recovered and host-verified at approximately 16:10 UTC.
Kaggle `indarkarhana/biohub-focus-joint-training-v1/1` is COMPLETE.

All 800 fitting-only updates completed. Both encoder and association head
changed; detector head and flow tensors remained unchanged. Original controls
replayed, all nine saved checkpoints passed the worker's exact real-image reload,
and the host verified actual checkpoint hashes, runtime sources, fitting queue,
diagnostic coverage and four guarded AMP overflow skips.

| Diagnostic | Original neural | Final joint | Required |
| --- | ---: | ---: | ---: |
| Unweighted NLL | 0.219008899 | 0.215789351 | Below both controls |
| Correct parent | 2585 / 2645 | 2588 / 2645 | At least 2585 |
| Correct absent parent | 12 / 27 | 11 / 27 | At least 18 |

The unchanged diagnostic gate FAILS. This is training-domain feasibility, not
independent whole-system validation or a leaderboard estimate. No source tracking
evaluation, new target opening, checkpoint selection or submission is authorized
by this result. Do not extend this failed configuration unchanged.

Worker time: 473.038 seconds. End-to-end launcher: 528.006 seconds (8.80 minutes),
within the declared 1800-second cap. Peak allocated memory: 2,565,336,576 bytes;
peak reserved: 3,474,980,864 bytes. No Biohub GPU run remains active.

All artifacts are under `.biohub/cache/kernel-outputs/focus-joint-training-v1`.
Host receipt: `focus-joint-training-v1-result.json`, SHA-256
`f68b07626b9105f66c3dfad11b64927e255189eb250732a4e9504fbe2fa1ba3b`.
Worker result SHA-256:
`b6e0a9e64dcd981ed502325d25497e563fb79fa1d0b9415f72168ae0fbc92283`.
Final checkpoint SHA-256:
`424eb1e47728cff4bbe0cfb7eb2a95531bfcd43a27c3d879620a22082b477368`.

Next research must address missing-parent discrimination with different evidence
or modeling, not tune the threshold against these 27 exposed diagnostic cases.
The conditional-motion component remains useful but also has not passed its
full-source individual-movie regression guard. Qualified submissions today: 0/5.
