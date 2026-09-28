# Parent-presence correction: better loss, feasibility still FAIL

September 10, 2026. Frozen summary job/1 completed in112.933s including
setup (worker32.442s). Actual notebook, runtime, model tensor identity,
summary hashes, target identities and labels verified. Previous whole
diagnostics replayed with exact classification counts; factorized NLL differed
by only2.94e-9 from the original GPU result.

CPU fit completed in5.078s including verification. Fixed ridge1 logistic
correction converged in34 iterations on2,870 fitting examples (2,818 present,
52 absent). Fitting-only feature standardization and parameters persisted
before corrected diagnostic evaluation. Unknown cells were never negatives.
Original real-parent ranking, raw nodes, image encoder and flow unchanged.

| Diagnostic arm | NLL | Correct parent /2645 | Correct absent /27 |
| --- | ---: | ---: | ---: |
| Physical-only | 0.333915 | 2500 | 18 |
| Original neural + physical | 0.219009 | 2585 | 12 |
| Learned presence correction | 0.193443 | 2589 | 12 |

The same frozen gate FAILS: required missing-parent count18 is not reached.
No source tracking extension, target access, coefficient adjustment, gate
relaxation, promotion or submission. Diagnostics remain original-training
domain, not independent validation or an official tracking score.

Verified resultSHA256:
`89a6f4fe7d9b2e1642c3c852ecd17aed15b89080511ecd489ec594a96b954da6`.
Saved fitSHA256:
`f0752ddf15f3de54ea8cc8234cf3f7919e25b14571ee3892573058ea5cac973a`.
Worker resultSHA256:
`4993810152196f08268f15e4a8c5a22f449c5ce3f3e1d042738745f479f06d55`.

Five tests pass, including zero-correction loss parity, fitting-role exclusion,
single-source behavior and correct joint-class decisions. Fresh quota10.48h;
all Biohub GPU jobs completed, no follow-up queued. Reusable summaries permit
further CPU-only analysis. A useful next diagnostic is whether cached image
features distinguish true parents from plausible alternatives; the fitted
nearest-feature cosine has mean0.99953, but that alone does not establish
feature collapse or its cause. No new experiment is implied by that hypothesis.
