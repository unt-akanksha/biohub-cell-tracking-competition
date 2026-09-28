# Fitting-only evidence: conservative dropout is too easy for the physical prior

Completed11.656 CPU seconds; no GPU or diagnostic/source/target access.
All original fitting packet hashes and all1,121 previous augmentation decisions
replayed before computing the original physical-prior null probabilities.

| Fitting cases | Count | Physical correct-null | Mean null NLL | Median best-parent minus null logit |
| --- | ---: | ---: | ---: | ---: |
| Natural missing parents | 161 | 115 (71.43%) | 0.961621 | -3.85575 |
| Conservative14um synthetic | 1136 | 1135 (99.91%) | 0.004479 | -31.19583 |

This supports a training-example difficulty mismatch: the conservative synthetic
null cases are nearly all already solved by the physical prior. It does not prove
the entire neural-training failure was caused by augmentation, because neural
logits were not recomputed in this audit. No diagnostic threshold was selected.

Also inspected official training and inference coordinate contracts. Official
training multiplies detected downsampled coordinates by downsample before edge
prediction; our cached linker passes native coordinates. No scale mismatch was
found. The constant -4.5 null score is not by itself proof of an expressivity bug:
the learned pair scorer can change real-parent logits relative to that constant.

Next justified action is an exact-GT7um training-only dropout audit, not a larger
repeat of14um dropout. Exact GT positions allow removing only the matching
neighborhood while retaining the same conservative missing-parent label meaning.

ResultSHA: `686ba7bc396c23740f1bf273c243b3d22d11dc22cdac22ad33a182d0248a433a`.
SourceSHA: `127a23b3bfacf49f44b53d471244f6d732b9699a18fc07990241b6ed52f19e1a`.
