# Dense deformation linker adaptation: finished, not submission-qualified

Update, September 14 approximately 09:18 UTC: the prospectively fixed complete-
movie diagnostic also finished and rejected BOTH final linkers. Four-movie
control .93839 versus source44 .92501/source6 .91854; both embryos regress.
See `dense-warp-movies-v1-findings.md`. The planned diagnostic mentioned below
is now completed, not a queued next step. No ensemble or retuning is authorized.

September 14, 2026. Two source-specific fits finished all 4,000 updates in
318.13 seconds including feature extraction. Both retain the original public
detector and temporal U-Net exactly; only the 580,353-parameter linker changed.
No new Kaggle job or submission was launched.

| Source | Correct parents, original -> adapted | Cross entropy, original -> adapted | Frozen screen |
| --- | --- | --- | --- |
| 44b6 | 18/18 -> 18/18 | 0.129182 -> 0.011973 | No additional correct parent |
| 6bba | 181/185 -> 181/185 | 0.150000 -> 0.054997 | No net gain; one movie loses one, another gains one |

Neither passed the predeclared source gate. Neither opposite-embryo model output
was opened, and no ensemble was evaluated. Do not silently relax the threshold,
change the loss/seed, promote these weights, or call lower cross entropy a
tracking-score gain. Public initialization overlaps competition training;
these are adaptation-held-out diagnostics, not independent model CV.

## What this changes

Dense image-transform supervision is technically viable and does transfer a
large likelihood improvement to these real pairs; the previous division-only
tiny-label recipe is not being repeated. The evidence does NOT show improved
parent ranking or better official tracking. Synthetic warps cannot represent
mitosis or unknown births, and learned confidence may increase false links.

The 44b6 screening design has a ceiling problem: only 18 eligible links across
four transitions per movie, and the original is already perfect. A requirement
of one extra correct parent is unattainable on that subset regardless of model
quality. Report that design limitation explicitly rather than interpreting it
as proof that the model cannot improve full-movie inference. 6bba has a genuine
per-movie regression under the unchanged screen, even though pooled rank count
is unchanged. Neither observation justifies a submission or an ensemble.

The next useful measurement is a prospectively specified complete-movie
comparison of the fixed checkpoints, not another training run. It must retain
the failed partial-screen record and disclose that these movie identities and
four timepoints have already been used. It must measure the downstream graph
and patched official score, since confidence affects edge retention whereas
top-1 ranking does not measure it. This is diagnostic work only until a complete
paired promotion gate passes; no model/threshold selection on the expanded
results. Preserve untouched validation scope for later confirmation. No such
complete-movie job is currently queued.

## Reproducibility and checks

- Eight CPU geometry/identity/screen tests pass; real GPU smoke passed strict
  loading, finite backward, 60 nonzero gradient tensors, exact frozen modules
  and exact checkpoint reload.
- An independent CPU tensor audit verified all 70,823 dense correspondences
  across 364 pairs from 91 optimization movies. Maximum analytic image-transform
  correspondence error was 1.907345e-6 isotropic voxels. All ten selection movies
  remained outside optimization. Both final frozen backbones equal the parent
  tensors byte-for-byte. Mixed-source smoke weights were not used.
- Full artifacts (78,393,697 bytes) and smoke artifacts (18,582,952 bytes) were
  copied locally and verified against remote SHA-256 manifests. Final model
  hashes: 44b6 `bbb447a02c7ae096de9cbfc5e0d98e1c6f16db20341cb32b36d466aa7718eb96`;
  6bba `ebece605a713211d06aa69f4df6b1635ce955f69d81ed26f23cf2053eba2452d`.
- An independent verifier recomputed the frozen source gates and confirmed
  zero qualified experts. Result SHA-256:
  `bd0960a7e731e796a0e8f064a78323b9debbc6f0f246189669f54037184b83f3`.

Artifacts: [result](dense-warp-v1-full-result.json),
[verification](dense-warp-v1-verification.json),
[backup](dense-warp-v1-full-harvest.json),
[frozen protocol](dense-warp-correspondence-v1-design.md).
Antelume's training process exited normally. No instance stop, system cache drop,
GPU reset, RSNA interruption or shared-environment modification occurred.
