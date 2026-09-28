# Additive supervision v5: calibration improves, candidate still rejected

Completed September14,2026,06:56UTC. No new submission or ensemble is qualified.
The existing trajectory-motion submission56219125 was observed PENDING again
at06:56UTC; its local0.9453907score remains a local diagnostic, not leaderboard.

## Executed experiment

Extracted all21additional source-optimization positive triplets:3from44b6,
18from6bba. Each raw frame hash matched the prior audit; every patch center
exactly matched fresh image-only proposals. No new negative labels were made.
All old3,087feature/label rows and every selection row remained exactly equal
after appending the21new rows. Totals are65optimization positives,2,305negative
examples; selection stays7positive/731negative, with its original movie roles.

The original v3training recipe was held fixed: four independent frozen-encoder
visual heads and two geometry controls, each2,000full-batch Adam updates. All
six original saved heads were replayed on the unchanged source selection arrays;
their previous metrics reproduced within1e-7. No target-pilot labels were read.

| Source | Head | Old balanced NLL | New balanced NLL | New source TP | Opposite gate |
| --- | --- | ---: | ---: | ---: | --- |
| 44b6 | Geometry control | 0.80754 | 0.69480 | 0/1 | Not opened |
| 44b6 | CNN-origin | 0.54857 | 0.49212 | 0/1 | Source failed |
| 44b6 | Transformer-origin | 0.35527 | 0.28721 | 0/1 | Source failed |
| 6bba | Geometry control | 0.18220 | 0.17208 | 3/6 | Comparator only |
| 6bba | CNN-origin | 0.10699 | 0.08713 | 4/6 | 0/1 TP; NLL1.09508vsgeometry0.49269: failed |
| 6bba | Transformer-origin | 0.08742 | 0.08465 | 5/6 | 0/1 TP; NLL1.31311vsgeometry0.49269: failed |

Source true-positive counts are unchanged. Source false positives are zero by
calibration; both opened opposite checks also have zero false positives but
miss their single positive. The tiny positive sample precludes broad statistical
claims. Better NLL is real paired evidence, but not a recovered division or an
accepted model. Both potential ensembles remain ineligible. Do not relax the
threshold or fit the exposed held-out error.

## Functionality and resources

The first extraction smoke stopped before reading images because a local loop
variable shadowed the time module. R2fixed only that name and passed before full
extraction. A later local packaging receipt-path variable error occurred after
archive completion; source hashes, exact archive members/bytes and data identity
were independently verified before recovering the receipt. Neither failure
triggered a duplicate full GPU run or changed the scientific protocol.

- Extraction smoke1.329s; full21events10.757s.
- New feature smoke4.723s; full63patches/21triplets4.451s.
- All six classifier fits, exact old-model replay and screening14.418s.
- Ten focused matching/data/audit tests pass; exact checkpoint reload,
  daughter-order invariance and both-class/single-class handling smokes pass.
- Data backup1,033,223bytes; feature/model backup46,656,924bytes. Every remote
  file was copied locally and verified by SHA-256. No files deleted this turn.
- At06:56UTC AntelumeGPUmemory0MiB, no compute processes. Instance remains
  running/billing; no queued run. RSNA/shared environments/unknown disks untouched.

## Evidence

Extraction contractcb5105bf5dac9f2585ff2e585789983bdb11d3d04eed235470a80c2c2c1a124c.
Extra DATAe358a984eade85b96fbf2d77937ba0d57307f4813f44fb6c339346024d17df87.
Head contract15ee3493c0f5acb5ac2fd1e94007297dd59966435d077538f31b170272da5b78.
Merged feature result9c45e07f422f36f5fbaa9df55751e6d0542a1cb19eb37e5a2ebfaa70ee78b373.
Terminal89e24752a67b47e0ebca6ec89bd2fd23a65d63360e965f8c0c63fa6b575d421a.
See native-division-additive-v5-head-full-result.json and the data/head harvest
receipts for full model hashes and recovery locations.

## Next scientific question, not a running job

More isolated examples improved all probability estimates but did not solve
cross-embryo division discrimination. Do not run another data-radius or
threshold variation. Test actual pre/post-split image context at the same
image-derived parent/daughter locations, preserving split/label identities.
Begin with bounded extraction and a functionality/throughput smoke before
committing to trainable temporal-encoder adaptation. This is not temporal
reversal of the same pair, which the earlier architecture audit found redundant.

Historical accepted external contextual pretraining is not a ready competition
solution: its completed reciprocal transfer terminal reports both_folds_improved
false; larger multiscale pretraining also failed its audit. Reuse only verified
artifacts with explicit source provenance, never a previously rejected transfer
checkpoint as an admitted expert. The overall top-five goal remains active.
