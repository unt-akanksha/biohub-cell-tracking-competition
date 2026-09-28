# Temporal division context v6: data and functionality protocol

Previous goal turn made progress by executing the additive-data experiment;
all six fits improved source NLL but neither visual pair passed cross-embryo
screening. The goal remains active, not blocked or achieved. Do not repeat
matching-radius, confidence, or failed-target filtering variants.

Preserve all3,108v5triplets,65optimization positives/2,305negatives, and the
unchanged7selection positives/731negatives. Infer each stored image patch's
parent/daughter role from its existing triple indices and reject ambiguous
identities. Sample one genuinely new frame at the identical image-derived
physical coordinate: parent at t-1, daughters at t+2. Combined with existing
parent t/daughter t+1images, this spans four adjacent frames. No future GT
position, reassociation, new label or optical-flow-derived center is used.

Boundary observations outside0..99are zero with an explicit validity mask;
retain the original example rather than fabricate an image or drop its label.
Existing original patches/coordinates/triples/GT identities/labels must remain
byte-equal. Use the same native normalization and physical3scale15cubed crops.

Freeze exact source/selection movie roles, packet hashes and archive members.
First smoke fixed positive and negative optimization packets from each embryo;
then full data only if geometry/label invariance, RAM guard and projected
throughput pass. Cap full extraction at1,200seconds/512MiB output, preserve
3GiBsystem available RAM, run GPU sequentially, and leave foreign jobs untouched.
All derived output is backed up locally before removal; no raw-movie cache.

Before a larger model run: verify daughter-order invariance, boundary masks,
finite forward/backward, exact saved reload and representative step timing.
Only source-trained initialization is allowed for reciprocal embryo claims.
The eventual temporal model must be compared with a no-extra-context control
under a frozen training/calibration protocol, then meet the same full-movie
patched scoring/runtime requirements. This document authorizes data extraction
and functionality testing, not automatic submission or an unbounded training job.
