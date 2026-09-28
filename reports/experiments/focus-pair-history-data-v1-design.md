# Complete history-head fitting array preparation

Prerequisites: completed history data audit79910268..., real15D optimizer
smoke5e3e9be9..., original12movie candidate data and all unchanged upstream
source hashes. This stage only serializes the already-defined15D features; no
new model/hyperparameter decision, optimizer, diagnostic/source/target evaluation
or promotion. Execute after the current tree fitting process is terminal to
avoid competing disk/memory loads during its remaining folds.

Only original12 fitting stems and99 transitions each. For every original known
target retain every real source and null; unknown targets never become negative
labels. Independently calculate all seven history features using a separate
per-target formula and node-ID lookup, then compare against the actual packer.
Compare every original8D feature, offset, group boundary, target ID, source frame
and label against the hash-verified original packed arrays, not just totals.

Save uncompressed NPZ arrays per movie in the ignored cache, with exact reload,
SHA256 and manifest/base identities. Expect10,915groups,4,691,320choices,
10,754parents/161absent; roughly600MB numeric arrays, check>=2GiB free disk.
Stream one movie at a time, preserve old artifacts, never overwrite any attempt.
600second direct-worker timeout,2CPUthreads, no GPU/cloud/RSNA changes.
This stage is not a quality screen. A larger training protocol/profile and
independent model evaluation remain required before any candidate promotion.
