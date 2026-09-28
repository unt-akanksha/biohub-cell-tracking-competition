# Cached FOCUS detections with independently trained image motion

This is a distinct combination, not a repeat of the rejected FOCUS physical
or public-neural linkers. Existing project-generated raw detections are reused;
the 1.1B-parameter detector is not rerun. The motion model is the independently
trained and frozen flow3006, using native single-view flow. This probe does
not inherit any unvalidated motion-TTA transfer result.

## Frozen small scope

- Three frames of `6bba_23af9eeb`, a recorded fold0 training movie only.
- 285 cached raw detections, including fractional and boundary coordinates.
- Exact raw terminal SHA
  `0609934b1e2a40473763acf521cfcf7120e418f5857c24d6f28c0e662638bd41`.
- Exact raw movie NPZ SHA
  `7d62559975797b50f9390ba68d2c76be358cd9c303b74459d1d871570111c7b2`.
- Owned flow checkpoint SHA
  `3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788`;
  unchanged tensor SHA
  `e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779`.

No labels, public association weights, new target movies, test predictions or
submission. The original 400-frame raw cache stays unchanged; only the named
training-movie NPZ is opened by the small worker. No node count, threshold or
postprocessing tuning is introduced.

## Adapter and functionality conditions

Flow channels remain backward ZYX displacements in microns. Original raw
centroids remain exact; they are never rounded onto the association grid.
Strided reads sample0,4,...,252 in a256-pixel axis. For legitimate centroids
in the remaining trailing strip, the declared adapter extends the last flow
value through the original voxel-center bound255. It does not move or remove
the centroid. The number of extended samples is reported per frame.

The GPU test uses two T4 devices for the two frame pairs. Interior sampling
must agree with the existing Torch sampler within2e-5 microns. Zero motion
must reproduce the frozen static linker's topology. Both arms retain every
coordinate through a real GEFF round trip, with only next-frame edges, one
parent and at most two children. Native flow must be finite and nontrivial;
weights remain unchanged. Variance/null/posterior constants are inherited,
not fit to this diagnostic.

Host verification independently replays both edge sets from the recovered
motion NPZ and compares them with actual serialized graph endpoints. It
checks exact raw coordinate order, input/output hashes, sample coverage,
border counts, scope flags and the frozen source-bundle receipts before any
full-movie continuation. This is functionality, not accuracy validation.

## Current status

31 local checks pass, including the real285-node training cache round trip,
zero-flow graph parity, actual CLI import context, malformed topology and
tampered-receipt rejection. Synthetic-flow verifier tests are explicitly
fixtures; they are not substituted for GPU execution evidence.

Staged private/offline notebook `biohub-focus-owned-flow-probe-v1`, SHA
`36c761d95cb598f743d6ff6cd9172b228b2c37404ac3d5621329314bb2722abb`.
Version1 has now completed57.023s total launcher time and passed host replay.
Its verified report SHA is
`645a7613cf8fc69e5fc8127d06f6a3ae2e2994d97dddf85e5e402a412d85ad05`.
No labels were opened by the smoke. The separate full-movie result is in
focus-owned-flow-full-v1-result.json; it does not retroactively turn this
functionality probe into an accuracy experiment. Do not rebuild this staging.
