# Multiscale division localization v1 — frozen design

Status: design frozen; no new external validation window has been extracted,
no GPU job has been launched, and no Biohub label has been scored for this
candidate.

## Why this lane exists

The refreshed public discussion suggests that division recovery can fail even
when ordinary node matching is strong because the local division neighborhood
is spatially less accurate. Public thresholds and leaderboard configurations
are excluded. The project-authored response is a learned physical recentering
task whose output cannot change association topology.

The label-free processed-graph audit found 218 predicted division parents in
97,971 nodes. An independent coordinate donor covered 217 complete events, so
only 651 nodes (`0.0066448235`) were eligible to move. Median candidate movement
was `0.9051 µm`, p90 `2.2262 µm`, and maximum `4.4276 µm`. This audit read no
truth and performed no scoring. Its immutable evidence is
`artifacts/profiles/division-localization-scope-v1.json`.

## Model

The localization model copies the complete 46,386,607-parameter v4 3D+axial
backbone into a separate network and adds a zero-initialized physical-offset
adapter. Total size is 47,964,082 parameters. The association checkpoint and
its graph predictions remain unchanged.

The existing external shards contain 17-cubed temporal patches sampled on an
isotropic grid from -8 to +8 µm. Training takes padding-free 13-cubed windows
whose centers are artificially displaced by at most two grid steps and learns
the inverse physical offset. The adapter is bounded to ±2 µm per axis. These
limits follow exactly from the available padding-free crop geometry; no public
distance value or leaderboard experiment selected them.

## Frozen evidence protocol

- Optimization uses the existing ZSNS004 training shards only.
- Checkpoint selection uses new ZSNS005 timepoints 156–159 and 456–459.
- One-shot audit uses new ZSNS005 timepoints 276–279 and 556–559.
- These 16 timepoints are disjoint from all v3/v4 ZSNS005 selection and audit
  timepoints. The lists were frozen before extraction.
- Selection and audit compare physical residual error with zero correction.
  Mean and p90 error must improve and every axis MAE must be non-regressive.
- The audit cannot redirect checkpoint selection.

After external gates, one frozen checkpoint may donate coordinates only to a
complete predicted division parent/daughter triplet. Events missing any donor
node, every non-division node, all IDs/times/counts, and all edges remain exact.
Submission-space integer projection and the exact processed control are required
before any Biohub comparison.

The four prior coordinate-acceptance movies are already open. They cannot
select a checkpoint, correction scale, bound, or graph scope, and their result
alone can never authorize promotion. V1 remains downstream of an accepted v4
terminal so it cannot delay the armed v3 → v4 submission chain.

## Staging result

The frozen external supplement was built and published privately as
`indarkarhana/biohub-division-localization-shards-v1`, version 1. Its 16 shards
occupy 69,010,862 bytes locally. Selection contains 504 source nodes, 768 target
nodes, and 57 division sources; audit contains 502 source nodes, 768 target
nodes, and 54 division sources. The manifest SHA-256 is
`e1f6eb8c6148c092f72e5eddc81d75f17b16b6f37ad61e35a9d6e652bfb81336`.
The private Kaggle artifact was downloaded again and passed the fail-closed
array, provenance, timepoint, and inventory verifier.

The offline runtime was published privately as
`indarkarhana/biohub-division-localization-runtime-v1`, version 1. Its manifest
SHA-256 is
`10a5c1bc27a9cabbae0f018c05f43165b6d93008abaa864fbbc5f11101090caa`;
the redownloaded eight-file runtime passed its hash verifier.

The staged kernel is
`indarkarhana/biohub-multiscale-division-localization-v1`. It requires exactly
two T4s, has internet/TPU disabled, attaches no competition input, and contains
no submission path. Two isolated workers train one model per accepted v4 fold.
Each selection evaluation exhaustively tests all 124 nonzero integer offsets
inside the padding-free `[-2,2]^3` cube. The one-shot audit inventory is not
addressed by a worker until its selected checkpoint has been serialized and
hash-frozen. The notebook is intentionally not pushed until the accepted v4
parent exists, so this lane cannot compete with the armed v3/v4 submission
sequence for Kaggle GPUs.
