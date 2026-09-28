# Exact-cache offline acceptance staging

Update09:21UTC: launched once after the Antelume experiment terminated. Actual
Kaggle title-derived slug is `indarkarhana/biohub-exact-cache-runtime-acceptance/1`
(numeric134314478), verified RUNNING. No duplicate push to fix the requested slug.
Remote code cells and offline inputs exactly match staging. Fresh quota28.97h,
twoGPU-hours reserved, no pending submissions. See launch/remote receipts and
latest readiness handoff for terminal-output verification commands. Older
not-launched statements below describe the prelaunch staging checkpoint.

The A10 proof covers four complete movies and eight graphs (original and
trajectory-repaired output for each), not eight movies. Exact duplicate-view
encoder caching plus two overlapping FP32 workers reduced measured wall time
from 642.874 to 435.400 seconds (32.273%) with exact graph identity. Original
eight-view contributions, inverse transforms and accumulation order remain
unchanged. This is not the previously rejected D4 correction.

The new private, Internet-disabled two-T4 acceptance notebook is staged locally
only. Its derived contract is
`99c9fb48404e31051ce3096a0901b32698d19742abb3f04d5f4dc0453f9ec411`;
build receipt: `trajectory-overlap-cache-kaggle-v1-build.json`.
All base weights/wheels remain unchanged, and the cached predictor is verified
against the A10 source, allowing only newline normalization. Static tests check
embedded bytes, hashes, compilability, unchanged assets and offline metadata.
Three staging tests and five complete-movie scoring tests passed together.

No Kaggle kernel was pushed or submitted. Two-T4 runtime acceptance remains
unverified. Launch requires a fresh quota/pending-reservation audit, the eight-hour
reserve, and no overlapping GPU experiment. The staged acceptance cap is one hour
on two GPUs. The current Antelume dense-warp diagnostic remains undisturbed.

Scoring tests additionally reject partial runs before opening labels, reject
nonfinite or skipped official score rows, and verify micro-aggregation and the
official no-division behavior. They do not score current candidate predictions.

The dedicated launcher now checks the known Antelume run's terminal artifact and
actual empty GPU process list, rejects pending/unknown competition submission
statuses, and refreshes quota immediately before the capped launch. An attempted
launch writes an immutable receipt before pushing to prevent blind duplicates.
The dedicated verifier preserves the old verifier source and pins its hash,
retains all eight-movie/two-T4/four-worker checks, and rejects any changed graph
before reopening truth. Fourteen combined launcher/overlay/verifier/scoring tests
passed. Neither new script has launched a GPU job at this checkpoint.
