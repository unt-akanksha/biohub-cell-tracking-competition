# Complete cached-detection motion diagnostic

The real small probe completed in57.023 total launcher seconds. Host replay
verified all285 exact centroids and both174-edge graphs; the latter equal
counts do not imply equal edge identities or an accuracy gain. Sampling
parity error was1.2517e-6 microns. Frozen probe report SHA
`645a7613cf8fc69e5fc8127d06f6a3ae2e2994d97dddf85e5e402a412d85ad05`.

## Fixed comparison before any new scoring

Four complete100-frame movies in original raw-cache order:
44b6_81c256f0,44b6_24264f12,6bba_f1fde7e0,6bba_23af9eeb.
All four were previously exposed during the older FOCUS diagnostic program;
the6bba pair is in the motion training role. Neither44b6 movie belongs to the
remaining65 closed audit movies. This is not fresh independent confirmation.

Both arms preserve every original raw FOCUS centroid, without rounding,
smoothing, deletion or graph pruning. Control is the frozen static physical
linker; candidate substitutes the already-trained native single-view image
flow. Both use unchanged variance, null-4.5, posterior>.5, maximum1 parent/
2 children and next-frame-only links. No flowD4, public association weight or
large detector inference. The exact small-probe motion cache must replay on
the same first three training frames. Both complete graph arms and motion
caches are saved and hashed before any GT scoring.

Use complete-movie patched-official micro-aggregation, per-embryo reporting,
and worst-movie analysis. The predeclared diagnostic conditions are: positive
aggregate score and raw-edge Jaccard gains over the static control; unchanged
node counts and detection recall; at least3/4 adjusted-edge movie gains; no
per-embryo score loss; each movie loss<=0.02; worst-movie loss<=0.01.
No post-result radius, threshold, flow coefficient or pruning sweep.

Historical public-control scores on these already-exposed movies are
contextual diagnostics only: their association model trained on those movies.
They are not leaderboard values or independent holdouts. A pass here cannot
authorize submission, establish superiority to public bests, or open the
remaining65 target movies.

## Execution

34 relevant checks pass. GPU notebook `biohub-focus-owned-flow-full-v1` is
staged with SHA
`5af97d02de71448af3d4f2367db307a77212fb59ae0d1389f0a6beffb7e3db9f`.
The verified original probe's runtime helpers are copied from its frozen
notebook, not rebuilt. The raw cache, flow fit and probe kernels are pinned
inputs.34 checks cover real cache/graph replay, source binding, CLI import,
geometry and malformed provenance. Internet/TPU off; two T4s,1h declared cap
with earlier watchdog; fresh quota and previous GPU completion required.

## Completed outcome

GPU version1 accepted with fresh12.11h quota and completed in144.589 total
launcher seconds. All400 frames/396 pairs and65,091 raw nodes were preserved;
the original three-frame GPU motion replay was exact. Local CPU harvest/
scoring controller PID29036 completed09:16:58UTC. All graph edges and cached
motion were verified/replayed before GT scoring. No GPU job remains active.

The diagnostic gate PASSED: score0.7857341462 ->0.8103292365;
raw-edgeJ0.8087621697 ->0.8339076499; recall unchanged0.9913392817. All four
adjusted-edge movie scores improved.44b6 score gain0.0028020270;6bba gain
0.0341741092; worst-movie gain0.0638039731. Counts1163TP75FP200FN ->
1210TP88FP153FN. False divisions11->22; this diagnostic has no annotated
divisions, so it does not establish division recall or reliable fork quality.

Report `focus-owned-flow-full-v1-result.json`, SHA
`079abeeccec5f7a76e27b40c88063fae30bca4a155f3dae3bf3bc0311e4f6685`.

The historical raw public-neural linker scored0.8802090945 on these same
movies, and its public-base control scored0.9440079104. Both use association
weights trained on these movies. The new0.810329 result is NOT stronger than
those historical diagnostics and is not a leaderboard score. The largest
gain comes from a flow-training movie. Do not promote or claim public-best
performance from this result; no new target movie or submission was made.

Next useful evidence would be a direct fixed source-selection comparison of
this detector/owned-flow combination against our independent parentD4. A new
FOCUS detector cache first needs an exact model-weight identity check; the
old raw-cache receipts bind centroid NPZ files, not the upstream weight bytes.
Do not mistake those NPZ hashes for checkpoint provenance. Preserve all
existing stages and do not repeat this unchanged four-movie experiment.
