# Learned fork16 worker packaging: draft only

The existing submitted fixed8worker calls event.refine without explicit stage
limits, inheriting2seconds/frame and120seconds/movie. That is NOT a bug in the
existing fixed8submission, which was verified with those defaults. However,
blindly swapping in learned fork16weights would mismatch their frozen local
evaluation policy of10seconds/frame and600seconds/movie and could truncate it.

The new local draft explicitly supplies10/600 and checks max_fork_children=16.
Exactly four overlay files differ: worker, event module, event model JSON,
NOTICE. Upstream image models, sharding, structured repair and dataset remain
unchanged. No notebook or kernel metadata was created, no GPU launched, no
pending submission changed, and no quality or runtime eligibility is implied.
Global runtime projection/platform acceptance/fresh quota checks remain required.

Draft .biohub/staging/trajectory-event-fork16-worker-44b6-v1:
receiptSHA`c061edfde8e22153ed36f38f1d843e85e68aae46dc7a71da168dee0bf77ba2b5`;
overlaySHA`4e1bebb0c1f27b626fc4b3cb0c26e86e9c08d7eef195c646476c8d7091cbaa2b`.
Model values equal the previously verified final-source44model exactly. Its
JSON line endings normalize when embedded, so JSON bytes/hash differ from the
source file; the source and embedded model values were explicitly compared.
Actual fitted weights remain17f60597888c426aba77992d50bd6dfecd1547c3f4356d4bdd2bf1755c731f01.

Six adapter tests pass: only intended worker edits, actual AST call passes10/600,
wrong/missing vocabulary rejected and double adaptation rejected. Real movie
worker-call smoke60326 is TERMINAL success:6bba_55b7eebe,99transitions,8.078s
total/4.813s inference, exact graph and solver records, no fallback/cap or GT.
Smoke receiptSHA`6cf9ce113eac09e1235a75c5b6a88e68f50ae1ce05de53e42323edec569b5391`.
This executes the actual worker's event-call AST with real frozen upstream
inputs, not the full GPU image worker. It is not Kaggle acceptance.
