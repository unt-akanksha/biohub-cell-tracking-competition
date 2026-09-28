# Exact-cache T4 acceptance: passed against the paired T4 control

Kaggle `indarkarhana/biohub-exact-cache-runtime-acceptance/1` completed. The
initial v1 verifier rejected comparison with the old A10 outputs. This was
not ignored: the previously accepted uncached T4 run already differs from A10
on44b6_267148e4, and its original acceptance fully rescored that difference.
The cached run is exactly identical to that accepted T4 control on all16graphs
(eight complete movies, original and repaired output each), including nodes,
edge identities and edge order. No graph difference was excused or rescored.

V2 corrects the reference platform, pins the accepted T4 report and every
reference graph hash, requires matching Torch versions, and retains strict
smoke/two-T4/four-worker/eight-movie validation. It records the initial A10
reference failure. Original verifier/source/artifacts remain unchanged.
Result: `trajectory-overlap-cache-kaggle-v2-result.json`, SHA-256
`2934de829c237906f214f7c6e4583b961b6e59aeb73c8a8adde05c52d69a1ede`.

Eight-movie wall time:1,099.5456seconds uncached overlap to1,016.8864seconds
cached overlap, **7.5176% faster**. This is the measured T4 result, not the
earlier32.27%A10 combined scheduling/cache result. The simple199movie cohort
projection is7.03hours; hidden movie complexity and imbalance remain unknown.
No hidden-runtime guarantee, new quality score, production update or duplicate
competition submission is claimed. The existing public0.946candidate is unchanged.

Full predictions, runtime sources and smoke outputs are recovered under
`.biohub/cache/trajectory-overlap-cache-kaggle-v1-output`. Two over-broad local
CLI downloads were cancelled after PID/command verification; neither interrupted
Kaggle or Antelume. Targeted recovery completed with page-size200 and the smoke
prefix. A partial local dependency tree is harmless and not used for validation.
Fresh quota after the completed test:28.65GPUhours remaining, no pending-test
reservation needed. The eight-hour reserve is intact.
