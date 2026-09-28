# Exact-GT dropout training: failed quality gate

Kaggle `indarkarhana/biohub-focus-gt-parent-dropout-training-v1/1` completed
all 800 planned updates. This is not a competition submission. Recovered all
nine checkpoints, runtime sources, result JSON and terminal launcher receipt.
Independent host verification completed successfully at 17:20 UTC (session26593
terminal). Executed runtime/spec, original controls, all 800 exact alternating
sample identities, all nine checkpoint hashes/reload receipts, frozen component
identity and the recomputed failed diagnostic gate were verified.
Host receipt SHA256:
e5995108f69215243783b102e1816666774f6065abc011fd2987e9ab4a7d3fb1.

| Fixed diagnostic | Original neural model | Physical control | Final trained model |
| --- | ---: | ---: | ---: |
| Unweighted NLL, lower better | 0.219009 | 0.333915 | 0.221171 |
| Correct parent, of 2,645 | 2,585 | 2,500 | 2,577 |
| Correct absent parent, of 27 | 12 | 18 | 11 |

All three frozen diagnostic gates fail. The more realistic exact-GT dropout
data did not translate into a better model in this fixed experiment. Do not
extend this failed configuration unchanged, choose an intermediate checkpoint
against these diagnostics, relax the gate, or submit the final model. No new
source/target movies were opened. These are training-domain diagnostics, not
independent full-movie tracking scores or predictions of leaderboard score.

Worker time: 122.092 seconds. Launcher time: 203.752 seconds (3.40 minutes),
within the 900-second declared cap. Peak allocated GPU memory: 164,237,312 bytes.
Thirty-five relevant local tests passed in 4.56 seconds. Kaggle reported 8.80h
remaining after completion, retaining the mandatory 8h reserve.

Result SHA256:
65dc6d1719a420919469732ae59c0daf2b238fa2aadaafab8a966c119fd730ad.
Final checkpoint SHA256:
ba0850b894843e46aae6378ba9876fd274b1734ff162842719c44204b7818230.
Final model tensor SHA256:
16985846df6d16c2fb2e0511d74cf3e8035cbac499da8f2a194efd6c3151436c.

No Biohub GPU worker remains active. Antelume was not used: a fresh AWS STS
identity request returned ExpiredToken for profile
148971207977_InventoryOptimization-EC2-Access. EC2 inventory returned
RequestExpired. Current remote GPU utilization cannot be established; RSNA and
the shared instance were untouched. Credentials must be refreshed before any
Antelume launch, then instance identity and shared GPU occupancy rechecked.

Today's request remains 0/5 new qualified submissions. More GPU availability
would enable new experiments but does not make this failed model submission-ready.
