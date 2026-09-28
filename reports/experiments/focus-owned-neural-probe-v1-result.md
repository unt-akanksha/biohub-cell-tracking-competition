# Owned learned linker on raw FOCUS nodes: rejected training probe

Kaggle `indarkarhana/biohub-focus-owned-neural-probe-v1/1` completed in
62.929s. Fresh quota before launch11.24h, declared one-hour cap, reserve8h;
after completion11.22h. No other Biohub GPU job overlaps. Remote metadata
verified private, offline, GPU enabled, exact version1 and owned checkpoint
input. No public linker checkpoint was loaded.

Notebook SHA256:9f52f7ba5a46b9ea5f75a36668d86e0453fc4119781448d29e759a36e7e7fe11.
Checkpoint SHA256:76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144.
Nine focused link-policy and decision-gate tests passed. Strict model load,
repeat neural-head inference, finite real logits and unchanged tensor hash
passed. Actual saved repository/runtime bytes were checked against the frozen
notebook. All1,070 raw detections and cached backward flows were preserved.
Host replay recovered exact edge identities and probabilities within1e-12
relative/1e-14 absolute numerical tolerance. Zero-neural control exactly
reproduced prior physical-flow edge identities. All predictions were verified
before fresh GT evaluation using the pinned patched official scorer.

| First-three-frame training scope | Control TP/FP/FN | Neural TP/FP/FN |
| --- | --- | --- |
| 6bba_f1fde7e0,785 nodes | 7 / 3 / 4 | 7 / 6 / 4 |
| 6bba_23af9eeb,285 nodes | 12 / 0 / 0 | 12 / 1 / 0 |

Pooled raw edge Jaccard0.7307692308 ->0.6333333333. Correct ordinary links
remain19; no true divisions are present in this tiny scope. False divisions
increase0->2. No correct-link recovery. Feasibility gate FAILS; no full
eight-movie inference of this configuration and no submission.

Candidate adds the neural score to the original physical prior at coefficient1,
matching the original owned training objective. It used official integer-floor
feature sampling with boundary feature-value extension, exact native geometry
and fractional downsampled positional features. This is not the public
integer-rounded FOCUS adapter or its public weight/postprocessing configuration.
The old public diagnostic score therefore does not predict this result.

Next hypothesis to investigate, not a staged run: the current head was trained
on annotated-node inputs rather than these dense predicted-node sets. Adapt
training to predicted FOCUS nodes, with annotated target supervision and
explicitly justified missing-parent nulls, while ignoring unknown targets.
Existing `annotated_missing_parent` and owned training contracts should be
reviewed before implementation. This probe does not establish the mismatch as
the cause; do not claim a fix or sweep the mixing coefficient on these results.

Current state: no Biohub jobs active or queued. Best development candidate
remains residual-calibrated FOCUS-flow0.7805420622, still not promoted. No new
source-eight/target access by this probe. Training score is not complete-movie
or independent validation. FOCUS pretraining overlap remains unresolved.

Authoritative result:`focus-owned-neural-probe-v1-result.json`, SHA256:
7a0a471de64e1de38c2e40f5c8f50c75f208be1093bb08db77aa0e9d89aa71cd.
Design and executed sources remain frozen separately.

Subsequent correction: the preceding 'annotated-node inputs' explanation was
incorrect. Inspection of the frozen training notebook shows native predicted
detections already feed the head, with GT used for matching/supervision.
Only FOCUS-specific proposal adaptation is new. See
`focus-predicted-node-labels-v1-result.md`. The measured failed probe result
and its frozen JSON/source/design hashes are unchanged.
