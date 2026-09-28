# Exact-GT missing-parent augmentation: fitting-only audit

The completed difficulty audit found the old 14um predicted-parent dropout
produces1135/1136 physically easy null decisions, versus115/161 natural nulls.
Do not repeat that training unchanged. Replace the conservative triangle-bound
neighborhood with the exact known GT parent's7um matching neighborhood.

Use only12 frozen fitting movies. Recompute official node matching and require
every original cached parent/null/unknown label to replay exactly. For each pair,
retain the previous seed/stem/frame-selected supervised parent identity. Delete
source candidates at distance<=7um from that parent's exact annotated location.
For each other removed supervised parent, assign null only if every retained
candidate is farther than7um from that parent's exact GT location; otherwise
ignore that target. Reindex retained labels, retain natural nulls, and never turn
original unknown targets into negatives. Source-candidate changes are for training
augmentation only; original packets, all target arrays and inference stay intact.

Persist exact GT-parent coordinates, original packet hashes and every removal/
label decision. Prove matched parent coordinates are within7um, selected labels
become null, and all newly null targets have no remaining candidate within7um of
their GT parent. Preserve two daughter targets sharing one annotated parent.

Compare fitting-only physical-null difficulty with the frozen14um augmentation:
require more than100 eligible pairs, strictly more wrong-parent argmax cases and
strictly higher mean null NLL. This is a data-feasibility gate only, not a model
quality claim. No radius/threshold selection:7um is fixed by the label semantics.
Only a pass justifies a small separately frozen training smoke. No GPU, diagnostic
or source-selection access, target expansion, checkpoint choice or submission.
