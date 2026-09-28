# Training-only missing-parent augmentation audit

This is a data/label feasibility audit, not training or submission. Use only the
12 verified fitting movies and their cached raw-detection feature packets.
Never alter original packets, diagnostic packets or inference node inventories.

For each fitting pair with a known parent, choose one unique supervised parent
uniformly via a deterministic SHA-256 of the fixed seed 244691, stem and frame.
Remove source candidates within 14 microns of that predicted parent, inclusive.
This is twice the official 7-micron matching distance: the selected matched
parent is within 7 microns of its GT parent, so every retained candidate is
strictly more than 7 microns from that GT parent by the triangle inequality.

Reindex retained known-parent labels. Original known-null labels stay null.
For another removed known parent, assign null only if ALL retained candidates
are farther than 14 microns from that parent's original predicted position;
otherwise mark that target unknown. Original unknown targets remain unknown.
Division daughters sharing a parent follow the same transformation. Retain all
target arrays and features exactly. Reject non-fitting roles before reading data.
Pairs losing every source are omitted from training eligibility, not truncated.

Audit every fitting pair, save decisions/hashes and total supervised counts,
and check no diagnostics entered. Require at least 100 nonempty-source augmented
pairs and at least 100 additional verified synthetic null labels to justify a
small real training smoke test. These are feasibility counts, not quality gates.

If feasible, a separately frozen training experiment may use a fixed 50/50
original/augmented fitting schedule, retaining the original unaugmented diagnostic
gate. The current audit does not authorize a long run. Synthetic occlusion in
candidate space may not match real missed detections; only unchanged real-data
validation can establish usefulness. No source/target-label access or GPU use.
