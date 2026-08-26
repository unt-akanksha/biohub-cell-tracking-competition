# hoct-probe-finetune-v1

Status: staged independent association candidate; waits for the active V6 GPU
job and creates no submission.

The public 0.927 notebook reproduction remains only a frozen comparator. This
lane uses the official July 2026 Higher-Order Cell Tracking Transformer, whose
edge-centric attention is designed for the two structures that matter here:
competing parental links and cell divisions. The pinned MIT checkpoint has
6,252,593 parameters and produces 288-dimensional edge representations. It is
not part of any public Biohub notebook.

The backbone stays frozen. A 289-parameter linear head is initialized from the
official classifier and trained on all 195 non-validation Biohub lineage
graphs. Candidate windows retain every positive edge and a 3:1 mixture of hard
and uniform negatives. Missing mask-derived morphology is filled with the
official training means, making those channels neutral after the checkpoint's
own standardization instead of fabricating measurements.

Dense final-topology inference is spatially tiled with unique target-core
ownership and an 80-scaled-voxel context halo. The two selection movies choose
between the official head and the Biohub probe, HOCT-only and probability-aware
hybrid association, and a bounded threshold grid. Only after that choice is
frozen are the other two complete movies inferred and scored once. Promotion
requires a positive clean acceptance delta without a material worst-movie
regression. Leaderboard scores and metric pathologies are excluded from
selection.
