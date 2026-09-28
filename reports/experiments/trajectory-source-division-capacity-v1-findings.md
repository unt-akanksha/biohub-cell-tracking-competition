# Next accuracy gap: source division capacity, not another weak-model ensemble

The frozen structured submission candidate remains unchanged. This new audit
opened only the eight already-designated source optimization movies. All source
prediction and truth files passed their recorded hashes; patched official
baseline counts reproduced exactly. No selection/validation movie was opened,
no model fitted, and no oracle prediction or changed graph exported.

Among seven annotated source forks, four have all three cells physically
matched in the final predictor graph and three lack at least one matched cell.
All four matched forks are absent from the final lineage, yet both daughter
links occur in our broader joint candidate sets. Only one of the seven has
both links in the raw neural candidate edge list. The official source score
records zero true divisions, two false divisions, and seven missed divisions.
Exact-frame coverage is not the scorer's plus/minus-one-frame event recall.

This identifies a candidate-graph capacity limitation as well as a data shortage.
The eight movies supply only four immediately usable fork examples. Training a
large division ensemble on those four would not resolve the generalization
failures already observed in native v3/v5/v6. Do not repurpose exposed selection
events as training data or loosen thresholds based on their errors.

A distinct next experiment would train event-structured decisions on the actual
public predictor candidate distribution, rather than isolated peak triplets.
Before any GPU training, expand source-only event coverage from the existing
optimization-role inventory, preserve ordinary nondivision contexts, and test
source-only feasible event constraints. Use a small image/prediction collection
smoke before a bounded larger collection. The current eight-movie sample alone
does not justify that larger training launch yet.

Research context: Hirsch et al. combine a learned division detector with graph
optimization and structured-SVM weight fitting in whole-embryo tracking. This
supports testing event-level structured learning, not a claim of Biohub transfer
or permission to import unverified weights. [Primary paper](https://arxiv.org/abs/2208.11467).

The refreshed [dim-node discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737896)
raises a participant's annotation-quality question; replies speculate about
interpolation. It is not organizer confirmation that annotations are invalid,
and provides no basis to remove difficult labels or modify ground-truth geometry.
No public predictions, unknown-license checkpoint or metric-hack notebook was
downloaded in this refresh. No newly verified clean public-best model was found.

Audit runtime: 24.312 seconds, CPU only. Machine-readable evidence:
`trajectory-source-division-capacity-v1.json`.
