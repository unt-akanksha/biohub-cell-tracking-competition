# Expanded-data nonlinear presence model

Prospective CPU-only follow-up after the twelve-movie linear model failed
missing-parent correctness (7/27). Refit the EXACT previously declared shallow
boosted-tree method on the twelve fitting movies; this is not extension of its
failed four-movie checkpoint or a sweep. Original eight input features,
100trees,depth2,learning rate0.05,min leaf20,subsample1,seed244691,log loss,
class-prior initializer,no class weighting,no early stopping. Same pinned
sklearn1.9.0 worker and portable float32-input/double-threshold evaluator.

Use the verified twelve fitting summaries (10754knownparent/161knownabsent)
and identical old diagnostic summaries (2645/27). No new labels, unknown
negatives, raw node changes, model/head changes or additional GPU extraction.
The worker receives only a hash-bound fitting file. Persist the full portable
tree model and verify native/export/reload before corrected diagnostic use.

Same original feasibility gates: NLL strictly below both controls, correct
parent>=2585, correct absent>=18. No posthoc changes if it fails. A pass would
only permit complete-source tracking evaluation with the existing graph and
promotion requirements, not immediate submission. Original diagnostic movies
were used by the pretrained owned encoder, so these are training-domain
feasibility results, not independent validation or leaderboard scores.

No GPU, cloud mutation, source/target evaluation or submission in this fit.
The prior linear and four-movie tree results remain failed and unchanged.
