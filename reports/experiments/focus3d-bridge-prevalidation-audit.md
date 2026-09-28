# FOCUS bridge audit — 2026-09-09

The proposal kernel version 1 completed in 1,009.22 notebook seconds. Its
terminal SHA-256 is `288eab6b9e2c568f38587fa51c0af0c2dd140443df8a890a05f8cf2d290ee985`.
Logs report zero failed frames for all four 100-frame inputs. Current Kaggle
quota is 25.50 hours remaining. No new GPU job was launched during this audit.

All four downloaded GEFF directory hashes match the terminal manifest. The
proposal run has not evaluated ground truth. This establishes transport
integrity, not detection correctness or a score improvement.

## Findings before paired validation

- The source applies trajectory smoothing before saving proposal GEFFs. These
  are postprocessed FOCUS coordinates, not raw segmentation centroids. Raw
  centroids were not persisted by version 1. Do not describe these artifacts
  as raw detections or claim that their complementarity is established.
- The frozen gate forbids any increase in node-count penalty while the
  experiment adds a node. The inspected scorer uses
  `(num_pred_nodes - n_total) / n_total`; adding a node increases that term by
  `1 / n_total` for every positive finite `n_total`. This gate cannot admit
  an active bridge on validation movies. Preserve the original policy as a
  historical artifact; a documented replacement contract must precede label
  access. Require actual edge recovery and improvement in the full adjusted
  score, including the added-node cost, rather than requiring zero cost.
- The previous 948TTA2 validator implements its own component-based division
  proxy. The packaged scorer extracts local division-event graphs. Arithmetic
  recomputation of proxy CSVs does not prove official-scorer equivalence.
  The next evaluator must invoke pinned official `evaluate`,
  `per_sample_metrics`, and `summarise` directly on fresh graphs for each arm.
- The previous handoff said the paired evaluator was ready. Only the bridge
  function, policy, and proposal builder existed. Integration and full paired
  validation remain outstanding.
- Bridge implementation now rejects invalid graph structure and coordinates;
  its tie-break now follows the written policy (residual, then FOCUS node ID).
  Eight behavioral tests pass. Upper image bounds still require image shape
  checks in the evaluator. These repairs were made without opening labels.

The one-bridge cap is a feasibility experiment, not evidence of reaching the
0.945 target. Neither this audit nor the completed proposal run authorizes a
submission. Next work is a corrected preregistered paired evaluator with
official scoring, per-embryo results, and persisted control/candidate graphs.
