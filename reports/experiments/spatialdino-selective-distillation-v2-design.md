# SpatialDINO selective-distillation v2 design

Status: local design only. Do not launch while
`spatialdino-pu-adaptation-v1` is active, and do not use unless v1 evidence
shows that sparse hard consensus is the limiting factor.

The active v1 learns only from high-threshold teacher consensus, organizer
positives, safe background, and weak/strong consistency. A possible v2 adds a
small soft loss over lower-confidence structure supported by the two teacher
seeds. It averages their probabilities, weights voxels by geometric joint
confidence, and quadratically discounts seed disagreement. Voxels outside
teacher support remain absent from this loss.

This is knowledge distillation into the independent 29.5M SpatialDINO/UNETR
architecture, not copied prediction output. The v2 loss should remain a
minority term (initial design weight 0.25) so organizer positives, PU masking,
and the microscopy foundation model can differ from the public teachers.

Launch decision after v1:

- If v1 has reasonable peak density but low clean recall, integrate and test
  selective distillation.
- If v1 collapses, first diagnose optimization and peak calibration rather than
  assuming more teacher supervision is sufficient.
- If v1 passes untouched acceptance, do not spend GPU on v2; advance the clean
  learned checkpoint toward a separately guarded two-GPU submission candidate.
