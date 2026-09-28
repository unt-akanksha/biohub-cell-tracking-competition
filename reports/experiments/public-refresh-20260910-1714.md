# Bounded public source refresh — September 10, 17:14 UTC

Latest listing included skomuro/biohub-gpu-rerun-20260905 (16:53), newer than
the previous audit. Pulled notebook source and metadata only; no execution,
weights, output predictions or settings adopted. Known metric-exploit notebooks
were not pulled. Official repository remote HEAD remains
075fc5f5a52d11077f9dc2b074644618f26939e2.

Source: https://www.kaggle.com/code/skomuro/biohub-gpu-rerun-20260905

Local notebook SHA256:
cccd990bab090da42dae36a5971da30241d289eb2c65951b99c57bbf78af0400.
Cache: `.biohub/cache/public-audit-20260910/skomuro-1653`.

Static inspection, not a full transitive runtime/metric audit:

- Explicit receipt says `leaderboard_feedback_used_for_configuration: True`.
  Its quoted leaderboard scores do not establish independent clean superiority.
- Combines public dual-seed/harmonic linking, DeepCenter gap/division veto and
  postprocessing; loads packaged external source/weights. Dependencies have not
  been fully audited in this refresh, so no clean-runtime certification.
- New drift method forms a displacement histogram between candidate point
  clouds, refines the mode by median, and re-evaluates associations after
  subtracting the estimated shift. It takes a maximum with original logits.
  Candidate counts and spatial thresholds gate activation. This is a possible
  label-free motion hypothesis, not evidence of a gain over our measured flow.
- Notebook labels its local validator a proxy score. It is not a substitute for
  our pinned patched official scorer and complete-movie evaluation.

No claim that leaderboard tuning itself is a metric exploit. No submission
produced from this source. Rules and discussion landing pages were opened again
through the web reader; both returned zero readable body lines. No fresh full
rules/discussion review can be claimed from those responses.

At approximately 17:17 UTC the newest returned submission-history rows still
begin with submission55784044 on August26 (COMPLETE, blank score). No September10
entry was present in these latest returned rows. This is a bounded latest-row
check, not an exhaustive paginated audit or a score-selection signal.
