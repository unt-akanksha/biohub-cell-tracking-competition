# Image-context pilot v2: source-split feasibility correction

September 13, 2026, before any optimizer step or model prediction for this family.
Retain the v1 image feature function, architecture, external warm starts, seeds,
2,000-step two-model pilot, small GPU smoke, resource limits and quality gates.

V1 data preparation exposed a mathematical feasibility failure: its source-44b6
selection packet contains only ONE eligible positive, while the unchanged gate
requires at least TWO. No model can pass it. V1 data/design remain immutable and
will not launch training; this is a sample-count finding, not a model-score result.

V2 uses the already existing ORIGINAL relational-data movie roles instead of
the later v3 roles. Source-44b6 original optimization has 543 rows, 19 positives,
5 eligible positives and 3 eligible negatives; selection has 53 rows, 2 eligible
positives and 2 eligible negatives. Source-6bba original optimization has 1,539
rows, 67 positives, 23 eligible positives and 80 eligible negatives; selection
has 345 rows, 13 eligible positives and 39 eligible negatives. Original audit
shards remain excluded. This reuses an established, event-bearing source split;
there is no random split search or gate reduction.

For each model, only ONE source embryo supplies all optimization, checkpoint
selection and threshold calibration. Its target embryo supplies none. The two
source datasets are loaded in separate worker stages. The source-44b6 model is
evaluated only on held-out 6bba, and vice versa, after BOTH source model/threshold
artifacts are fixed. The prior v3 audit is historically exposed and is NOT
represented as sealed in v2; some of those source-only movies fall in the
original training roles. Component-level embryo exclusion, not a pristine
project-wide audit, is the claim to verify.

No existing Biohub-trained weights may initialize these models. Exact external
warm starts remain a0a1794134893d191d896b8b83abee61a754bc3852e2b8f74dacd353ce118a76
and 9e8af9aeb247297d3ed09bda3bcc2b5af413d78c6bf2546eff5f4898667e074d.
All old rejection gates and decisions remain unchanged. V2 does not open the
original audit, competition test, public predictions, or leaderboard data.

Before pilot launch require a successful actual 20-step model/optimizer smoke,
strict checkpoint reload, finite gradients/logits and measured 4,000-step runtime
with margin under a 40-minute bound. Keep a rolling atomic optimizer/EMA/RNG
checkpoint and best source-selected model. Never touch RSNA or stop the instance.
Only explicitly generated smoke-roundtrip temporary files may be removed after
verification to preserve disk headroom; their result/hash receipt must remain.

The eligible AP>=0.55 and >=2 source TPs at zero FP gates remain. Held-out
embryo AP>=0.55, >=1 TP each, >=3 pooled TP and <=1 pooled FP at the unchanged
source thresholds remain. This small event count is an important limitation.
Full predicted-candidate, complete-movie patched-official, worst-movie and offline
runtime/non-replica checks still precede any submission. No promotion from a
feature dataset, smoke, patch AP, or historical training diagnostic alone.
