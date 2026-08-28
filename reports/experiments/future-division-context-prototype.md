# Future division context prototype

Date: 2026-08-27

Status: isolated project-authored research component implemented and locally
tested. It is not integrated into contextual v3 or multiscale v4, is not in a
published runtime, and has not been trained, selected, launched, or submitted.

Current public discussion independently reinforces a failure mode already
visible in our architecture audit: a post-hoc division threshold can propose a
second child without checking whether both daughters have distinct plausible
continuations. The sampled public notebooks use closely related hand-written
rules and leaderboard-touched constants, so none of their code or values is
admissible.

`future_division_context.py` instead implements a threshold-free physical
feature extractor. Given two proposed daughters at `t+1`, their learned
association scores to `t+2`, and an eligibility mask, it:

- solves one global two-row injective continuation assignment, so daughters
  cannot claim the same future node;
- measures current and future daughter separation and normalized gain;
- records the minimum/mean learned continuation score;
- records the selected-versus-swapped assignment margin; and
- records directional consistency of the daughter-separation vector.

The resulting eight-feature vector makes no accept/reject decision. Any future
reranker weights or thresholds must be learned only on reciprocal training
movies, frozen on reserved full-movie calibration, and pass untouched exact
acceptance. The component cannot alter or delay an accepted v3 candidate.

Evidence:

- Source SHA-256:
  `a775f5ee340eeb87be201abf09083c33737b58f15b5b150f7eab538a38746734`
- Test SHA-256:
  `7085b925aa337bd1a87fef55cb33a6b9fc6e2263dc25318edb788418dd35c91a`
- Five tests pass for outward continuation, daughter-order invariance,
  convergence evidence, unavailable distinct continuations, and non-finite
  input rejection.
- GPU used: no
- Public code or predictions copied: no
- Leaderboard used for selection: no
- Submission command: absent

Promotion is intentionally deferred. The next scientific step, only after the
active v3 gate, is to add a small sibling-pair head to the high-capacity lane,
train it with real division pairs and hard crowded-frame negatives, and compare
the complete graph against the same zero-feature control under official
full-movie scoring.
