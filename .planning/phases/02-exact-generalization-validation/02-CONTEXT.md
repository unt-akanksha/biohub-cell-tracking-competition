# Phase 2: Exact Generalization Validation - Context

**Gathered:** 2026-08-23
**Status:** Ready for planning
**Mode:** Autonomous smart discuss (`--auto`); recommendations accepted under the user's full-build authorization

<domain>
## Phase Boundary

Deliver the authoritative offline decision system for Biohub candidates: a pinned organizer scorer, immutable overlap-safe embryo folds, complete-movie evaluation, diagnostic decomposition, and an executable promotion policy. This phase does not train a new model or use Kaggle GPU time.

</domain>

<decisions>
## Implementation Decisions

### Scorer Fidelity and Provenance
- Pin an exact commit of `royerlab/kaggle-cell-tracking-competition`; record the upstream URL, commit SHA, source hashes, license, and retrieval timestamp.
- Reuse the organizer implementation behind a thin local adapter. Any pure-Python aggregation helper is secondary and must agree with the organizer scorer on regression fixtures.
- Treat the post-patch local division-window topology, 7-micron timepoint-aware node assignment, adjusted node-count penalty, and complete-split micro aggregation as authoritative.
- Add adversarial fixtures for the former fake-division exploit, negative/sentinel time or coordinates, fork reuse, merged daughter branches, coordinate-axis/voxel-scale drift, and per-movie averaging drift.

### Split Identity and Leakage Control
- Freeze two reciprocal leave-one-embryo-out folds: train/calibrate on `44b6`, evaluate all complete `6bba` movies; then reverse the direction.
- Derive sample identity from official metadata and paths, fail closed on missing or conflicting embryo/source identity, and never fall back to a random movie split.
- Store canonical sorted manifests with data hashes, source-group labels, train/evaluation membership, and an overlap audit; hash the manifest into every evaluation and experiment event.
- Keep complete movies intact. Calibration, threshold selection, and early stopping occur only on the training side of a fold.

### Exact Report and Diagnostics
- Produce pooled, per-embryo, per-fold, and per-movie adjusted-edge, raw-edge, division, node, and final scores with underlying TP/FP/FN and node-count penalty terms.
- Add endpoint availability, oracle-link, conditional-link, displacement-bin, density-bin, division-window, and worst-movie diagnostics so detection and association ceilings are separable.
- Compare candidates against a named baseline on identical manifests and emit deltas plus deterministic movie bootstrap intervals; public leaderboard scores stay in a separate non-authoritative field.
- Validate GEFF directly and after GEFF-to-CSV-to-GEFF round trips; reject missing movies, non-finite values, invalid time edges, malformed forks, or schema drift.

### Promotion Policy
- Default `promote` requires a positive pooled exact delta, nonnegative bilateral embryo evidence within configured tolerance, no material worst-movie collapse, stable node recall, and no material division regression.
- Marginal or intentionally traded-off results become `review_required`, never automatic promotion; the exact thresholds, inputs, evidence hashes, and approved exception must be appended to the ledger.
- Any metric exploit signature, incomplete movie coverage, manifest mismatch, scorer/source mismatch, or non-finite metric yields deterministic rejection.
- Public score can corroborate a promoted candidate after submission but cannot cause promotion.

### the Agent's Discretion
- Choose internal dataclasses, CLI names, report serialization details, bootstrap seed/count, and lightweight dependency boundaries while preserving exact scorer fidelity and reproducibility.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/biohub_tracker/io.py` already provides canonical JSON, atomic/immutable publishing, path containment, and SHA-256 helpers.
- `src/biohub_tracker/ledger.py` already validates complete metric payloads and appends immutable experiment events.
- `src/biohub_tracker/cli.py` and `__main__.py` provide the command surface to extend with manifest, evaluate, and promote operations.
- Phase 1 fixtures and pytest conventions provide deterministic offline test patterns.

### Established Patterns
- External state is explicit (`--live`) and uncertainty fails closed.
- Compact JSON is authoritative; Markdown is a deterministic projection.
- Numeric evidence is normalized rather than stored as binary floating-point artifacts in ledger decisions.
- Generated biological data, GEFF files, model weights, and downloaded assets remain outside Git; small manifests, policies, fixtures, and reports are tracked.

### Integration Points
- Exact reports must satisfy the metric schema already enforced by `validate_complete_metrics`.
- Promotion events append through the existing experiment ledger and update `biohub progress`.
- Later ZebraHub and strong-model phases consume frozen manifest hashes, scorer provenance, exact-report schema, and promotion policy from this phase.

</code_context>

<specifics>
## Specific Ideas

- The user wants strong generalizable networks, but compute is intentionally deferred until this validation authority prevents expensive optimization of leaky proxies.
- Current evidence keeps ZebraHub `selective_ssm_medium` as the first candidate to pass through this gate.
- No metric hacks, stale notebook scores, hand labels, or public-score imitation may enter candidate selection.

</specifics>

<deferred>
## Deferred Ideas

- Reciprocal ZebraHub calibration and exact candidate evaluation belong to Phase 3.
- GPU capacity benchmarks, training, external pretraining, and higher-capacity architecture selection belong to Phase 4.
- Kaggle inference packaging, submission execution, and final-slot selection belong to Phase 5.

</deferred>
