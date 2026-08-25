# Exact validation and Phase 2 CPU acceptance

The authoritative candidate path is local and ledger-bound. It requires four
complete prediction sets, a started aggregate evaluation, the pinned patched
organizer scorer, and the frozen reciprocal manifest:

```powershell
biohub evaluate exact `
  --ledger experiments/events.jsonl `
  --evaluation-run-id EVALUATION_ID `
  --truth-dir DATA_ROOT `
  --manifest manifests/reciprocal-embryo-v1.json `
  --scorer-lock config/official-scorer.lock.json `
  --evaluation-policy config/evaluation-policy.json `
  --baseline-set fold-44b6-to-6bba=BASELINE_44_DIR `
  --baseline-set fold-6bba-to-44b6=BASELINE_6_DIR `
  --candidate-set fold-44b6-to-6bba=CANDIDATE_44_DIR `
  --candidate-set fold-6bba-to-44b6=CANDIDATE_6_DIR `
  --output-dir reports/exact/CANDIDATE_ID
```

Each directory must contain `prediction-set.json`. The evaluator resolves every
producer against immutable registration, optional CPU input-binding, terminal
inventory, and artifact event hashes. Missing ledger, missing aggregate start,
unknown producer, wrong fold, incomplete coverage, non-integral submission
round trip, or any source/hash mismatch exits nonzero and attaches a legal
aggregate failure where possible. There is no ledger-free fallback.

Evaluate a completed learned candidate under the frozen policy:

```powershell
biohub promote evaluate --ledger experiments/events.jsonl --report reports/exact/CANDIDATE_ID --policy config/promotion-policy.json --evaluation-run-id EVALUATION_ID
biohub promote evaluate --ledger experiments/events.jsonl --report reports/exact/CANDIDATE_ID --policy config/promotion-policy.json --evaluation-run-id EVALUATION_ID --record
```

Public score is not a policy input. A hard integrity failure rejects. An
integrity-clean soft failure records `review_required`; any approved trade-off
is a second append-only exception event with the quantitative failed values,
approver, reason, downstream authorization, and all exact evidence hashes.

## Official-data CPU control

Dry preflight validates owned paths and CPU metadata without issuing a request
or changing a Kaggle asset:

```powershell
powershell -File scripts/run-phase2-cpu-acceptance.ps1
```

The explicit live command is the only permitted mutation path:

```powershell
powershell -File scripts/run-phase2-cpu-acceptance.ps1 -Execute
```

It reads quota/competition state, pre-registers a one-use request, versions only
`indarkarhana/biohub-phase2-runtime`, pushes only the private CPU script kernel
`indarkarhana/biohub-phase-2-cpu-acceptance`, polls it to a terminal state,
downloads one compact pending report, re-reads GPU quota, and reconciles locally.
Kernel metadata fixes GPU, TPU, and Internet to false. No competition submission,
vote, comment, fork, or unrelated asset operation is reachable.

Remote output is `pending_reconciliation`, not an `ExactReport`, signature, or
authority. Local reconciliation verifies the nonce/request hash, owned versions,
source identities, manifest/folds, round-trip inventories, output hashes, CPU
runtime declaration, no-submission declaration, and unchanged GPU quota. It then
legally completes the CPU producer and a separate four-slot aggregate evaluation.
The accepted control remains `official_data_control`,
`authorized_for_promotion=false`, and `authorized_for_submission=false`.

Inspect the resulting evidence:

```powershell
biohub progress --json
python -m pytest -q tests/test_exact_cli.py tests/test_promotion.py tests/test_ledger.py
```

The Phase 3 entry gate is unchanged: ZebraHub or any learned candidate needs real
completed evidence-eligible reciprocal producers, four producer-bound prediction
sets, a new completed aggregate exact evaluation, and its own policy decision.
The truth/self control proves infrastructure only.
