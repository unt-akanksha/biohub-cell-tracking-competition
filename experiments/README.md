# Experiment event ledger

`events.jsonl` is append-only. Every line is one immutable event; never edit, reorder, compact, or delete historical lines. Corrections use `experiment amend`. A malformed final fragment blocks all appends until `experiment repair --reason ...` explicitly quarantines and acknowledges it.

Register and run a new experiment:

```powershell
biohub experiment register --hypothesis "reciprocal ZebraHub calibration" --max-runtime-hours 1 --seed 1 --split embryo-held-out
biohub experiment start RUN_ID --kaggle-ref owner/kernel --authorization-id AUTH_ID --quota-before-hours 30
biohub experiment finish RUN_ID --actual-runtime-hours 0.8 --quota-after-hours 29.2 --metrics-report reports/metrics.json --report reports/metrics.json
biohub experiment decide RUN_ID --decision retain --evidence exact_oof:reports/metrics.json
```

Record failures and rejected gates instead of deleting them:

```powershell
biohub experiment fail RUN_ID --actual-runtime-hours 2 --quota-after-hours 28 --reason "OOM on dense movie"
biohub experiment reject RUN_ID --actual-runtime-hours 0.2 --quota-after-hours 29.8 --gate bilateral_oof --reason "division regression"
```

Append a correction:

```powershell
biohub experiment amend RUN_ID --target-event-id EVENT_ID --reason "split label typo" --replacement '{"split":"leave-one-embryo-out"}'
```

Render the full history, including failures and rejections:

```powershell
biohub progress
biohub progress --json
```

Public score is recorded separately and is never sufficient evidence for promotion.

## CPU controls and aggregate exact decisions

The Phase 2 official-data control uses a separate append-only lifecycle:
`cpu_acceptance_registered -> cpu_acceptance_started ->
cpu_acceptance_inputs_bound -> cpu_acceptance_completed|failed`. These events
contain CPU runtime only. They are excluded from Phase 1 GPU run reconstruction,
quota accounting, launch authorization, and submission authorization.

An exact comparison is also a separate aggregate run. Its registration freezes
the four reciprocal role/fold members and each producer's immutable registration,
optional CPU input-binding, terminal inventory, and artifact hashes. The aggregate
must complete with the report attachment before a `model_candidate` can receive
an `exact_promotion_decision`. An `official_data_control` can prove the evaluator
works on mounted official data, but it cannot receive a promotion decision.

Soft scientific failures record `review_required`. Any exception is a second
event with the original decision hash, failed quantitative values, approver,
reason, downstream authorization, and report/policy/scorer/manifest hashes. It
does not rewrite the original decision and cannot override a hard integrity
reject.
