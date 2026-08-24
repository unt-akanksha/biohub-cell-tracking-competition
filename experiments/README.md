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
