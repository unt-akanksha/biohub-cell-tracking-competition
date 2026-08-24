# Guarded Kaggle GPU Runbook

Every GPU experiment follows this order: register, preflight, authorize, execute, monitor, then finish or fail. Authorization and execution always re-read Kaggle quota and owned competition-kernel statuses. Tests use fixtures and an injected runner; setup never pushes a kernel.

## Hard resource rules

- Reserve formula: `projected_remaining = live_remaining - declared_max_runtime`.
- A launch is allowed only when `projected_remaining >= 8.00 hours`. Exactly 8.00 passes; 7.99 does not. There is no reserve override.
- The registered declared maximum is authoritative and cannot differ from the guarded runtime.
- Kaggle's competition notebook maximum is 12.00 hours.
- A run strictly greater than 1.00 hour requires both `dense_memory` and `dataset_coverage` preflight evidence in addition to the smoke checks.
- When Kaggle reaches the protected 8.00-hour reserve, stop Kaggle GPU work and hand the same registered manifest to the user's cloud GPU. Never spend the reserve while waiting for cloud access.

## 1. Register

```powershell
python -m biohub_tracker experiment register --run-id zebra-reciprocal-calibration-v1 --hypothesis "Reciprocal calibration improves complete-graph exact OOF" --max-runtime-hours 2.00 --seed 1 --seed 2 --seed 3 --split leave-one-embryo-out
```

Record the configuration, data/model artifacts, and hashes at registration. Do not shorten the declared maximum later to fit quota.

## 2. Produce and validate preflight evidence

Create one immutable version-1 report bound to the run. It must contain passed, hashed evidence for `imports`, `inputs`, `single_batch`, `checkpoint_roundtrip`, and `output_location`. `model_step` must pass forward/backward or explicitly say why it is not applicable. Long runs also require `dense_memory` on the worst movie and `dataset_coverage` for the complete intended manifest.

```powershell
python -m biohub_tracker preflight validate --run-id zebra-reciprocal-calibration-v1 --max-runtime-hours 2.00 --report artifacts/preflights/zebra-reciprocal-calibration-v1.json
```

## 3. Authorize without launching

The following command reads live quota/status and emits a short-lived authorization plus a random nonce. It does not launch.

```powershell
.\scripts\guarded-kaggle-launch.ps1 -RunId zebra-reciprocal-calibration-v1 -KernelDir notebooks/zebra-reciprocal-calibration-v1 -KernelRef indarkarhana/zebra-reciprocal-calibration-v1 -PreflightReport artifacts/preflights/zebra-reciprocal-calibration-v1.json
```

Inspect `authorized`, `projected_remaining_hours`, the kernel/source hashes, expiry, authorization ID, and nonce.

## 4. Execute with explicit single-use confirmation

```powershell
.\scripts\guarded-kaggle-launch.ps1 -AuthorizationId auth-REPLACE -Nonce REPLACE -Execute
```

Execution rechecks live quota, the 20 newest owned competition kernel statuses (far above Kaggle's concurrency capacity), policy, source tree, preflight evidence, expiry, run/path/ref, and nonce. The authorization is marked consumed before `kaggle kernels push` and cannot be replayed.

## 5. Integrate the notebook watchdog

Instantiate this near the first executable cell and call `pulse()` inside every bounded training/inference loop. The safety margin must leave enough time to checkpoint and flush outputs.

```python
from pathlib import Path
from biohub_tracker.watchdog import BudgetExpired, BudgetWatchdog

watchdog = BudgetWatchdog(
    run_id="zebra-reciprocal-calibration-v1",
    declared_budget_seconds=2.00 * 3600,
    safety_margin_seconds=10 * 60,
    output_dir=Path("/kaggle/working/watchdog"),
)
watchdog.register_checkpoint("model", save_resumable_checkpoint)
watchdog.register_flush("metrics", flush_metrics_and_manifest)

try:
    for batch in loader:
        train_one_batch(batch)
        watchdog.pulse()
except BudgetExpired:
    raise SystemExit("declared budget reached; checkpoint and terminal status written")
```

On ordinary completion, call `watchdog.shutdown("completed")`. Always download/inspect `watchdog-terminal.json`, coverage evidence, the checkpoint manifest, and Kaggle logs before recording the terminal experiment event.

## 6. Finish or fail

Append actual quota/runtime and exact evidence with `experiment finish`, or preserve the failure with `experiment fail`. Never edit an earlier event; use `experiment amend` for corrections. Then regenerate `biohub progress`.
