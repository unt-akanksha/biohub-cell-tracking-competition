# LSM-FM PU adaptation v2 staging

## Decision

Retry the unchanged LSM-FM training/validation experiment after one
subprocess-environment repair. No competition submission is created or made.

## Attempt-1 evidence

Version 1 stopped before model construction and optimizer step zero. Its
notebook process successfully:

- verified the private runtime manifest and all 428 Kaggle-extracted MONAI
  members;
- imported MONAI 1.5.1;
- verified both teacher checkpoints and the stripped LSM-FM checkpoint;
- loaded the competition and graph inputs; and
- confirmed a Tesla T4.

The spawned trainer then raised `ModuleNotFoundError: monai` because its
explicit `PYTHONPATH` contained the runtime and support repository but omitted
the already verified extracted MONAI root. It produced no learned checkpoint,
validation result, public prediction, or submission.

## V2 delta

The single code change is:

`trainer/evaluator PYTHONPATH += monai_import_root`

The same environment object is used by both subprocesses. The runtime dataset,
model and teacher hashes, 187-movie inventory, seed 20260827, 768-step target,
learning rates, unfreeze schedule, serialized activation strategy, inference,
eight-movie selection set, four untouched acceptance movies, and all gates are
unchanged.

## Verification

- V1 result SHA-256: `736b044f59ca73edfae4070e4fd1de470d1feef7d4d44b31a7e406ea25bc6da0`
- V1 launcher terminal SHA-256: `5157b3b1ffef4864c3ae632eae8887430d0129bf2f9cdf262911691d1b18b14c`
- V1 log SHA-256: `00c2e419d8243d59e08d848d0ad24852378429cfe249b84b60c04f73866df86a`
- V2 notebook SHA-256: `aa4b87ca6e5c0cbd5e181c0030f5ce07483c04a311185b6572ec22a6e7420f89`
- V2 strict preflight logical SHA-256: `cb754b0b28f5fa0f803c0cd4bde6fc662cbbc9d5f09d81777713ee2b359bba86`
- Base focused suite: 36 passed
- V2 repair test: 1 passed
- Internet and TPU: disabled
- Submission path/API: absent and guarded

Quota after v1 was 26.37 hours. A 2.00-hour declared v2 remains safely above
the protected 8.00-hour reserve.
