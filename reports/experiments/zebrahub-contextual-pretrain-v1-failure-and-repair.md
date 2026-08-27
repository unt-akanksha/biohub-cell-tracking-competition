# ZebraHub contextual pretraining v1 failure and repair

Status: Kaggle version 1 failed before optimizer step 1; the defect is repaired,
covered by an autocast regression test, and published in private runtime version
9. The retry is staged but not yet launched.

## Failure evidence

- Kernel: `indarkarhana/biohub-zebrahub-contextual-pretrain-v1/1`
- Kaggle state: `KernelWorkerStatus.ERROR`
- Launcher elapsed time: `48.016` seconds
- GPU quota before: `17.45` hours
- GPU quota after: `17.43` hours
- Accounted use: `0.02` GPU hours
- Failure event: `evt-e3cae75ef74f4ddd8f3bf9d186a06f30`
- Launcher terminal SHA-256:
  `6b59fdd9139873d55286d752794b1d523500b5071804f88a42f4e726806b1a50`
- Both worker logs SHA-256:
  `f60de9e2d07e3098347654be3c95900bf28491e53e356f583f85a60b31fb2966`
- Kernel log SHA-256:
  `eaf47cfbd9d433d93467cf907a5bbde19971e680be00df23ad202c20d9b0b979`

Both T4 workers independently failed during the initial ZSNS005 validation:

```text
RuntimeError: Index put requires the source and destination dtypes match,
got Float for the destination and Half for the source.
```

All prior boundaries passed first: two T4s were visible, runtime version 8
verified all 36 files, data version 4 verified all 64 training and 16 validation
shards, and no competition data or submission path was present. No checkpoint
or training metric was produced.

## Root cause and repair

`candidate_pair_logits` deliberately creates its dense sentinel output from
the float32 node embeddings. Under CUDA autocast, the learned edge head returns
float16 compact logits. PyTorch index scatter requires exact source/destination
dtypes, so assignment failed.

The repair casts only the compact learned logits back to the stable output dtype
immediately before scatter. This is differentiable and keeps downstream ranking
losses in float32. It does not change architecture, parameters, initialization,
data, split, augmentation, objective, optimizer, or promotion thresholds.

A CPU bfloat16 autocast regression now reproduces the same mixed-dtype path and
verifies finite float32 output plus finite backward gradients. The focused
model/pretrainer suite passes `20` tests.

## Immutable retry package

- Retry run: `zebrahub-contextual-pretrain-v1-mp-repair`
- Existing kernel slug, next version: `2`
- Private runtime version: `9`
- Runtime manifest SHA-256:
  `aff4e21f675848f94cbfd42e4d1a43ebc5b470fbc21db3934f85797066fc65cb`
- Runtime remote re-download: verified, 36 files, 551,635 bytes
- Repaired model source SHA-256:
  `ec3f0e503af039afd879c757a80d805362c36bd66aff44bd588d12c3f5012e7f`
- Notebook SHA-256:
  `962a921ac2b13401aa8da63553c4bc5fb0f333ff8ecb853886f97a5a01dadae4`
- Metadata SHA-256:
  `0abddc3553ac9954f9af7ee9022d80bebbc2af65d824c33d9d14ea2b78e7ae01`

Quota at diagnosis was `17.43` hours. A fresh `6.67`-hour guarded retry projects
`10.76` hours remaining, above the mandatory `8.00`-hour reserve. Launch still
requires a new mount-aware preflight and single-use authorization.
