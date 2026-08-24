# Project Research Summary

**Project:** Biohub Cell Tracking Competition  
**Domain:** Clean, resource-safe score improvement for sparse 3D+time cell tracking  
**Researched:** 2026-08-23

## Key Findings

### Stack

Extend a pinned version of the official PyTorch baseline rather than replacing its I/O, metric, and graph contract. Use the Kaggle-compatible Python/PyTorch environment, `torch.amp` FP16, OME-Zarr/GEFF, chunk-aware caching, pytest/Ruff, and a Git-tracked offline experiment ledger. Package promoted weights as hashed Kaggle datasets for Internet-disabled inference.

### Table Stakes

Before another GPU hour is spent, the project needs a live competition watch, immutable experiment ledger, exact scorer regression, held-out-embryo split, quota guard, notebook watchdog, and output-coverage preflight. These are score-enabling infrastructure because prior runs have failed late, leaked across source material, or optimized an unreliable proxy.

### Differentiator

The likely competitive advantage is not another public-checkpoint graph patch. It is a stronger learned detector/affinity system trained with sparse-label awareness, temporal-artifact augmentation, predicted-node association training, public external pretraining, and bilateral full-movie validation. ZebraHub selective-SSM is the best low-cost bridge into this architecture, but its gain remains unproven at the exact graph score.

### Watch Out For

The critical risks are stale notebook scores, the patched division exploit, source-overlap leakage, node-recall ceilings disguised as association errors, rare-division variance, Zarr throughput, dense-movie OOM, late coverage failure, public-LB overfitting, and accidental use of the 8-hour GPU reserve.

## Recommended Strategy

1. Build the operating guardrails and evidence ledger.
2. Pin and regression-test the official metric; create two embryo-held-out folds.
3. Run reciprocal competition calibration and exact graph OOF for ZebraHub selective-SSM.
4. Benchmark one baseline and one higher-capacity model for examples/second and peak memory.
5. Train sequential fold models with AMP, checkpoints, temporal-artifact augmentation, predicted-node association, and division sampling.
6. Spend remaining Kaggle budget only on a promoted calibration/ensemble and complete offline inference.
7. Preserve at least 8 hours; transition heavy follow-up work to the user's cloud GPU.

## GPU Portfolio

The current quota snapshot permits at most 22 hours of new Kaggle GPU use. This is a ceiling, not a target. A reasonable initial allocation is:

- Up to 1 hour: adversarial dense-movie and training-throughput smoke.
- Up to 3 hours: ZebraHub reciprocal calibration and exact graph evaluation.
- Up to 5 hours: first held-out-embryo fold.
- Up to 5 hours: reciprocal fold.
- Up to 3 hours: calibration or one high-information ablation.
- Up to 5 hours: final full inference and packaging.

Every launch recomputes the live quota and may shrink or reject this plan. No job receives the entire remaining allowance, and no failed smoke advances to a long run.

## Implications for Roadmap

- Phase 1 must deliver a working watch/ledger/quota/coverage foundation.
- Phase 2 must establish exact, leakage-resistant evaluation.
- Phase 3 should mature the existing ZebraHub lead as the first end-to-end evidence slice.
- Phase 4 should deliver a trained higher-capacity detector/linker with two reciprocal folds.
- Phase 5 should package, audit, submit, and select robust finals.

## Sources

- [Biohub competition overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview)
- [Biohub competition rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
- [Official baseline repository](https://github.com/royerlab/kaggle-cell-tracking-competition)
- [Official patched metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- Competition discussion audit links are cataloged in `FEATURES.md` and `PITFALLS.md`.
