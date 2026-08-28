# Multiscale contextual v4 kernel staging

Date: 2026-08-27

Status: deterministic two-GPU pretraining and reciprocal-transfer notebooks
built and locally tested. Neither kernel is registered, preflight-authorized,
launched, scored, or submitted. The passing v3 lane remains first priority.

## External pretraining kernel

- Local kernel: `biohub-zebrahub-multiscale-pretrain-v1`
- Remote identifier reserved for future registration:
  `indarkarhana/biohub-zebrahub-multiscale-pretrain-v1`
- Notebook SHA-256:
  `c15f7e1d9080067bec78e7567b1aba8c95132b135854cedd56951954083ed4ec`
- Metadata SHA-256:
  `dfb9857ae52874f80bc8138d1a7afd724992df3b2865681428098ccfe6306fdd`
- Builder SHA-256:
  `47de52faa2baf4e9db2f730ea7b7fd9bd3d292b2118ad3da7956d1fab3adff21`
- Builder test SHA-256:
  `060cb6523a5c076cf17fa5b8cc767c7add5937e9be3b087c9686c406c6eade5a`

The notebook requires exactly two GPUs, internet off, the verified v4 runtime
dataset version 3, frozen ZebraHub shards version 4, both original v3
pretraining folds, and the passing one-shot v3 acceptance output. It recomputes
both acceptance gates, binds aggregate and child terminals, checks launcher
provenance, and strict-loads both 20,747,761-parameter v3 checkpoints before
allowing the zero-residual v4 warm start. Training uses a 32-patch microbatch
with three-step accumulation to hold the effective batch at 96 while reducing
46.4M-model activation pressure on each T4.

## Reciprocal transfer kernel

- Local kernel: `biohub-temporal-multiscale-transfer-v4`
- Remote identifier reserved for future registration:
  `indarkarhana/biohub-temporal-multiscale-transfer-v4`
- Notebook SHA-256:
  `fd749af69a332f06c012983dd8cb588e4363a7e3d3e9d07103ca6f544e14d71c`
- Metadata SHA-256:
  `5b327b04b58ee8f013679548e811fafb349b4e62ff600b265bb56a9e9ee28e01`
- Builder SHA-256:
  `60a54b4955c0b1cf9c55bcd44b5e8fb2f6cf12f725affd0d4d588fffddfe21da`
- Builder test SHA-256:
  `33319ff4949d10d920621e4e4455ed5061197954bee9f2c7f9d62edaf6ee3686`

The transfer notebook independently rechecks the accepted-v3 chain, then
requires two complete v4 pretraining folds whose checkpoints, worker terminals,
selection/audit gates, and v3 initialization records agree by hash. It
strict-loads both 46,386,607-parameter checkpoints before reciprocal Biohub
training. Promotion still requires both real folds to improve and every
synthetic metric regression to remain at or below 0.01.

Both notebook builders fail if their reviewed v3 base templates drift, and both
generated notebooks are byte-deterministic. Four focused builder tests and all
24 combined v3-template, v4 model/trainer, runtime, and kernel tests pass. There
is no Kaggle submission command in either notebook, and this staging work used
no GPU quota.
