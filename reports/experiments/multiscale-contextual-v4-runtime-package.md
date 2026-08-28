# Multiscale contextual v4 runtime package

Date: 2026-08-27

Status: locally built, privately published as dataset version 3, and
independently verified from a clean remote re-download. The experiment is not
registered or launched, and no submission was created.

The 46,386,607-parameter-per-fold v4 fallback is now portable rather than only
repository-local. The package contains 39 hash-bound source files covering:

- the prediction-preserving multiscale 3D/axial model;
- accepted-v3 warm-start loading;
- two-GPU ZebraHub pretraining;
- two-GPU reciprocal Biohub transfer;
- transition context and bidirectional candidate ranking;
- calibration, processed acceptance, and whole-movie inference;
- strict output/runtime verifiers; and
- the pinned Trackastra source needed by the downstream global linker.

The packaged runtime retains exactly two required GPUs, internet disabled,
leaderboard selection forbidden, public prediction copying forbidden, and no
competition submission command. V4 remains conditional on a passing v3
ZSNS001 acceptance gate and cannot delay an accepted v3 submission.

## Verified package

- Local directory:
  `.biohub/staging/biohub-temporal-multiscale-contextual-runtime-v4`
- Private dataset:
  `indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4`
- Admissible dataset version: `3`
- Clean re-download:
  `.biohub/cache/dataset-redownloads/biohub-temporal-multiscale-contextual-runtime-v4-version3`
- Source files: `39`
- Verified source bytes: `572,230`
- Total staged bytes: `582,195`
- Runtime manifest SHA-256:
  `ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3`
- Dataset metadata SHA-256:
  `a10a507e61ba492c19a0a9fd9239a8f1f211343ecff077ebdb72d26f607c84b1`
- Packaged experiment SHA-256:
  `3c9a4750d53d9037e700fbdec0653e129d96cb40037b0b9d4e1a6350f812fef3`
- Source experiment SHA-256:
  `022cd4853bf6f3df9b1fa459d8cc076b28a2c4e0d6380beb5953e33248b16812`
- Runtime builder SHA-256:
  `a687cba7c350c29c88e6541b1816bd3440bcd528a1f3e008762f07c114a3dfc0`
- Runtime builder test SHA-256:
  `0399f866b080e2897a85c0e721d569b39a9dda164722f4d349e4bb326871e14c`

Version 1 is excluded because the default Kaggle upload mode omitted the nested
Trackastra source. Version 2 is excluded because it was the remote verification
provenance used to freeze self-reference-free package metadata. Version 3 is
the only admissible runtime. Its clean download verifier checked all 39 byte
counts and hashes and returned `status: verified`.

Updating the repository-side experiment record with the verified version and
manifest did not change any packaged byte: rebuilding reproduced the exact
`ac1c32...b8b3` manifest. The combined runtime and multiscale suites passed 18
tests. All packaging and verification used no GPU.
