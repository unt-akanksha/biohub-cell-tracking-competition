# Biohub rules refresh — 2026-08-27

Checked at: `2026-08-27T05:09:40Z`

The authenticated Kaggle CLI fetched the live `rules` and `Code Requirements`
pages without querying the active kernel. Both pages are byte-for-byte unchanged
from the stored 2026-08-26 snapshot:

- rules SHA-256:
  `14a61abea978cda14209163cd047ac8167b3d8f5dc3d450d2e21412f026686bb`;
- Code Requirements SHA-256:
  `556ed778342f2f9ad5265d9e21aeb6e607281a836cab196412b0dc23a4682191`.

Official pages:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/code-requirements>

## Binding constraints relevant to this candidate

- Notebook submissions must run in at most 12 hours on CPU or GPU, have
  internet disabled, and create `submission.csv`.
- Freely and publicly available external data and pretrained models are
  allowed. External resources must be reasonably accessible at minimal cost.
- A participant may make at most five submissions per day and select at most
  two final submissions. Maximum team size is five.
- Test/validation hand-labeling and human prediction are prohibited. Private
  code or data sharing outside an official team is prohibited.
- Open-source code used to generate a submission must use an OSI-approved
  license that does not prohibit commercial use.
- A winning solution and the code used to generate it must be deliverable and
  reproducible, including training, inference, environment, architecture,
  preprocessing, loss, and hyperparameter documentation. The winner license is
  MIT, subject to the stated exceptions for separately licensed external data
  or pretrained models.

## Current candidate compliance

- The appearance model and training code are Biohub-owned; no public Kaggle
  code or predictions are copied.
- Trackastra source is pinned at commit
  `6a8ce94ee7c5a1f22c8eb77229ea5a0bc95a7b5b` under BSD-3-Clause. The corrected
  public synthetic corpus is CC0. Both are commercially usable and recorded in
  the runtime manifest.
- GPL CELLECT implementation and weights, gated/non-commercial checkpoints,
  and hand-labeled test data are excluded from the candidate.
- The future submission notebook remains internet-disabled and has a 39,000
  second inference hard stop, below the 12-hour Kaggle limit. Exactly two GPUs
  and whole-movie sharding are additional internal timeout controls.
- Training, clean calibration, processed materialization, exact scoring, and
  final competition upload remain separate. The portable runtime contains no
  Kaggle submit command, and any upload still requires explicit authorization.

Result: no rules-driven model or runtime change is required.
