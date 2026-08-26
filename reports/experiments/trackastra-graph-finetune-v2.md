# trackastra-graph-finetune-v2

Status: staged guarded retry

This retry preserves the new Trackastra/Biohub scientific experiment from `v1`
and changes only two operational inputs discovered during the 32-second setup
failure:

- Pin NumPy to the Kaggle image's SciPy-compatible `2.0.2` during offline install.
- Replace an incomplete cached validation graph with a fresh download, then
  require exactly 33 GEFF files and load all four graphs through `tracksdata`
  before packaging.

The model, training split, augmentation, resource envelope, clean promotion gate,
and no-submission policy are unchanged. This remains our own learned candidate,
not a reproduction or calibration of the public submission.
