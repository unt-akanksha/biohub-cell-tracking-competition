# Resident CPU cache admission, unchanged modeling

The exact prepared CPU smoke passed, but repeated mapping/objective calls remain
slow. Reuse readonly maps across optimizer evaluations; keep arrays in RAM only
when the entire prepared arm fits512MiB (the9D LDA arm). The72D full arm remains
readonly memory-mapped. Preserve64-group blocks, original order and FP64 values.
No model, projection, loss, solver, data, gate or selection changes.

Replay every prepared feature/label array against the frozen original transform
before profiling. Use only the same first-fold11 fitting movies; compare zero
objective/gradient norm with the completed previous profile,3 timed calls per
arm. This does not evaluate the held-out movie. Two CPU threads,300second cap,
no GPU or cloud/RSNA mutations. Separately declare full-fitting cap after timing.
