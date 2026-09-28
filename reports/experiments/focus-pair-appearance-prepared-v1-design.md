# Exact CPU transform-cache admission

The GPU smoke passed in kernel version1, but the shared account quota continues
to decline independently of Biohub. Before committing another GPU run, remove
repeated transformation work from the CPU optimizer. This changes computation,
not data, supervision, projection, objective, optimization or acceptance gates.

Write each training movie's transformed FP64 features once, in bounded64-group
blocks, to new memory-mapped files. Preserve the exact original block order and
all candidate/null rows. Never allocate the entire fold as a dense RAM array.
Cap disk cache per fold/arm at3GiB; retain original bounded active feature blocks.
Use2 numeric threads and a300second subprocess watchdog for this profile.

First replay the existing20-target real fitting smoke. Require every transformed
array, objective and gradient at zero/saved coefficients to match exactly. Refit
both heads with the frozen solver/settings and require exact coefficients and
objective. Then use only the original first fold's eleven training movies,
rebuild the same projection, and measure3 zero-objective passes per arm. Compare
the complete objective/gradient norm with the already frozen CPU profile. No
held-out scoring or full head optimization in the throughput profile.

Persist artifacts without overwriting prior runs. No GPU, AWS or RSNA action.
Only after measured admission may a separately capped full12fold comparison
proceed. This is not a relaxed quality gate or a new hyperparameter experiment.
