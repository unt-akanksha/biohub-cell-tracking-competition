# Sequential full pair-appearance comparison: runtime admission

Run the previously frozen two-arm pair-appearance protocol without changing any
feature, label, loss, optimizer setting or quality gate. Start with the cheaper
LDA arm, then the full-feature arm,12 movie-held-out fits each. Both arms run
regardless of the other's quality; an optimizer failure stops only its own arm,
persists the reason and is not repaired by changing hyperparameters mid-run.

Evidence before launch: complete descriptor provenance independently replayed;
CPU/GPU smoke exact decisions and equivalent objectives; prepared CPU smoke
exact coefficients; resident arrays exactly replayed on first-fold11 movies.
Resident first-fold objective times: full0.672-0.906s; LDA0.328-0.453s. The smoke
solver used325/100 evaluations. With preparation and24folds this suggests about
1-1.5CPU hours, not a guarantee. Declare a9000second parent subprocess cap with
2 numeric threads. Save each model before scoring its held-out movie, then save
the fold result immediately. Keep completed fold artifacts if interrupted.

The9D arm may hold its <=512MiB feature arrays in RAM; the72D arm uses readonly
persistent memory maps and bounded64-group operations. Disk prepared cache
cap3GiB per fold; <=36GiB for the full24fold comparison, within observed449GB
free. No new GPU launch, cloud start/stop, shared RSNA or environment mutation.

Use only twelve original fitting movies and their unchanged controls. Each
fold's projection/scaler/weight/head uses the other eleven movies only. Encoder
training included these movies: this is correction screening, not evidence of
new-embryo generalization. Unknown annotations remain excluded, all candidates
and fixed null preserved. No leaderboard feedback or metric exploit.

Apply the original gate separately to each arm: pooled NLL below both physical
and neural; parents>=9835; absent>=115; >=8/12 neural NLL wins; per-movie NLL
regression<=0.02. Compare full/LDA directly. Even a passing arm requires host
verification before all-fit export and subsequent diagnostic/full-movie/embryo/
runtime gates. No source/diagnostic/new target access or submission in this run.
