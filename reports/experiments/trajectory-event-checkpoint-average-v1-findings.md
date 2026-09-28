# Source optimization audit and fixed averages: not quality-validated

All5780source case targets were read with their original file hashes verified.
Source44has10forced-fork cases/1223,10forced forks,4148known child constraints.
Source6has50forced-fork cases/4557,50forced forks,43788known child constraints.
These are retained matched partial-label training constraints, not all105
annotated source divisions. Original sparse-unknown handling is unchanged.

Checkpoint inspection used79source44and280source6saved states. Source6six
division-geometry weights had norm0.790811559357at13400and0.000000599363at13671.
An independent271step replay using ONLY original anchored regularization and
saved Adam moments reproduces the final six values exactly(maxdifference0).
This is strong evidence consistent with no effective division-geometry data
gradient in that tail. It is not an exact replay of all training-case oracles.
Negative-control source44tail3400..3669 differs0.558618757116from that
regularization-only simulation, so the control is not trivially always exact.

One post-hoc, outcome-independent averaging rule was frozen: all regular50update
snapshots from epoch3, excluding irregular epoch-end extras. Six tests pass
0.88s. Builder completed and exact serialization verified. No final model,
failed score, training contract or snapshot is replaced; all prior exposure
is disclosed in the design. Build reportSHA
`75dc1dcf4253dd2e964dc4acccbbbb396124afa05cfa526677120e7a418d8e6d`.

| Source | Snapshots | Geometry norm final | Geometry norm average | New model SHA |
| --- | ---: | ---: | ---: | --- |
|44b6|25|0.9348415844|0.7243436448|65f8b8874ba2939bdfe460cdf76984579639e10a870c5f9c428a10a61df1266f|
|6bba|91|0.0000005994|0.1209924626|cf39a432b110bf3bca05e040af990d1c8f8e8c0ff9897141adceccad5d05d1a9|

Then a small source-only exact-hinge probe tested the first two ordinary cases
and first two forced-fork cases in each source's frozen order (8cases total).
Actual source6two-case smoke passed before fullprobe. Three arms: initializer,
originalfinal, fixedaverage. Fullprobe6.063seconds, no timeout/fallback or new
GT files. This is a functionality/optimization diagnostic, NOT a representative
full training-risk estimate, held-out quality score, or selection test.

| Source/stratum | Final regularized objective | Average objective |
| --- | ---: | ---: |
|44 ordinary|0.035213899|0.021875820|
|44 division|0.805071704|0.987632737|
|6 ordinary|0.022259056|0.010823851|
|6 division|0.671545965|0.638072900|

Ordinary sampled hinges are zero for all arms; their reported objective
improvement is ONLY reduced regularization, not extra correct edges.
Source44rare objective worsens under averaging. Source6rare mean improves but
one of its two raw hinges worsens slightly. Do not claim both averages are
better or ensemble them automatically. A prospective source6averaged candidate
still needs complete-movie scoring/runtime gates; none is promoted here.
ProbeSHA`fe41370a1a61f10fea0c41ba2440119abc2f225ebb79fc9febfa04078f4dc883`.

The next optimization hypothesis is to preserve rare-event learning during
ordinary updates (e.g. a fixed balanced minibatch objective), not secretly
select a favorable intermediate checkpoint. The separate locked-division
scope limitation is documented independently; it does not justify blanket
division removal. All this work used localCPU, no AntelumeGPU.
