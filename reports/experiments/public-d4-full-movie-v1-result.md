# Public D4 correction: pooled gain, failed embryo gate

Completed September10,2026. **Not promoted or submitted.** This is an enhancement
to the pinned LF-DCTTA source, not a claim of a new Kaggle leaderboard score.
Four previously exposed public-checkpoint training diagnostics are not independent
CV. Both source weights and quality gates remained fixed throughout the experiment.

## Measured outcome

| Complete movie | Original combined | Corrected combined | Change |
|---|---:|---:|---:|
|44b6_24264f12|0.895361|0.923089|+0.027728|
|44b6_81c256f0|0.987744|0.989007|+0.001263|
|6bba_23af9eeb|1.004163|0.998102|-0.006062|
|6bba_f1fde7e0|0.889318|0.888963|-0.000355|
|Official pooled summary|0.944839|0.947619|+0.002781|

Pooled raw edge Jaccard also improves:0.941720→0.944365. Counts are original
TP1325/FP44/FN38 versus correctedTP1324/FP39/FN39: the net gain primarily comes
from fewer false edges, not more true edges. Embryo44b6 gains0.016012;
embryo6bba loses0.003052. Both movie and embryo nonregression gates FAIL.
Worst movie remains6bba_f1fde7e0 and slightly worsens. Do not relax these gates
or route by embryo identity.0.947619 is a local diagnostic result, not proof
of meeting the user's leaderboard target or beating the public0.947 reference.

No annotated positive divisions are present in this four-movie subset. Each
arm has one false division and no true divisions; the subset therefore cannot
establish positive-division recovery. A future independent evaluation also
needs adequate division-event coverage. A score above1 for one movie is a
consequence of the official count adjustment, not fabricated graph content.

Result SHA256: `04458d9d43caef023ab26de663f9c830618e5744b8d9b748c1d9164762720c0b`.
Separate saved-count micro-aggregation exactly reproduces both official summaries.
Metric commit: `075fc5f5a52d11077f9dc2b074644618f26939e2`.

## Runtime, recovery and scoring correction

Real encoder/DeepCenter smoke21.614s; end-to-end8-frame paired smoke28.496s;
full4movie×2arm GPU job932.901s (**15m33s**). Total GPU-job wall time983.012s
(16.38minutes), not the AWS billed instance uptime: preparation and idle time
also cost money. No extra GPU run after inference. No Kaggle GPU hours used.
RSNA and the shared environment were untouched. At22:43–22:44UTC the A10G had
no remaining compute process; the shared instance was left running, not stopped.

All57 terminal/prediction/log/raw-candidate/pre-postprocess files are backed up
locally and independently agree with remote sizes/SHA256. Remote manifest has
one additional transient progress file; exact inventory is in
`public-d4-full-movie-v1-artifact-manifest.json`. About26.47MB retained locally.
The terminal receipt SHA256 is
`61429f28d6fa3c42ee761d38456f9daca66986bd5845e38162f8baefcaf45683`.
No model, source, graph, data or old receipt was deleted or rewritten.

Scorer wrapperV1 stopped **before evaluate()** because it incorrectly required
GT annotations at all100 image times.44b6_24264f12's complete sparse annotations
end at86;6bba_f1fde7e0 has no annotations at87. All100 image frames and all
original annotation files are nevertheless present. V1 remains unchanged.
New wrapperV2 verifies all84 selected GEFF files against the historical cache
manifest, then scores the full100-frame predictions without trimming anything.
Official metric, weights, predictions and numerical acceptance gates are unchanged.
Truth manifest SHA256: `744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9`.
V2 SHA256: `4165dbfb9b96d82e25aeaaa534ba0688a522a54b0723b7bb570e60da8db92667`.
Final combined focused suite: **25 passed in9.77s**. The new sparse-truth tests
plus scorer/gate subset also passed10/10. See execution report for earlier
timings and the frozen contract. Scoped Git diff whitespace checks were clean.

## Concrete next bottleneck from saved outputs

Local CPU node-matching attribution, not a new selected candidate:

|Corrected arm|Raw detector matched / annotated|After ILP|Final|
|---|---:|---:|---:|
|44b6_24264f12|230/230|228|228|
|44b6_81c256f0|199/199|199|199|
|6bba_23af9eeb|513/513|512|512|
|6bba_f1fde7e0|476/476|473|464|

All1,418 annotated nodes have a raw corrected detection within the official
matching tolerance. This does **not** imply precision, exhaustive cell recall,
or independent detector generalization. It does show that increasing raw
detector recall is not the immediate bottleneck on these diagnostic movies.

On the weakest movie the original arm matches476→475→467 cells through these
stages, while corrected matches476→473→464. The post-ILP stage loses11 previously
matched annotations and gains2, for net9 lost. This attribution combines repair,
pruning and smoothing; it does not yet isolate which substep causes each loss.

Next bounded work: separate pruning/localization losses from association losses,
using cached genuine detections and learned edge probabilities. Investigate
temporally supported recovery of real discarded detections, not arbitrary
node insertion, count manipulation, embryo-specific routing or threshold sweeps.
The cached artifacts support initial CPU tests without another detector run.
Any new candidate needs its own frozen design, complete-movie checks and genuinely
independent generalization evidence before promotion. Do not resubmit a replica.

Qualified submissions today remain **0/5**; overall goal remains unfinished.
