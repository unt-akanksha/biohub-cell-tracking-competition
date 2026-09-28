# Structured assignment: real pooled gain, one unresolved movie regression

Source feasibility audit: at most16additional known positive parent choices are
jointly recoverable in the fixed protected-degree assignment family. This is
not a learned-model score. No oracle graph or submission was generated.

The partial-label structured hinge passed synthetic coupled-assignment/gradient/
ambiguity tests and a two-update whole-source-data smoke. Ten fixed CPU fits
(eight whole-movie folds and two full-source fits),300updates each, completed in
47.25s. Only source6passed both internal source and opposite-embryo screens.

Complete8source-movie scorer test of source6weights:

| Metric | Current | Structured v1 |
|---|---:|---:|
| Patched combined score |.8772907140|.8812336418|
| True edges |3713|3719|
| False edges |261|249|
| Missing edges |243|237|
| Divisions TP/FP/FN |0/2/7|0/2/7|

Both embryo aggregates improve (.84188->.84947 and .88456->.88773), but
44b6_c50204e0 regresses .81940->.81215. Four movies improve, three are neutral,
one regresses. Thus the predefined movie-nonregression gate FAILS. No promotion,
selection-label test, ensemble or submission for these weights. The method has
real source-level signal; the candidate is not qualified. Some source6data were
used for fitting, so even a source pass would not prove independent generalization.

Follow-up source-label coverage audit (same unchanged predictions and candidate
sets, no selection/validation GT): the official unique physical matcher provides
3,838positive parent groups, versus2,871under the conservative3.25um training
matcher. All original2,871positive identities are unchanged;967previously omitted
groups are added. The regressing movie alone gains105supervised groups. This is
evidence of incomplete supervision, not proof that it caused the regression.

V2 tests the same structured loss/optimizer/feature set/budget with those complete
physical source matches. Positives use unique geometric matching, negatives still
require distance>7um; no node movement, count manipulation, duplicate edge logic,
hidden-label use, metric bypass, or threshold sweep. Initial weights remain the
original source-only fold weights; held movies do not enter their fold fits.
Smoke6.58s passed. Full CPU session76917launched; inspect terminal before claims.

Selection collection session46526 is terminal:10complete100-frame movies in
1,595.22s/26.59min. All90,246,284output bytes are locally backed up/hash-verified.
Antelume GPU is free; instance still bills, raw image cache remains recoverable.
Selection feature preparation session47895is CPU-only and opens no GT. Selection
truth remains unopened by this lane. Public submission remains0.946.
