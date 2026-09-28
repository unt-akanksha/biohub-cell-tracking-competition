# Division proposal audit: strict matching, not a missing large detector

Completed September14,2026. This is source-optimization evidence only, not
complete-movie performance or a submission-ready model.

The fixed audit covered112division transitions in63optimization movies and223
native frames. Both image generators ran successfully in181.128seconds on
Antelume after a5.697second/four-transition smoke. No validation, target-pilot,
sealed or unassigned movies were opened.

| Embryo | Annotated divisions | Current usable | Finer XY usable | Current: every cell has a peak within7um |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 19 | 6 | 9 | 19 |
| 6bba | 93 | 38 | 40 | 87 |
| Total | 112 | 44 | 49 | 106 |

The finer sampler gains8but loses3previously usable events: net+5, not a
coverage-preserving improvement. It yields71,903rather than70,263image proposals
(+2.334%) and adds133.924seconds of CPU proposal work over223already-normalized
frames. No deployment or training sampler was replaced.

All68current strict failures arise from parent/daughter image matches beyond
3.25um. None arose from one-to-one conflicts within that radius or displacement
over20um. Proximity within7um alone is NOT an identity guarantee: one division
loses a distinct daughter assignment, leaving105/112with complete one-to-one
matches; some nearest candidates are not isolated from other image peaks.
Do not simply widen every positive label or call all106divisions recovered.

## Tested additive-match hypothesis, not promoted labels

The new pure matching helper first retains every current3.25um assignment. It
may add an unmatched annotation only when its nearest image peak is within7um,
the next image peak is at least3.25um farther away, the chosen peak is unused,
and no other new annotation claims it. It cannot move or replace an old match.
This is a training-data hypothesis using image-only proposal locations, not an
inference metric trick. Unknown cells are still never negative labels.

A CPU dry run on the cached source image proposals yields:

| Embryo | Old usable | Additive usable | Newly available | Old events lost |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 6 | 9 | 3 | 0 |
| 6bba | 38 | 56 | 18 | 0 |
| Total | 44 | 65 | 21 | 0 |

This differs from requiring isolation on every cell: old strict matches remain
allowed, while only added matches must pass isolation. Four unit tests pin
extension, ambiguous-neighbor rejection, competing-annotation rejection, and
protection of an existing strict match. The seven-test audit/matcher suite passes.
No canonical training labels, model weights, selection sets or submissions were
changed by the dry run.

## Evidence and recovery

- Audit contract55e22648d8d9477277874b23a0bbb0965534753f980a5c3080de00897a12e8db.
- Source plan30c914750af6ce3d03af2e39cf9a172cd6dd95817dbd1bfe266b275468346281.
- Full result0f12f8a011c0575f63f9160e97db2dcc2dc8ccab6cb8c898e3963cc18e6a2d4e.
- Candidate matchera4283493608f0aa3867d6ff79d400c6a52c7cc1151f4a2e0ecd7106c6d0a6fee.
- All source points, smoke/full results and progress logs are SHA-verified
  locally:1,381,515bytes. See native-division-proposal-audit-v4-harvest.json.
- Reports native-division-proposal-audit-v4-summary.json,
  native-division-match-isolation-v4.json and
  native-division-additive-matches-v5-dry-run.json contain per-event evidence.

At06:37UTC Antelume had0MiBGPUmemory and no compute processes. No queued job;
the instance remains running/billing. Nothing was deleted this turn. Prior
RAM cleanup remains recoverable; RSNA/shared environments/unknown disks untouched.

## Research and next experiment boundary

The official [metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
allows a one-frame division timing offset. Same-transition triplet coverage is
therefore not a bound on official division recall. The current native labeling
radius is deliberately stricter than the scorer; changing it requires separate
identity and downstream quality evidence, not just an attractive count.

The primary-source refresh rediscovered [HOCT](https://arxiv.org/abs/2607.11754)
and [Ultrack](https://github.com/royerlab/ultrack). Existing local reports already
record rejected HOCT linear-probe and positive-unlabeled detector experiments;
they are not new ready-made solutions. HOCT full-backprop feasibility is a
distinct staged hypothesis, not proof that its transfer works. Direct Kaggle
discussion pages were unreadable in this refresh, so no claim of a newly
verified public-best model is made.

Next: extract only the additional source-optimization examples under the pinned
additive policy while preserving existing training/selection assets. Freeze a
paired retraining comparison before opening model results. The rejected native
v3heads stay rejected; no target-error fitting, threshold relaxation, old-failure
filtering or neutral Kaggle submission. The top-five goal remains active.
