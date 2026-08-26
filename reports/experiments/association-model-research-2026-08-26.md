# Association-model research — 2026-08-26

This note records model choices made without leaderboard feedback.

## Current evidence

- [HOCT](https://arxiv.org/abs/2607.11754) is the newest directly applicable
  lineage-association architecture found. Its edge-centric attention explicitly
  models competing links and divisions, reports state-of-the-art Cell Tracking
  Challenge results without a deep image encoder, and reports rapid adaptation
  from a few hundred annotations. This supports the dual-checkpoint frozen-
  backbone probe lane now staged for Biohub.
- [Trackastra](https://github.com/weigertlab/trackastra) remains a strong
  general tracker and won the generalizable-linking task in the seventh Cell
  Tracking Challenge. Its newer SAM2-feature option is not scheduled here: it
  assumes a reliable instance-segmentation feature path that the centroid-only
  Biohub artifacts do not provide, while the independently fine-tuned 27.5M
  coordinate model already degraded on provisional complete-movie validation.
- The ICCV 2025 paper
  [How To Make Your Cell Tracker Say “I dunno!”](https://openaccess.thecvf.com/content/ICCV2025/html/Paul_How_To_Make_Your_Cell_Tracker_Say_I_dunno_ICCV_2025_paper.html)
  shows that uncertainty and disagreement are useful for transformer-based
  trackers. The staged HOCT grid therefore includes a minimum-consensus variant
  in addition to log-odds blends. This is a bounded selection option, not an
  extra acceptance-time choice.

## Decision

Run the dual-checkpoint HOCT experiment directly after the active sequential
gate. It strictly contains the general-only HOCT candidate, adds the
CTC-specialized checkpoint, and avoids spending a separate four-hour GPU budget
on duplicate feature extraction. Selection maximizes the worse per-embryo gain,
and acceptance labels—including base-comparator acceptance scores—remain
unloaded until the model, blend, linker, and thresholds are frozen.

The eventual submission kernel is already upgraded to load only the
backbone(s) required by the clean winner, preserve every detector node, and
refuse an edge-identical public replica.
