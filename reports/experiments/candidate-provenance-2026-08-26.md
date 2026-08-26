# Biohub candidate provenance — 2026-08-26

## Benchmark versus candidate

`public-0927-clean-repro-v2` is an exact, attributed reproduction of
`evgendvorkin/biohub-0-927-lb`. It is a score floor and frozen comparator only.
It is not an independently developed candidate and must never be presented as
one.

`hoct-multibackbone-probe-v1` is not an exact public-code replica. It keeps the
public pipeline's detector nodes because that detector has 0.969 annotated-node
recall on the four candidate-disjoint validation movies and the independent
Spotiflow replacements were materially worse. From those fixed nodes onward it
uses a different association system:

- official MIT-licensed HOCT `general_v1` and `ctc_v0` edge-centric backbones;
- two independently fitted 289-parameter Biohub probes;
- twelve bounded single/backbone-ensemble variants;
- a probability-aware linker selected by worst-embryo gain on two movies; and
- a disjoint two-movie acceptance read after the full choice is frozen.

The submission kernel refuses to emit a candidate if its edge set is identical
to the public comparator. Therefore a promoted CSV must differ through learned
HOCT association decisions, not through copied public thresholds or metric
artifacts.

## Honest limitation

The current candidate is independently learned for lineage association, but it
is not yet a fully owned end-to-end detector. Its node coordinates inherit the
strong public detector and postprocessor. This is deliberate evidence-based
risk control: the tested 35.5M Spotiflow detector lost 0.191 annotated recall,
and its synthetic-only fine-tune lost 0.877 versus the public detector. A later
positive-unlabeled real-image detector adaptation remains the higher-upside
end-to-end lane; replacing nodes without a clean recall win is prohibited.

Public leaderboard feedback is not used to select HOCT models, blends,
thresholds, or promotion. Explicit metric-hack notebooks remain excluded.
