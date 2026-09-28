# Public refresh: September 10, approximately 16:10 UTC

Read-only Kaggle dateRun listing returned two notebooks newer than the previously
reviewed sjlee 12:57 version. Downloaded source and metadata only; no notebook,
bundled payload, model weights or public prediction was executed or adopted.
Known metric-exploit notebook slugs were not downloaded.

- `canhtoanle/biohub-b4-dual-v17`, last run 15:40 UTC. Notebook SHA-256
  `990535244994e532f1eaf5891324fcc6128d07db0213aa2221e1147581f7784e`.
  Inspected wrapper uses two existing Pilkwang temporal model datasets, invokes
  a bundled driver, and advertises detector weight 0.25 plus ILP. Metadata enables
  internet and setup requests Torch from an internet package index. Final CSV
  check hard-codes four visible test IDs. These are offline/hidden-coverage
  acceptance concerns, not proof of a metric exploit. The base64/tar driver has
  NOT been decoded or audited; do not certify the full method as clean.
- `dhiaalhemdani/biohub-competition-solution`, last run 15:22 UTC. Notebook SHA-256
  `1792d746c4aceb17f18b9778ad0ee1a549682925d4423df0ae062d5e562360ce`.
  Selected source inspection shows an existing 50-epoch checkpoint, local mode
  and three fixed validation IDs. The image loader unconditionally uses the
  training directory even though a submit-mode ID selector exists. Checkpoint
  loading permits missing/unexpected keys (`strict=False`). Selected settings
  include 0.5 detection threshold, (1,2,2) sampling and image-based gap closing.
  No clean, independently validated improvement is established by this inspection.

Kaggle discussion search surfaces “Public Notebook Rankings Need a Metric
Refresh,” FOCUS3D and detector-overfitting topics. The direct discussion page
returned no readable body; snippets have inconsistent crawl ages. This refresh
does NOT claim individual discussion threads were fully read or verified.

Opened the author's July working note:
https://pilkwangkim.github.io/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs/
It discusses learned temporal graphs, sparse objectives and image-supported
repair, but also reports public-score-driven postprocessing comparisons. Treat
ideas as hypotheses; its historical metric description is not the authoritative
patched scorer. No reported setting or score was used for model selection here.

Source cache: `.biohub/cache/public-audit-20260910/{canhtoanle-1540,dhiaalhemdani-1522}`.
Neither notebook is promoted or submission-ready from this source-only refresh.
