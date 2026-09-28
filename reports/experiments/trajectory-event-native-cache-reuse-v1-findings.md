# Native-cache reuse audit, September14,2026

Read-only metadata/schema audit while fixed event fits continue. No data/model
was changed, no source/held labels or learned predictions were loaded for this
audit, and no GPU or image download was launched.

The verified native-correspondence-v2 cache contains2,103packets/1,216,826,026
bytes; its terminalSHA is
`f2861ce9f21307507ed521716ca3bf9a7adf3b76ce9f8216e4407233616c693a`.
Metadata overlap with the immutable fork16 training contracts is:

| Source | Event movies | Native-overlap movies | Event transitions | Matching cached transitions | Matching-role packets |
| --- | ---: | ---: | ---: | ---: | ---: |
|44b6|13|13|1,223|194|195optimization|
|6bba|47|47|4,557|673|675optimization|

Parent-time native transition is compared with event child-time minus one.
867/5,780event transitions overlap, about15.0%. The overlapping native packets
occupy503,141,770compressed bytes; all corresponding movie roles agree as
optimization. Missing transitions are not absent biological links.

`native_correspondence_data_v2.groups` first matches annotated child queries to
image proposals, then `pack_groups` stores only those queries and their candidate
parents. Coordinates are physical absolute ZYX; patch IDs are packet-local, not
public detection IDs. The15-cube image patches use0.8125/1.625/3.25um scales.

Consequently these packets are useful supervised correspondence examples, but
NOT a complete prediction-only feature table for the event graph. Naively joining
them to all event candidates and filling missing entries with zeros would make
feature availability depend on annotation-selected query coverage. It could
create an annotation-availability shortcut and training/deployment mismatch.
No such join or missingness feature is authorized by this audit.

A legitimate dense native-feature extension requires image-derived features at
the event graph's prediction-only nodes/candidate locations, at the required
timepoints, with a source-only encoder contract and fold-safe training. Existing
patches alone cannot establish that coverage. No broad source image download is
justified merely by this inventory; finish the running event-head comparison
before expanding the image collection/model lane.
