# FOCUS-specific label inventory: more coverage needed

CPU audit completed12.204s; seven label-policy tests passed, nine including
the subsequent training-cache split contract. No optimizer or GPU work in
this inventory. Both original complete training control scores replayed
exactly before their official node match maps were consumed.

| Role, across two movies | Known parent | Known parent absent | Unknown targets |
| --- | ---: | ---: | ---: |
| Fitting frames0..69 | 708 | 3 | 12,906 |
| Diagnostic frames80..99 | 112 | 0 | 1,515 |
| Embargo transitions | 108 | 1 | 949 |

Additional birth/nonadjacent-parent/ambiguous cases are also ignored; see JSON
for full counts. Diagnostic frames are disjoint from adaptation-fitting
frames, but both movies were in the original checkpoint's training pool.
These are not independent-validation examples.

The available cache is heavily skewed toward present-parent supervision. It
does not support a reliable missing-parent diagnostic or an automatic fit.
Unknown target cells must not become null labels. Store these audited labels
for reproducibility; broaden training-only proposal coverage before choosing
the actual adaptation experiment. No prediction, source or target score changed.

Correction of previous explanation: the frozen training notebook already
detects cells, matches detections to GT, indexes features at predicted native
detector positions and builds matched edge targets. The runtime's guarded
matcher only enforces an attention-size limit; it returns those detections
unchanged. Known missing-parent null supervision is also already installed.
Thus 'trained only on annotated nodes' was incorrect. The new possible
intervention is FOCUS-specific proposal-domain adaptation, not predicted-node
training in general. Whether that intervention helps remains untested.

Inspected frozen training notebookSHA:
aabe6fbb399ddd1cb8a935ca04d0b0bcc610421799e51bb6d2b5b2d293ab919a.
Bundled training scriptSHA:
83e3f30a3313ea452b5072b118bfd06539302cf6886e317c542a602deb27d281.
Bundled run_associationSHA:
fb07930e28545d3ff0286ec3fcf96b4814f3c4c190624039d03ea717c59ca289.

Authoritative inventory:`focus-predicted-node-labels-v1-result.json`, SHA256:
1ea573dfaa1e7075d4c2c486bcca8c7bd27e592ce233dc4afdcc844d8dedbbd1.
Label artifacts stay in ignored cache; source/design hashes remain frozen.
