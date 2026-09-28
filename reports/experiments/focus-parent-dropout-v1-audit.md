# Fitting-only parent-dropout audit passed

Completed all 12 fitting movies in 75.062 CPU seconds. Original raw feature
packets were hash-checked and remained unchanged. Diagnostics, source selection
and target movies were not augmented or opened by the audit. No GPU used.

- 1,121 eligible augmented pairs, all retaining at least one source candidate.
- 1,136 synthetic known-null labels, geometrically certified by the conservative
  14-micron removal rule and original <=7-micron parent matching contract.
- 161 existing known-null labels retained.
- 9,539 real-parent labels retained after source reindexing.
- Removed-parent labels without the conservative absence guarantee became
  unknown rather than fabricated negatives.

The minimum-100-pair/minimum-100-null feasibility gate passed. This justifies a
small training smoke, not a quality claim or submission. The planned combined
original/augmented inventory has 20,293 parent and 1,458 null labels; any loss
weight must use these fitting counts, not diagnostic outcomes.

Audit JSON SHA-256:
`dd75e5643a84645e0bef2b6d7a3f80d09347b646fdcb61794d887437ca115225`.

Smoke v1 upload was rejected with HTTP400, and the kernel status returned404:
no run existed. Its notebook was1,337,539bytes, including a1,008,208byte audit.
Source-size limit is the suspected upload cause; server response exposed no
specific reason. A separate v2 packages the same audit with lossless gzip/base64.
Tests require byte-identical decoded runtime and notebook size under950,000bytes.
No experiment, budget, label or validation semantics changed in the packaging retry.
