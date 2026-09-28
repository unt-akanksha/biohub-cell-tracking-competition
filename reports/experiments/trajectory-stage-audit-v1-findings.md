# Where the current candidate loses tracking accuracy

CPU audit of three archived stages on four complete 100-frame movies. All
input files matched the verified terminal backup before labels opened. The
submitted-stage counts exactly reproduce the prior patched official evaluation.
No predictions changed, no failed dense-warp weights included, no parameters fit.

| Stage | Combined score | Raw edge Jaccard | True divisions / annotated |
|---|---:|---:|---:|
| Neural detector/linker + ILP, before public post-processing | .920107 | .920000 | 0 / 5 |
| Public post-processing | .937988 | .915900 | 1 / 5 |
| Submitted trajectory repair | .938392 | .916318 | 1 / 5 |

Post-processing is beneficial overall because it recovers a true division, not
because it uniformly improves ordinary association. Removing it wholesale is
unsupported. Its impact differs materially across movies: .90451 to .94359 on
44b6_12dfb391, but .85058 to .79453 on 44b6_267148e4. These exposed diagnostics
cannot justify an embryo/movie-ID router or a threshold fitted to these cases.

In the final submitted graphs,21true and36false evaluated edges were absent
from the initial ILP edge list. Another2169true and61false edges have the same
endpoint identities as initial ILP edges. Of103missing true edges,52have both
endpoints matched somewhere in the final graph;51lack a matched endpoint.
These are exhaustive provenance categories, not a causal intervention: public
centroid smoothing changes node matching, so an inherited edge's correctness
can change without changing its endpoints. One must not equate all newly
introduced links with mistakes or label every missing endpoint a detector miss.

Source inspection confirms public motion relinking reconstructs the edge list
and supplies a learned-probability bonus only for retained ILP edges. The logged
`motion_relink_replaced_raw_edges` is the size of the replaced list, NOT the
number of biologically different links. Many endpoint pairs remain identical.
This helps explain why training a sharper linker can mainly affect node
retention and may not translate directly into better final lineage decisions.

Next modeling question: discriminate real neural-versus-motion disagreements
with multi-frame trajectory evidence, while preserving validated division/gap
repairs. Train/evaluate on that decision distribution, not just easy synthetic
one-to-one warps. First inventory label-free disagreement coverage and check
that source-only fitting data actually supplies both outcomes. Do not perform
an oracle splice of the audited stages or tune on these known errors.

The two prior missing-parent dropout fits and the prior joint encoder/linker
fit also failed their recorded gates; simply repeating those is not a new plan.
No large model or ensemble is currently qualified by this audit.

Two pure attribution tests pass (exhaustive missing-edge categories and duplicate
match rejection). Audit runtime36.22s. Official scorer commit
075fc5f5a52d11077f9dc2b074644618f26939e2. Full artifact:
`trajectory-stage-audit-v1-result.json`, SHA-256
`80d98eb39a498b8ad065d04d83a26c788c4f4e3d6358a2df052c5fa146eb07ca`.
No GPU usage, remote mutation, new submission or quality promotion in this audit.
