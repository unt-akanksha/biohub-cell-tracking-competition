# Existing-division scope audit: a limitation, not a removal policy

Full60source-movie audit80227 TERMINAL0,202.937seconds. Every movie reproduces
the earlier patched official divisionTP/FP/FNcounts. Prediction-only eligibility
was frozen before GT; only aggregate class counts are exported, not per-fork
GT labels, oracle graphs or a deployment rule. Four unit tests passed22.58s;
first real smoke98063 passed51.219s, correctly retaining3unscored forks as
unknown rather than negative. No GPU, candidate modification or held-out data.

| Source | TP | FP | Unscored (not negative) | FP in observed consecutive scope |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 4 | 8 | 282 | 8 |
| 6bba | 10 | 31 | 688 | 30 |

The current event head protects every existing division and its incident
edges. Thus38of39verified baseline false divisions are outside its editable
scope even though their cells and adjacent edges are observed/consecutive.
All14true divisions are ALSO in that same observed scope, along with962of970
unscored divisions. Observation/temporal persistence is NOT a discriminating
removal rule:37of38eligible false divisions and all14true divisions have the
tested mother/daughter temporal context. A model would need actual independent
evidence; blanket pruning, treating unscored as negative, or GT-ID routing is
not authorized. The one hard-protected FP has gap/synthetic incidence and is
outside the proposed ordinary observed-edge reconciliation scope.

An independent graph-ID mapping was added for the audit because the scorer's
graph constructor renumbers source IDs. Coordinate/topology values are unchanged,
mapping verified through an audit-only node attribute, and all old counts match.

Full reportSHA`0e781f703c449f8039d1b5fdf048f8b2bca12e0d8f7c1da8ec93b0afad09c9ab`.
Smoke reportSHA`7b4e696a1c3fa196071338b47eb8466674e968841e0d91b552d1e16b77fb4266`.
Audit scriptSHA`f94aa2a843fece1b1ffa0edc53327aaf99c4ec4ce324b730aed951ed5ed7b40f`;
helperSHA`1c101aa89f27db4a1c0e04bf92676bdfcbce26a838ad854c91e921a2003285f2`.
