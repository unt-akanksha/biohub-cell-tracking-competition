# API Coverage — Kaggle CLI Read/Launch Surface

> Full coverage by default for the Phase 1 competition-control capability. Opt-outs are explicit and reasoned.

| capability | decision | reason |
|---|---|---|
| quota read | INTEGRATE | Required for the 8-hour reserve guard |
| personal submissions list | INTEGRATE | Required for score and daily submission accounting |
| public leaderboard show/download | INTEGRATE | Required for current rank and competitive context |
| competition topics list/show | INTEGRATE | Required for discussion and organizer-change monitoring |
| competition kernels list | INTEGRATE | Required for current notebook discovery |
| kernel pull metadata/source | INTEGRATE | Required for opt-in static provenance audit |
| kernel status | INTEGRATE | Required for conflicting active-run detection |
| kernel push | OPT-OUT | Phase 1 creates and tests authorization only; no real GPU launch is permitted during control-plane setup |
| competition submit | OPT-OUT | Submission belongs to Phase 5 and must consume a promoted artifact |
| dataset create/version | OPT-OUT | Weight packaging belongs to Phase 5 |
| kernel delete | OPT-OUT | Destructive operation is not required by this project |
| competition team operations | OPT-OUT | Team membership changes require explicit user action outside the control plane |
