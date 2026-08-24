# Walking Skeleton — Biohub Competition Control Plane

**Phase:** 1  
**Generated:** 2026-08-23

## Capability Proven End-to-End

A competitor can run one terminal command that reads authenticated Kaggle state, writes an atomic timestamped snapshot, applies the competition policy registry, and renders the current session status without consuming GPU.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Python standard-library-first package with `argparse` CLI | Works locally and on Kaggle without a service runtime |
| Data layer | Atomic JSON snapshots plus append-only JSONL experiment events | Offline, diffable, hashable, and sufficient for one competitor |
| Auth | Delegate authentication to the installed Kaggle CLI | The project never reads or stores Kaggle credentials |
| Deployment target | Documented local CLI plus fixture-driven tests; later copied into Kaggle notebooks | The control plane runs on the user's machine, while model jobs run on Kaggle |
| Directory layout | `src/biohub_tracker`, `tests`, `config`, `policies`, `experiments`, `reports`, `.biohub` | Separates immutable policy/evidence from ignored runtime cache |

## Stack Touched in Phase 1

- [ ] Project scaffold — package, test runner, lint-compatible layout, and local command.
- [ ] Routing — CLI subcommands for watch, experiments, progress, guard, and launch authorization.
- [ ] Persistent data — real atomic snapshot write/read and append-only ledger write/read.
- [ ] UI — terminal command and Markdown/JSON status output.
- [ ] Deployment — documented local invocation that exercises the authenticated Kaggle read path.

## Out of Scope

- Training, inference, kernel push, and competition submission.
- Hosted tracker or web dashboard.
- Exact Biohub model scoring and held-out split construction.
- Cloud GPU backend.

## Subsequent Slice Plan

- Phase 2: score and diagnose a candidate with pinned exact held-out validation.
- Phase 3: calibrate and decide the ZebraHub selective-SSM candidate.
- Phase 4: train reciprocal stronger spatiotemporal models.
- Phase 5: run offline inference, submit, and select finals.

