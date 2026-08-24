# Exact evaluation reports

This directory may contain compact, content-addressed exact-report cores and their evidence envelopes. A core is authoritative only when its scorer, environment, policy, manifest, round-trip inventories, and four producer members re-resolve to a completed aggregate exact-evaluation event in the local append-only ledger.

Evidence kinds are deliberately distinct:

- `synthetic_fixture` proves deterministic implementation behavior only.
- A remote CPU result is provisional until Plan 02-04 reconciles it locally; it is not an exact report or promotion evidence.
- `official_data_control` is a reconciled truth/self or controlled-reference infrastructure check, never a learned candidate or submission candidate.
- `model_candidate` requires real evidence-eligible reciprocal producers and is the only learned-candidate report kind.

Runtime, peak memory, creation time, and optional public-score presentation metadata belong only in the envelope. Public leaderboard score is never a canonical comparison or promotion input. GEFF graphs, prediction CSV files, model weights, downloaded data, and other large artifacts must not be committed here.
