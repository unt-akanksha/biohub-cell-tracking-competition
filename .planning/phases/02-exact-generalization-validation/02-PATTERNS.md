# Phase 2: Exact Generalization Validation — Pattern Map

**Mapped:** 2026-08-24  
**Scope:** Repository-local implementation patterns only  
**GPU used:** 0 hours

## Mapping Conclusion

Phase 2 should extend the Phase 1 control-plane architecture, not introduce a
second framework. The repository's stable pattern is:

1. immutable, frozen dataclasses at module boundaries;
2. explicit `from_dict` validation and deterministic `to_dict` projection;
3. semantic SHA-256 over compact canonical JSON;
4. immutable evidence artifacts and replaceable Markdown/current-state views;
5. fail-closed validation before an append-only ledger event;
6. a single `argparse` command tree with thin handlers in `cli._main`;
7. fixture-first pytest tests using `tmp_path`, direct function calls, and CLI
   smoke tests through `main([...])`.

The new graph/scorer dependencies have no existing repository analog. They
should be isolated behind narrow adapters and imported lazily so the control
plane and `biohub --help` remain usable when the CPU evaluation environment is
not installed.

## Intended Data Flow

```text
official-scorer.lock.json -> scorer_lock.verify()
                                      |
mounted train root -> manifests.build()/verify() -> reciprocal manifest + hash
                                      |
baseline/candidate GEFF inventories --+--> graphs.validate_complete_set()
                                             |
                                             v
                              submission_io.project_and_roundtrip()
                                             |
                                             v
                          evaluation.evaluate_exact() per sorted movie
                                             |
                       +---------------------+---------------------+
                       v                     v                     v
                 official counts      diagnostics rows      comparison/bootstrap
                       +---------------------+---------------------+
                                             v
                       immutable exact report core + evidence envelope
                                             |
                                             v
                           promotion.evaluate_policy() -> ledger event
                                             |
                                             v
                              progress JSON/Markdown projection
```

All paths to policy application must pass through scorer, manifest, exact
coverage, graph-integrity, submission-round-trip, and finite-metric checks. A
diagnostic quantity must never flow back into an organizer score component.

## Existing Patterns to Preserve

### Canonical identity and immutable publication

`io.py` already defines the semantic byte representation and hashing boundary:

```python
# src/biohub_tracker/io.py:16-26
def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
```

New manifests, locks, policies, report cores, and inventory records should use
these helpers after normalizing all exact numeric values to canonical decimal
strings. Immutable files should use `atomic_write_json`; current projections may
use `atomic_replace_json`/`atomic_replace_text`. The distinction is established
at `src/biohub_tracker/io.py:37-58` and `src/biohub_tracker/io.py:61-81`.

The closest self-hash pattern is `PreflightReport`:

```python
# src/biohub_tracker/preflight.py:142-160
def unsigned_dict(self) -> dict[str, Any]: ...
def computed_sha256(self) -> str:
    return sha256_bytes(canonical_json_bytes(self.unsigned_dict()))

def write_preflight_report(path, report):
    if not secrets.compare_digest(report.report_sha256, report.computed_sha256()):
        raise PreflightError("preflight report hash is invalid")
    return atomic_write_json(path, report.to_dict())
```

For semantic identity, volatile evidence metadata must be excluded. The quota
snapshot already demonstrates that rule by removing `captured_at` before hashing
(`src/biohub_tracker/guard.py:196-207`). Use the same separation for manifest and
exact-report cores: timestamps, runtime, and peak memory belong in an envelope
that references the core hash.

### Frozen typed boundaries and reason-coded decisions

`GuardDecision` is the nearest pattern for a deterministic gate:

```python
# src/biohub_tracker/guard.py:90-119
@dataclass(frozen=True)
class GuardDecision:
    run_id: str
    authorized: bool
    reason_codes: tuple[str, ...]
    ...
    def to_dict(self) -> dict[str, Any]: ...
```

Phase 2 types should likewise be frozen and serialize tuples as sorted JSON
arrays. Promotion should return a decision object with a stable state and ordered,
deduplicated reason codes, following `evaluate_guard`'s accumulation and
`tuple(dict.fromkeys(reasons))` at `src/biohub_tracker/guard.py:210-260`.

### Append-only evidence and deterministic projection

The ledger validates the entire event stream under an exclusive lock and appends
one canonical JSON line with an `fsync`:

```python
# src/biohub_tracker/ledger.py:417-427
def append(self, event: ExperimentEvent) -> None:
    ...
    events = self._decode(existing_bytes)
    validate_transition(events, event)
    payload = canonical_json_bytes(event.to_dict()) + b"\n"
    with self.path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
```

Exact evidence should be persisted first, hashed, and only then referenced from a
ledger event. Progress should remain a deterministic projection, like
`render_progress_json` and `write_progress_reports` at
`src/biohub_tracker/progress.py:114-131` and
`src/biohub_tracker/progress.py:208-216`.

### Test conventions

Tests favor small synthetic evidence, direct construction, and explicit failure
assertions. `tests/test_preflight.py:21-32` builds a complete hashed report in a
`tmp_path`; `tests/test_io.py:8-30` verifies concurrent immutable publication;
`tests/test_ledger.py:80-92` proves rejected operations do not change bytes; and
`tests/test_ledger.py:152-164` provides the current minimal exact-metric fixture.
Phase 2 tests should follow these patterns and assert both the exception/reason
code and unchanged output/ledger bytes on failure.

## File-by-File Mapping

### `config/official-scorer.lock.json`

- **Role:** Immutable declaration of organizer repository/commit, patch commit,
  critical source hashes, dependency/environment identity, license, constants,
  adapter schema, and frozen fixture-result hash.
- **Flow:** Loaded by `scorer_lock.py`; verified before data or candidate graphs
  are opened; its semantic SHA-256 enters every manifest-independent exact report.
- **Closest analog:** `config/competition.json` is a checked-in JSON authority
  (`config/competition.json:1-27`), while `PreflightReport.unsigned_dict()` and
  `computed_sha256()` provide the self-hash shape
  (`src/biohub_tracker/preflight.py:142-154`).
- **Reuse:** `canonical_json_bytes`, `sha256_bytes`, `sha256_file`, and
  `secrets.compare_digest` as used in preflight validation
  (`src/biohub_tracker/preflight.py:182-204`).
- **Hazards:** The research and AI-SPEC use both `official-baseline.lock.json` and
  `official-scorer.lock.json`; choose the latter everywhere. A repository commit
  alone is insufficient because upstream `tracksdata@main` floats. Retrieval time
  and local checkout path must not affect semantic identity. Do not claim the
  unpublished private Kaggle container is byte-identical.

### `config/evaluation-policy.json`

- **Role:** Versioned non-model evaluation constants: axes/scale, match threshold,
  diagnostic bins, bootstrap seed/count, report schema, and tolerances used outside
  the official scorer.
- **Flow:** Validated once, hashed, then consumed by manifests, diagnostics,
  comparison, and evaluation; its hash enters the report core.
- **Closest analog:** Competition policy fields are projected and hashed together
  in `collect_snapshot` (`src/biohub_tracker/watch.py:264-291`).
- **Reuse:** `provenance.load_policy(path)` supplies the minimal JSON-object loader
  (`src/biohub_tracker/provenance.py:49-54`), but Phase 2 needs strict required-key,
  type, finite-number, and unknown-key validation.
- **Hazards:** Do not duplicate organizer semantics as tunable policy. The 7 µm
  threshold and official scale are locked provenance; bootstrap and density-bin
  choices are project policy. Freeze bin boundaries on the training side, never
  after seeing held-out candidate results.

### `config/promotion-policy.json`

- **Role:** Versioned hard integrity gates and soft scientific thresholds, with
  exact report field names and an intrinsic semantic hash.
- **Flow:** `promotion.py` validates policy and report identities, returns
  `promote`, `review_required`, or `reject` plus reason codes, then appends hashes
  to the ledger.
- **Closest analog:** `evaluate_guard(...) -> GuardDecision` is the deterministic
  policy engine (`src/biohub_tracker/guard.py:210-260`); `decision_payload` is the
  existing ledger bridge (`src/biohub_tracker/ledger.py:615-624`).
- **Reuse:** Canonical decimal strings via `ledger.decimal_text`
  (`src/biohub_tracker/ledger.py:142-150`) and ordered reason codes via the guard.
- **Hazards:** Existing decision choices omit `review_required` in both CLI
  (`src/biohub_tracker/cli.py:124-131`) and ledger validation/payload construction
  (`src/biohub_tracker/ledger.py:306-316`, `615-624`). Update all three together.
  Public score must not be a policy input, even though terminal events retain it
  separately at `src/biohub_tracker/ledger.py:552-570`.

### `vendor/kaggle-cell-tracking-competition/`

- **Role:** Exact checkout/submodule of organizer source at
  `075fc5f5a52d11077f9dc2b074644618f26939e2`; scorer execution authority, not a
  place for local patches.
- **Flow:** Verified by `scorer_lock.py`, then imported by the scorer adapter.
- **Closest analog:** No vendored dependency exists. The nearest external-source
  audit hashes stable relative names plus bytes in sorted order
  (`src/biohub_tracker/provenance.py:57-101`).
- **Reuse:** The relative-name-plus-NUL-plus-content hashing convention is useful
  for tree identity, but implement a general binary tree hash rather than reuse
  `audit_source_tree`, which deliberately filters to text suffixes.
- **Hazards:** `pyproject.toml` currently has no runtime dependencies
  (`pyproject.toml:5-13`). An eager top-level `tracking_cellmot`, GEFF, Polars,
  SciPy, NumPy, or Torch import would break all current CLI/tests in the base
  environment. Pin TracksData separately, preserve the organizer checkout clean,
  verify `module.__file__` is inside the verified checkout, and avoid committing
  generated vendor caches or weights. Decide submodule versus ignored checkout
  explicitly; the current `.gitignore` has no `vendor/` rule.

### `src/biohub_tracker/scorer_lock.py`

- **Role:** Parse/validate the scorer lock, hash critical checkout files and the
  environment lock, confirm imports resolve to the checkout, and expose only the
  pinned organizer `evaluate`, `per_sample_metrics`, `summarise`, and
  `node_recall` API.
- **Flow:** First gate for `scorer verify` and every exact evaluation. Returns a
  frozen verified provenance object used by `evaluation.py`.
- **Closest analog:** `validate_preflight` loads, reconstructs, verifies a self
  hash, checks freshness/identity, and re-hashes evidence before returning a typed
  report (`src/biohub_tracker/preflight.py:163-215`).
- **Reuse:** `sha256_file`, `canonical_json_bytes`, `sha256_bytes`,
  `atomic_write_json`, and `workspace_file` only for lock/evidence files within the
  repository.
- **Hazards:** Do not call or reimplement `evaluate_datasets`; do not accept a
  checkout just because `git rev-parse` matches; verify critical file bytes and
  imported module paths. Local scorer verification should be deterministic and
  offline by default; `--live` may compare upstream provenance but cannot mutate
  the lock. Load heavy dependencies inside verification/evaluation functions.

### `src/biohub_tracker/manifests.py`

- **Role:** Discover paired Zarr/GEFF samples, derive official embryo identity,
  validate metadata, create the two reciprocal complete-movie folds, perform
  overlap/coverage audits, and publish a semantic self-hashed manifest.
- **Flow:** Mounted data root -> sorted sample records -> reciprocal membership ->
  overlap audit -> immutable manifest. Training/evaluation/report events consume
  only the manifest hash and declared memberships.
- **Closest analog:** `PreflightReport` is a frozen self-hashed evidence document
  (`src/biohub_tracker/preflight.py:104-160`); `collect_snapshot` shows ordered
  collection followed by one content hash (`src/biohub_tracker/watch.py:224-292`).
- **Reuse:** Canonical JSON/hash and immutable publication. Use the snapshot rule
  that evidence time is distinct from state identity
  (`src/biohub_tracker/guard.py:196-207`).
- **Hazards:** `canonical_json_bytes` sorts object keys, not list members; sort
  every sample/fold/membership array explicitly. `workspace_file` rejects external
  mounts and directories (`src/biohub_tracker/ledger.py:159-167`), so do not use it
  for `--data-root`; establish containment under the resolved supplied root and
  store portable relative paths. Never hard-code 199 movies or fall back to a
  random split. Hash Zarr metadata used for scoring, not all ~88 GB of chunks, and
  label a remote inventory fingerprint distinctly from a content hash.

### `src/biohub_tracker/graphs.py`

- **Role:** Load GEFF through the pinned stack and enforce project integrity:
  schema, finite/integer submission fields, bounds, endpoints, time direction,
  degree, fork, cycle, duplicate, sentinel, and manifest identity checks.
- **Flow:** Manifest movie record + prediction/truth path -> validated graph value
  or deterministic hard failure. It must not silently repair candidates or alter
  organizer scoring semantics.
- **Closest analog:** No graph-domain analog exists. Structurally,
  `PreflightCheck.create` validates allowed states and requires hashed evidence
  before constructing a frozen value (`src/biohub_tracker/preflight.py:46-84`),
  while `GuardInputError` carries a stable reason code
  (`src/biohub_tracker/guard.py:18-21`).
- **Reuse:** Define a Phase 2 base error derived from `ValueError` with a reason
  code, frozen validation-result dataclasses, sorted issue lists, and finite
  Decimal/number validation.
- **Hazards:** Some rejected topologies are tolerated/ignored/capped upstream;
  label these checks as project integrity policy, not organizer behavior. The
  organizer scorer mutates prediction match attributes, so validation and every
  independent evaluation/diagnostic require fresh graph loads/copies. Do not load
  image volumes; read only metadata needed for shape/scale.

### `src/biohub_tracker/submission_io.py`

- **Role:** Apply the pinned GEFF->CSV integer-coordinate projection, validate the
  exact CSV schema/coverage, rebuild GEFF, and prove semantic graph and metric
  parity after node-ID remapping.
- **Flow:** Valid native GEFF -> canonical submission-space CSV -> rebuilt GEFF ->
  graph validation -> exact parity evidence. Promotion consumes the rebuilt graph;
  native subvoxel results remain diagnostic only.
- **Closest analog:** `KaggleRunner.download_leaderboard` safely materializes and
  parses an external CSV/ZIP without shell interpolation
  (`src/biohub_tracker/kaggle.py:98-131`); immutable publication comes from
  `atomic_write_json`.
- **Reuse:** `csv` from the standard library, validated argument arrays rather
  than shell strings, sorted output records, SHA helpers, and temporary directories
  that are published only after complete validation.
- **Hazards:** Byte identity and original node IDs are intentionally not preserved;
  compare semantic node/edge multisets through the converter mapping. Validate all
  expected movies as one dataset, exact header/order, row types, consecutive IDs,
  `-1` placeholders, and references. Never invoke converter scripts through an
  interpolated shell command. Keep generated `.csv`/`.geff` artifacts out of Git.

### `src/biohub_tracker/evaluation.py`

- **Role:** Orchestrate the fail-closed exact pipeline, call organizer scoring on
  each complete movie in canonical order, preserve integer sufficient statistics,
  apply official `summarise`, validate a strict report core, and publish the core
  plus a non-semantic runtime envelope.
- **Flow:** Verified scorer + manifest + baseline/candidate round-tripped graphs ->
  paired per-movie official rows -> pooled/by-fold/by-embryo result -> diagnostics
  and comparison -> immutable exact report.
- **Closest analog:** `collect_snapshot` is the current multi-step collector that
  normalizes sources then hashes one result (`src/biohub_tracker/watch.py:187-292`),
  but exact evaluation must reverse its best-effort behavior at lines 203-208 and
  fail immediately. `validate_complete_metrics` is the ledger compatibility
  boundary (`src/biohub_tracker/ledger.py:519-539`).
- **Reuse:** `normalize_exact_values` for the final ledger projection
  (`src/biohub_tracker/ledger.py:507-516`), canonical hashes, immutable output,
  and deterministic sorted iteration.
- **Hazards:** Add a stricter exact-report validator; do not reduce the new schema
  to `validate_complete_metrics`. That existing validator does not reject
  non-finite floats and `canonical_json_bytes` does not set `allow_nan=False`, so
  Phase 2 must reject NaN/Inf before serialization. Preserve integer counts and
  decimal strings, not derived binary floats. Never average movie scores. Separate
  deterministic report core from timestamps/runtime/peak-memory envelope.

### `src/biohub_tracker/diagnostics.py`

- **Role:** Compute endpoint availability, oracle-link ceiling, conditional
  association measures, displacement/density/division strata, and worst-case
  explanatory rows from copied organizer match state.
- **Flow:** Fresh/copy match state + official sufficient statistics + frozen bins
  -> reconciled non-authoritative diagnostics embedded in the exact report.
- **Closest analog:** `progress._summarize_state` derives a read-only view from
  authoritative events without modifying them (`src/biohub_tracker/progress.py:38-89`).
- **Reuse:** Pure functions, mappings in/JSON-compatible mappings out, stable sort
  order, and explicit field names. Markdown escaping is already centralized for
  projections at `src/biohub_tracker/progress.py:17-20`.
- **Hazards:** Diagnostics cannot replace official edge/division counts. Any
  reproduction of private organizer helper behavior must be labeled
  non-authoritative and reconcile totals against organizer output. Define no-event
  values explicitly rather than emitting NaN. Density boundaries must come from
  the training side and already be frozen in policy/manifest.

### `src/biohub_tracker/comparison.py`

- **Role:** Require identical evidence identities, pair baseline/candidate rows by
  sample ID, compute component deltas and worst movies, and perform the seeded
  embryo-stratified paired movie bootstrap.
- **Flow:** Two exact per-movie tables with identical scorer/manifest/policy and
  coverage -> paired deltas -> deterministic interval/probability rows -> report.
- **Closest analog:** `render_progress_json` sorts reconstructed run summaries
  before selection/projection (`src/biohub_tracker/progress.py:114-131`), and
  `snapshot_hash` sorts active kernels before hashing
  (`src/biohub_tracker/guard.py:196-207`).
- **Reuse:** Explicit sort keys, Decimal string normalization, and pure output
  mappings. Treat identical inputs and seed as a byte-reproducibility test.
- **Hazards:** Pair on sample ID and reject mismatched identities before any math.
  Resample complete movies within embryo with the same indices for baseline and
  candidate; never bootstrap edges independently. Use NumPy `Generator(PCG64)`,
  record the seed/repetitions, re-run official aggregation per replicate, and
  preserve the official no-division behavior. Percentiles are diagnostics, not
  proof of population generalization.

### `src/biohub_tracker/promotion.py`

- **Role:** Validate report/policy/scorer/manifest identities, evaluate hard gates
  before soft gates, produce a stable decision record, publish it immutably, and
  append evidence hashes to the existing experiment ledger.
- **Flow:** Strict exact report + frozen policy + candidate run state -> promotion
  decision -> immutable decision JSON -> ledger DECISION -> progress projection.
- **Closest analog:** `evaluate_guard` creates an auditable fail-closed decision
  (`src/biohub_tracker/guard.py:210-260`); ledger transition validation prevents a
  decision before a terminal event (`src/biohub_tracker/ledger.py:251-255` and
  `306-316`).
- **Reuse:** `ExperimentEvent.create`, `Ledger.append`, `decision_payload`, artifact
  hashes, and reason-coded frozen dataclasses.
- **Hazards:** Extend `review_required` consistently through parser choices,
  `validate_transition`, `decision_payload`, `RunState`/progress rendering, and
  tests. Decide explicitly whether policy `reject` is stored as `retire` or becomes
  a new ledger decision; do not overload the lifecycle `REJECTED` event, which is
  only legal from RUNNING (`src/biohub_tracker/ledger.py:294-305`). Existing
  completion requires a positive `actual_runtime_hours` and names quota fields
  (`src/biohub_tracker/ledger.py:552-570`); a zero-GPU CPU evaluation should attach
  to an already completed model run or use a purpose-built evidence event rather
  than falsely recording GPU runtime. Enforce report/policy/hash evidence in this
  module because the generic ledger accepts arbitrary non-public evidence strings.

### `tests/fixtures/metric/graph_specs/`, `csv_roundtrip/`, and `expected/`

- **Role:** Small licensed synthetic/adversarial inputs and frozen authoritative
  expected counts/parity outputs.
- **Flow:** Fixture specs -> graph builders/converters -> organizer adapter ->
  checked-in expected JSON. They exercise code paths without data download or GPU.
- **Closest analog:** Static Kaggle JSON fixtures are supplied by
  `tests/conftest.py:9-17`; test-local builders create minimal complete workspaces
  in `tests/test_cli.py:27-61`.
- **Reuse:** Keep binary/large graphs out of Git; prefer compact declarative JSON
  graph specs plus deterministic builders. Expected JSON should retain TP/FP/FN,
  not only final floats.
- **Hazards:** Preserve BSD attribution for organizer-derived fixtures. The
  pre-patch exploit scorer may be a test-only proof and must not be reachable from
  the normal CLI. A synthetic pass cannot be marked as a real model promotion.

### `tests/test_scorer.py`, `test_manifests.py`, `test_graphs.py`, and `test_submission_io.py`

- **Role:** Wave 0 plus Plans 02-01/02 verification: lock/import integrity,
  organizer parity, adversarial topology, identity/overlap/coverage, graph
  integrity, and submission-space round trip.
- **Closest analog:** Failure tests assert unchanged bytes and stable exceptions in
  `tests/test_ledger.py:80-129`; concurrency/immutability is tested at
  `tests/test_io.py:8-30`.
- **Conventions:** Direct function calls first; CLI smoke separately. Use
  `pytest.mark.parametrize` for topology matrices, `tmp_path` for generated trees,
  exact integer-count assertions, and monkeypatch only import/source boundaries.
- **Hazards:** Tests must work without official 88 GB data. Separate dependency
  parity tests from pure manifest/schema tests, and fail clearly—not silently
  skip—inside the dedicated evaluation environment.

### `tests/test_evaluation.py`, `test_diagnostics.py`, and `test_comparison.py`

- **Role:** Plans 02-03 verification for exact aggregation, fresh-copy semantics,
  diagnostic reconciliation, deterministic report regeneration, paired bootstrap,
  and worst-movie behavior.
- **Closest analog:** `tests/test_progress.py:55-63` verifies deterministic output
  after legal event reordering and safe projection of external text.
- **Conventions:** Construct imbalanced multi-movie/multi-embryo fixtures where a
  simple per-movie mean is provably different; call evaluation twice and compare
  canonical core bytes/hashes; assert no-event outputs are explicit and finite.
- **Hazards:** Do not compare only rounded final scores. Freeze all sufficient
  counts, weights, macro/micro node recall names, no-division aggregation, and
  candidate-minus-baseline direction.

### `tests/test_promotion.py` and `tests/test_exact_cli.py`

- **Role:** Plan 02-04 policy/ledger integration and CPU-only end-to-end command
  verification.
- **Closest analog:** CLI tests call `main([...])`, inspect exit code/output, and
  inject runners rather than spawn subprocesses (`tests/test_ledger.py:56-77`,
  `tests/test_cli.py:64-78`).
- **Conventions:** Test hard rejection before soft gates, deterministic reason
  order, `review_required`, public-score exclusion, hash mismatch, unchanged ledger
  on rejection/error, and one successful immutable report/decision append.
- **Hazards:** The end-to-end synthetic tracer must say `synthetic`; no test fixture
  can authorize a real submission. Assert no Kaggle runner or GPU-launch function
  is called.

### `manifests/reciprocal-embryo-v1.json` and `manifests/README.md`

- **Role:** The JSON is the frozen, self-hashed evaluation identity; the README
  documents how to regenerate/verify it and distinguishes semantic hashes from
  inventory fingerprints.
- **Flow:** Produced only from a mounted official train root by `manifest build`,
  verified by `manifest verify`, then referenced by hash everywhere downstream.
- **Closest analog:** Immutable snapshots are named with time and hash and written
  without overwrite (`src/biohub_tracker/watch.py:295-303`); for manifests, prefer
  a stable versioned name and refuse overwrite if semantic content differs.
- **Hazards:** No real dataset is mounted now, so planning may add schema fixtures
  and README but must not fabricate the production manifest or assert the movie
  count. Generated GEFF/Zarr remain ignored by `.gitignore:16-24`.

### `reports/exact/`

- **Role:** Tracked compact exact report cores/decision summaries when they contain
  no large graphs, plus references to ignored large artifacts.
- **Flow:** `evaluation.py` publishes a content-addressed immutable core and
  envelope; `promotion.py` consumes its hash; progress creates the current view.
- **Closest analog:** Immutable snapshots use `atomic_write_json`
  (`src/biohub_tracker/watch.py:295-303`), while current status/progress reports
  use atomic replacement (`src/biohub_tracker/watch.py:543-553` and
  `src/biohub_tracker/progress.py:208-216`).
- **Hazards:** Never overwrite a report core. Avoid putting timestamps in the core
  filename identity unless the semantic hash remains present. Store no GEFF,
  prediction CSV, model weights, or secrets. The report envelope must not be
  mistaken for the byte-stable core.

## CLI Attachment Map

The parser is built in one function (`src/biohub_tracker/cli.py:67-171`) and
handlers are a top-level `if args.command == ...` chain
(`src/biohub_tracker/cli.py:174-421`). Attach the Phase 2 groups before the current
`return parser` and dispatch them before the final `parser.error`:

| Command | Parser destination | Handler boundary | Primary module |
|---|---|---|---|
| `biohub scorer verify [--lock ...] [--checkout ...] [--live]` | `command=scorer`, `scorer_command=verify` | Load lock, offline verify by default; `--live` only corroborates remote provenance | `scorer_lock.py` |
| `biohub manifest build --data-root ... --output ...` | `command=manifest`, `manifest_command=build` | Discover/validate all samples, create semantic self-hash, immutable publish | `manifests.py` |
| `biohub manifest verify --manifest ... --data-root ...` | same group, `verify` | Recompute manifest/hash/overlap without mutation | `manifests.py` |
| `biohub graph validate --pred-dir ... --manifest ... --fold ...` | `command=graph`, `graph_command=validate` | Require exact expected inventory, then validate each graph in sorted order | `graphs.py` |
| `biohub evaluate exact --baseline-dir ... --candidate-dir ... --truth-dir ... --manifest ... --output-dir ...` | `command=evaluate`, `evaluate_command=exact` | Run full fail-closed pipeline and immutable report publication | `evaluation.py` |
| `biohub promote evaluate --report ... --policy ... --run-id ...` | `command=promote`, `promote_command=evaluate` | Validate identities, decide, publish decision, append ledger evidence | `promotion.py` |

Use `Path` arguments consistently with existing parser declarations such as
`--metrics-report` (`src/biohub_tracker/cli.py:104-110`). Print the authoritative
core/evidence hash in machine-readable JSON. Phase errors should derive from
`ValueError` or a shared caught base; `main` currently converts only
`GuardInputError`, `LedgerError`, `OSError`, `ValueError`, and JSON errors to exit
code 2 (`src/biohub_tracker/cli.py:425-430`). Keep heavy dependency imports inside
the selected command handler or adapter function.

## Cross-Cutting Integration Hazards

1. **Dependency availability:** The base package is stdlib-only. Eager scientific
   imports will regress every Phase 1 command. Use an explicit CPU-eval extra or
   locked environment and lazy adapter imports.
2. **Float/non-finite serialization:** Existing canonical JSON permits Python's
   non-standard NaN tokens and cannot serialize `Decimal` directly. Strictly
   validate finiteness and normalize exact metrics to decimal strings before
   hashing.
3. **List determinism:** JSON key sorting does not sort movie/fold lists. Every
   identity-bearing collection needs a declared canonical sort key.
4. **Directory hashing:** `sha256_file` and `artifact_record` only accept files;
   implement a binary-safe, sorted relative-path tree/Merkle inventory for GEFF
   and metadata directories.
5. **External mounts:** `workspace_file` correctly rejects paths outside the
   repository, but official data may be mounted elsewhere. Data-root containment
   needs its own explicit resolver; evidence published to Git remains inside the
   workspace.
6. **Upstream fail-open behavior:** Do not imitate `collect_snapshot`'s useful
   best-effort status collection. Exact scoring must stop before scoring on any
   missing/unreadable/extra movie or provenance uncertainty.
7. **Graph mutation:** Organizer evaluation writes match attributes. Load/copy
   afresh for baseline, candidate, round-trip parity, and diagnostics.
8. **Submission-space authority:** Native subvoxel scores are diagnostics only.
   Promotion uses integer CSV-round-tripped graphs after semantic parity.
9. **Ledger semantics:** `review_required` is absent, lifecycle rejection is not a
   promotion rejection, and completion fields currently assume positive runtime
   plus quota. Resolve these explicitly rather than fabricating GPU usage.
10. **No real-artifact overclaim:** Synthetic implementation and parity can pass
    before official data/prediction graphs are mounted, but the phase must retain a
    visible manual blocker for a genuine reciprocal candidate report.

## Recommended Implementation Order

1. Add Wave 0 graph specs and exact expected count fixtures.
2. Implement strict shared schemas/numeric validation and `scorer_lock.py` without
   eager scientific imports.
3. Implement manifest identity/coverage, graph validation, and round-trip I/O.
4. Implement sequential exact evaluation and strict report-core validation.
5. Add pure diagnostics and paired comparison/bootstrap over per-movie sufficient
   rows.
6. Implement promotion policy, `review_required`, ledger projection, and progress
   rendering.
7. Attach CLI commands and run the synthetic CPU end-to-end tracer.
8. Keep real reciprocal scoring incomplete until mounted official data and actual
   baseline/candidate prediction graphs are available.

This order respects the Phase 2 fail-closed sequence and keeps every later layer
dependent on already-proven identities rather than circularly validating its own
output.
