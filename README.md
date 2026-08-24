# Biohub competition tracker

This repository is an offline-first competition control plane for Kaggle's Biohub Cell Tracking During Development challenge. Phase 1 provides a read-only competition watch; later slices add immutable experiment tracking and fail-closed GPU authorization.

Install locally with:

```powershell
python -m pip install -e ".[dev]"
```

Run the deterministic fixture watch:

```powershell
biohub watch --fixture-dir tests/fixtures/kaggle
```

Live Kaggle access is always explicit:

```powershell
biohub watch --live
```

At the beginning of every work session, refresh live evidence and statically audit the current top notebook sources:

```powershell
biohub watch --live --audit-notebook-sources --top 20
```

Source classification is not score reproduction. `research_candidate` means no known exploit was found in the reviewed source; only an exact post-patch rerun can establish `reproduced_post_patch` evidence. Unknown, changed, or stale-source notebooks remain review items and cannot be treated as clean submissions.

The watch rewrites the compact current projections at `reports/competition-status.md` and `reports/competition-status.json`, while preserving every raw snapshot under `.biohub/snapshots/`.

The tracker delegates authentication to the installed Kaggle CLI. It never reads or records Kaggle credential files.
