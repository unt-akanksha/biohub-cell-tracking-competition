# Biohub competition tracker

This repository is an offline-first competition control plane for Kaggle's Biohub Cell Tracking During Development challenge. Phase 1 provides the read-only competition watch, immutable experiment tracking, and fail-closed GPU authorization.

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
biohub progress
```

Source classification is not score reproduction. `research_candidate` means no known exploit was found in the reviewed source; only an exact post-patch rerun can establish `reproduced_post_patch` evidence. Unknown, changed, or stale-source notebooks remain review items and cannot be treated as clean submissions.

The watch rewrites the compact current projections at `reports/competition-status.md` and `reports/competition-status.json`, while preserving every raw snapshot under `.biohub/snapshots/`.

The tracker delegates authentication to the installed Kaggle CLI. It never reads or records Kaggle credential files.

Every GPU run must use the register â†’ preflight â†’ authorize â†’ explicit execute â†’ watchdog â†’ terminal workflow in `docs/KAGGLE_RUNBOOK.md`. Direct `kaggle kernels push` is outside project policy. Stop Kaggle GPU work at the protected 8.00-hour reserve and hand the registered manifest to the cloud backend instead.
