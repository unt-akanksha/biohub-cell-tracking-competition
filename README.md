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

The tracker delegates authentication to the installed Kaggle CLI. It never reads or records Kaggle credential files.

