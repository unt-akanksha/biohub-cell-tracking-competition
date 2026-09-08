#!/usr/bin/env python
"""Submit one externally verified v2 LSM-consensus candidate exactly once."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(
    str(ROOT / "scripts/submit-948tta2-lsm-consensus-candidate.py")
)
RUN_ID = "948tta2-lsm-consensus-v2"
_main = BASE["main"]
_main.__globals__["RUN_ID"] = RUN_ID


if __name__ == "__main__":
    _main()
