#!/usr/bin/env python
"""Verify v2 LSM consensus, including bounds-provenance accounting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(
    str(ROOT / "scripts/verify-948tta2-lsm-consensus-candidate.py")
)
RUN_ID = "948tta2-lsm-consensus-v2"
_base_verify_candidate = BASE["verify_candidate"]
_base_verify_candidate.__globals__["RUN_ID"] = RUN_ID


def verify_candidate(output_root: Path) -> dict:
    result = _base_verify_candidate(output_root)
    run_stats_path = BASE["unique_file"](output_root, "run_stats.csv")
    with run_stats_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "lsm_consensus_preexisting_out_of_bounds",
        "lsm_consensus_remaining_out_of_bounds",
        "lsm_consensus_out_of_bounds",
    }
    if not rows or not required <= set(rows[0]):
        raise RuntimeError("Bounds-provenance statistics are incomplete")
    preexisting = sum(
        int(float(row["lsm_consensus_preexisting_out_of_bounds"])) for row in rows
    )
    remaining = sum(
        int(float(row["lsm_consensus_remaining_out_of_bounds"])) for row in rows
    )
    introduced = sum(
        int(float(row["lsm_consensus_out_of_bounds"])) for row in rows
    )
    if min(preexisting, remaining, introduced) < 0:
        raise RuntimeError("Bounds-provenance statistics cannot be negative")
    if introduced != 0 or remaining > preexisting:
        raise RuntimeError("Candidate introduced or increased out-of-bounds coordinates")
    return {
        **result,
        "preexisting_out_of_bounds": preexisting,
        "remaining_out_of_bounds": remaining,
        "newly_introduced_out_of_bounds": introduced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(args.output_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
