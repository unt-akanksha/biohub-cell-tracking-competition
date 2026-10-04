#!/usr/bin/env python
"""Submit a completed kernel to the code competition.

Two things this exists for. Kaggle's CLI reports a bare "400 Bad Request" and
hides the reason; the real message lives in the response body and is worth
seeing ("Notebook is still running", "Did not find provided Notebook Output
File"). And a slug pushed more than once has several versions, only some of
which completed, so the version has to be discovered rather than assumed --
submitting version 1 blindly once tried to submit a run that was still going.
"""
from __future__ import annotations

import argparse
import sys

COMP = "biohub-cell-tracking-during-development"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kernel")
    ap.add_argument("message")
    ap.add_argument("--max-version", type=int, default=6)
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    ref = args.kernel if "/" in args.kernel else f"indarkarhana/{args.kernel}"

    # Highest version first: that is the most recent push, hence the run we
    # actually intend to submit.
    last = ""
    for version in range(args.max_version, 0, -1):
        try:
            response = api.competition_submit_code(
                file_name="submission.csv", message=args.message,
                competition=COMP, kernel=ref, kernel_version=version,
            )
            print(f"SUBMITTED {ref} v{version} -> {response}")
            return 0
        except Exception as exc:  # noqa: BLE001 - surface whatever Kaggle said
            body = ""
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    body = resp.text[:300]
                except Exception:
                    pass
            last = f"v{version}: {type(exc).__name__} {exc} | {body}"
    print(f"SUBMIT FAILED for {ref}; last error {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
