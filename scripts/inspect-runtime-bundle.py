from __future__ import annotations

import argparse
import json

from biohub_tracker.acceptance import inspect_runtime_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_runtime_bundle(args.bundle),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
