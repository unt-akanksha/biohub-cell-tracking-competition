from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.temporal_contrastive.zebrahub_external import (  # noqa: E402
    build_temporal_patch_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a provenance-bound public ZebraHub temporal patch shard"
    )
    parser.add_argument("--source", choices=("ZSNS004", "ZSNS005"), required=True)
    parser.add_argument("--timepoint", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44004)
    parser.add_argument("--max-sources", type=int, default=64)
    parser.add_argument("--max-targets", type=int, default=96)
    parser.add_argument("--candidate-radius-um", type=float, default=32.0)
    args = parser.parse_args()
    manifest = build_temporal_patch_shard(
        args.source,
        csv_timepoint=args.timepoint,
        cache_root=args.cache_root.resolve(),
        output_path=args.output.resolve(),
        seed=args.seed,
        max_sources=args.max_sources,
        max_targets=args.max_targets,
        radius_um=args.candidate_radius_um,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
