"""Measure normalized source overlap among public Kaggle notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_code_lines(path: Path) -> frozenset[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"notebook has no cell list: {path}")
    lines: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(map(str, source))
        if not isinstance(source, str):
            raise ValueError(f"code cell source is malformed: {path}")
        for line in source.splitlines():
            normalized = line.strip()
            if normalized:
                lines.add(normalized)
    if not lines:
        raise ValueError(f"notebook contains no nonempty code: {path}")
    return frozenset(lines)


def line_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def audit_lineage(paths: Sequence[Path], *, cluster_threshold: float = 0.70) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two notebooks are required")
    if not 0.0 <= cluster_threshold <= 1.0:
        raise ValueError("cluster_threshold must lie in [0, 1]")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("notebook paths must be unique")
    line_sets = {path: normalized_code_lines(path) for path in resolved}
    records = [
        {
            "path": path.as_posix(),
            "source_sha256": sha256_file(path),
            "unique_code_lines": len(line_sets[path]),
        }
        for path in resolved
    ]
    pairs = []
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            score = line_jaccard(line_sets[left], line_sets[right])
            pairs.append(
                {
                    "left": left.as_posix(),
                    "right": right.as_posix(),
                    "line_jaccard": score,
                    "shared_lineage": score >= cluster_threshold,
                }
            )
    return {
        "schema_version": 1,
        "method": "trim nonempty code lines, deduplicate within each notebook, Jaccard overlap",
        "cluster_threshold": cluster_threshold,
        "notebooks": records,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", type=Path, nargs="+")
    parser.add_argument("--cluster-threshold", type=float, default=0.70)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_lineage(args.notebooks, cluster_threshold=args.cluster_threshold),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
