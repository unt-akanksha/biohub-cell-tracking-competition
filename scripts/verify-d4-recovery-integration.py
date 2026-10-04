#!/usr/bin/env python
"""Replay the frozen 2026-09-10 recovery inputs through the deployed call path.

This is the test that matters for the integration: it feeds the exact cached
inputs of the scored run into the same `recover()` the kernel will execute, with
the config the kernel will build, and requires the output graphs to be byte
identical to the frozen candidate graphs that produced 0.9493103203140519.

If this passes, the deployed recovery is the scored recovery. It says nothing
about whether that result generalizes.

CPU only, no GPU, no Kaggle, no truth labels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".biohub/cache/public-pruned-track-recovery-v1"
PARENTS = ROOT / ".biohub/cache/public-d4-full-movie-v1-output"
RESULT = ROOT / "reports/experiments/public-pruned-track-recovery-v1-result.json"

_SPEC = importlib.util.spec_from_file_location(
    "public_d4_recovery_v1", ROOT / "research" / "public_d4_recovery_v1.py"
)
_integration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_integration)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_graph(nodes_by_id, edges):
    """Same transform the kernel installs, applied here for parity."""
    out = {"nodes": {}, "edges": []}
    for ident, node in nodes_by_id.items():
        ident = int(ident)
        row = {"node_id": ident, "t": int(node["t"])}
        for axis, bound in (("z", 64), ("y", 256), ("x", 256)):
            value = max(0, int(round(float(node[axis]))))
            if value >= bound:
                raise ValueError("Out-of-volume predicted coordinate")
            row[axis] = value
        out["nodes"][str(ident)] = row
    for edge in edges:
        out["edges"].append(
            {"source_id": int(edge["source_id"]), "target_id": int(edge["target_id"])}
        )
    return out


def main() -> int:
    if not CACHE.exists() or not PARENTS.exists():
        print("Frozen 2026-09-10 cache is not installed; cannot verify.", file=sys.stderr)
        return 2

    # The verbatim, SHA-pinned recovery, loaded exactly as the kernel will load it.
    source = _integration.recovery_source()
    namespace: dict = {}
    exec(compile(source, "public_pruned_track_recovery.py", "exec"), namespace)
    recover = namespace["recover"]

    # The config the kernel builds from notebook globals.
    config = dict(_integration.FROZEN_RESCUE_CONSTANTS)

    # The public linefit smoother, taken from the same frozen postprocess source
    # the scored run used.
    import math

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    sys.path.insert(0, str(ROOT))
    from research.public_d4_preflight import named_definitions

    postprocess = ROOT / ".biohub/cache/public-d4-correction-v1/public-postprocess-d4-corrected.py"
    bundle = ROOT / ".biohub/cache/public-d4-full-movie-v1-bundle/resolved-public-config.json"
    globals_ = json.loads(bundle.read_text())["globals"]
    env = dict(
        globals_,
        np=np,
        math=math,
        linear_sum_assignment=linear_sum_assignment,
        VOXEL_SCALE_UM=(1.625, 0.40625, 0.40625),
    )
    exec(
        compile(
            named_definitions(
                postprocess.read_text(),
                ("edge_distance_um", "_position_um", "motion_relink_edges",
                 "linefit_smooth_output_graph"),
            ),
            "frozen-public-smoothing",
            "exec",
        ),
        env,
    )
    smooth = env["linefit_smooth_output_graph"]

    frozen = json.loads(RESULT.read_text())
    from research.public_d4_full_movie import STEMS
    stems = list(STEMS)
    checked = 0
    failures = []

    for stem in stems:
        for parent in ("original", "corrected"):
            source_dir = PARENTS / f"{stem}-{parent}"
            pre = json.loads((source_dir / "pre-postprocess.json").read_text())
            saved_stats = json.loads((source_dir / "postprocess-stats.json").read_text())
            prediction = json.loads((source_dir / "prediction.json").read_text())
            motion = json.loads((CACHE / f"{stem}-{parent}-motion.json").read_text())

            base = csv_graph(prediction["nodes"], prediction["edges"])
            pre_count = len(base["nodes"]) + int(saved_stats["short_track_nodes_removed"])
            graph, details = recover(base, pre, motion, config, smooth, pre_count)

            expected_path = CACHE / f"{stem}-{parent}-recovery.json"
            expected = json.loads(expected_path.read_text())
            record = frozen["receipt"]["records"][stem][parent]

            ok = graph == expected
            same_counts = (
                details["recovered_components"] == record["recovered_components"]
                and details["added_nodes"] == record["added_nodes"]
                and details["added_edges"] == record["added_edges"]
                and details["node_budget"] == record["node_budget"]
            )
            checked += 1
            status = "OK " if (ok and same_counts) else "FAIL"
            print(
                f"  {status} {stem:16s} {parent:9s} "
                f"components={details['recovered_components']:3d} "
                f"+nodes={details['added_nodes']:4d} +edges={details['added_edges']:4d} "
                f"budget={details['node_budget']:4d} graph_identical={ok}"
            )
            if not (ok and same_counts):
                failures.append((stem, parent, ok, same_counts))

    print()
    print(f"arms checked: {checked}")
    if failures:
        print(f"FAILURES: {failures}", file=sys.stderr)
        return 1
    print("Every arm reproduces the frozen scored candidate graph byte for byte.")
    print(f"Scored pooled (corrected_recovery): {frozen['summaries']['corrected_recovery']['score']}")
    print(f"Recovery source SHA-256 pinned:     {_integration.RECOVERY_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
