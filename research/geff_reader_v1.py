"""Minimal reader for the competition's ground-truth .geff graphs.

The Kaggle stack reads these with tracksdata; we only need the arrays, and the
files are plain zarr v3 -- one chunk per array, little-endian raw bytes behind a
zstd codec. Decoding them directly keeps this analysis on CPU with no Kaggle
dependency, which matters because the point is to study the labels, not to run
the pipeline.

Coordinates are stored as voxel indices. The per-axis `scale` in the root
metadata converts them to microns, which is the unit every gate in the pipeline
is expressed in, so the conversion has to happen here or nothing is comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zstandard


def _read_array(path: Path) -> np.ndarray:
    meta = json.loads((path / "zarr.json").read_text())
    dtype = np.dtype(meta["data_type"]).newbyteorder(
        "<" if meta["codecs"][0]["configuration"]["endian"] == "little" else ">"
    )
    shape = tuple(meta["shape"])
    codecs = [c["name"] for c in meta["codecs"]]
    # Single chunk: the chunk grid always matches the shape in these files.
    chunk = path / "c"
    for _ in shape:
        entries = sorted(p for p in chunk.iterdir() if p.name.isdigit())
        chunk = entries[0]
    raw = chunk.read_bytes()
    if "zstd" in codecs:
        raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 28)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def read_geff(path: str | Path) -> dict:
    """Return nodes (id, t, z, y, x in microns) and directed edges."""
    path = Path(path)
    root = json.loads((path / "zarr.json").read_text())
    scales = {a["name"]: float(a.get("scale") or 1.0) for a in root["attributes"]["geff"]["axes"]}

    node_ids = _read_array(path / "nodes" / "ids")
    props = {
        name: _read_array(path / "nodes" / "props" / name / "values")
        for name in ("t", "z", "y", "x")
    }
    edges = _read_array(path / "edges" / "ids")
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)

    return {
        "node_ids": node_ids,
        "t": props["t"].astype(np.int64),
        # Microns, matching the units every pipeline gate uses.
        "z": props["z"].astype(np.float64) * scales["z"],
        "y": props["y"].astype(np.float64) * scales["y"],
        "x": props["x"].astype(np.float64) * scales["x"],
        "edges": edges.astype(np.int64),
        "scales": scales,
    }
