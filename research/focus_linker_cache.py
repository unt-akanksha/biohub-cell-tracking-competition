"""Label-free adapter from recovered FOCUS points to the learned linker.

All points are retained, including duplicates and empty frames. FOCUS edges
are deliberately not accepted. Coordinate provenance must be recorded; raw
centroids and postprocessed outputs differ. This does not establish validation
independence for the association weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class FocusLinkerCache:
    node_ids: tuple[int, ...]
    points: np.ndarray
    movie_shape: tuple[int, int, int, int]
    downsample: tuple[int, int, int] = (1, 4, 4)
    coordinate_source: str = 'postprocessed FOCUS proposals'

    def __post_init__(self):
        shape = tuple(self.movie_shape)
        grid = tuple(self.downsample)
        if self.coordinate_source not in {'postprocessed FOCUS proposals', 'raw instance centroids'}:
            raise ValueError('unsupported coordinate provenance')
        if len(shape) != 4 or any(int(v) != v or v <= 0 for v in shape):
            raise ValueError('movie_shape must contain four positive integers')
        if grid != (1, 4, 4):
            raise ValueError('only the audited (1,4,4) linker grid is supported')
        points = np.array(self.points, dtype=np.float64, copy=True)
        if points.shape != (len(self.node_ids), 4) or not np.isfinite(points).all():
            raise ValueError('points must be finite (N,4) coordinates')
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError('duplicate node IDs')
        if np.any(points < 0) or np.any(points >= np.asarray(shape)):
            raise ValueError('coordinates outside movie bounds; no silent clipping')
        if np.any(points[:, 0] != np.floor(points[:, 0])):
            raise ValueError('fractional frame index')
        if np.any(np.diff(points[:, 0]) < 0):
            raise ValueError('points must be ordered by frame')
        points.setflags(write=False)
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, 'movie_shape', shape)

    @classmethod
    def from_nodes(cls, nodes: Iterable[Mapping], movie_shape, *,
                   coordinate_source='postprocessed FOCUS proposals'):
        rows = list(nodes)
        for row in rows:
            if int(row['node_id']) != row['node_id']:
                raise ValueError('node IDs must be integers')
        rows.sort(key=lambda row: (row['t'], row['node_id']))
        points = np.asarray([[r[k] for k in ('t', 'z', 'y', 'x')] for r in rows])
        return cls(tuple(int(r['node_id']) for r in rows), points.reshape(-1, 4),
                   tuple(movie_shape), coordinate_source=coordinate_source)

    def association_coords(self, frame: int, *, mode='rounded'):
        if int(frame) != frame or not 0 <= frame < self.movie_shape[0]:
            raise ValueError('frame outside movie bounds')
        if mode not in {'rounded', 'subvoxel'}:
            raise ValueError('unsupported association coordinate mode')
        points = self.points[self.points[:, 0] == frame].copy()
        points[:, 1:] /= np.asarray(self.downsample)
        if mode == 'rounded':
            points[:, 1:] = np.rint(points[:, 1:])
        if np.any(points > np.iinfo(np.int16).max):
            raise ValueError('coordinates exceed linker int16 contract')
        return points.astype(np.int16 if mode == 'rounded' else np.float32)

    def precise_output_coords(self):
        return self.points.copy()

    def restore_precise_output(self, linked_coords):
        linked = np.asarray(linked_coords)
        if linked.shape != self.points.shape or not np.isfinite(linked).all():
            raise ValueError('linker changed node count or returned invalid coordinates')
        # The official predictor rescales grid coordinates to image space.
        # Check positions as well as times: matching frame counts alone can
        # silently attach edges to reordered detections in the same frame.
        expected = self.points.copy()
        expected[:, 1:] = np.rint(expected[:, 1:] / self.downsample) * self.downsample
        if not (np.allclose(linked, expected, atol=1e-4, rtol=0) or
                np.allclose(linked, self.points, atol=1e-4, rtol=0)):
            raise ValueError('linker changed external node coordinates/order')
        return self.precise_output_coords()

    def manifest(self):
        return {'candidate': 'focus-points-learned-linker-v1',
                'coordinate_source': self.coordinate_source,
                'selected_node_count': len(self.node_ids),
                'frame_counts': [int(np.sum(self.points[:, 0] == t))
                                 for t in range(self.movie_shape[0])],
                'source_node_ids': list(self.node_ids),
                'spatial_downsample': list(self.downsample),
                'node_filtering': False, 'focus_edges_used': False,
                'authorized_for_submission': False}
