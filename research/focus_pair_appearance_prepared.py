"""Cache training-fold transforms once; preserve the exact CPU block objective."""
import json
from pathlib import Path

import numpy as np

from research.focus_pair_appearance_head import blocks, validate_samples


class Prepared:
    def __init__(self, samples, projection, arm, directory, role):
        if role != 'fitting':
            raise ValueError('Only fitting data may prepare an optimizer cache')
        if arm not in ('full', 'lda'):
            raise ValueError('Only the two frozen appearance arms are supported')
        counts = validate_samples(samples)
        expected = [counts['choices'] - counts['groups'] - counts['parents'], counts['parents']]
        if projection['real_pair_counts'] != expected:
            raise ValueError('Projection and prepared data must share exactly the training pairs')
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.samples = samples
        self.paths = []
        self.dimension = 72 if arm == 'full' else 9
        self.bytes = counts['choices'] * self.dimension * 8
        if self.bytes > 3 * 1024**3:
            raise ValueError('Prepared fold disk-cache cap is3GiB')
        for index, sample in enumerate(samples):
            base, _ = sample
            path = self.directory / f'{index:02d}.npy'
            target = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                             shape=(len(base['offset']), self.dimension))
            cursor = 0
            for block in blocks([sample], projection, arm):
                end = cursor + len(block['x'])
                target[cursor:end] = block['x']
                cursor = end
            if cursor != len(target):
                raise ValueError('Complete prepared candidate coverage required')
            target.flush()
            del target
            self.paths.append(path)
        manifest = dict(arm=arm, counts=counts, projection=projection,
                        dimension=self.dimension, bytes=self.bytes,
                        optimization_changes=False, source_group_block=64)
        (self.directory / 'manifest.json').write_text(json.dumps(manifest, allow_nan=False))

    def __call__(self):
        # Same movie/group/block order as the frozen streamed implementation.
        for sample, path in zip(self.samples, self.paths):
            base, _ = sample
            cached = np.load(path, mmap_mode='r', allow_pickle=False)
            cursor = 0
            for block in blocks([(base, None)], None, 'base'):
                end = cursor + len(block['x'])
                yield dict(block, x=cached[cursor:end])
                cursor = end
            if cursor != len(cached):
                raise ValueError('Prepared cache row coverage changed')
            del cached
