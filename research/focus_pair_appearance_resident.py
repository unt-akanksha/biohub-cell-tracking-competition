"""Reuse prepared arrays across optimizer calls without repeated file mapping."""
import numpy as np

from research.focus_pair_appearance_head import blocks


class Resident:
    def __init__(self, prepared):
        self.prepared = prepared
        self.mode = 'ram' if prepared.bytes <= 512 * 1024**2 else 'persistent_readonly_mmap'
        self.arrays = [np.load(path, mmap_mode=None if self.mode == 'ram' else 'r', allow_pickle=False)
                       for path in prepared.paths]
        if sum(a.nbytes for a in self.arrays) != prepared.bytes:
            raise ValueError('Exact prepared resident feature coverage required')
        for a in self.arrays:
            a.flags.writeable = False

    def __call__(self):
        for sample, cached in zip(self.prepared.samples, self.arrays):
            base, _ = sample
            cursor = 0
            for block in blocks([(base, None)], None, 'base'):
                end = cursor + len(block['x'])
                yield dict(block, x=cached[cursor:end])
                cursor = end
            if cursor != len(cached):
                raise ValueError('Exact resident block coverage required')
