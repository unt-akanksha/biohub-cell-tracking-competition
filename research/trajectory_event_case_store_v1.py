"""Bounded-memory, hash-checked replay of frozen event-training cases."""
from collections import OrderedDict
from collections.abc import Sequence
import hashlib
from pathlib import Path

import numpy as np

from research.trajectory_event_assignment_v1 import problem, prepare, validate_choice


class CaseStore(Sequence):
    def __init__(self, root, records, cache_size=2):
        self.root=Path(root).resolve(strict=True)
        self.records=list(records)
        if not self.records or not isinstance(cache_size,int) or cache_size<1:
            raise ValueError('Nonempty case manifest and positive cache bound required')
        if len({r['path'] for r in self.records})!=len(self.records):
            raise ValueError('Duplicate training case')
        self.cache_size=cache_size
        self.cache=OrderedDict()

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        if not isinstance(index,(int,np.integer)):raise TypeError('Integer case index required')
        index=int(index)
        if index<0:index+=len(self.records)
        if not 0<=index<len(self.records):raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row=self.records[index]
        path=self.root/row['path']
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root):
            raise ValueError('Case path escapes the frozen store')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Frozen event case hash changed')
        with np.load(path,allow_pickle=False) as data:
            values=dict(data)
        p=problem(values['options'],int(values['nparents']),int(values['nchildren']))
        validate_choice(p,values['incumbent'])
        case,reason=prepare(p,values['features'],values['targets'],values['safe'])
        if (case is None or reason!='prepared' or case['n_constraints']!=row['n_constraints']
                or not np.array_equal(case['allowed'],values['allowed'])
                or not np.array_equal(case['margin'],values['margin'])):
            raise ValueError('Stored partial supervision failed exact reconstruction')
        self.cache[index]=case
        while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return case
