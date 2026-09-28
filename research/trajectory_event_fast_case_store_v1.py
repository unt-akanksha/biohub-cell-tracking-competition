"""Hash-checked source cases with constructive feasibility and bounded caching."""
import hashlib
import numpy as np

from research.trajectory_event_case_store_v1 import CaseStore as ReferenceStore
from research.trajectory_event_assignment_v1 import problem,validate_choice
from research.trajectory_event_fast_training_v1 import prepare


class CaseStore(ReferenceStore):
    def __init__(self,root,records,cache_size=2,max_cache_bytes=1536*1024**2):
        super().__init__(root,records,cache_size=cache_size)
        if not isinstance(max_cache_bytes,int) or max_cache_bytes<=0:raise ValueError('Positive cache byte cap required')
        self.max_cache_bytes=max_cache_bytes;self.cache_bytes=0;self.sizes={}

    def __getitem__(self,index):
        if not isinstance(index,(int,np.integer)):raise TypeError('Integer case index required')
        index=int(index)
        if index<0:index+=len(self.records)
        if not 0<=index<len(self.records):raise IndexError(index)
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        row=self.records[index];path=self.root/row['path']
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root):raise ValueError('Case path escapes the frozen store')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Frozen event case hash changed')
        with np.load(path,allow_pickle=False) as data:values=dict(data)
        p=problem(values['options'],int(values['nparents']),int(values['nchildren']));validate_choice(p,values['incumbent'])
        case,reason=prepare(p,values['features'],values['targets'],values['safe'])
        if (case is None or reason!='prepared' or case['n_constraints']!=row['n_constraints']
                or not np.array_equal(case['allowed'],values['allowed']) or not np.array_equal(case['margin'],values['margin'])):
            raise ValueError('Stored partial supervision failed exact reconstruction')
        size=sum(case[k].nbytes for k in ('x','allowed','margin'))+p['options'].nbytes
        size+=p['matrix'].data.nbytes+p['matrix'].indices.nbytes+p['matrix'].indptr.nbytes
        while self.cache and (len(self.cache)>=self.cache_size or self.cache_bytes+size>self.max_cache_bytes):
            old,_=self.cache.popitem(last=False);self.cache_bytes-=self.sizes.pop(old)
        if size<=self.max_cache_bytes:
            self.cache[index]=case;self.sizes[index]=size;self.cache_bytes+=size
        return case
