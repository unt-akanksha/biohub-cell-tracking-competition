"""Own deterministic augmentation stream and input fingerprints for paired fits."""
import hashlib
import inspect
import textwrap


def seed_dataset(official,dataset,seed):
    import numpy as np
    cls = type(dataset)
    if not getattr(cls,'_biohub_seeded_augmentation',False):
        source = textwrap.dedent(inspect.getsource(cls.__getitem__))
        if source.count('np.random.default_rng()') != 1:
            raise ValueError('Unrecognized organizer augmentation RNG source')
        source = source.replace('np.random.default_rng()','self.augmentation_rng')
        scope = dict(vars(official))
        exec(compile(source,'<seeded-augmentation>','exec'),scope)
        getitem = scope['__getitem__']
        def traced(self,index):
            result = getitem(self,index)
            digest = hashlib.sha256()
            for key in ('imgs','coords','masks'):
                value = result[key].detach().cpu().contiguous().numpy()
                digest.update(key.encode()+b'\0'+str(value.shape).encode()+b'\0')
                digest.update(value.tobytes())
            self.sample_hashes.append(digest.hexdigest())
            return result
        cls.__getitem__ = traced
        cls._biohub_seeded_augmentation = True
    dataset.augmentation_rng = np.random.default_rng(seed)
    dataset.sample_hashes = []
    return dataset
