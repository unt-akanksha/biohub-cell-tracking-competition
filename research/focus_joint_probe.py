"""Training-only stress examples and bounded, original-normalization image IO."""
from collections import OrderedDict
import hashlib
import numpy as np

SETTINGS=dict(steps=4,encoder_learning_rate=1e-5,head_learning_rate=1e-4,weight_decay=1e-4,
    gradient_clip=1.,seed=244691,amp_dtype='float16',encoder_batchnorm_mode='eval')


def select_samples(samples):
    if len(samples)<4 or any(s['role']!='fitting' for s in samples):raise ValueError('At least four fitting-only samples required')
    def size(s):p=s['packet'];return len(p['source_indices'])*len(p['target_indices'])
    def absent(s):p=s['packet'];return int((p['labels']==len(p['source_indices'])).sum())
    def parent(s):p=s['packet'];return int(((p['labels']>=0)&(p['labels']<len(p['source_indices']))).sum())
    chosen=[];seen=set()
    for sample in [samples[0],max(samples,key=size),max(samples,key=absent),max(samples,key=parent)]+samples:
        key=(sample['stem'],int(sample['packet']['source_frame']))
        if key not in seen:chosen.append(sample);seen.add(key)
        if len(chosen)==4:return chosen
    raise ValueError('Four unique fitting pairs required')


def normalize_pair(raw,low,high):
    values=np.asarray(raw,dtype=np.float32)
    if values.shape!=(2,64,64,64) or not np.isfinite(values).all() or not np.isfinite([low,high]).all() or high<=low:raise ValueError('Finite original downsampled two-frame images and quantiles required')
    return np.maximum((values-low)/(high-low+1e-6),0)


class ImagePairs:
    def __init__(self,data,allowed_stems):
        self.data=data;self.allowed=set(allowed_stems);self.arrays={};self.cache=OrderedDict();self.records={}

    def get(self,sample):
        stem=sample['stem'];t=int(sample['packet']['source_frame']);key=(stem,t)
        if stem not in self.allowed or not 0<=t<99:raise ValueError('Original declared training image scope required')
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        if stem not in self.arrays:
            import zarr
            from tracking_cellmot.io import open_dataset
            ds=open_dataset(self.data/stem,normalize=False,load_image=False,require_tracks=False,downsample=(1,4,4))
            if ds.tracks is not None:raise ValueError('No GPU raw annotation graph access')
            array=zarr.open_group(str(ds.zarr_path),mode='r')['0']
            if tuple(array.shape)!=(100,64,256,256):raise ValueError('Original complete native image shape required')
            self.arrays[stem]=(array,float(ds.quantiles['0.001']),float(ds.quantiles['0.999']))
        array,low,high=self.arrays[stem]
        image=normalize_pair(array[t:t+2,::1,::4,::4],low,high)
        self.cache[key]=image
        if len(self.cache)>16:self.cache.popitem(last=False)
        self.records[key]=dict(stem=stem,source_frame=t,normalized_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        return image
