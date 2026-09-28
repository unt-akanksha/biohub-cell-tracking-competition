"""Small 3D blind-spot CNN and bounded tiled inference, without normalization.

Own architecture: masked dilation-1 convolution, then dilation 2 and 4,
pointwise activations and a 1x1 head. No residual input path or spatial norms.
At least one first-layer offset is odd; all later spatial offsets are even,
so the input center cannot reach its corresponding output. Constant padding
also cannot introduce such a dependency. This statement conditions on fixed
weights and fixed normalization; it does not certify microscope noise or
any data-dependent preprocessing/training procedure as independent.
"""
from itertools import product


def receptive_offsets():
    offsets = {p for p in product((-1, 0, 1), repeat=3) if p != (0, 0, 0)}
    for dilation in (2, 4):
        offsets = {tuple(a + dilation*b for a,b in zip(p,q))
                   for p in offsets for q in product((-1,0,1), repeat=3)}
    return offsets


def tiles(shape, core=(32,64,64)):
    if (len(shape) != 3 or len(core) != 3
            or any(int(v) != v or v <= 0 for v in (*shape,*core))):
        raise ValueError('Positive integral three-axis shape/core required')
    for start in product(*(range(0,int(n),int(c)) for n,c in zip(shape,core))):
        end = tuple(min(s+c,n) for s,c,n in zip(start,core,shape))
        lo = tuple(max(0,s-7) for s in start)
        hi = tuple(min(n,e+7) for n,e in zip(shape,end))
        yield dict(read=tuple(slice(a,b) for a,b in zip(lo,hi)),
                   write=tuple(slice(a,b) for a,b in zip(start,end)),
                   crop=tuple(slice(s-a,e-a) for s,e,a in zip(start,end,lo)))


def build_model(width=16):
    if not isinstance(width,int) or isinstance(width,bool) or width <= 0:
        raise ValueError('Positive integer channel width required')
    import torch
    from torch import nn
    from torch.nn import functional as F

    class BlindSpotRestorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Conv3d(1,width,3,padding=1)
            self.second = nn.Conv3d(width,width,3,padding=2,dilation=2)
            self.third = nn.Conv3d(width,width,3,padding=4,dilation=4)
            self.head = nn.Conv3d(width,1,1)

        def forward(self, x):
            if x.ndim != 5 or x.shape[1] != 1 or min(x.shape) <= 0:
                raise ValueError('Nonempty N,1,Z,Y,X tensor required')
            # Construct the mask, rather than loading a mutable checkpoint buffer.
            mask = torch.ones_like(self.first.weight)
            mask[:,:,1,1,1] = 0
            y = F.relu(F.conv3d(x,self.first.weight*mask,self.first.bias,padding=1))
            y = F.relu(self.second(y))
            y = F.relu(self.third(y))
            return self.head(y)

    return BlindSpotRestorer()


def predict_tiled(model, volume, core=(32,64,64)):
    import torch
    if (volume.ndim != 5 or volume.shape[1] != 1 or min(volume.shape) <= 0
            or not torch.isfinite(volume).all()):
        raise ValueError('Finite nonempty N,1,Z,Y,X tensor required')
    output = torch.empty_like(volume)
    with torch.inference_mode():
        for tile in tiles(volume.shape[-3:],core):
            prediction = model(volume[(slice(None),slice(None))+tile['read']])
            output[(slice(None),slice(None))+tile['write']] = prediction[(slice(None),slice(None))+tile['crop']]
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite restoration output')
    return output
