"""Two independently trained visual association architectures, not detectors."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ('resnet3d', 'token_transformer3d')


class Residual3D(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.SiLU(),
            nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing))
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False)

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


class VisualCorrespondence(nn.Module):
    def __init__(self, family, input_channels=2):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError('Unknown architecture')
        self.family = family
        if input_channels not in (2, 3):
            raise ValueError('Expected two legacy or three native scales')
        self.input_channels = input_channels
        blocks = [nn.Conv3d(input_channels, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.SiLU()]
        previous = 32
        for width in (32, 64, 128, 256):
            blocks.extend([Residual3D(previous, width, 1 if width == 32 else 2), Residual3D(width, width)])
            previous = width
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.null_head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
        if family == 'resnet3d':
            self.edge_head = nn.Sequential(nn.Linear(1032, 512), nn.SiLU(), nn.Dropout(.1),
                                           nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1))
        else:
            self.token_projection = nn.Linear(264, 384)
            layer = nn.TransformerEncoderLayer(384, 8, 1536, .1, activation='gelu', batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(384), enable_nested_tensor=False)
            self.edge_head = nn.Sequential(nn.Linear(768, 384), nn.GELU(), nn.Linear(384, 1))
        for head in (self.null_head, self.edge_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, patches, coords, valid):
        # First entry is the child query; remaining entries are image proposals.
        b, n = patches.shape[:2]
        if patches.shape[2:] != (self.input_channels, 11, 11, 11) or coords.shape != (b, n, 3) or valid.shape != (b, n):
            raise ValueError('Visual input contract changed')
        if not bool(valid[:, 0].all()):
            raise ValueError('Every group needs an image query')
        flat_valid = valid.flatten()
        embeddings = self.encoder(patches.flatten(0, 1)[flat_valid])
        features = embeddings.new_zeros((b*n, 256)).index_copy(0, flat_valid.nonzero().flatten(), embeddings).reshape(b, n, 256)
        query, parents = features[:, :1], features[:, 1:]
        delta = (coords[:, 1:] - coords[:, :1]) / 10.
        radius = delta.square().sum(-1, keepdim=True)
        geometry = torch.cat((delta, delta.abs(), radius, radius.sqrt()), -1)
        prior = -2. * radius.squeeze(-1)
        if self.family == 'resnet3d':
            q = query.expand_as(parents)
            pair = torch.cat((q, parents, q*parents, (q-parents).abs(), geometry), -1)
            correction = self.edge_head(pair).squeeze(-1)
            null = self.null_head(query[:, 0]) - 8.
        else:
            geometry = F.pad(geometry, (0, 0, 1, 0))
            tokens = self.token_projection(torch.cat((features, geometry), -1))
            tokens = self.context(tokens, src_key_padding_mask=~valid)
            query_token = tokens[:, :1].expand(-1, n-1, -1)
            correction = self.edge_head(torch.cat((query_token, tokens[:, 1:]), -1)).squeeze(-1)
            null = self.null_head(query[:, 0]) - 8.
        scores = torch.cat((prior + correction.float(), null.float()), -1)
        return scores.masked_fill(~torch.cat((valid[:, 1:], valid[:, :1]), -1), -1e4)


def prepare_patches(stored, coords, valid, augment=False, generator=None):
    """Crop two scales with the same physical center jitter; preserve units."""
    b, n = stored.shape[:2]
    device = stored.device
    coords = coords.clone()
    shift = torch.zeros((b, n, 3), device=device)
    if augment:
        shift = torch.rand((b, n, 3), device=device, generator=generator)*2-1
        coords += shift * 1.625
    axis = torch.linspace(-5/7, 5/7, 11, device=device)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing='ij')
    grid = torch.stack((x, y, z), -1).view(1, 1, 11, 11, 11, 3)
    shift_xyz = shift.flip(-1).flatten(0, 1)[:, None, None, None, None, :]
    scales = torch.tensor((1., 2.), device=device).view(1, 2, 1, 1, 1, 1)
    grid = (grid + shift_xyz / (7*scales)).expand(b*n, 2, 11, 11, 11, 3)
    images = F.grid_sample(stored.float().reshape(b*n*2, 1, 15, 15, 15),
                           grid.reshape(b*n*2, 11, 11, 11, 3), mode='bilinear',
                           padding_mode='border', align_corners=True).reshape(b, n, 2, 11, 11, 11)
    if augment:
        gain = .8 + .4*torch.rand((b, 1, 1, 1, 1, 1), device=device, generator=generator)
        images = images*gain
        # Same XY rotation on all candidates and physical displacement vectors.
        k = int(torch.randint(4, (), device=device, generator=generator).item())
        images = torch.rot90(images, k, (-2, -1))
        delta = coords - coords[:, :1].clone()
        for _ in range(k):
            old_y = delta[..., 1].clone()
            delta[..., 1] = -delta[..., 2]; delta[..., 2] = old_y
        coords = delta
    return images * valid[:, :, None, None, None, None], coords


def mask_scores(scores, loss_mask):
    return scores.masked_fill(~torch.cat((loss_mask, torch.ones_like(loss_mask[:, :1])), -1), -1e4)
