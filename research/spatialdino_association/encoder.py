from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 384, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("volume must have shape (B, 1, Z, Y, X)")
        if any(int(size) % self.patch_size for size in volume.shape[-3:]):
            raise ValueError("volume dimensions must be divisible by the patch size")
        return self.proj(volume).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, channels = tokens.shape
        qkv = (
            self.qkv(tokens)
            .reshape(batch, count, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        weights = (query @ key.transpose(-2, -1)).mul(self.scale).softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, count, channels)
        return self.proj(attended)


class LayerScale(nn.Module):
    def __init__(self, dim: int = 384, initial_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(initial_value)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 1536) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(tokens)))


class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)
        self.ls2 = LayerScale(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.ls1(self.attn(self.norm1(tokens)))
        return tokens + self.ls2(self.mlp(self.norm2(tokens)))


class SpatialDinoViTS8(nn.Module):
    """Dependency-light exact architecture for the released SpatialDINO ViT-S/8.

    The upstream checkpoint uses no positional embeddings and its ordinary MLP
    feed-forward blocks, so inference needs only PyTorch. Keeping the original
    module names makes strict checkpoint loading an integrity check.
    """

    embed_dim = 384
    patch_size = 8
    depth = 12
    num_heads = 6

    def __init__(self) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_embed = PatchEmbed(self.embed_dim, self.patch_size)
        self.blocks = nn.ModuleList(
            Block(self.embed_dim, self.num_heads) for _ in range(self.depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward_intermediates(
        self,
        volume: torch.Tensor,
        *,
        block_indices: tuple[int, ...] = (2, 5, 8, 11),
    ) -> tuple[torch.Tensor, ...]:
        """Return patch-token grids after selected zero-based transformer blocks.

        The released checkpoint has no learned positional embedding, so the same
        token sequence can be exposed at several depths without changing the
        checkpoint-compatible module structure.  This supports UNETR-style
        decoders while keeping the original ``forward`` output bit-identical.
        """

        if not block_indices:
            raise ValueError("block_indices must not be empty")
        if tuple(sorted(set(block_indices))) != block_indices:
            raise ValueError("block_indices must be unique and increasing")
        if block_indices[0] < 0 or block_indices[-1] >= self.depth:
            raise ValueError("block_indices are outside the transformer depth")
        patch_tokens = self.patch_embed(volume)
        tokens = torch.cat(
            (self.cls_token.expand(volume.shape[0], -1, -1), patch_tokens),
            dim=1,
        )
        selected: list[torch.Tensor] = []
        grid_shape = tuple(int(size) // self.patch_size for size in volume.shape[-3:])
        selected_indices = set(block_indices)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in selected_indices:
                patch_grid = tokens[:, 1:]
                if index == self.depth - 1:
                    patch_grid = self.norm(patch_grid)
                selected.append(
                    patch_grid.transpose(1, 2).reshape(
                        volume.shape[0], self.embed_dim, *grid_shape
                    )
                )
        return tuple(selected)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediates(volume, block_indices=(self.depth - 1,))[0]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spatialdino_vits8(
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> SpatialDinoViTS8:
    path = Path(checkpoint)
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"SpatialDINO checkpoint hash mismatch: {actual_sha256}"
        )
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or len(state) != 174:
        raise ValueError("SpatialDINO checkpoint must contain exactly 174 tensors")
    model = SpatialDinoViTS8()
    model.load_state_dict(state, strict=True)
    return model.eval()
