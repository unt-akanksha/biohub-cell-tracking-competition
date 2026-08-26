"""Appearance features for Biohub association from a frozen SpatialDINO encoder."""

from .appearance import (
    edge_cosine_scores,
    parent_choice_margins,
    sample_patch_embeddings,
)
from .encoder import SpatialDinoViTS8, load_spatialdino_vits8
from .correction import SwapConfig, appearance_pair_swaps

__all__ = [
    "edge_cosine_scores",
    "parent_choice_margins",
    "sample_patch_embeddings",
    "SpatialDinoViTS8",
    "load_spatialdino_vits8",
    "SwapConfig",
    "appearance_pair_swaps",
]
