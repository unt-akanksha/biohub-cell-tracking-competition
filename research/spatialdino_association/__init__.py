"""Appearance features for Biohub association from a frozen SpatialDINO encoder."""

from .appearance import (
    edge_cosine_scores,
    parent_choice_margins,
    sample_patch_embeddings,
)

__all__ = [
    "edge_cosine_scores",
    "parent_choice_margins",
    "sample_patch_embeddings",
]
