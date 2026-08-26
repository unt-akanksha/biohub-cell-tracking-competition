from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

try:
    from association_ensemble import PairScores, blend_pair_scores
except ModuleNotFoundError:
    from research.association_ensemble import PairScores, blend_pair_scores


@dataclass(frozen=True)
class BackboneBlend:
    head_variant: str
    general_weight: float

    @property
    def name(self) -> str:
        weight = f"{self.general_weight:.2f}".rstrip("0").rstrip(".")
        return f"general_ctc_blend_w{weight}:{self.head_variant}"


def default_blends() -> tuple[BackboneBlend, ...]:
    return tuple(
        BackboneBlend(head_variant=head, general_weight=weight)
        for head in ("pretrained", "biohub_probe")
        for weight in (0.25, 0.5, 0.75)
    )


def build_multibackbone_variants(
    backbone_scores: Mapping[str, Mapping[str, PairScores]],
    *,
    blends: Iterable[BackboneBlend] | None = None,
) -> dict[str, PairScores]:
    """Expose single-backbone scores plus support-aware general/CTC blends.

    The variant names are stable experiment identifiers. Only like-for-like
    heads are blended: official with official, and Biohub probe with Biohub
    probe. This keeps the selection grid small enough for two complete movies
    while still testing the complementary checkpoint families.
    """
    expected_backbones = {"general", "ctc"}
    if set(backbone_scores) != expected_backbones:
        raise ValueError("Expected exactly the general and ctc HOCT backbones")
    expected_heads = {"pretrained", "biohub_probe"}
    for backbone, scores in backbone_scores.items():
        if set(scores) != expected_heads:
            raise ValueError(
                f"HOCT {backbone} scores must contain exactly {sorted(expected_heads)}"
            )

    variants = {
        f"{backbone}:{head}": scores
        for backbone, by_head in sorted(backbone_scores.items())
        for head, scores in sorted(by_head.items())
    }
    for spec in tuple(blends) if blends is not None else default_blends():
        if spec.head_variant not in expected_heads:
            raise ValueError(f"Unknown HOCT head variant: {spec.head_variant}")
        if spec.name in variants:
            raise ValueError(f"Duplicate HOCT variant name: {spec.name}")
        variants[spec.name] = blend_pair_scores(
            backbone_scores["general"][spec.head_variant],
            backbone_scores["ctc"][spec.head_variant],
            trackastra_weight=spec.general_weight,
            support_aware=True,
        )
    return variants


def required_backbones(variant_name: str) -> frozenset[str]:
    """Return the minimum inference set for a frozen selected variant."""
    prefix, separator, _head = variant_name.partition(":")
    if not separator:
        raise ValueError(f"Malformed HOCT variant name: {variant_name}")
    if prefix == "general":
        return frozenset({"general"})
    if prefix == "ctc":
        return frozenset({"ctc"})
    if prefix.startswith("general_ctc_blend_w"):
        return frozenset({"general", "ctc"})
    raise ValueError(f"Unknown HOCT variant name: {variant_name}")


def materialize_variant(
    variant_name: str,
    backbone_scores: Mapping[str, Mapping[str, PairScores]],
) -> PairScores:
    """Materialize one frozen variant without constructing unused blends."""
    prefix, separator, head = variant_name.partition(":")
    if not separator or head not in {"pretrained", "biohub_probe"}:
        raise ValueError(f"Malformed HOCT variant name: {variant_name}")
    required = required_backbones(variant_name)
    missing = required - set(backbone_scores)
    if missing:
        raise ValueError(f"Missing HOCT backbone scores: {sorted(missing)}")
    if prefix in {"general", "ctc"}:
        return backbone_scores[prefix][head]
    weight_text = prefix.removeprefix("general_ctc_blend_w")
    try:
        general_weight = float(weight_text)
    except ValueError as exc:
        raise ValueError(f"Malformed HOCT blend weight: {variant_name}") from exc
    if not 0 <= general_weight <= 1:
        raise ValueError(f"HOCT blend weight is outside [0, 1]: {general_weight}")
    return blend_pair_scores(
        backbone_scores["general"][head],
        backbone_scores["ctc"][head],
        trackastra_weight=general_weight,
        support_aware=True,
    )
