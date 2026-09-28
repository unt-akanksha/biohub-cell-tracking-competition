"""Density-conditional motion-relink gates for the stock public pipeline.

Our own 40-movie decomposition found that the score deficit is a tail of ten
movies whose linker output is fragmented -- edges per node 0.93-0.95 against
0.99 on the best movies -- while their detector node counts are already correct
(6bba_474be664 sits at 0.98x truth and still scores 0.7508 edge Jaccard). Cell
density correlates -0.32 with edge Jaccard: the sparsest thirteen movies average
79.6 cells per frame and 0.9353, the densest thirteen average 509.1 and 0.8620.

The first wide sweep probed only gates TIGHTER than the defaults and found every
candidate within 0.0009 of base. That was the wrong direction for sparse movies:
where cells are far apart and move freely, a 6.0 um tight gate and 10.0 um
relaxed gate drop true links, which is exactly the fragmentation observed.

The mechanism here is taken from the public notebook
`haideptry/biohub-sota-0-948-density-adaptive-2xt4-22m`, which conditions the
gates on measured density rather than applying one global constant. The band
thresholds and per-band gates are that notebook's; the reason to believe them is
our own measurement above, not its advertised score, which is not evidence.

Selection is by computed density -- mean nodes per frame of the movie's own
predicted graph -- never by dataset name, so it generalises to the hidden test
set and is not movie-identity routing.

Nothing else changes: same models, same detector, same thresholds elsewhere,
same post-process stages in the same order.

Two of the knobs are emitted as plain module-level floats, DENSITY_LOW_BAND and
DENSITY_TIGHT_CONTINUOUS, rather than being frozen into the injected code. That
is deliberate: the notebook's own post-process sweep rebinds sweepable globals
and rescores the SAME cached prediction graphs, so registering these two makes
band120-vs-band165 and banded-vs-continuous answerable in a single kernel run,
paired per movie and therefore splittable by embryo. Nothing about the shipped
default changes -- production pins them through the same two literals.
"""
from __future__ import annotations

import ast

# Per-band gates as published in the source notebook.
# Defaults for reference: tight 6.0, relaxed 10.0, velocity 0.5, bonus 1.0.
#
# The published band boundaries are 120 and 400. The 120 boundary is not where
# our own data breaks. Across the 40 held-out movies, mean edge Jaccard in
# sliding windows of eight holds at 0.954-0.956 up to ~158 nodes per frame and
# then falls to 0.904 and 0.825:
#
#     50.5- 85.2  0.9559      176.1-239.8  0.8249
#     63.3-102.4  0.9548      228.3-284.2  0.8425
#     85.3-158.4  0.9539      253.8-299.6  0.8785
#    125.3-221.4  0.9035      289.3-443.9  0.8588
#
# Four movies sit between 120 and 160 and behave like low-density ones while
# receiving middle-band gates: 6bba_2312ac41 (0.9877), 44b6_aaf8b0ea (0.9854),
# 44b6_deabac95 (0.9609), 44b6_d2f34f90 (0.9414). The break lies in the gap
# between 158.4 and 176.1, so every boundary in that interval reclassifies our
# movies identically; 165 is its midpoint and is not otherwise special.
#
# The 400 boundary is left alone: the 289-444 and 308-558 windows sit at 0.859
# and 0.838, essentially flat across it, so our data says nothing about it.
#
# THAT HYPOTHESIS WAS TESTED AND REFUTED. Moving the boundary to 165 shipped as
# submission 56334433 and scored 0.948 against 56318569's 0.949 on the published
# 120. The 40-movie sweep (density-config-sweep-v1, 2026-09-18) then found band120
# ahead of band165 by +0.000012 pooled, with only THREE of forty movies differing
# at all -- 44b6_aaf8b0ea +0.00001, 44b6_d2f34f90 +0.00012, 6bba_2312ac41 +0.00022.
# So the train set agrees in direction but cannot resolve the size, and the
# leaderboard, which sees a proportionate slice of ~199 movies rather than three,
# is the better evidence. The window analysis above was sound about where OUR
# movies break and wrong about what to do with it: reclassifying that window
# changes the gates for a handful of movies and buys nothing.
#
# Back to the published boundary. The reasoning is kept rather than deleted
# because it is the record of a refuted hypothesis, not a live justification.
DENSITY_BANDS = {"low": 120.0, "middle": 400.0}
PUBLISHED_BANDS = {"low": 120.0, "middle": 400.0}

# Continuous variant for tight_um only.
#
# The three published tight_um values are almost exactly linear in density.
# Taking the outer two band midpoints as anchors gives
# tight_um = 7.6645 - 0.004605 * density, which reproduces the middle band's
# 6.5 to within 0.013. The steps are approximating a line, so the boundaries
# are an artefact of expressing it as bands rather than something meaningful.
#
# Replacing the steps with the line removes both boundaries for the one
# parameter that the 40-movie sweep showed is actually sensitive -- every other
# candidate there clustered within 0.0009 of base. It also extrapolates, which
# matters because the test set is embryo-disjoint and unseen embryos may sit
# outside our observed 50.5-557.8 nodes/frame range.
#
# The clamp is deliberately set to the range our data covers: 7.5 binds below
# 35.7 and 5.0 binds above 578, so the line is only used where we have evidence
# and is held flat beyond it rather than extrapolated without support.
#
# Only tight_um becomes continuous. relaxed_um, velocity_weight and
# learned_bonus are NOT monotonic across the three bands (11.0/9.0/10.0,
# 0.5/0.0/0.5, 3.0/6.0/1.0), so there is no line to fit and interpolating them
# would be inventing structure the published table does not contain.
#
# DENSITY_TIGHT_CONTINUOUS selects among three rules:
#   0 = the published banded step (what 0.949 shipped)
#   1 = the line, wherever it falls
#   2 = max(step, line) -- the line may only LOOSEN, never tighten
#
# Mode 2 exists because the 12-movie pilot (density-config-pilot-v1,
# 2026-09-18) showed mode 1 helping and hurting for opposite reasons. Per-movie
# adjusted edge Jaccard against the banded base:
#
#     190.1 nodes/frame  6.500 -> 6.789 (looser)  +0.02252
#     297.5 nodes/frame  6.500 -> 6.294 (tighter) -0.00721
#     299.6 nodes/frame  6.500 -> 6.285 (tighter) -0.00481
#     the other nine                              within 0.00004
#
# Every gain came from loosening and every loss from tightening, which is what
# the original 40-movie diagnosis predicted: the failing tail is fragmented, and
# fragmentation is true links dropped for exceeding a distance gate. Mode 2 is
# that reading applied consistently.
#
# Mode 2 was chosen AFTER seeing those twelve movies, so those twelve cannot
# also test it. It is only evidence if it holds on movies the pilot never saw.
#
# IT DID NOT. On the 40-movie sweep, `loosen` scored +0.001101 on the twelve
# movies that suggested it and -0.000330 on the twenty-eight it had never seen;
# per-embryo on those held-out movies it was -0.000002 (44b6) and -0.000437
# (6bba), and NOT ONE of the twenty-eight improved (0 better / 4 worse / 24
# tied). The +0.02252 on 44b6_587a1e22 that motivated the whole idea was 8.8
# edges out of that movie's 389, and nothing like it recurred. Mode 1 was worse
# still, -0.000791 overall and -0.000935 held out.
#
# Both modes stay implemented and both stay unshipped: DENSITY_TIGHT_CONTINUOUS
# defaults to 0.0 and every production build pins it there. They are kept as the
# executable record of a measurement, not as options waiting to be switched on.
CONTINUOUS_MODES = {"banded": 0.0, "line": 1.0, "line_loosen_only": 2.0}
CONTINUOUS_TIGHT = {"intercept": 7.6645, "slope": -0.004605, "min": 5.0, "max": 7.5}
DENSITY_GROUP_OVERRIDES = {
    "low": {"tight_um": 7.25, "relaxed_um": 11.0, "velocity_weight": 0.5, "learned_bonus": 3.0},
    "middle": {"tight_um": 6.5, "relaxed_um": 9.0, "velocity_weight": 0.0, "learned_bonus": 6.0},
    "high": {"tight_um": 5.5, "relaxed_um": 10.0, "velocity_weight": 0.5, "learned_bonus": 1.0},
}

# Names the notebook's post-process sweep may rebind, in the order appended.
SWEEPABLE_GLOBALS = ("DENSITY_LOW_BAND", "DENSITY_TIGHT_CONTINUOUS")

_CALL_ANCHOR = (
    "        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)\n"
)
_DEF_ANCHOR = "def filter_output_graph(\n"
_SWEEP_KEYS_ANCHOR = '    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",\n]\n'


def preamble(continuous: bool = False) -> str:
    return "\n".join([
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Density-conditional motion-relink gates.",
        "# Band membership is computed from the movie's own predicted graph, never",
        "# from its name, so this generalises to the hidden test set.",
        "#",
        "# DENSITY_LOW_BAND and DENSITY_TIGHT_CONTINUOUS are plain floats read from",
        "# globals() at call time so the post-process sweep can rebind them and",
        "# rescore the same cached graphs. Their values here are the shipped ones.",
        "# ---------------------------------------------------------------------------",
        f"DENSITY_LOW_BAND = {float(DENSITY_BANDS['low'])!r}",
        f"DENSITY_MIDDLE_BAND = {float(DENSITY_BANDS['middle'])!r}",
        f"DENSITY_TIGHT_CONTINUOUS = {(1.0 if continuous else 0.0)!r}",
        f"_DENSITY_GROUP_OVERRIDES = {DENSITY_GROUP_OVERRIDES!r}",
        f"_CONTINUOUS_TIGHT = {CONTINUOUS_TIGHT!r}",
        "",
        "",
        "def _density_nodes_per_frame(nodes_by_id):",
        '    """Mean nodes per frame of this movie\'s own predicted graph."""',
        "    if not nodes_by_id:",
        "        return 0.0",
        "    times = [int(n['t']) for n in nodes_by_id.values()]",
        "    return len(nodes_by_id) / max(len(set(times)), 1)",
        "",
        "",
        "def _determine_density_group(nodes_by_id):",
        "    if not nodes_by_id:",
        "        return 'middle'",
        "    avg_per_frame = _density_nodes_per_frame(nodes_by_id)",
        "    if avg_per_frame < float(globals()['DENSITY_LOW_BAND']):",
        "        return 'low'",
        "    if avg_per_frame < float(globals()['DENSITY_MIDDLE_BAND']):",
        "        return 'middle'",
        "    return 'high'",
        "",
        "",
        "def _density_tight_um(per_frame, cfg):",
        '    """0 = published banded step, 1 = the line, 2 = the looser of the two."""',
        "    mode = float(globals()['DENSITY_TIGHT_CONTINUOUS'])",
        "    banded = float(cfg['tight_um'])",
        "    if mode < 0.5:",
        "        return banded",
        "    raw = _CONTINUOUS_TIGHT['intercept'] + _CONTINUOUS_TIGHT['slope'] * per_frame",
        "    line = float(min(_CONTINUOUS_TIGHT['max'],",
        "                     max(_CONTINUOUS_TIGHT['min'], raw)))",
        "    if mode < 1.5:",
        "        return line",
        "    return max(banded, line)",
        "",
        "",
        "def _apply_density_group(nodes_by_id, stats, dataset=None):",
        '    """Rebind the four motion-relink globals for this movie only."""',
        "    name = _determine_density_group(nodes_by_id)",
        "    cfg = _DENSITY_GROUP_OVERRIDES.get(name, _DENSITY_GROUP_OVERRIDES['middle'])",
        "    per_frame = _density_nodes_per_frame(nodes_by_id)",
        "    scope = globals()",
        "    continuous = int(float(scope['DENSITY_TIGHT_CONTINUOUS']))",
        "    tight = _density_tight_um(per_frame, cfg)",
        "    scope['MOTION_RELINK_TIGHT_UM'] = tight",
        "    scope['MOTION_RELINK_RELAXED_UM'] = float(cfg['relaxed_um'])",
        "    scope['MOTION_RELINK_VELOCITY_WEIGHT'] = float(cfg['velocity_weight'])",
        "    scope['MOTION_RELINK_LEARNED_BONUS'] = float(cfg['learned_bonus'])",
        "    stats['density_group_low'] = int(name == 'low')",
        "    stats['density_group_middle'] = int(name == 'middle')",
        "    stats['density_group_high'] = int(name == 'high')",
        "    stats['density_nodes_per_frame'] = int(round(per_frame))",
        "    stats['density_tight_um_milli'] = int(round(tight * 1000))",
        "    stats['density_low_band_milli'] = int(round(float(scope['DENSITY_LOW_BAND']) * 1000))",
        "    stats['density_tight_continuous'] = continuous",
        "    print(",
        "        f'  [{dataset}] DENSITY_ADAPTIVE: {per_frame:.1f} nodes/frame -> {name} '",
        "        f'(tight={tight:.3f} relaxed={cfg[\"relaxed_um\"]} '",
        "        f'velocity={cfg[\"velocity_weight\"]} bonus={cfg[\"learned_bonus\"]} '",
        "        f'low_band={scope[\"DENSITY_LOW_BAND\"]} continuous={continuous})',",
        "        flush=True,",
        "    )",
        "    return name",
        "",
    ])


def register_sweep_keys(cell_source: str) -> str:
    """Let the notebook's own sweep rebind the two density switches.

    PP_BASE_CONFIG is built by reading each name out of globals(), so this is
    only valid because the preamble lands in an earlier cell. Fails closed.
    """
    if cell_source.count(_SWEEP_KEYS_ANCHOR) != 1:
        raise ValueError("PP_SWEEP_KEYS anchor is not unique")
    added = "    " + ", ".join(repr(name) for name in SWEEPABLE_GLOBALS) + ",\n"
    patched = cell_source.replace(
        _SWEEP_KEYS_ANCHOR, _SWEEP_KEYS_ANCHOR[:-2] + added + "]\n", 1
    )
    ast.parse(patched)
    for name in SWEEPABLE_GLOBALS:
        if patched.count(repr(name)) != 1:
            raise ValueError(f"{name} is not registered exactly once")
    return patched


def install(cell_source: str, continuous: bool = False) -> tuple[str, dict]:
    """Insert the band definitions and condition the relink call. Fails closed."""
    for name, anchor in (("definition", _DEF_ANCHOR), ("relink call", _CALL_ANCHOR)):
        if cell_source.count(anchor) != 1:
            raise ValueError(f"Density-adaptive {name} anchor is not unique")

    patched = cell_source.replace(
        _DEF_ANCHOR, preamble(continuous) + "\n" + _DEF_ANCHOR, 1
    )
    patched = patched.replace(
        _CALL_ANCHOR,
        "        _apply_density_group(nodes_by_id, stats, dataset)\n" + _CALL_ANCHOR,
        1,
    )
    ast.parse(patched)
    report = {
        "density_bands": dict(DENSITY_BANDS),
        "group_overrides": {k: dict(v) for k, v in DENSITY_GROUP_OVERRIDES.items()},
        "selected_by": "computed nodes-per-frame of the movie's own predicted graph",
        "movie_identity_routing": False,
        "published_bands": dict(PUBLISHED_BANDS),
        "continuous_tight_um": dict(CONTINUOUS_TIGHT),
        "tight_um_is_continuous": bool(continuous),
        "sweepable_globals": list(SWEEPABLE_GLOBALS),
        "low_boundary_changed_from_published": DENSITY_BANDS["low"] != PUBLISHED_BANDS["low"],
        "low_boundary_basis": ("40 held-out movies: edge Jaccard holds ~0.954 to 158 nodes/frame "
                               "then falls to 0.904 and 0.825; break lies in the 158.4-176.1 gap"),
        "source_notebook": "haideptry/biohub-sota-0-948-density-adaptive-2xt4-22m",
        "advertised_score_used_as_evidence": False,
        "added_lines": len(patched.splitlines()) - len(cell_source.splitlines()),
    }
    return patched, report
