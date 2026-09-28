"""Integrate the frozen strong pruned-track recovery into the deployed notebook.

The recovery logic itself is NOT reimplemented here. `research/public_pruned_track_recovery.py`
is embedded verbatim, SHA-pinned, and executed in its own namespace inside the
kernel, so the deployed behaviour is the same code that produced the 2026-09-10
four-movie diagnostic. This module only supplies the plumbing: it hands that
function the same four inputs the offline driver did, taken from live notebook
state instead of saved JSON.

Mapping from the offline driver to the notebook, verified by source inspection:

  driver `pre`                  -> `nodes_by_id` at the motion-relink call. Only
                                   edges are filtered between function entry and
                                   that call, so the node set is the input graph.
  driver `motion`               -> the notebook's own `motion_relink_edges(...)`
                                   result, which already carries edge_prob and
                                   distance_um.
  driver `base`                 -> the final post-smoothing graph, in the CSV
                                   integer form the recovery expects.
  driver `pre_filter_node_count`-> len(final nodes) + stats short_track_nodes_removed.

Two deliberate deviations from the offline driver, both fail-safe:

  * The recovery raises on eight different contract violations and hardcodes the
    (64, 256, 256) volume. Offline that is correct: a violation means the
    experiment is invalid. On 199 unseen movies a single violation would abort
    the whole submission, so the call is wrapped and falls back to the unmodified
    base graph for that movie, recording the reason in stats.
  * It is gated behind BIOHUB_D4R_ENABLE so the same builder can emit a
    recovery-free kernel.

No threshold is retuned. All six rescue constants are read from the notebook's
own configuration and were verified equal to the 2026-09-10 frozen values.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SOURCE = ROOT / "research" / "public_pruned_track_recovery.py"

# The exact recovery implementation scored on 2026-09-10. This value is the pin
# recorded by that run itself, in
# reports/experiments/public-pruned-track-recovery-v1-result.json -> receipt.pins,
# alongside its pooled corrected_recovery score of 0.9493103203140519.
RECOVERY_SHA256 = "68827c90bda26f77eded76cc3222d29d4d31f8faf4dd957ad21e82dabf79980d"

# Rescue constants frozen in .biohub/cache/public-d4-full-movie-v1-bundle/resolved-public-config.json
FROZEN_RESCUE_CONSTANTS = {
    "SHORT_TRACK_RESCUE_MIN_LEN": 4,
    "OUTPUT_MIN_TRACK_LEN": 6,
    "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": 0.88,
    "SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM": 3.0,
    "SHORT_TRACK_RESCUE_MAX_NODES_FRAC": 0.012,
    "SHORT_TRACK_RESCUE_MAX_NODES_ABS": 120,
}

_CAPTURE_ANCHOR = (
    "        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)\n"
)
_INIT_ANCHOR = "    edges: list[dict[str, object]] = []\n    for edge in raw_edges:\n"
_RETURN_ANCHOR = (
    '    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")\n'
    "\n"
    "    return nodes_by_id, edges, stats\n"
)


def recovery_source() -> str:
    text = RECOVERY_SOURCE.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != RECOVERY_SHA256:
        raise ValueError(
            f"Recovery implementation drift: expected {RECOVERY_SHA256}, read {digest}. "
            "The deployed logic must be the code that was scored; re-verify before relaxing."
        )
    return text


def preamble() -> str:
    """Module-level block installed once, before filter_output_graph is defined."""
    source = recovery_source()
    return "\n".join(
        [
            "",
            "",
            "# ---------------------------------------------------------------------------",
            "# Project-authored strong pruned-track recovery.",
            "# research/public_pruned_track_recovery.py embedded verbatim and SHA-pinned;",
            "# it runs in its own namespace so nothing here shadows notebook globals.",
            "# ---------------------------------------------------------------------------",
            "import hashlib as _d4r_hashlib",
            "",
            f"_D4R_SOURCE = {ascii(source)}",
            f"if _d4r_hashlib.sha256(_D4R_SOURCE.encode('utf-8')).hexdigest() != {RECOVERY_SHA256!r}:",
            "    raise RuntimeError('D4R recovery source drift')",
            "_d4r_namespace = {}",
            "exec(compile(_D4R_SOURCE, 'public_pruned_track_recovery.py', 'exec'), _d4r_namespace)",
            "_d4r_recover = _d4r_namespace['recover']",
            "",
            "_D4R_ENABLE = os.environ.get('BIOHUB_D4R_ENABLE', '0') != '0'",
            "_D4R_IMAGE_GATE = os.environ.get('BIOHUB_D4R_IMAGE_GATE', '0') != '0'",
            "_D4R_CONFIG = {",
            "    'SHORT_TRACK_RESCUE_MIN_LEN': SHORT_TRACK_RESCUE_MIN_LEN,",
            "    'OUTPUT_MIN_TRACK_LEN': OUTPUT_MIN_TRACK_LEN,",
            "    'SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB': SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB,",
            "    'SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM': SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM,",
            "    'SHORT_TRACK_RESCUE_MAX_NODES_ABS': SHORT_TRACK_RESCUE_MAX_NODES_ABS,",
            "    'SHORT_TRACK_RESCUE_MAX_NODES_FRAC': SHORT_TRACK_RESCUE_MAX_NODES_FRAC,",
            "}",
            f"_D4R_FROZEN = {FROZEN_RESCUE_CONSTANTS!r}",
            "if _D4R_ENABLE and _D4R_CONFIG != _D4R_FROZEN:",
            "    raise RuntimeError(",
            "        'D4R rescue constants differ from the 2026-09-10 frozen values: '",
            "        f'{_D4R_CONFIG} vs {_D4R_FROZEN}'",
            "    )",
            "",
            "",
            "def _d4r_csv_graph(nodes_by_id, edges):",
            '    """Final graph in the integer CSV form the frozen recovery expects."""',
            "    out = {'nodes': {}, 'edges': []}",
            "    for _ident, _node in nodes_by_id.items():",
            "        _ident = int(_ident)",
            "        _row = {'node_id': _ident, 't': int(_node['t'])}",
            "        for _axis, _bound in (('z', 64), ('y', 256), ('x', 256)):",
            "            _value = max(0, int(round(float(_node[_axis]))))",
            "            if _value >= _bound:",
            "                raise ValueError('Out-of-volume predicted coordinate')",
            "            _row[_axis] = _value",
            "        out['nodes'][str(_ident)] = _row",
            "    for _edge in edges:",
            "        out['edges'].append({",
            "            'source_id': int(_edge['source_id']),",
            "            'target_id': int(_edge['target_id']),",
            "        })",
            "    return out",
            "",
        ]
    )


def _capture_block() -> str:
    return (
        "        _d4r_reference = {int(_i): dict(_n) for _i, _n in nodes_by_id.items()}\n"
        "        _d4r_motion = list(motion_edges)\n"
    )


def _init_block() -> str:
    return (
        "    _d4r_reference = None\n"
        "    _d4r_motion = None\n"
    )


def _recovery_block() -> str:
    return "\n".join(
        [
            "",
            "    # Strong pruned-track recovery. Falls back to the base graph on any",
            "    # contract violation so one movie cannot abort a 199-movie submission.",
            "    if _D4R_ENABLE and _d4r_reference and _d4r_motion:",
            "        stats['d4r_attempted'] = 1",
            "        try:",
            "            _d4r_base = _d4r_csv_graph(nodes_by_id, edges)",
            "            _d4r_ref = {'nodes': {str(_i): _n for _i, _n in _d4r_reference.items()}}",
            "            _d4r_pre_count = len(nodes_by_id) + int(stats.get('short_track_nodes_removed', 0))",
            "            _d4r_out, _d4r_details = _d4r_recover(",
            "                _d4r_base, _d4r_ref, _d4r_motion, _D4R_CONFIG,",
            "                linefit_smooth_output_graph, _d4r_pre_count,",
            "            )",
            "            # Conservative image-supported gate. The unguarded recovery",
            "            # restored 379 nodes across four movies on edge probability and",
            "            # distance alone and scored exactly 0.000 on the leaderboard. A",
            "            # component is now kept only if EVERY restored node carries",
            "            # DeepCenter evidence at the same threshold the pipeline already",
            "            # uses to accept a gap-repair point, so a track is restored only",
            "            # where the image says a cell is.",
            "            _d4r_keep = set()",
            "            _d4r_rejected = 0",
            "            # deepcenter_accept_repair_point does stats[key] += 1, and the",
            "            # notebook pre-seeds only the 'gap' and 'safe_div' prefixes, so a",
            "            # new prefix must seed its own counters or the first call raises.",
            "            for _k in ('checked', 'accepted', 'rejected', 'missing'):",
            "                stats.setdefault('deepcenter_d4r_' + _k, 0)",
            "            if _D4R_IMAGE_GATE and deepcenter_bundle is not None:",
            "                _d4r_fc, _d4r_hc = {}, {}",
            "                for _comp in _d4r_details['components']:",
            "                    _ok = True",
            "                    for _nid in _comp['node_ids']:",
            "                        _n = _d4r_out['nodes'].get(str(int(_nid)))",
            "                        if _n is None:",
            "                            continue",
            "                        if not deepcenter_accept_repair_point(",
            "                            dataset, int(_n['t']),",
            "                            (float(_n['z']), float(_n['y']), float(_n['x'])),",
            "                            deepcenter_bundle, _d4r_fc, _d4r_hc, stats,",
            "                            'd4r', DEEPCENTER_GAP_THRESHOLD,",
            "                        ):",
            "                            _ok = False",
            "                            break",
            "                    if _ok:",
            "                        _d4r_keep.update(int(_i) for _i in _comp['node_ids'])",
            "                    else:",
            "                        _d4r_rejected += 1",
            "            else:",
            "                for _comp in _d4r_details['components']:",
            "                    _d4r_keep.update(int(_i) for _i in _comp['node_ids'])",
            "            stats['d4r_components_image_rejected'] = _d4r_rejected",
            "            stats['d4r_image_gate'] = int(bool(_D4R_IMAGE_GATE))",
            "            _d4r_new_nodes = 0",
            "            for _sid, _row in _d4r_out['nodes'].items():",
            "                _iid = int(_sid)",
            "                if _iid in nodes_by_id or _iid not in _d4r_keep:",
            "                    continue",
            "                nodes_by_id[_iid] = {",
            "                    'node_id': _iid, 't': int(_row['t']),",
            "                    'z': float(_row['z']), 'y': float(_row['y']), 'x': float(_row['x']),",
            "                }",
            "                _d4r_new_nodes += 1",
            "            for _edge in _d4r_out['edges'][len(_d4r_base['edges']):]:",
            "                if (int(_edge['source_id']) not in _d4r_keep",
            "                        or int(_edge['target_id']) not in _d4r_keep):",
            "                    continue",
            "                edges.append({",
            "                    'source_id': int(_edge['source_id']),",
            "                    'target_id': int(_edge['target_id']),",
            "                })",
            "            stats['d4r_recovered_components'] = int(_d4r_details['recovered_components'])",
            "            stats['d4r_added_nodes'] = _d4r_new_nodes",
            "            stats['d4r_added_edges'] = int(_d4r_details['added_edges'])",
            "            stats['d4r_node_budget'] = int(_d4r_details['node_budget'])",
            "            stats['d4r_eligible_components'] = int(_d4r_details['eligible_components'])",
            "            print(",
            "                f\"  [{dataset}] D4R recovery: +{_d4r_new_nodes} nodes \"",
            "                f\"+{_d4r_details['added_edges']} edges from \"",
            "                f\"{_d4r_details['recovered_components']}/{_d4r_details['eligible_components']} \"",
            "                f\"components (budget {_d4r_details['node_budget']}) \"",
            "                f\"image_gate={_D4R_IMAGE_GATE} rejected={_d4r_rejected}\",",
            "                flush=True,",
            "            )",
            "        except Exception as _d4r_exc:",
            "            stats['d4r_failed'] = 1",
            "            print(",
            "                f'  [{dataset}] D4R recovery SKIPPED, base graph kept: '",
            "                f'{type(_d4r_exc).__name__}: {_d4r_exc}',",
            "                flush=True,",
            "            )",
            "",
        ]
    )


def install(cell_source: str) -> tuple[str, dict]:
    """Return (patched cell 5 source, report). Fails closed on any anchor drift."""
    for name, anchor in (
        ("init", _INIT_ANCHOR),
        ("capture", _CAPTURE_ANCHOR),
        ("return", _RETURN_ANCHOR),
    ):
        if cell_source.count(anchor) != 1:
            raise ValueError(f"D4R {name} anchor is not unique in the postprocess cell")

    patched = cell_source.replace(_INIT_ANCHOR, _init_block() + _INIT_ANCHOR, 1)
    patched = patched.replace(_CAPTURE_ANCHOR, _CAPTURE_ANCHOR + _capture_block(), 1)
    patched = patched.replace(
        _RETURN_ANCHOR,
        _RETURN_ANCHOR.replace(
            "\n    return nodes_by_id, edges, stats\n",
            _recovery_block() + "\n    return nodes_by_id, edges, stats\n",
        ),
        1,
    )
    # The preamble must precede filter_output_graph's definition.
    marker = "def filter_output_graph(\n"
    if patched.count(marker) != 1:
        raise ValueError("filter_output_graph definition is not unique")
    patched = patched.replace(marker, preamble() + "\n" + marker, 1)

    ast.parse(patched)
    report = {
        "recovery_sha256": RECOVERY_SHA256,
        "frozen_rescue_constants": dict(FROZEN_RESCUE_CONSTANTS),
        "anchors_installed": ["init", "capture", "recovery", "preamble"],
        "logic_reimplemented": False,
        "thresholds_retuned": False,
        "fail_safe_fallback": True,
        "image_gate_available": True,
        "added_lines": len(patched.splitlines()) - len(cell_source.splitlines()),
    }
    return patched, report
