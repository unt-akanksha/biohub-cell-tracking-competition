"""Harvest raw prediction graphs so config selection can move off the kernel.

The published notebook selects its post-process configuration with an in-kernel
sweep over 8 held-out TRAIN movies -- 4 per embryo prefix. That number is not
arbitrary: the validator, its base scoring and the 7-candidate sweep cost about
75 minutes on two T4s, and they run *inside* the submission kernel, so they are
charged against the competition's 12-hour cap. Widening the selection set in
place would push a full run over that cap.

That single selection step is nonetheless worth about +0.009 on the leaderboard,
which is an order of magnitude more than anything this project has measured on
its 4-movie diagnostic. Selecting it on 8 of 199 available labelled movies, all
of which the public checkpoints were trained on, is the weakest link in the only
chain that has actually paid.

This mode breaks the coupling. It runs the validator's *prediction* over a much
larger held-out set and persists each raw graph as compact JSON. Scoring and the
config search then happen offline, on CPU, against the locally cached ground
truth, with no runtime cap and no obligation to stop at 7 candidates -- and,
critically, with the freedom to hold out an entire embryo, which an in-kernel
sweep structurally cannot do.

Nothing is tuned here and no configuration is selected. This only produces
evidence for a later, separate decision.
"""
from __future__ import annotations

import ast

_CACHE_ANCHOR = (
    '    print(f"VALIDATOR: cached {len(VAL_RAW_GRAPHS)} raw prediction graphs + GT")\n'
)


def harvest_block() -> str:
    return "\n".join(
        [
            "",
            "    # ---------------------------------------------------------------",
            "    # Harvest: persist the raw prediction graphs and the ground-truth",
            "    # node-count estimate for each held-out movie, so post-process",
            "    # configuration selection can run offline over far more movies",
            "    # than fit inside the submission runtime cap.",
            "    # ---------------------------------------------------------------",
            "    _harvest_dir = WORKING_DIR / 'harvest'",
            "    _harvest_dir.mkdir(parents=True, exist_ok=True)",
            "    _harvest_records = []",
            "    for _stem in val_stems:",
            "        _raw_nodes, _raw_edges = VAL_RAW_GRAPHS[_stem]",
            "        _gt_nodes, _gt_edges, _t_true = VAL_GT[_stem]",
            "        _payload = {",
            "            'stem': _stem,",
            "            'embryo': _stem.split('_')[0],",
            "            't_true': _t_true,",
            "            'nodes': {",
            "                str(int(_i)): {",
            "                    'node_id': int(_n['node_id']), 't': int(_n['t']),",
            "                    'z': float(_n['z']), 'y': float(_n['y']), 'x': float(_n['x']),",
            "                }",
            "                for _i, _n in _raw_nodes.items()",
            "            },",
            "            'edges': [",
            "                {",
            "                    'source_id': int(_e['source_id']),",
            "                    'target_id': int(_e['target_id']),",
            "                    'edge_prob': (",
            "                        None if _e.get('edge_prob') is None else float(_e['edge_prob'])",
            "                    ),",
            "                }",
            "                for _e in _raw_edges",
            "            ],",
            "        }",
            "        _path = _harvest_dir / (_stem + '.json')",
            "        _path.write_text(json.dumps(_payload, sort_keys=True, allow_nan=False))",
            "        _harvest_records.append({",
            "            'stem': _stem, 'embryo': _payload['embryo'], 't_true': _t_true,",
            "            'nodes': len(_payload['nodes']), 'edges': len(_payload['edges']),",
            "            'bytes': _path.stat().st_size,",
            "        })",
            "        print(f\"  HARVEST {_stem}: {len(_payload['nodes'])} nodes \"",
            "              f\"{len(_payload['edges'])} edges\", flush=True)",
            "    (_harvest_dir / 'index.json').write_text(json.dumps({",
            "        'stems': list(val_stems),",
            "        'records': _harvest_records,",
            "        'total_bytes': sum(r['bytes'] for r in _harvest_records),",
            "        'selection_performed': False,",
            "        'configuration_tuned': False,",
            "    }, indent=2, sort_keys=True) + '\\n')",
            "    print(f'HARVEST: wrote {len(_harvest_records)} raw graphs '",
            "          f\"({sum(r['bytes'] for r in _harvest_records) / 1e6:.1f} MB)\", flush=True)",
            "",
        ]
    )


def install(cell_source: str) -> tuple[str, dict]:
    """Insert the harvest dump right after the validator caches its graphs."""
    if cell_source.count(_CACHE_ANCHOR) != 1:
        raise ValueError("Validator raw-graph cache anchor is not unique")
    patched = cell_source.replace(_CACHE_ANCHOR, _CACHE_ANCHOR + harvest_block(), 1)
    ast.parse(patched)
    report = {
        "harvest_installed": True,
        "selection_performed_in_kernel": False,
        "sweep_disabled": True,
        "added_lines": len(patched.splitlines()) - len(cell_source.splitlines()),
    }
    return patched, report
