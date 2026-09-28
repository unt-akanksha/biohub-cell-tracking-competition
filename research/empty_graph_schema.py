"""Restore declared spatial schema after an empty GEFF round trip."""


def restore_empty_spatial_schema(graph, expected_axes=('z', 'y', 'x')):
    import polars as pl
    if tuple(expected_axes) != ('z', 'y', 'x'):
        raise ValueError('Only the known Biohub spatial schema may be restored')
    missing = [axis for axis in expected_axes if axis not in graph.node_attr_keys()]
    if missing and (graph.num_nodes() or graph.num_edges()):
        raise ValueError('Nonempty graph is missing coordinates; refusing to invent values')
    for axis in missing:
        graph.add_node_attr_key(axis, pl.Float64, 0.)
    return missing
