"""Hash-verify frozen complete selection graphs before current-official scoring."""
import argparse
import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import runpy
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PILOT = runpy.run_path(str(ROOT/'scripts/verify-independent-real-pilot.py'))
INFER = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))


def verify_scorer_sources(folder):
    # LF-normalized bytes of the clean pinned organizer commit. Windows
    # checkout uses CRLF; notebook source embedding writes LF on Linux.
    hashes = {
        'metrics.py':'cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444',
        'division_metrics.py':'0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9'}
    for filename,digest in hashes.items():
        if hashlib.sha256((folder/filename).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
            raise ValueError('Current official scorer hash mismatch')


def load_scorer(folder):
    verify_scorer_sources(folder)
    name = '_independent_selection_metric'
    package = ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    return importlib.import_module(name+'.metrics')


def diagnostic_node_recall(scorer,graph,truth):
    # Official evaluate returns early for edgeless predictions, without node
    # matching. Compute only the supplemental detection recall in that case.
    # The already-computed official TP/FP/FN counts are not altered.
    import tracksdata as td
    if graph.num_nodes() == 0:
        return 0.
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID not in graph.node_attr_keys():
        from tracksdata.metrics import DistanceMatching
        graph.match(truth,matching=DistanceMatching(max_distance=7.,scale=(1.625,.40625,.40625)))
    return scorer.node_recall(graph,truth)


def smoke_empty_edges(scorer):
    import tracksdata as td
    import polars as pl
    graphs = [td.graph.InMemoryGraph(),td.graph.InMemoryGraph()]
    for graph in graphs:
        for axis in ('z','y','x'):
            graph.add_node_attr_key(axis,pl.Float64,0.)
        graph.bulk_add_nodes([dict(t=t,z=2.,y=2.,x=2.) for t in (0,1)])
    pred,truth = graphs
    ids = list(truth.node_attrs()['node_id'])
    truth.bulk_add_edges([dict(source_id=ids[0],target_id=ids[1])])
    er = scorer.evaluate(pred,truth,scale=(1.625,.40625,.40625),max_distance=7.)
    assert (er.edge_tp,er.edge_fp,er.edge_fn) == (0,0,1)
    assert diagnostic_node_recall(scorer,pred,truth) == 1.
    assert pred.num_nodes() == 2 and pred.num_edges() == 0


def emitted_manifest_run_id(runtime_source):
    """Read the literal receipt identity from already hash-verified run code.

    A composed launcher can have a different name from its embedded runner.
    Never edit old receipts or infer this value from the receipt itself.
    """
    values = []
    for node in ast.walk(ast.parse(runtime_source)):
        if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'result' for t in node.targets)
            and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
            and node.value.func.id == 'dict'):
            for keyword in node.value.keywords:
                if keyword.arg == 'run_id':
                    values.append(ast.literal_eval(keyword.value))
    if len(values) != 1 or not isinstance(values[0],str) or not values[0]:
        raise ValueError('Unambiguous literal manifest run identity required')
    return values[0]


def verify_manifest(manifest, terminal, codex, split, *, expected_manifest_run_id=None):
    if terminal['status'] != 'completed' or terminal['run_id'] != codex['run_id']:
        raise ValueError('Completed selection terminal required')
    if not 0 < terminal['elapsed_seconds'] <= 3600:
        raise ValueError('Selection budget exceeded')
    expected_id = codex['run_id'] if expected_manifest_run_id is None else expected_manifest_run_id
    if manifest['status'] != 'completed' or manifest['run_id'] != expected_id:
        raise ValueError('Incomplete selection manifest')
    if manifest['checkpoint_sha256'] != codex['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    audit = codex.get('embryo_audit')
    transfer = codex.get('owned_transfer_diagnostic')
    if any(manifest[k] is not False for k in ('ground_truth_opened','authorized_for_submission')):
        raise ValueError('Unexpected selection scope')
    expected = split['folds'][0]['selection']
    if codex.get('owned_detector_fit') is not None:
        receipt=manifest.get('owned_detector_fit')
        if (receipt!=codex['owned_detector_fit'] or receipt.get('objective') not in ('sparse','pu')
            or receipt.get('steps')!=1000 or receipt.get('version')!=1 or not codex.get('detector_spatial_tta')
            or receipt.get('probe_result_sha256')!='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba'):
            raise ValueError('Registered owned-detector fit receipt required')
    elif manifest.get('owned_detector_fit') is not None:
        raise ValueError('Unregistered trained detector profile')
    detector_calibrated=codex.get('detector_confidence_calibration')
    if detector_calibrated is not None:
        if (manifest.get('detector_confidence_calibration')!=detector_calibrated
            or codex.get('owned_detector_fit',{}).get('objective')!='sparse'
            or not codex.get('detector_spatial_tta') or audit is not None or transfer is not None
            or detector_calibrated.get('checkpoint_sha256')!=manifest['checkpoint_sha256']
            or detector_calibrated.get('authorized_for_submission') is not False):
            raise ValueError('Registered source-only training-calibrated sparse detector required')
    elif manifest.get('detector_confidence_calibration') is not None:
        raise ValueError('Unregistered detector confidence threshold change')
    ensemble=codex.get('owned_detector_ensemble')
    if ensemble is not None:
        row=manifest.get('owned_detector_ensemble',{})
        execution=row.get('execution',{})
        calls=sum(r['processed_frames']-1 for r in manifest['records'])
        frozen=row.get('frozen_hashes',[])
        if (row.get('contract')!=ensemble or not codex.get('detector_spatial_tta')
            or audit is not None or transfer is not None or detector_calibrated is not None
            or codex.get('association_calibration') is not None
            or ensemble.get('parent_sha256')!=manifest['checkpoint_sha256']
            or ensemble.get('weights')!=[.5,.5] or ensemble.get('source_validation_only') is not True
            or execution.get('weights')!=[.5,.5] or execution.get('probability_mixture') is not True
            or execution.get('features')!='parent native unchanged' or execution.get('encode_calls')!=calls
            or execution.get('parent_d4')!=manifest.get('detector_spatial_tta')
            or execution.get('secondary_d4',{}).get('views')!=8
            or execution.get('secondary_d4',{}).get('encode_calls')!=calls
            or not math.isfinite(execution.get('maximum_mean_absolute_logit_delta',float('nan')))
            or execution.get('maximum_mean_absolute_logit_delta',0)<=0
            or len(frozen)!=2 or any(r['before']!=r['after'] or len(r['before'])!=64 for r in frozen)
            or row.get('secondary_provenance',{}).get('objective')!='pu'):
            raise ValueError('Registered fixed frozen source-only ensemble required')
    elif manifest.get('owned_detector_ensemble') is not None:
        raise ValueError('Unregistered detector ensemble')
    flow_contract=codex.get('flow_spatial_tta')
    flow_transfer=flow_contract if flow_contract and flow_contract.get('exposed_transfer_only') is True else None
    if flow_contract is not None:
        row=manifest.get('flow_spatial_tta',{}); execution=row.get('execution',{})
        motion=row.get('motion_execution',{}); frozen=row.get('frozen_hashes',{})
        calls=sum(r['processed_frames']-1 for r in manifest['records'])
        if (row.get('contract')!=flow_contract or not codex.get('detector_spatial_tta')
            or audit is not None or transfer is not None or detector_calibrated is not None or ensemble is not None
            or codex.get('association_calibration') is not None
            or flow_contract.get('checkpoint_sha256')!=manifest['checkpoint_sha256']
            or (flow_transfer is None and flow_contract.get('source_validation_only') is not True)
            or execution.get('views')!=8 or execution.get('forward_calls')!=calls
            or execution.get('output_precision')!='FP32 arithmetic mean'
            or not 0<execution.get('maximum_mean_absolute_flow_delta_um',0)<float('inf')
            or motion.get('skip_zero_neural') is not True or motion.get('neural_forward_calls')!=0
            or motion.get('zero_weight_skips')!=calls
            or frozen.get('flow_before')!=flow_contract.get('flow_tensor_sha256')
            or frozen.get('flow_after')!=flow_contract.get('flow_tensor_sha256')
            or frozen.get('neural_before')!=frozen.get('neural_after') or len(frozen.get('neural_before',''))!=64
            or manifest.get('node_reference_manifest_sha256')!=flow_contract.get('reference_manifest_sha256')
            or any(r.get('reference_nodes_identical') is not True for r in manifest['records'])):
            raise ValueError('Registered fixed-node frozen motion-only source recipe required')
    elif manifest.get('flow_spatial_tta') is not None:
        raise ValueError('Unregistered motion-field averaging')
    if flow_transfer is not None:
        from research.flow_transfer_contract import STEMS,REFERENCE_SHA,SOURCE_SHA
        if (flow_transfer.get('source_validation_only') is not False
            or flow_transfer.get('stems')!=STEMS or flow_transfer.get('source_comparison_sha256')!=SOURCE_SHA
            or flow_transfer.get('reference_manifest_sha256')!=REFERENCE_SHA
            or flow_transfer.get('independent_confirmation') is not False
            or flow_transfer.get('new_target_movies_opened')!=0
            or manifest['target_audit_opened'] is not True or codex.get('target_audit_opened') is not True
            or manifest.get('embryo_audit') is not None or manifest.get('owned_transfer_diagnostic') is not None):
            raise ValueError('Exact previously exposed motion-transfer scope required')
        expected=STEMS
    elif transfer is not None:
        from research.owned_detector_transfer_contract import contract
        if (audit is not None or manifest.get('embryo_audit') is not None
            or transfer!=contract(split) or manifest.get('owned_transfer_diagnostic')!=transfer
            or manifest['target_audit_opened'] is not True or codex.get('target_audit_opened') is not True
            or manifest['checkpoint_sha256']!=transfer['checkpoint_sha256']
            or codex.get('owned_detector_fit',{}).get('objective')!='pu'):
            raise ValueError('Exact previously exposed diagnostic scope and frozen PU required')
        expected = transfer['stems']
    elif manifest.get('owned_transfer_diagnostic') is not None:
        raise ValueError('Unregistered transfer diagnostic')
    elif audit is not None:
        from research.embryo_audit_contract import contract
        if (audit!=contract(split) or manifest.get('embryo_audit')!=audit
            or manifest['target_audit_opened'] is not True or codex.get('target_audit_opened') is not True
            or not codex.get('detector_spatial_tta') or manifest['checkpoint_sha256']!=audit['checkpoint_sha256']):
            raise ValueError('Exact frozen detector candidate and explicit embryo audit required')
        expected = audit['stems']
    elif manifest['target_audit_opened'] is not False or manifest.get('embryo_audit') is not None:
        raise ValueError('Unregistered target embryo audit')
    if codex.get('association_calibration') is not None:
        if manifest.get('association_calibration')!=codex['association_calibration']:
            raise ValueError('Exact fitted association calibration receipt required')
    elif manifest.get('association_calibration') is not None:
        raise ValueError('Unregistered association calibration')
    if codex.get('detector_spatial_tta'):
        receipt = manifest.get('detector_spatial_tta',{})
        expected_flow = dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
            flow_checkpoint_sha256=INFER['image_motion_contract']()['flow_checkpoint_sha256'])
        if (manifest.get('standalone_image_flow')!=expected_flow or codex.get('standalone_image_flow')!=expected_flow
            or receipt.get('version')!=1 or receipt.get('views')!=8
            or receipt.get('features')!='native unchanged'
            or receipt.get('logits')!='inverse-aligned XY D4 arithmetic mean'
            or receipt.get('maximum_mean_absolute_logit_delta',0)<=0
            or receipt.get('encode_calls')!=sum(r['processed_frames']-1 for r in manifest['records'])):
            raise ValueError('Exact detector-only D4 and frozen standalone flow receipts required')
    elif manifest.get('detector_spatial_tta') is not None or manifest.get('standalone_image_flow') is not None:
        raise ValueError('Unregistered changed-detector/standalone-flow policy')
    if codex.get('image_motion'):
        if (manifest.get('image_motion_training') != INFER['image_motion_contract']()
            or len(manifest.get('frozen_flow_sha256','')) != 64
            or manifest.get('frozen_flow_sha256') != codex.get('frozen_flow_sha256')):
            raise ValueError('Frozen image-motion training contract mismatch')
    if codex.get('division_specialist'):
        if (manifest.get('division_specialist_training') != dict(version=1,division_window_mass=.5)
            or not math.isclose(manifest.get('division_sampling',{}).get('division_mass',0),.5,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('Missing division-specialist training receipt')
    if codex.get('known_null'):
        if manifest.get('known_null_training') != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False):
            raise ValueError('Known-null training provenance mismatch')
        if not codex.get('detector_spatial_tta') and (not manifest.get('node_reference_manifest_sha256')
            or any(r.get('reference_nodes_identical') is not True for r in manifest['records'])):
            raise ValueError('Missing known-null native-node preservation receipt')
    if codex.get('edge_feature_tta'):
        receipt = manifest.get('edge_feature_tta',{})
        if (receipt.get('views') != 8 or receipt.get('encode_calls',0) < 1
            or receipt.get('maximum_mean_absolute_feature_delta',0) <= 0
            or not manifest.get('node_reference_manifest_sha256')
            or any(r.get('reference_nodes_identical') is not True for r in manifest['records'])):
            raise ValueError('Missing native-node preservation or feature TTA receipt')
    if [r['stem'] for r in manifest['records']] != expected:
        raise ValueError('Incomplete or unexpected movie coverage')
    for record in manifest['records']:
        shape = record['image_shape']
        if len(shape) != 4 or any(not isinstance(n,int) or n <= 0 for n in shape):
            raise ValueError('Invalid image shape')
        if record['processed_frames'] != shape[0]:
            raise ValueError('Incomplete movie')


def prepare(root, notebook):
    nb = json.loads(notebook.read_text())
    bundles = PILOT['embedded_sources'](notebook)
    for name, folder, filename in [('sources','repo','source_hashes.json'),
                                  ('runtime_sources','runtime','runtime_hashes.json')]:
        expected = {p: hashlib.sha256(s.encode()).hexdigest() for p,s in bundles[name].items()}
        if json.loads((root/filename).read_text()) != expected:
            raise ValueError('Source manifest mismatch')
        for path,digest in expected.items():
            if PILOT['sha'](root/folder/path) != digest:
                raise ValueError('Source file mismatch')
    split_path = root/'runtime/split.json'
    split = json.loads(split_path.read_text())
    manifest = json.loads((root/'outputs/selection_manifest.json').read_text())
    terminal = json.loads((root/'launcher_terminal.json').read_text())
    if nb['metadata']['codex'].get('embryo_audit') is not None:
        from research.embryo_audit_contract import verify_gate
        if verify_gate((root/'runtime/frozen_selection_report.json').read_bytes(),split_path.read_bytes())!=nb['metadata']['codex']['embryo_audit']:
            raise ValueError('Frozen source-selection audit authorization mismatch')
    if nb['metadata']['codex'].get('owned_transfer_diagnostic') is not None:
        from research.owned_detector_transfer_contract import verify
        if verify((root/'runtime/frozen_owned_comparison.json').read_bytes(),split_path.read_bytes())!=nb['metadata']['codex']['owned_transfer_diagnostic']:
            raise ValueError('Frozen failed-promotion diagnostic provenance mismatch')
    if nb['metadata']['codex'].get('detector_confidence_calibration') is not None:
        from research.detector_confidence_calibration import receipt
        if receipt((root/'runtime/detector_calibration_result.json').read_bytes(),split,manifest['checkpoint_sha256'])!=nb['metadata']['codex']['detector_confidence_calibration']:
            raise ValueError('Detector cutoff lacks verified full training calibration')
    # Sources and their manifests have both been checked above. Bind each
    if nb['metadata']['codex'].get('flow_spatial_tta') is not None:
        flow_policy=nb['metadata']['codex']['flow_spatial_tta']
        if flow_policy.get('exposed_transfer_only') is True:
            from research.flow_transfer_contract import verify
            expected_flow=verify((root/'runtime/frozen_flow_source_comparison.json').read_bytes(),
                (root/'runtime/flow_spatial_tta_probe.json').read_bytes(),split_path.read_bytes())
        else:
            from research.flow_spatial_tta_contract import receipt
            expected_flow=receipt((root/'runtime/flow_spatial_tta_probe.json').read_bytes(),split_path.read_bytes())
        if expected_flow!=flow_policy:
            raise ValueError('Motion-field smoke report provenance mismatch')
    if nb['metadata']['codex'].get('owned_detector_ensemble') is not None:
        from research.owned_detector_ensemble_contract import receipt
        if receipt((root/'runtime/ensemble_probe_report.json').read_bytes(),split_path.read_bytes())!=nb['metadata']['codex']['owned_detector_ensemble']:
            raise ValueError('Ensemble smoke report provenance mismatch')
    # identity to its actual producer, preserving all original artifact bytes.
    expected_id = emitted_manifest_run_id(bundles['runtime_sources']['run_selection.py'])
    verify_manifest(manifest,terminal,nb['metadata']['codex'],split,
                    expected_manifest_run_id=expected_id)
    if manifest['manifest_sha256'] != PILOT['sha'](split_path):
        raise ValueError('Split hash mismatch')
    import numpy as np
    import tracksdata as td
    from geff import GeffMetadata
    from research.empty_graph_schema import restore_empty_spatial_schema
    from research.focus3d_bridge_rescue import _validate_graph
    prepared = {}
    # Every prediction is validated before any ground truth is opened.
    for record in manifest['records']:
        stem = record['stem']
        path = root/'outputs'/(stem+'.geff')
        if INFER['tree_hash'](path) != record['graph_sha256']:
            raise ValueError('Graph checksum mismatch')
        graph = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        restore_empty_spatial_schema(graph)
        if (graph.num_nodes(),graph.num_edges()) != (record['predicted_nodes'],record['predicted_edges']):
            raise ValueError('Graph counts mismatch')
        nodes = {int(r['node_id']):r for r in graph.node_attrs().iter_rows(named=True)}
        edges = list(graph.edge_attrs().iter_rows(named=True))
        _validate_graph(nodes,edges)
        coords = np.asarray([[r[k] for k in ('t','z','y','x')] for r in nodes.values()]).reshape(-1,4)
        if not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords >= record['image_shape']):
            raise ValueError('Coordinates outside movie')
        prepared[stem] = graph
    return prepared,manifest


def score(root, notebook, truth_root):
    import tracksdata as td
    from geff import GeffMetadata
    prepared,manifest = prepare(root,notebook)
    scorer = load_scorer(ROOT/'.biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot')
    smoke_empty_edges(scorer)
    rows = []
    for stem,graph in prepared.items():
        path = truth_root/(stem+'.geff')
        truth = td.graph.IndexedRXGraph.from_geff(str(path))[0]
        er = scorer.evaluate(graph,truth,scale=(1.625,.40625,.40625),max_distance=7.)
        total = float(GeffMetadata.read(str(path)).extra['estimated_number_of_nodes'])
        rows.append(dict(scorer.per_sample_metrics(er,total,diagnostic_node_recall(scorer,graph,truth)),
                         stem=stem,embryo=stem.split('_')[0]))
    audit = manifest.get('embryo_audit')
    transfer = manifest.get('owned_transfer_diagnostic')
    flow_transfer = manifest.get('flow_spatial_tta',{}).get('contract',{}).get('exposed_transfer_only',False)
    return dict(status='scored_exposed_flow_transfer_diagnostic' if flow_transfer else 'scored_exposed_transfer_diagnostic' if transfer is not None else 'scored_complete_embryo_audit' if audit is not None else 'scored_complete_selection',per_movie=rows,
        source_manifest_run_id=manifest['run_id'],
        summary=scorer.summarise(rows),by_embryo={embryo:scorer.summarise([r for r in rows if r['embryo']==embryo]) for embryo in sorted({r['embryo'] for r in rows})},
        checkpoint_sha256=manifest['checkpoint_sha256'],
        validation_scope=manifest['validation_scope'],
        target_audit_opened=audit is not None or transfer is not None or flow_transfer,embryo_audit=audit,
        flow_transfer_diagnostic=manifest['flow_spatial_tta']['contract'] if flow_transfer else None,
        owned_transfer_diagnostic=transfer,authorized_for_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_root',type=Path)
    parser.add_argument('--notebook',type=Path,default=ROOT/'kaggle/biohub-independent-selection-inference-v1/biohub-independent-selection-inference-v1.ipynb')
    parser.add_argument('--truth-root',type=Path,default=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train')
    args = parser.parse_args()
    print(json.dumps(score(args.output_root,args.notebook,args.truth_root),indent=2))
