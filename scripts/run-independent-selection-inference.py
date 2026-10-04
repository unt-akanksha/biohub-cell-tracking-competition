"""Frozen full-movie source-selection inference; no GT scoring on GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def selection_scope(identity, manifest):
    fold = manifest['folds'][0]
    profile = identity.get('training_profile','pilot')
    expected = fold['train'] if profile == 'full' else fold['train'][:4]
    if profile not in ('pilot','full') or identity['training_stems'] != expected:
        raise ValueError('Unexpected checkpoint training scope')
    if set(expected) & set(fold['audit_order']):
        raise ValueError('Target audit overlap in checkpoint training scope')
    selection = fold['selection']
    if len(selection) != 8 or set(selection) & set(identity['training_stems']):
        raise ValueError('Expected eight disjoint source-selection movies')
    if any(s.split('_')[0] != fold['training_embryo'] for s in selection):
        raise ValueError('Target embryo audit must remain closed')
    return selection


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode()+b'\0')
            digest.update(path.read_bytes()+b'\0')
    return digest.hexdigest()


def image_motion_contract():
    return dict(version=1,flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788',
        flow_frozen=True,displacement='child_to_parent_um_zyx',variance='unchanged_static_control',
        downsample=[1,4,4],neural_residual='preserved_then_trained')


def calibration_receipt(result,split,checkpoint_sha256,result_sha256):
    fold = split['folds'][0]
    diagnostic = fold['train'][::5]
    fitting = [s for s in fold['train'] if s not in diagnostic]
    parameters = result['fit']['parameters']
    bounds = [[0.,1.],[.25,4.],[-12.,4.]]
    if (result['status']!='completed_calibration_not_tracking_validation' or result['profile']!='full'
        or result['checkpoint_sha256']!=checkpoint_sha256 or result['fitting_stems']!=fitting
        or result['diagnostic_stems']!=diagnostic or len(fitting)!=96 or len(diagnostic)!=24
        or set(fold['train'])&set(fold['selection']+fold['audit_order'])
        or not result['functionality_passed'] or not result['probe_inputs_replayed']
        or result['frozen_before']!=result['frozen_after'] or not result['fit']['converged']
        or result['fit']['bounds']!=bounds or len(parameters)!=3
        or any(not math.isfinite(v) or not lo<=v<=hi for v,(lo,hi) in zip(parameters,bounds))
        or result['fit']['fit_metrics']['known_null_columns']<=0
        or any(result[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Completed training-only frozen calibration required')
    return dict(version=1,parameters=parameters,result_sha256=result_sha256,
        checkpoint_sha256=checkpoint_sha256,split_sha256=result['split_sha256'],
        frozen_hashes=result['frozen_after'])


def owned_detector_fit_receipt(state,probe):
    from owned_detector_logit_targets import contract as target_contract
    identity=state['identity']; objective=identity.get('owned_detector_objective')
    expected_loss='Original sparse detector BCE control' if objective=='sparse' else 'Owned frozen view-consensus positive/unlabeled detector BCE'
    if (objective not in ('sparse','pu') or identity.get('owned_detector_pu')!=target_contract()
        or identity.get('fine_tuning_profile')!='full' or identity.get('training_profile')!='full'
        or identity.get('fine_tuning_stems')!=identity['training_stems'] or len(identity['training_stems'])!=120
        or identity.get('initialization_sha256')!='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
        or identity.get('frozen_modules')!=['transformer','flow','batchnorm_statistics'] or identity.get('loss')!=expected_loss
        or identity.get('teacher_peak_order')!='raw_logits' or identity.get('teacher_probability_precision')!='float32_before_sigmoid'
        or identity.get('image_motion')!=image_motion_contract() or not state.get('frozen_flow_model')
        or identity.get('max_steps')!=1000 or state['step']!=1000 or probe is None
        or len(state.get('input_hashes',[]))!=2000 or len(state.get('target_hashes',[]))!=4000
        or state['input_hashes'][:20]!=probe['input_hashes'] or state['target_hashes'][:40]!=probe['target_hashes']
        or any(identity[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Complete source-only owned-detector profile and raw-probe replay required')
    updates=[int(v['step']) for v in state.get('optimizer',{}).get('state',{}).values() if 'step' in v]
    if not updates or min(updates)!=1000 or max(updates)!=1000:
        raise ValueError('Complete detector optimizer updates required')
    for key in ('parent_identity','known_null','image_motion','motion_residual','frozen_flow_sha256'):
        if identity.get(key) is None or identity[key]!=probe['identity'][key]:
            raise ValueError('Owned detector inherited frozen provenance changed')
    return dict(version=1,objective=objective,steps=1000,initialization_sha256=identity['initialization_sha256'],
        target_contract=identity['owned_detector_pu'],frozen_modules=identity['frozen_modules'],
        probe_result_sha256='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba')


def check_completed_profile(state,owned_probe=None):
    if state['identity'].get('owned_detector_pu') is not None:
        owned_detector_fit_receipt(state,owned_probe)
        return
    maximum = state['identity']['max_steps']
    allowed = (100,1000) if state['identity'].get('motion_residual') else (1000,)
    if state['identity'].get('image_motion'):
        identity = state['identity']
        if (identity['image_motion'] != image_motion_contract()
            or identity.get('known_null') != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False)
            or identity.get('joint_training') or identity.get('division_specialist')
            or identity.get('training_profile') != 'full' or identity.get('frozen_modules') != ['unet','detect_head']
            or identity.get('initialization_sha256') != 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'
            or identity.get('loss') != 'sparse_parent_with_annotated_missing_parent_null_v1'
            or not identity.get('motion_residual') or not state.get('frozen_flow_model')
            or len(identity.get('frozen_flow_sha256','')) != 64):
            raise ValueError('Unexpected frozen image-motion linker profile')
        allowed = (1000,)
        updates = [int(v['step']) for v in state.get('optimizer',{}).get('state',{}).values() if 'step' in v]
        if not updates or min(updates) < 900 or max(updates) > 1000:
            raise ValueError('Incomplete image-motion linker optimizer updates')
    elif state['identity'].get('division_specialist'):
        identity = state['identity']
        sampling = identity.get('division_sampling',{})
        if (identity['division_specialist'] != dict(version=1,division_window_mass=.5)
            or identity.get('known_null') != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False)
            or identity.get('joint_training') or identity.get('training_profile') != 'full'
            or identity.get('frozen_modules') != ['unet','detect_head']
            or identity.get('initialization_sha256') != 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'
            or identity.get('loss') != 'division_balanced_with_annotated_missing_parent_null_v1'
            or not identity.get('motion_residual') or sampling.get('replacement') is not True
            or sampling.get('division_windows',0) <= 0 or sampling.get('ordinary_windows',0) <= 0
            or not math.isclose(sampling.get('division_mass',0),.5,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('Unexpected division-specialist profile')
        allowed = (1000,)
        updates = [int(v['step']) for v in state.get('optimizer',{}).get('state',{}).values() if 'step' in v]
        if not updates or min(updates) < 900 or max(updates) > 1000:
            raise ValueError('Incomplete division-specialist optimizer updates')
    elif state['identity'].get('known_null'):
        identity = state['identity']
        if (identity['known_null'] != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False)
            or identity.get('joint_training') or identity.get('training_profile') != 'full'
            or identity.get('frozen_modules') != ['unet','detect_head']
            or identity.get('initialization_sha256') != 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'
            or identity.get('loss') != 'sparse_parent_with_annotated_missing_parent_null_v1'
            or not identity.get('motion_residual')):
            raise ValueError('Unexpected annotated missing-parent null profile')
        allowed = (1000,)
        updates = [int(v['step']) for v in state.get('optimizer',{}).get('state',{}).values() if 'step' in v]
        if not updates or min(updates) < 900 or max(updates) > 1000:
            raise ValueError('Incomplete known-null optimizer updates')
    if state['identity'].get('checkpoint_bn_once'):
        identity = state['identity']
        if (not identity.get('joint_training') or identity.get('training_profile') != 'full'
            or not identity.get('motion_residual') or identity.get('bn_recalibration')
            or identity.get('loss') != 'sparse_parent_classification_with_null_v1'):
            raise ValueError('Unexpected guarded longer-fit profile')
        allowed = (6000,)
        updates = [int(v['step']) for v in state.get('optimizer',{}).get('state',{}).values() if 'step' in v]
        if not updates or min(updates) < 5400 or max(updates) > 6000:
            raise ValueError('Incomplete longer-fit optimizer updates')
    if maximum not in allowed or state['step'] != maximum:
        raise ValueError('Completed frozen optimization checkpoint required')
    calibration = state['identity'].get('bn_recalibration')
    if calibration and (calibration['status'] != 'completed' or calibration['movie_count'] != 120
        or calibration['batches'] != 120 or calibration['parameters_unchanged'] is not True):
        raise ValueError('Completed full-training BatchNorm recalibration required')


def main(args):
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('Checkpoint checksum mismatch before load')
    import numpy as np
    import torch
    import zarr
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    from train_unet_transformer import UNetNodeTransformer, TemporalUNet3D
    from predict_unet_transformer import predict_video, PredictConfig, build_graph
    from independent_real_baseline import install_empty_attention_guard
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    owned_probe=None; owned_receipt=None
    if state['identity'].get('owned_detector_pu') is not None:
        raw=(args.runtime/'raw_probe_result.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba':
            raise ValueError('Original detector raw-probe receipt required')
        owned_probe=json.loads(raw)
    check_completed_profile(state,owned_probe)
    if owned_probe is not None:
        owned_receipt=owned_detector_fit_receipt(state,owned_probe)
    manifest = json.loads(args.manifest.read_text())
    movies = selection_scope(state['identity'], manifest)
    embryo_audit = None
    transfer_diagnostic = None
    if getattr(args,'owned_transfer_diagnostic',False):
        from owned_detector_transfer_contract import verify
        transfer_diagnostic = verify((args.runtime/'frozen_owned_comparison.json').read_bytes(),args.manifest.read_bytes())
        if (getattr(args,'embryo_audit',False) or args.sha256!=transfer_diagnostic['checkpoint_sha256']
            or owned_receipt is None or owned_receipt['objective']!='pu'
            or not getattr(args,'detector_spatial_tta',False) or not getattr(args,'standalone_image_flow',False)):
            raise ValueError('Frozen PU diagnostic only, distinct from promotion or new holdout audit')
        movies = transfer_diagnostic['stems']
    if getattr(args,'embryo_audit',False):
        from embryo_audit_contract import verify_gate
        embryo_audit = verify_gate((args.runtime/'frozen_selection_report.json').read_bytes(),args.manifest.read_bytes())
        if (args.sha256!=embryo_audit['checkpoint_sha256'] or not getattr(args,'detector_spatial_tta',False)
            or not getattr(args,'standalone_image_flow',False)):
            raise ValueError('Audit requires the exact frozen detector-D4/standalone-flow candidate')
        movies = embryo_audit['stems']
    calibrated = None
    detector_calibrated = None
    calibration_path = getattr(args,'calibration_json',None)
    standalone = getattr(args,'standalone_image_flow',False)
    detector_tta = getattr(args,'detector_spatial_tta',False)
    detector_calibration_path = getattr(args,'detector_calibration_json',None)
    ensemble_path = getattr(args,'detector_ensemble_secondary',None)
    ensemble_contract = None
    flow_reference = getattr(args,'flow_spatial_tta_reference',None)
    flow_contract = None
    flow_transfer = getattr(args,'flow_transfer_diagnostic',False)
    if flow_transfer and flow_reference is None:
        raise ValueError('Flow transfer requires its immutable detector audit reference')
    if flow_reference is not None:
        if flow_transfer:
            from flow_transfer_contract import verify,validate_request
            validate_request(args)
            flow_contract=verify((args.runtime/'frozen_flow_source_comparison.json').read_bytes(),
                (args.runtime/'flow_spatial_tta_probe.json').read_bytes(),args.manifest.read_bytes())
            movies=flow_contract['stems']
        else:
            from flow_spatial_tta_contract import receipt,validate_request
            validate_request(args)
            flow_contract=receipt((args.runtime/'flow_spatial_tta_probe.json').read_bytes(),args.manifest.read_bytes())
    if ensemble_path is not None:
        from owned_detector_ensemble_contract import receipt,PU_SHA,PARENT_SHA
        if (not standalone or not detector_tta or args.sha256!=PARENT_SHA
            or embryo_audit is not None or transfer_diagnostic is not None
            or calibration_path is not None or detector_calibration_path is not None
            or getattr(args,'node_reference',None) is not None or getattr(args,'edge_feature_tta',False)):
            raise ValueError('Fixed owned ensemble permits source validation only')
        ensemble_contract=receipt((args.runtime/'ensemble_probe_report.json').read_bytes(),args.manifest.read_bytes())
        if hashlib.sha256(ensemble_path.read_bytes()).hexdigest()!=PU_SHA:
            raise ValueError('Exact frozen owned PU secondary required')
    if detector_calibration_path is not None:
        from detector_confidence_calibration import receipt
        if (not standalone or not detector_tta or owned_receipt is None or owned_receipt['objective']!='sparse'
            or embryo_audit is not None or transfer_diagnostic is not None or calibration_path is not None
            or getattr(args,'node_reference',None) is not None or getattr(args,'edge_feature_tta',False)):
            raise ValueError('Calibrated sparse detector is a separately validated source-only recipe')
        detector_calibrated=receipt(detector_calibration_path.read_bytes(),manifest,args.sha256)
    if (standalone and (calibration_path is not None or not state['identity'].get('image_motion'))
        or detector_tta and (not standalone or getattr(args,'node_reference',None) is not None
                            or getattr(args,'edge_feature_tta',False))):
        raise ValueError('Detector-only TTA requires uncalibrated standalone flow and no fixed-node reference')
    if calibration_path is not None:
        if not state['identity'].get('image_motion'):
            raise ValueError('Calibration requires the frozen integrated image model')
        payload = calibration_path.read_bytes()
        calibrated = calibration_receipt(json.loads(payload),manifest,args.sha256,hashlib.sha256(payload).hexdigest())
        if calibrated['split_sha256']!=hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('Calibration split mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for volume inference')
    model = UNetNodeTransformer(TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),
        unet_out_channels=32,pos_feat_dim=32).cuda()
    model.load_state_dict(state['model'], strict=True)
    if calibrated is not None:
        from image_motion_residual import flow_hash
        if flow_hash(model)!=calibrated['frozen_hashes']['neural']:
            raise ValueError('Calibration neural component hash mismatch')
    install_empty_attention_guard(model)
    model.eval()
    ensemble_models = None
    if ensemble_contract is not None:
        import copy
        from owned_detector_ensemble import install_owned_detector_ensemble
        from image_motion_residual import flow_hash
        raw=(args.runtime/'raw_probe_result.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba':
            raise ValueError('Original owned PU provenance required')
        secondary_state=torch.load(ensemble_path,map_location='cpu',weights_only=True)
        secondary_provenance=owned_detector_fit_receipt(secondary_state,json.loads(raw))
        if secondary_state['identity']['training_stems']!=manifest['folds'][0]['train']:
            raise ValueError('Secondary training scope mismatch')
        secondary=copy.deepcopy(model)
        secondary.load_state_dict(secondary_state['model'],strict=True)
        secondary.eval()
        ensemble_models=[(model,flow_hash(model)),(secondary,flow_hash(secondary))]
        install_owned_detector_ensemble(model,secondary)
        del secondary_state
    elif detector_tta:
        from detector_spatial_tta import install_detector_spatial_tta
        install_detector_spatial_tta(model)
    if state['identity'].get('motion_residual'):
        from motion_residual import contract,install_motion_residual
        if state['identity']['motion_residual'] != contract():
            raise ValueError('Motion residual inference contract mismatch')
        install_motion_residual(model,inference=not bool(state['identity'].get('image_motion')))
    if state['identity'].get('image_motion'):
        from image_motion_residual import embedded_flow,install_image_motion_residual
        if calibrated is not None and calibrated['frozen_hashes']['flow']!=state['identity']['frozen_flow_sha256']:
            raise ValueError('Calibration flow component hash mismatch')
        flow_model=embedded_flow(state,'cuda')
        if flow_contract is not None:
            from image_motion_residual import flow_hash
            from backward_flow_spatial_tta import install_backward_flow_spatial_tta
            flow_before=flow_hash(flow_model); neural_before=flow_hash(model)
            install_backward_flow_spatial_tta(flow_model)
        install_image_motion_residual(model,flow_model,inference=True,
            calibration=[0.,1.,-4.5] if standalone else None if calibrated is None else calibrated['parameters'],
            skip_zero_neural=flow_contract is not None)
    model.eval()
    feature_tta = getattr(args,'edge_feature_tta',False)
    reference = None
    reference_root = flow_reference if flow_reference is not None else getattr(args,'node_reference',None)
    if flow_contract is not None:
        if flow_transfer:
            from flow_transfer_contract import load_reference
            reference=load_reference(flow_reference,flow_contract,hashlib.sha256(args.manifest.read_bytes()).hexdigest())
        else:
            from feature_tta_reference import load_reference
            reference=load_reference(flow_reference,args.sha256,hashlib.sha256(args.manifest.read_bytes()).hexdigest(),movies)
        if reference['manifest_sha256']!=flow_contract['reference_manifest_sha256']:
            raise ValueError('Exact retained parent detector D4 reference required')
    elif feature_tta:
        from edge_feature_tta import install_edge_feature_tta
        from feature_tta_reference import load_reference
        reference = load_reference(args.node_reference,args.sha256,
            hashlib.sha256(args.manifest.read_bytes()).hexdigest(),movies)
        install_edge_feature_tta(model)
    elif getattr(args,'node_reference',None) is not None:
        if not state['identity'].get('known_null'):
            raise ValueError('Cross-checkpoint node reference requires a frozen known-null detector')
        from feature_tta_reference import load_reference
        reference = load_reference(args.node_reference,state['identity']['initialization_sha256'],
            hashlib.sha256(args.manifest.read_bytes()).hexdigest(),movies)
    original = model.predict_edges
    def bounded_edges(*values):
        if max(values[0].shape[1],values[1].shape[1]) > 2048:
            raise RuntimeError('Selection attention guard exceeded; no node truncation allowed')
        return original(*values)
    model.predict_edges = bounded_edges
    config = PredictConfig(det_threshold=float(torch.sigmoid(torch.tensor(.3))),
        det_tta=False,pool_kernel_um=5.,max_parents_per_node=1,max_children_per_node=2)
    if detector_calibrated is not None:
        config.det_threshold=detector_calibrated['threshold']
    if state['identity'].get('motion_residual'):
        config.edge_activation = 'sigmoid'
    args.output.mkdir(parents=True,exist_ok=True)
    records = []
    started = time.monotonic()
    for stem in movies:
        shape = tuple(zarr.open_group(str(args.data/(stem+'.zarr')),mode='r')['0'].shape)
        with torch.no_grad():
            coords, edges = predict_video(model,args.data/stem,torch.device('cuda'),config,
                window_size=2,unet_batch_size=1,downsample=(1,4,4))
        if not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords >= np.asarray(shape)):
            raise ValueError('Invalid predicted coordinates')
        if reference is not None:
            from feature_tta_reference import verify_coordinates
            verify_coordinates(coords,reference_root/'outputs'/(stem+'.geff'),
                               reference['records'][stem]['graph_sha256'],tree_hash)
        graph = build_graph(coords,edges)
        path = args.output/(stem+'.geff')
        graph.to_geff(path)
        records.append(dict(stem=stem,image_shape=list(shape),processed_frames=shape[0],
            predicted_nodes=len(coords),predicted_edges=len(edges),graph_sha256=tree_hash(path)))
        if reference is not None:
            records[-1]['reference_nodes_identical'] = True
            records[-1]['reference_graph_sha256'] = reference['records'][stem]['graph_sha256']
        print(json.dumps(records[-1]),flush=True)
    result = dict(status='completed',run_id='independent-selection-inference-v1',
        checkpoint_sha256=args.sha256,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        records=records,elapsed_seconds=time.monotonic()-started,
        validation_scope=f"Eight source-embryo selection movies excluded from this {len(state['identity']['training_stems'])}-movie fit",
        target_audit_opened=False,ground_truth_opened=False,authorized_for_submission=False)
    if feature_tta:
        result.update(edge_feature_tta=model._biohub_edge_feature_tta)
    if reference is not None:
        result['node_reference_manifest_sha256'] = reference['manifest_sha256']
    if state['identity'].get('known_null'):
        result['known_null_training'] = state['identity']['known_null']
    if state['identity'].get('division_specialist'):
        result['division_specialist_training'] = state['identity']['division_specialist']
        result['division_sampling'] = state['identity']['division_sampling']
    if state['identity'].get('image_motion'):
        result['image_motion_training'] = state['identity']['image_motion']
        result['frozen_flow_sha256'] = state['identity']['frozen_flow_sha256']
    if calibrated is not None:
        result['association_calibration'] = calibrated
    if standalone:
        result['standalone_image_flow'] = dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
            flow_checkpoint_sha256=image_motion_contract()['flow_checkpoint_sha256'])
    if detector_tta:
        result['detector_spatial_tta'] = model._biohub_detector_tta
    if flow_contract is not None:
        frozen=dict(flow_before=flow_before,flow_after=flow_hash(flow_model),
            neural_before=neural_before,neural_after=flow_hash(model))
        if frozen['flow_before']!=frozen['flow_after'] or frozen['neural_before']!=frozen['neural_after']:
            raise ValueError('Motion-only validation changed frozen model tensors')
        result['flow_spatial_tta']=dict(contract=flow_contract,execution=flow_model._biohub_flow_tta,
            motion_execution=model._image_motion_execution,frozen_hashes=frozen)
        if flow_transfer:
            result.update(target_audit_opened=True,
                validation_scope='Four previously exposed44b6 movies; fixed-node flowD4 transfer diagnostic, not independent confirmation')
    if ensemble_contract is not None:
        frozen=[dict(before=before,after=flow_hash(item)) for item,before in ensemble_models]
        if any(r['before']!=r['after'] for r in frozen):
            raise ValueError('Frozen ensemble component mutated during inference')
        result['owned_detector_ensemble']=dict(contract=ensemble_contract,
            execution=model._biohub_detector_ensemble,frozen_hashes=frozen,
            secondary_provenance=secondary_provenance)
    if owned_receipt is not None:
        result['owned_detector_fit']=owned_receipt
    if detector_calibrated is not None:
        result['detector_confidence_calibration']=detector_calibrated
    if embryo_audit is not None:
        result.update(embryo_audit=embryo_audit,target_audit_opened=True,
            validation_scope='First four complete44b6 movies: embryo excluded from training and source selection; frozen candidate audit, not tuning')
    if transfer_diagnostic is not None:
        result.update(owned_transfer_diagnostic=transfer_diagnostic,target_audit_opened=True,
            validation_scope='Four previously exposed44b6 movies; frozen PU transfer diagnostic, NOT independent confirmation or promotion')
    (args.output/'selection_manifest.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--edge-feature-tta',action='store_true')
    parser.add_argument('--node-reference',type=Path)
    parser.add_argument('--calibration-json',type=Path)
    parser.add_argument('--detector-calibration-json',type=Path)
    parser.add_argument('--detector-ensemble-secondary',type=Path)
    parser.add_argument('--flow-spatial-tta-reference',type=Path)
    parser.add_argument('--flow-transfer-diagnostic',action='store_true')
    parser.add_argument('--standalone-image-flow',action='store_true')
    parser.add_argument('--detector-spatial-tta',action='store_true')
    parser.add_argument('--embryo-audit',action='store_true')
    parser.add_argument('--owned-transfer-diagnostic',action='store_true')
    main(parser.parse_args())
