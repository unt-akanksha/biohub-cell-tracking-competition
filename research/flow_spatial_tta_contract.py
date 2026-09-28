"""Frozen smoke authorizes only source-only motion validation with fixed nodes."""
import hashlib
import json

PROBE_SHA='a6a26aafabcaaff8c775e4af37b779bf6959805948097dc0b147e8779b0c93fe'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
PARENT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
REFERENCE_SHA='f77b9eff4b965afcd9e4f571e4e32f09f23c894e962bb712ed66b83e75b35679'
FLOW_TENSOR_SHA='e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779'


def receipt(payload,split_payload):
    if hashlib.sha256(payload).hexdigest()!=PROBE_SHA or hashlib.sha256(split_payload).hexdigest()!=SPLIT_SHA:
        raise ValueError('Exact verified three-arm flow probe and split required')
    report=json.loads(payload); result=report['result']
    if (report['status']!='verified_flow_spatial_tta_probe_not_selection'
        or result['checkpoint_sha256']!=PARENT_SHA or result['zero_weight_graph_identical'] is not True
        or result['detections_identical'] is not True or report['authorized_for_submission'] is not False):
        raise ValueError('Verified graph-preserving execution and flow probe required')
    return dict(version=1,checkpoint_sha256=PARENT_SHA,probe_report_sha256=PROBE_SHA,
        views=8,flow_tensor_sha256=FLOW_TENSOR_SHA,reference_manifest_sha256=REFERENCE_SHA,
        skip_zero_neural=True,detector_coordinates='exact retained parent D4 reference',
        flow_precision='FP32 arithmetic mean',source_validation_only=True,authorized_for_submission=False)


def validate_request(args):
    if (getattr(args,'flow_spatial_tta_reference',None) is None
        or args.sha256!=PARENT_SHA or not getattr(args,'standalone_image_flow',False)
        or not getattr(args,'detector_spatial_tta',False)
        or any(getattr(args,key,None) for key in ('calibration_json','detector_calibration_json',
            'detector_ensemble_secondary','node_reference','edge_feature_tta','embryo_audit','owned_transfer_diagnostic'))):
        raise ValueError('Flow-only source validation requires unchanged parent D4 and fixed flow linking')
    if getattr(args,'flow_transfer_diagnostic',False):
        raise ValueError('Transfer requires its separate passed-source contract')
