"""One frozen flow-D4 transfer check on four already-exposed target movies."""
import hashlib
import json

SOURCE_SHA = '624d0fce503dac81b93e6f2ff6f7113ad62ec4f3999c7200dc943a1afe64206f'
REFERENCE_SHA = 'cbc49f2ba442b9ff1f8cfc276e7e6291d852bc22929f362e540bae486bd84ac5'
STEMS = ['44b6_66f9292d','44b6_40c45f5a','44b6_3bb3690f','44b6_0c582fdc']


def verify(source_payload, probe_payload, split_payload):
    try:
        from flow_spatial_tta_contract import receipt
    except ModuleNotFoundError:
        from research.flow_spatial_tta_contract import receipt
    base = receipt(probe_payload, split_payload)
    if hashlib.sha256(source_payload).hexdigest() != SOURCE_SHA:
        raise ValueError('Exact passed motion source comparison required')
    source = json.loads(source_payload)
    if (source['status'] != 'verified_flow_spatial_tta_source_comparison'
        or source['versus_frozen_d4']['passes_predeclared_source_gate'] is not True
        or source['cpu_status'] != 'COMPLETE' or source['authorized_for_submission'] is not False):
        raise ValueError('Completed positive source gate required')
    fold = json.loads(split_payload)['folds'][0]
    if (fold['initial_complete_movie_audit'] != STEMS or fold['audit_order'][:4] != STEMS
        or set(STEMS) & set(fold['train']+fold['selection'])):
        raise ValueError('Only the four already-exposed target movies allowed')
    base.update(source_validation_only=False,exposed_transfer_only=True,
        stems=STEMS.copy(),source_comparison_sha256=SOURCE_SHA,
        reference_manifest_sha256=REFERENCE_SHA,new_target_movies_opened=0,
        independent_confirmation=False)
    return base


def validate_request(args):
    try:
        from flow_spatial_tta_contract import PARENT_SHA
    except ModuleNotFoundError:
        from research.flow_spatial_tta_contract import PARENT_SHA
    if (getattr(args,'flow_transfer_diagnostic',False) is not True
        or getattr(args,'flow_spatial_tta_reference',None) is None
        or args.sha256 != PARENT_SHA or not getattr(args,'standalone_image_flow',False)
        or not getattr(args,'detector_spatial_tta',False)
        or any(getattr(args,k,None) for k in ('calibration_json','detector_calibration_json',
            'detector_ensemble_secondary','node_reference','edge_feature_tta','embryo_audit','owned_transfer_diagnostic'))):
        raise ValueError('Exact fixed-node motion transfer request required')


def load_reference(root, policy, split_sha):
    path = root/'outputs/selection_manifest.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise ValueError('Exact original four-movie detector-D4 audit reference required')
    manifest = json.loads(path.read_text())
    terminal = json.loads((root/'launcher_terminal.json').read_text())
    if (terminal['status'] != 'completed' or manifest['status'] != 'completed'
        or manifest['checkpoint_sha256'] != policy['checkpoint_sha256']
        or manifest['manifest_sha256'] != split_sha
        or manifest['target_audit_opened'] is not True
        or manifest['ground_truth_opened'] is not False
        or manifest['authorized_for_submission'] is not False
        or [r['stem'] for r in manifest['records']] != STEMS
        or any(r['processed_frames'] != 100 or r['image_shape'][0] != 100 for r in manifest['records'])):
        raise ValueError('Complete immutable previously exposed detector audit required')
    return dict(records={r['stem']:r for r in manifest['records']},manifest_sha256=REFERENCE_SHA)
