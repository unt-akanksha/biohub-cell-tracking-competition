"""Frozen small-head screening, with source calibration before held-out access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from torch.nn import functional as F
from importlib.machinery import SourceFileLoader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fit(features, labels, columns, steps, deadline):
    x = torch.as_tensor(features[:, columns], device='cuda', dtype=torch.float32)
    y = torch.as_tensor(labels, device='cuda', dtype=torch.float32)
    if not bool((y == 0).any() and (y == 1).any()) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError('Both explicitly labeled classes required')
    mean = x.mean(0); scale = x.std(0, correction=0).clamp_min(.1)
    x = ((x - mean) / scale).clamp(-10, 10)
    weight = torch.zeros(len(columns), device='cuda', requires_grad=True)
    bias = torch.zeros((), device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=.01)
    balance = torch.where(y == 1, .5 / (y == 1).sum(), .5 / (y == 0).sum())
    for step in range(steps):
        if step % 100 == 0 and time.monotonic() > deadline:
            raise TimeoutError('Head fitting budget exhausted')
        optimizer.zero_grad(set_to_none=True)
        loss = (F.binary_cross_entropy_with_logits(x @ weight + bias, y, reduction='none') * balance).sum() + .01 * weight.square().sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite classifier loss')
        loss.backward(); optimizer.step()
    return dict(weight=weight.detach().cpu(), bias=bias.detach().cpu(), mean=mean.cpu(), scale=scale.cpu(),
                columns=torch.tensor(columns), steps=steps, final_regularized_loss=float(loss))


def predict(checkpoint, features):
    # Inference/threshold calibration on CPU avoids device-specific threshold drift.
    x = torch.as_tensor(features[:, checkpoint['columns'].numpy()], dtype=torch.float32)
    x = ((x - checkpoint['mean']) / checkpoint['scale']).clamp(-10, 10)
    return torch.sigmoid(x @ checkpoint['weight'] + checkpoint['bias']).numpy()


def calibrate(labels, probability):
    negatives = probability[labels == 0]
    if not len(negatives):
        raise ValueError('Source calibration needs known negative examples')
    return float(np.nextafter(negatives.max(), np.float32(np.inf), dtype=np.float32))


def metrics(labels, probability, threshold):
    y = labels.astype(bool); decision = probability >= threshold
    if not y.any() or y.all() or not np.isfinite(probability).all():
        raise ValueError('Invalid evaluation classes/probability')
    p = np.clip(probability.astype(np.float64), 1e-7, 1 - 1e-7)
    nll = -(labels * np.log(p) + (1 - labels) * np.log1p(-p))
    return dict(positive=int(y.sum()), negative=int((~y).sum()), tp=int((decision & y).sum()),
                fp=int((decision & ~y).sum()), fn=int((~decision & y).sum()),
                recall=float((decision & y).sum() / y.sum()), balanced_nll=float(.5 * nll[y].mean() + .5 * nll[~y].mean()),
                threshold=threshold)


def gate(result, control):
    return result['fp'] == 0 and result['recall'] >= .5 and result['balanced_nll'] < control['balanced_nll']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-proof', type=Path)
    parser.add_argument('--paired-reference', type=Path)
    args = parser.parse_args()
    contract_path = ROOT / 'CONTRACT.json'; contract = json.loads(contract_path.read_text())
    for record in contract['files']:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Classifier code contract changed')
    extraction = SourceFileLoader('division_extract_v3', str(ROOT / 'scripts/extract-division-features-v3.py')).load_module()
    extraction.gpu_guard()
    torch.set_num_threads(2)
    args.output.mkdir(exist_ok=False)
    started = time.monotonic(); deadline = started + contract['head_max_seconds']
    if args.smoke:
        x = np.array([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        checkpoint = fit(x, y, [0, 1], 200, deadline)
        p = predict(checkpoint, x)
        threshold = calibrate(y, p)
        result = metrics(y, p, threshold)
        if result['tp'] != 2 or result['fp'] != 0:
            raise ValueError('Synthetic joint classifier failed')
        path = args.output / 'smoke.pt'; torch.save(checkpoint, path)
        np.testing.assert_array_equal(p, predict(torch.load(path, weights_only=True), x))
        try:
            fit(x, np.zeros(4), [0], 2, deadline)
        except ValueError:
            pass
        else:
            raise ValueError('Single-class handling failed')
        save(args.output / 'RESULT.json', dict(status='head_smoke_passed', contract_sha256=sha(contract_path),
                                             seconds=time.monotonic() - started, metrics=result, exact_reload=True))
        print(json.dumps(result), flush=True)
        return
    proof = json.loads(args.smoke_proof.read_text())
    if proof['status'] != 'head_smoke_passed' or proof['contract_sha256'] != sha(contract_path):
        raise ValueError('Matching head functionality smoke required')
    feature_path = args.features / 'RESULT.json'; manifest = json.loads(feature_path.read_text())
    if manifest['status'] != 'features_complete' or manifest['contract_sha256'] != sha(contract_path):
        raise ValueError('Unverified features')
    results = []
    reference=None
    if args.paired_reference:
        reference=json.loads((args.paired_reference/'RESULT.json').read_text())
        if reference['contract_sha256']!='e263730431b25c681218f21d2af1673f6f5be3c85f33266aa86986a0904a3e96' or not manifest.get('selection_exact'):
            raise ValueError('Paired reference or unchanged selection proof missing')
    for member in manifest['members']:
        extraction.gpu_guard()
        source = member['source']; path = args.features / member['path']
        if sha(path) != member['sha256']:
            raise ValueError('Feature artifact changed')
        with np.load(path, allow_pickle=False) as packet:
            x = packet['features']; y = packet['labels']; embryo = packet['embryo']; role = packet['role']
        train = (embryo == source) & (role == 'optimization')
        selection = (embryo == source) & (role == 'selection')
        opposite = (embryo != source) & (role == 'selection')
        configurations = [('geometry', list(range(6))), ('cnn_origin', list(range(1310))),
                          ('transformer_origin', list(range(30)) + list(range(1310, 2590)))]
        states = {}; source_p = {}; members = []
        for name, columns in configurations:
            checkpoint = fit(x[train], y[train], columns, 2000, deadline)
            source_p[name] = predict(checkpoint, x[selection])
            checkpoint['threshold'] = calibrate(y[selection], source_p[name])
            checkpoint.update(source=source, family=name, feature_sha256=member['sha256'], encoder_provenance=member['encoders'])
            state_path = args.output / (source + '-' + name + '.pt'); torch.save(checkpoint, state_path)
            checkpoint = torch.load(state_path, weights_only=True)
            np.testing.assert_array_equal(source_p[name], predict(checkpoint, x[selection]))
            states[name] = checkpoint
            result = dict(family=name, source=metrics(y[selection], source_p[name], checkpoint['threshold']),
                          path=state_path.name, sha256=sha(state_path), source_pass=False, opposite_opened=False)
            if reference is not None:
                previous=next(m for f in reference['folds'] if f['source']==source for m in f['members'] if m['family']==name)
                reference_path=args.paired_reference/previous['path']
                if sha(reference_path)!=previous['sha256']:raise ValueError('Old checkpoint changed')
                old_state=torch.load(reference_path,weights_only=True)
                old_metrics=metrics(y[selection],predict(old_state,x[selection]),old_state['threshold'])
                for key,value in old_metrics.items():
                    if abs(value-previous['source'][key])>1e-7:raise ValueError('Old source result not reproduced')
                result.update(reference_source=old_metrics,source_balanced_nll_delta=result['source']['balanced_nll']-old_metrics['balanced_nll'])
            if name != 'geometry':
                result['source_pass'] = gate(result['source'], members[0]['source'])
                if result['source_pass']:
                    control = metrics(y[opposite], predict(states['geometry'], x[opposite]), states['geometry']['threshold'])
                    result['opposite'] = metrics(y[opposite], predict(checkpoint, x[opposite]), checkpoint['threshold'])
                    result['opposite_geometry'] = control
                    result['opposite_opened'] = True
                    result['opposite_pass'] = gate(result['opposite'], control)
            members.append(result)
            print(json.dumps(dict(source=source, member=result)), flush=True)
        ensemble = dict(status='not_eligible_individual_gates')
        if all(r['source_pass'] and r.get('opposite_pass', False) for r in members[1:]):
            probability = (source_p['cnn_origin'] + source_p['transformer_origin']) * .5
            threshold = calibrate(y[selection], probability)
            target = (predict(states['cnn_origin'], x[opposite]) + predict(states['transformer_origin'], x[opposite])) * .5
            source_result = metrics(y[selection], probability, threshold)
            opposite_result = metrics(y[opposite], target, threshold)
            passed = gate(source_result, members[0]['source']) and gate(opposite_result, members[1]['opposite_geometry'])
            ensemble = dict(status='screen_passed' if passed else 'screen_rejected', weights=[.5, .5],
                            threshold=threshold, source=source_result, opposite=opposite_result)
        results.append(dict(source=source, members=members, ensemble=ensemble))
    result = dict(status='division_head_screen_complete', seconds=time.monotonic() - started,
                  contract_sha256=sha(contract_path), feature_manifest_sha256=sha(feature_path), folds=results,
                  target_pilot_labels_used=False, authorized_for_submission=False)
    save(args.output / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
