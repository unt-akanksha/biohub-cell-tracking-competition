"""Read-only-in-models cross-embryo selection diagnostic for frozen members.

Run only after the sequential trainer terminates. No optimization, thresholds,
public scores, model updates, sealed-audit shards, or submission graphs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    start = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve completed diagnostic')
    pids = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    if pids:
        raise ValueError('GPU in use; do not overlap experiments')
    expected = 'a3a59450791c4d6aef7e75f8d7e2a40c5a41f9dc1a4a14d0440f0a1f0e1712d0'
    if sha(args.bundle/'BUNDLE.json') != expected:
        raise ValueError('Exact repaired bundle required')
    contract = json.loads((args.bundle/'BUNDLE.json').read_text())
    terminal = json.loads((args.training/'result.json').read_text())
    if not (terminal['status'] == 'training_completed_requires_validation' and len(terminal['members']) == 4
            and terminal['bundle_sha256'] == expected):
        raise ValueError('Complete four-member training terminal required')
    for name, digest in contract['files'].items():
        if sha(args.bundle/name) != digest:
            raise ValueError('Frozen source changed')
    sys.path.insert(0, str(args.bundle))
    from visual_correspondence_models_v1 import VisualCorrespondence, prepare_patches, mask_scores
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.9)
    device = torch.device('cuda')
    result = dict(run_id='visual-correspondence-ensemble-v1-cross-embryo',
                  training_terminal_sha256=sha(args.training/'result.json'), bundle_sha256=expected,
                  policy='fixed equal-probability mixture; only source-screen-admitted members',
                  members=[], folds=[], complete_movie_metric_gate_passed=False,
                  authorized_for_submission=False, weights_updated=False, sealed_audit_opened=False)

    def packet(embryo):
        parts = {k: [] for k in ('patches', 'ids', 'coords', 'masks', 'targets')}
        identities, offset = [], 0
        for record in contract['records']:
            if record['role'] != 'selection' or record['embryo'] != embryo:
                continue
            path = args.bundle/record['path']
            if sha(path) != record['sha256']:
                raise ValueError('Selection packet changed')
            with np.load(path, allow_pickle=False) as a:
                arrays = {k: a[k].copy() for k in parts}
            arrays['ids'][arrays['ids'] >= 0] += offset
            offset += len(arrays['patches'])
            for key, value in arrays.items():
                parts[key].append(torch.from_numpy(value))
            identities.extend([record['stem']]*len(arrays['targets']))
        return {k: torch.cat(v).to(device) for k, v in parts.items()}, identities

    @torch.inference_mode()
    def predict(model, data, missing=False):
        if model is not None:
            model.eval()
        probabilities, targets = [], []
        for first in range(0, len(data['targets']), 12):
            ids = data['ids'][first:first+12].long(); valid = ids >= 0
            coords = data['coords'][first:first+12].clone()
            mask = data['masks'][first:first+12].clone(); target = data['targets'][first:first+12].long().clone()
            if missing:
                rows = torch.nonzero(target < 16).flatten()
                valid[rows, target[rows]+1] = False; mask[rows, target[rows]] = False; target[rows] = 16
            x, coords = prepare_patches(data['patches'][ids.clamp_min(0)], coords, valid)
            if model is None:
                delta2 = ((coords[:, 1:]-coords[:, :1])/10).square().sum(-1)
                scores = torch.cat((-2*delta2, torch.full_like(delta2[:, :1], -8)), -1)
                scores.masked_fill_(~torch.cat((valid[:, 1:], valid[:, :1]), -1), -1e4)
            else:
                with torch.autocast('cuda', dtype=torch.float16):
                    scores = model(x, coords, valid)
            scores = mask_scores(scores.float(), mask)
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite cross-embryo predictions')
            probabilities.append(torch.softmax(scores, -1).cpu().numpy()); targets.append(target.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(targets)

    def metrics(probabilities, target, identities):
        chosen = probabilities.argmax(-1)
        losses = -np.log(np.maximum(probabilities[np.arange(len(target)), target], 1e-30))
        return dict(nll=float(losses.mean()), correct=int(np.sum(chosen == target)), total=len(target),
                    per_movie={stem: dict(nll=float(losses[np.array(identities) == stem].mean()),
                        correct=int(np.sum((chosen == target)[np.array(identities) == stem])),
                        total=int(np.sum(np.array(identities) == stem))) for stem in sorted(set(identities))})

    for source in contract['source_embryos']:
        members = [m for m in terminal['members'] if m['embryo'] == source]
        admitted = [m for m in members if m['source_screen_passed']]
        result['members'].extend(dict(source_embryo=source, family=m['family'],
             source_screen_passed=m['source_screen_passed'], best_step=m['best_step'],
             checkpoint_sha256=m['checkpoint_sha256'], source_selected=m['selected'],
             source_missing=m['missing']) for m in members)
        if len(admitted) != 2:
            result['folds'].append(dict(source=source, status='no_two_strong_source_members', target_opened=False))
            continue
        target_embryo = '44b6' if source == '6bba' else '6bba'
        data, identities = packet(target_embryo)
        baseline_p, target = predict(None, data)
        baseline_missing, missing_target = predict(None, data, True)
        fold = dict(source=source, target=target_embryo, target_opened=True,
                    baseline=metrics(baseline_p, target, identities),
                    missing_baseline=metrics(baseline_missing, missing_target, identities), members=[])
        all_p, all_missing = [], []
        for member in admitted:
            checkpoint = args.training/f"{source}-{member['family']}"/'best.pt'
            if sha(checkpoint) != member['checkpoint_sha256']:
                raise ValueError('Frozen selected checkpoint changed')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = VisualCorrespondence(member['family']).to(device)
            model.load_state_dict(payload['state_dict'], strict=True)
            p, truth = predict(model, data); pm, tm = predict(model, data, True)
            assert np.array_equal(truth, target) and np.array_equal(tm, missing_target)
            all_p.append(p); all_missing.append(pm)
            fold['members'].append(dict(family=member['family'], metrics=metrics(p, target, identities),
                                       missing=metrics(pm, missing_target, identities)))
            del model, payload; torch.cuda.empty_cache()
        mixture = np.mean(all_p, axis=0); mixture_missing = np.mean(all_missing, axis=0)
        fold['mixture'] = metrics(mixture, target, identities)
        fold['mixture_missing'] = metrics(mixture_missing, missing_target, identities)
        correct = [p.argmax(-1) == target for p in all_p]
        fold['complementarity'] = dict(first_only_correct=int(np.sum(correct[0] & ~correct[1])),
            second_only_correct=int(np.sum(correct[1] & ~correct[0])), both_wrong=int(np.sum(~correct[0] & ~correct[1])))
        fold['conditional_screen_passed'] = (
            all(m['metrics']['correct'] >= fold['baseline']['correct'] and m['metrics']['nll'] < fold['baseline']['nll'] for m in fold['members'])
            and fold['mixture']['correct'] >= max(m['metrics']['correct'] for m in fold['members'])
            and fold['mixture']['nll'] < min(m['metrics']['nll'] for m in fold['members'])
            and fold['mixture_missing']['nll'] <= fold['missing_baseline']['nll'])
        fold['status'] = 'conditional_pass_requires_full_movie' if fold['conditional_screen_passed'] else 'conditional_rejected'
        result['folds'].append(fold)
        del data; torch.cuda.empty_cache()
    result.update(status='completed_diagnostic', elapsed_seconds=time.monotonic()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial'); temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.output)
    print(json.dumps(dict(status=result['status'], seconds=result['elapsed_seconds'],
                          folds=[dict(source=f['source'], status=f['status']) for f in result['folds']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    run(args)
