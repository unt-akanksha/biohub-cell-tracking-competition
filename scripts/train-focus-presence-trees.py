"""CPU worker receives fitting data only and writes a portable JSON model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_presence_trees import fit,features,predict


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fitting',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Never overwrite fitted model')
    if hashlib.sha256(args.fitting.read_bytes()).hexdigest()!=args.sha256:raise ValueError('Exact fitting data required')
    with np.load(args.fitting,allow_pickle=False) as saved:data={k:saved[k].copy() for k in saved.files}
    model=fit(data,'fitting');payload=json.dumps(model,indent=2,allow_nan=False)
    restored=json.loads(payload)
    if not np.array_equal(predict(model,features(data)),predict(restored,features(data))):raise ValueError('Portable JSON round trip failed')
    args.output.write_text(payload,encoding='utf-8')
    print(json.dumps(dict(trees=len(model['trees']),fitting_examples=model['fitting_examples'],portable_replay_max_abs=model['fitting_replay_max_abs'])),flush=True)
