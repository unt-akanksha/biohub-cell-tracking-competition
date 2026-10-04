"""Small isolated cross-environment head replay; no training or promotion."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;started=time.monotonic()
    output=root/'result.json'
    if output.exists():raise ValueError('Never overwrite a GPU probe result')
    manifest=json.loads((root/'manifest.json').read_text())
    if any(sha(root/p)!=v for p,v in manifest['files'].items()):raise ValueError('Exact transferred probe files required')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU already has compute processes; do not disturb other projects')
    sys.path[:0]=[str(root/'repo/scripts'),str(root/'repo/src'),str(root/'runtime')]
    import numpy as np
    import torch
    from train_unet_transformer import UNetNodeTransformer,TemporalUNet3D
    from independent_real_baseline import install_empty_attention_guard
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.device_count()!=1:raise ValueError('Declared single A10G environment required')
    torch.cuda.set_per_process_memory_fraction(.2,0)
    checkpoint=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=UNetNodeTransformer(TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda(0)
    model.load_state_dict(checkpoint['model'],strict=True);install_empty_attention_guard(model);model.requires_grad_(False).eval()
    def tensor_hash():
        digest=hashlib.sha256()
        for k,v in model.state_dict().items():digest.update(k.encode()+b'\0'+v.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before=tensor_hash()
    if before!=manifest['model_tensor_sha256']:raise ValueError('Original model tensors required')
    records=[]
    with torch.no_grad():
        for item in manifest['pairs']:
            with np.load(root/item['packet'],allow_pickle=False) as loaded:packet={k:loaded[k].copy() for k in loaded.files}
            with np.load(root/item['reference'],allow_pickle=False) as loaded:reference=loaded['neural_'+str(item['frame'])].copy()
            values=[torch.from_numpy(packet[k]).unsqueeze(0).cuda(0) for k in
                ('source_features','target_features','source_coords','target_coords','source_pos','target_pos')]
            masks=[torch.ones((1,len(packet[k])),dtype=torch.bool,device='cuda:0') for k in ('source_indices','target_indices')]
            actual=model.predict_edges(*values,*masks)[0].float().cpu().numpy()
            repeated=model.predict_edges(*values,*masks)[0].float().cpu().numpy()
            if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.array_equal(actual,repeated):
                raise ValueError('Finite shaped deterministic real-packet replay required')
            np.savez_compressed(root/('actual_'+item['stem']+'_'+str(item['frame'])+'.npz'),logits=actual)
            records.append(dict(stem=item['stem'],frame=item['frame'],shape=list(actual.shape),
                exact_kaggle_replay=bool(np.array_equal(actual,reference)),max_abs_error=float(np.abs(actual-reference).max()),
                real_parent_argmax_changes=int((actual.argmax(0)!=reference.argmax(0)).sum())))
    after=tensor_hash()
    if after!=before:raise ValueError('Frozen head mutated')
    result=dict(status='completed_antelume_frozen_head_replay',records=records,model_before=before,model_after=after,
        exact_kaggle_replay=all(r['exact_kaggle_replay'] for r in records),torch_version=torch.__version__,numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_memory_fraction_cap=.2,optimizer_run=False,new_training_data_opened=False,source_selection_opened=False,
        new_target_movies_opened=0,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
