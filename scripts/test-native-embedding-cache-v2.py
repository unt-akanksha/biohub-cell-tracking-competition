"""CPU factorization checks: cached visual embeddings must preserve logits."""
import argparse
import json
from pathlib import Path
import sys
import torch

parser=argparse.ArgumentParser(); parser.add_argument('--module-root',type=Path,required=True)
args=parser.parse_args(); sys.path.insert(0,str(args.module_root))
from research.native_correspondence_models_v2 import VisualCorrespondence
from research.native_correspondence_inference_v2 import score_embeddings

torch.set_num_threads(2); torch.manual_seed(102)
patches=torch.randn(2,5,3,11,11,11); coords=torch.randn(2,5,3)
valid=torch.tensor([[True,True,False,True,False],[True,False,False,False,False]])
result={}
with torch.inference_mode():
    for family in ('resnet3d','token_transformer3d'):
        model=VisualCorrespondence(family,input_channels=3).eval()
        torch.nn.init.normal_(model.edge_head[-1].weight,std=.1)
        torch.nn.init.normal_(model.null_head[-1].weight,std=.1)
        reference=model(patches,coords,valid)
        flat=model.encoder(patches.flatten(0,1)[valid.flatten()])
        features=flat.new_zeros((10,256)).index_copy(0,valid.flatten().nonzero().flatten(),flat).reshape(2,5,256)
        cached=score_embeddings(model,features,coords,valid)
        error=float((reference-cached).abs().max())
        assert torch.equal(reference,cached), (family,error)
        # Encoder is normalization-batch independent: node-once extraction can
        # use a different batch partition without changing association choices.
        single=torch.cat([model.encoder(patches.flatten(0,1)[i:i+1]) for i in valid.flatten().nonzero().flatten().tolist()])
        independent=single.new_zeros((10,256)).index_copy(0,valid.flatten().nonzero().flatten(),single).reshape(2,5,256)
        independent_logits=score_embeddings(model,independent,coords,valid)
        independent_error=float((reference-independent_logits).abs().max())
        assert torch.allclose(reference,independent_logits,atol=2e-5,rtol=1e-5)
        assert torch.equal(reference.argmax(-1),independent_logits.argmax(-1))
        result[family]=dict(factorization_max_error=error,single_node_batch_max_error=independent_error)
print(json.dumps(dict(status='passed',results=result,complete_movie_runtime_tested=False)))
