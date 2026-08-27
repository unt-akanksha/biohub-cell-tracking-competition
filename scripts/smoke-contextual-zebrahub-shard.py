from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.temporal_contrastive.contextual_pair_fusion import (  # noqa: E402
    CONTEXTUAL_PAIR_FUSION_FAMILY,
    EXPECTED_PARAMETER_COUNT,
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.model import (  # noqa: E402
    masked_multi_positive_info_nce,
)
from research.temporal_contrastive.pair_fusion import (  # noqa: E402
    masked_multi_positive_pair_nll,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU forward/backward smoke for one frozen ZebraHub v3 shard"
    )
    parser.add_argument("--shard", type=Path, required=True)
    args = parser.parse_args()
    shard = args.shard.resolve()
    seed = 31_003
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    with np.load(shard) as data:
        model = ContextualPairFusionAssociationModel()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(f"unexpected v3 parameter count: {parameter_count}")
        source_patches = torch.as_tensor(
            data["source_patches"].astype(np.float32)
        )
        target_patches = torch.as_tensor(
            data["target_patches"].astype(np.float32)
        )
        source_embeddings, division_logits = model(source_patches)
        target_embeddings, _ = model(target_patches)
        candidates = torch.as_tensor(data["candidate_mask"], dtype=torch.bool)
        positives = torch.as_tensor(data["positive_mask"], dtype=torch.bool)
        logits = model.candidate_pair_logits(
            source_embeddings,
            target_embeddings,
            torch.as_tensor(data["source_coords_um"], dtype=torch.float32),
            torch.as_tensor(data["target_coords_um"], dtype=torch.float32),
            division_logits,
            candidates,
            torch.as_tensor(data["candidate_context"], dtype=torch.float32),
        )
        pair_loss = masked_multi_positive_pair_nll(logits, positives, candidates)
        embedding_loss = masked_multi_positive_info_nce(
            source_embeddings,
            target_embeddings,
            positives,
            candidates,
            temperature=0.10,
        )
        combined_loss = pair_loss + 0.25 * embedding_loss
        combined_loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        report = {
            "schema_version": 1,
            "family": CONTEXTUAL_PAIR_FUSION_FAMILY,
            "gpu_used": False,
            "seed": seed,
            "shard_path": str(shard),
            "shard_sha256": sha256_file(shard),
            "parameter_count": parameter_count,
            "source_embedding_shape": list(source_embeddings.shape),
            "target_embedding_shape": list(target_embeddings.shape),
            "candidate_context_shape": list(data["candidate_context"].shape),
            "transition_context": data["transition_context"].astype(float).tolist(),
            "finite_pair_logits": int(torch.isfinite(logits).sum()),
            "pair_loss": float(pair_loss.detach()),
            "embedding_loss": float(embedding_loss.detach()),
            "combined_loss": float(combined_loss.detach()),
            "gradient_tensors": len(gradients),
            "all_gradients_finite": bool(
                gradients and all(torch.isfinite(value).all() for value in gradients)
            ),
            "cpu_forward_backward_seconds": time.monotonic() - started,
            "competition_test_data_read": False,
            "public_competition_predictions_read": False,
            "leaderboard_used": False,
            "submission_created": False,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
