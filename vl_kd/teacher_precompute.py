"""Run the teacher model once to cache interaction features for every sample."""

import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration
from tqdm import tqdm

from .data import CholecFrameDataset
from .processor_utils import (
    build_batch_inputs,
    extract_visual_interaction_features,
    make_teacher_messages,
)

logger = logging.getLogger(__name__)


def precompute_teacher_features(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    dataset: CholecFrameDataset,
    batch_size: int = 1,
    device: torch.device = None,
    use_amp: bool = False,
    accelerator: Optional[Accelerator] = None,
) -> torch.Tensor:
    """
    Forward the teacher on teacher-style prompts and store interaction features.

    Single process: sequential over the full dataset.

    Multi-GPU (``accelerator`` with ``world_size > 1``): shard indices by ``i % world_size``,
    each rank computes its shard, then ``all_reduce`` sums into a full ``(N, H)`` tensor.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_samples = len(dataset)
    world_size = accelerator.num_processes if accelerator is not None else 1
    rank = accelerator.process_index if accelerator is not None else 0
    use_shard = (
        accelerator is not None
        and world_size > 1
        and dist.is_available()
        and dist.is_initialized()
    )

    autocast_enabled = use_amp and device.type == "cuda"

    def _forward_batch(batch: List[Tuple[str, int]]) -> torch.Tensor:
        teacher_msgs = [make_teacher_messages(p, lb) for p, lb in batch]
        inputs = build_batch_inputs(processor, teacher_msgs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            feats = extract_visual_interaction_features(outputs)
        del inputs, outputs
        return feats.cpu()

    with torch.no_grad():
        p0, lb0 = dataset.samples[0]
        probe_feats = _forward_batch([(p0, lb0)])
        H = int(probe_feats.shape[-1])
        del probe_feats

    if not use_shard:
        all_feats: List[torch.Tensor] = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        logger.info(
            "Starting teacher feature precomputation (single process): samples=%d, batch_size=%d, about %d steps",
            n_samples,
            batch_size,
            n_batches,
        )
        with torch.no_grad():
            for i in tqdm(
                range(0, n_samples, batch_size),
                desc="Precompute teacher features",
                total=n_batches,
                leave=True,
            ):
                batch = dataset.samples[i : i + batch_size]
                if not batch:
                    continue
                feats = _forward_batch(batch)
                all_feats.append(feats)
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        teacher_feats = torch.cat(all_feats, dim=0)
        model.train()
        return teacher_feats

    indices_per_rank = [i for i in range(n_samples) if i % world_size == rank]
    n_local = len(indices_per_rank)
    n_local_batches = (n_local + batch_size - 1) // batch_size if n_local else 0
    if accelerator is not None and accelerator.is_main_process:
        logger.info(
            "Starting teacher feature precomputation (sharded parallel): world_size=%d, total_samples=%d, about %d samples and %d micro-batches per rank",
            world_size,
            n_samples,
            (n_samples + world_size - 1) // world_size,
            (n_local + batch_size - 1) // batch_size if batch_size else 0,
        )

    local_feats: List[torch.Tensor] = []

    with torch.no_grad():
        for start in tqdm(
            range(0, n_local, batch_size),
            desc=f"Teacher precompute r{rank}/{world_size}",
            total=n_local_batches,
            leave=True,
            disable=accelerator is not None and not accelerator.is_local_main_process,
        ):
            idx_chunk = indices_per_rank[start : start + batch_size]
            batch = [dataset.samples[j] for j in idx_chunk]
            if not batch:
                continue
            feats = _forward_batch(batch)
            local_feats.append(feats)
            if start % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if local_feats:
        local_feats_cat = torch.cat(local_feats, dim=0)
    else:
        local_feats_cat = torch.empty(0, H)

    teacher_feats_full = torch.zeros(n_samples, H, dtype=torch.float32, device=device)
    for j, global_idx in enumerate(indices_per_rank):
        teacher_feats_full[global_idx] = local_feats_cat[j].to(device)

    accelerator.wait_for_everyone()
    dist.all_reduce(teacher_feats_full, op=dist.ReduceOp.SUM)

    teacher_feats = teacher_feats_full.cpu()
    del teacher_feats_full

    model.train()
    return teacher_feats
