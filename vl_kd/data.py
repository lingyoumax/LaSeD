"""Cholec80 frame dataset, label reweighting, stratified subsampling, and distillation collation."""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .processor_utils import build_batch_inputs, make_student_messages


class CholecFrameDataset(Dataset):
    """
    Pairs video frames under ``data_dir/video*`` with per-frame labels from matching CSVs.

    Each row in ``{video_folder}.csv`` maps frame index ``idx`` to label; expects ``{idx}.jpg``.
    """

    def __init__(self, data_dir: str, csv_dir: str):
        self.samples: List[Tuple[str, int]] = []
        video_folders = sorted([f for f in os.listdir(data_dir) if f.startswith("video")])
        for video_folder in video_folders:
            video_path = os.path.join(data_dir, video_folder)
            csv_path = os.path.join(csv_dir, f"{video_folder}.csv")
            if not (os.path.exists(video_path) and os.path.exists(csv_path)):
                continue
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()[1:]  # skip header
            for idx, line in enumerate(lines):
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    label = int(parts[1])
                except Exception:
                    continue
                img_file = os.path.join(video_path, f"{idx}.jpg")
                if os.path.exists(img_file):
                    self.samples.append((img_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_label_weights_tensor(dataset: CholecFrameDataset) -> torch.Tensor:
    """
    Per-phase sample weights: ``w_k = D_k / mean(D_1..D_7)`` where ``D_k`` is the class count.

    Returns a float32 tensor of shape ``(7,)`` with indices ``0..6`` for phases ``1..7``.
    """
    counts = torch.zeros(7, dtype=torch.float64)
    for _, lb in dataset.samples:
        if 1 <= lb <= 7:
            counts[lb - 1] += 1.0
    mean_d = counts.mean().clamp_min(1e-8)
    return (counts / mean_d).float()


def stratified_phase_subset_indices(
    dataset: CholecFrameDataset, max_samples: int, seed: int
) -> List[int]:
    """
    Stratified subsample indices so each surgical phase is represented proportionally.

    Allocates floor/ceil counts by phase size, then distributes remaining slots by largest fractional parts.
    """
    total = len(dataset)
    if max_samples <= 0 or max_samples >= total:
        return list(range(total))

    phase_to_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        phase_to_indices.setdefault(label, []).append(idx)

    target_per_phase: Dict[int, int] = {}
    remainders: List[Tuple[float, int]] = []
    assigned = 0

    for phase, indices in phase_to_indices.items():
        exact = len(indices) * max_samples / total
        base = int(exact)
        target = min(base, len(indices))
        target_per_phase[phase] = target
        assigned += target
        remainders.append((exact - base, phase))

    remaining = max_samples - assigned
    for _, phase in sorted(remainders, key=lambda x: x[0], reverse=True):
        if remaining <= 0:
            break
        if target_per_phase[phase] < len(phase_to_indices[phase]):
            target_per_phase[phase] += 1
            remaining -= 1

    if remaining > 0:
        phases_by_capacity = sorted(
            phase_to_indices.keys(),
            key=lambda p: len(phase_to_indices[p]) - target_per_phase[p],
            reverse=True,
        )
        for phase in phases_by_capacity:
            if remaining <= 0:
                break
            capacity = len(phase_to_indices[phase]) - target_per_phase[phase]
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            target_per_phase[phase] += add
            remaining -= add

    rng = random.Random(seed)
    sampled_indices: List[int] = []
    for phase, indices in phase_to_indices.items():
        k = target_per_phase.get(phase, 0)
        if k > 0:
            sampled_indices.extend(rng.sample(indices, k))

    sampled_indices.sort()
    return sampled_indices


class DistillationDataset(Dataset):
    """Wraps ``CholecFrameDataset`` so each item is an index; enables shuffling ``samples`` in place."""

    def __init__(self, base_dataset: CholecFrameDataset):
        self.base_dataset = base_dataset
        self._samples = base_dataset.samples

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return {"idx": idx}

    def shuffle(self):
        """Shuffle the underlying sample list (same object as ``base_dataset.samples``)."""
        random.shuffle(self._samples)


class DistillationDataCollator:
    """
    Builds student-side batch tensors and attaches precomputed teacher features by index.
    """

    def __init__(
        self,
        base_dataset: CholecFrameDataset,
        processor,
        teacher_visual_feats: Optional[torch.Tensor] = None,
        lambda_ce: float = 1.0,
        lambda_feat: float = 1.0,
        label_weights: Optional[torch.Tensor] = None,
    ):
        self.base_dataset = base_dataset
        self.processor = processor
        self.teacher_visual_feats = teacher_visual_feats
        self.lambda_ce = lambda_ce
        self.lambda_feat = lambda_feat
        self.label_weights = label_weights

    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        batch_indices = [f["idx"] for f in features]
        batch_samples = [self.base_dataset.samples[idx] for idx in batch_indices]

        student_msgs = [make_student_messages(p) for p, _ in batch_samples]
        student_inputs = build_batch_inputs(self.processor, student_msgs)

        labels = torch.tensor([lb for _, lb in batch_samples], dtype=torch.long) - 1
        if torch.any((labels < 0) | (labels > 6)):
            raise ValueError(f"Labels after shift are out of range 0~6: {labels}")

        out: Dict[str, Any] = {
            "student_inputs": student_inputs,
            "labels": labels,
            "lambda_ce": self.lambda_ce,
            "lambda_feat": self.lambda_feat,
        }
        if self.teacher_visual_feats is not None:
            out["teacher_visual_feats"] = self.teacher_visual_feats[batch_indices]
        if self.label_weights is not None:
            csv_labels = [lb for _, lb in batch_samples]
            idx = torch.tensor([lb - 1 for lb in csv_labels], dtype=torch.long)
            out["sample_weights"] = self.label_weights[idx].clone()

        return out
