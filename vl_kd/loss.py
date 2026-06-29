"""Training loss for LaSeD: cross-entropy on digit logits + MSE on teacher interaction features."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import AutoProcessor

from .processor_utils import extract_visual_interaction_features, get_class_logits_probs

logger = logging.getLogger(__name__)

_logged_loss_stats_once = False


def compute_distillation_loss(
    model: nn.Module,
    processor: AutoProcessor,
    student_inputs: Dict[str, torch.Tensor],
    teacher_visual_feats: torch.Tensor,
    labels: torch.Tensor,
    lambda_ce: float = 1.0,
    lambda_feat: float = 1.0,
    log_kd_batch: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute batch loss = lambda_ce * CE(logits, labels) + lambda_feat * MSE(features).

    ``sample_weights`` reweights the mean when ``use_label_weight`` is enabled.
    """
    global _logged_loss_stats_once

    s_outputs = model(**student_inputs, output_hidden_states=True, return_dict=True)
    cls_logits, _ = get_class_logits_probs(processor, s_outputs.logits[:, -1, :])
    s_visual_interaction_feats = extract_visual_interaction_features(s_outputs)

    if hasattr(s_outputs, "hidden_states"):
        del s_outputs.hidden_states

    lab = labels.to(cls_logits.device).long()
    ce_per = F.cross_entropy(cls_logits, lab, reduction="none")

    teacher_visual_feats = teacher_visual_feats.to(s_visual_interaction_feats.device)
    feat_per = F.mse_loss(s_visual_interaction_feats, teacher_visual_feats, reduction="none").mean(
        dim=-1
    )

    if sample_weights is not None:
        w = sample_weights.to(ce_per.device).float()
        denom = w.sum().clamp_min(1e-8)
        ce_loss = (ce_per * w).sum() / denom
        feat_loss = (feat_per * w).sum() / denom
    else:
        ce_loss = ce_per.mean()
        feat_loss = feat_per.mean()

    if not _logged_loss_stats_once:
        with torch.no_grad():
            preds = cls_logits.argmax(dim=-1)
            acc = (preds == lab).float().mean().item()
        logger.info("[Train] loss stats (first batch): CE=%.6f, feat=%.6f, acc=%.4f", ce_loss.item(), feat_loss.item(), acc)
        _logged_loss_stats_once = True

    loss = lambda_ce * ce_loss + lambda_feat * feat_loss

    if log_kd_batch:
        with torch.no_grad():
            preds = cls_logits.argmax(dim=-1)
            acc = (preds == lab).float().mean().item()
            logger.info(
                "[Train] batch dbg: ce=%.6f feat=%.6f total=%.6f | acc=%.4f",
                ce_loss.item(),
                feat_loss.item(),
                loss.item(),
                acc,
            )

    return loss, ce_loss, feat_loss
