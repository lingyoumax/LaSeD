"""Distributed evaluation, confusion matrix, and heatmap artifacts."""

import logging
import os
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import seaborn as sns  # noqa: E402
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from .processor_utils import get_class_logits_probs

logger = logging.getLogger(__name__)


def build_confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Count co-occurrences of true vs predicted class indices (0-based)."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_mask = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    for true_label, pred_label in zip(labels[valid_mask], preds[valid_mask]):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def save_confusion_matrix_artifacts(output_dir: str, eval_index: int, confusion_matrix: np.ndarray) -> None:
    """Write raw counts to ``.txt`` and a row-normalized heatmap PDF under ``output_dir/eval``."""
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    txt_path = os.path.join(eval_dir, f"{eval_index}.txt")
    np.savetxt(txt_path, confusion_matrix, fmt="%d")

    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion_matrix.astype(np.float32),
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=np.float32),
        where=row_sums != 0,
    )

    labels = [f"Phase {i}" for i in range(1, confusion_matrix.shape[0] + 1)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("Predicted Phase", fontsize=12)
    plt.ylabel("True Phase", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.tight_layout()
    pdf_path = os.path.join(eval_dir, f"{eval_index}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()


def evaluate_student(
    model: torch.nn.Module,
    processor,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    Run the student on the eval loader; gather predictions on the main process.

    Returns macro-averaged recall/precision/F1/Jaccard, accuracy, and mean logit margin for the predicted class.
    """
    model.eval()
    device = accelerator.device

    all_preds = []
    all_labels = []
    logit_diff_sum = 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"Evaluating (rank {accelerator.process_index})",
            disable=not accelerator.is_local_main_process,
        ):
            student_inputs = batch["student_inputs"]
            labels = batch["labels"]

            outputs = model(**student_inputs, output_hidden_states=False, return_dict=True)
            cls_logits, _ = get_class_logits_probs(processor, outputs.logits[:, -1, :])

            preds = cls_logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total += len(labels)

            batch_size = cls_logits.size(0)
            for b in range(batch_size):
                pred_logits = cls_logits[b]
                pred_class = preds[b].item()

                pred_logit = pred_logits[pred_class]
                other_logits = torch.cat([pred_logits[:pred_class], pred_logits[pred_class + 1 :]])
                other_mean_logit = other_logits.mean() if len(other_logits) > 0 else 0.0
                logit_diff_sum += (pred_logit - other_mean_logit).item()

    if len(all_preds) > 0:
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
    else:
        all_preds_tensor = torch.tensor([], dtype=torch.long)
        all_labels_tensor = torch.tensor([], dtype=torch.long)

    all_preds_tensor = all_preds_tensor.to(device)
    all_labels_tensor = all_labels_tensor.to(device)
    gathered_preds = accelerator.gather(all_preds_tensor.unsqueeze(0))
    gathered_labels = accelerator.gather(all_labels_tensor.unsqueeze(0))
    gathered_logit_diff = torch.tensor([logit_diff_sum], device=device, dtype=torch.float32)
    gathered_logit_diff = accelerator.gather(gathered_logit_diff.unsqueeze(0))
    gathered_total = torch.tensor([total], device=device, dtype=torch.long)
    gathered_total = accelerator.gather(gathered_total.unsqueeze(0))

    if accelerator.is_main_process:
        all_preds_flat = gathered_preds.flatten().cpu().numpy()
        all_labels_flat = gathered_labels.flatten().cpu().numpy()
        all_logit_diff_sum = gathered_logit_diff.sum().item()
        all_total = gathered_total.sum().item()

        if all_total == 0:
            return (
                {
                    "accuracy": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "f1": 0.0,
                    "jaccard": 0.0,
                    "logit_diff": 0.0,
                },
                np.zeros((7, 7), dtype=np.int64),
            )

        accuracy = (all_preds_flat == all_labels_flat).mean()
        confusion_matrix = build_confusion_matrix(all_labels_flat, all_preds_flat, num_classes=7)

        num_classes = (
            max(int(all_labels_flat.max()) + 1, int(all_preds_flat.max()) + 1)
            if len(all_labels_flat) > 0
            else 7
        )

        tp_per_class = np.zeros(num_classes)
        fp_per_class = np.zeros(num_classes)
        fn_per_class = np.zeros(num_classes)

        for cls in range(num_classes):
            tp_per_class[cls] = np.sum((all_preds_flat == cls) & (all_labels_flat == cls))
            fp_per_class[cls] = np.sum((all_preds_flat == cls) & (all_labels_flat != cls))
            fn_per_class[cls] = np.sum((all_preds_flat != cls) & (all_labels_flat == cls))

        recall_per_class = np.zeros(num_classes)
        precision_per_class = np.zeros(num_classes)
        f1_per_class = np.zeros(num_classes)
        jaccard_per_class = np.zeros(num_classes)

        for cls in range(num_classes):
            if tp_per_class[cls] + fn_per_class[cls] > 0:
                recall_per_class[cls] = tp_per_class[cls] / (tp_per_class[cls] + fn_per_class[cls])
            else:
                recall_per_class[cls] = 0.0

            if tp_per_class[cls] + fp_per_class[cls] > 0:
                precision_per_class[cls] = tp_per_class[cls] / (tp_per_class[cls] + fp_per_class[cls])
            else:
                precision_per_class[cls] = 0.0

            if precision_per_class[cls] + recall_per_class[cls] > 0:
                f1_per_class[cls] = (
                    2
                    * (precision_per_class[cls] * recall_per_class[cls])
                    / (precision_per_class[cls] + recall_per_class[cls])
                )
            else:
                f1_per_class[cls] = 0.0

            if tp_per_class[cls] + fp_per_class[cls] + fn_per_class[cls] > 0:
                jaccard_per_class[cls] = tp_per_class[cls] / (
                    tp_per_class[cls] + fp_per_class[cls] + fn_per_class[cls]
                )
            else:
                jaccard_per_class[cls] = 0.0

        recall_macro = recall_per_class.mean()
        precision_macro = precision_per_class.mean()
        f1_macro = f1_per_class.mean()
        jaccard_macro = jaccard_per_class.mean()

        logit_diff_avg = all_logit_diff_sum / all_total

        return {
            "accuracy": float(accuracy),
            "recall": float(recall_macro),
            "precision": float(precision_macro),
            "f1": float(f1_macro),
            "jaccard": float(jaccard_macro),
            "logit_diff": float(logit_diff_avg),
        }, confusion_matrix

    return {
        "accuracy": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
        "jaccard": 0.0,
        "logit_diff": 0.0,
    }, None
