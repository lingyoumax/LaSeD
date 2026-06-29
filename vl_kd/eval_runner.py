"""Load a trained student and run visual-only evaluation on Cholec80-style data."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, BitsAndBytesConfig

from .data import CholecFrameDataset
from .defaults import CSV_DIR, MODEL_NAME, TEST_DIR, cache_dir
from .evaluation import build_confusion_matrix, save_confusion_matrix_artifacts
from .logging_utils import setup_logging
from .processor_utils import build_batch_inputs, get_class_logits_probs, make_student_messages

logger = logging.getLogger(__name__)


class EvalFrameDataset(Dataset):
    def __init__(self, base: CholecFrameDataset):
        self.samples = base.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


class EvalCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Tuple[str, int]]) -> Dict[str, Any]:
        student_msgs = [make_student_messages(path) for path, _ in features]
        student_inputs = build_batch_inputs(self.processor, student_msgs)
        labels = torch.tensor([label for _, label in features], dtype=torch.long) - 1
        image_paths = [path for path, _ in features]
        return {
            "student_inputs": student_inputs,
            "labels": labels,
            "image_paths": image_paths,
        }


def get_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen3-VL student")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=TEST_DIR)
    parser.add_argument("--csv_dir", type=str, default=CSV_DIR)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--save_confusion_matrix", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def _load_model(args, device: torch.device) -> nn.Module:
    model_path = args.model_name_or_path
    is_adapter = os.path.isfile(os.path.join(model_path, "adapter_config.json"))
    base_path = args.base_model_name_or_path or MODEL_NAME

    quantization_config = None
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if args.use_qlora:
        compute_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        compute_dtype = compute_dtype_map.get(args.bnb_4bit_compute_dtype, torch.bfloat16)
        if args.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

    load_source = base_path if is_adapter else model_path
    config = AutoConfig.from_pretrained(load_source, trust_remote_code=True)
    kwargs = {
        "config": config,
        "quantization_config": quantization_config,
        "dtype": model_dtype,
        "device_map": None,
        "trust_remote_code": True,
    }
    if args.use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    if is_adapter:
        model = Qwen3VLForConditionalGeneration.from_pretrained(base_path, **kwargs)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)

    model.config.use_cache = False
    return model.to(device)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, logit_diff_sum: float, total: int) -> Dict[str, float]:
    if total == 0:
        return {k: 0.0 for k in ("accuracy", "recall", "precision", "f1", "jaccard", "logit_diff")}

    accuracy = float((preds == labels).mean())
    num_classes = 7
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for cls in range(num_classes):
        tp[cls] = np.sum((preds == cls) & (labels == cls))
        fp[cls] = np.sum((preds == cls) & (labels != cls))
        fn[cls] = np.sum((preds != cls) & (labels == cls))

    recall = np.array([tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] > 0 else 0.0 for c in range(num_classes)])
    precision = np.array([tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] > 0 else 0.0 for c in range(num_classes)])
    f1 = np.array([
        2 * precision[c] * recall[c] / (precision[c] + recall[c]) if precision[c] + recall[c] > 0 else 0.0
        for c in range(num_classes)
    ])
    jaccard = np.array([
        tp[c] / (tp[c] + fp[c] + fn[c]) if tp[c] + fp[c] + fn[c] > 0 else 0.0 for c in range(num_classes)
    ])

    return {
        "accuracy": accuracy,
        "recall": float(recall.mean()),
        "precision": float(precision.mean()),
        "f1": float(f1.mean()),
        "jaccard": float(jaccard.mean()),
        "logit_diff": float(logit_diff_sum / total),
    }


def run_evaluation(model, processor, dataloader, accelerator, save_predictions=False):
    model.eval()
    device = accelerator.device
    all_preds, all_labels, all_paths = [], [], []
    logit_diff_sum, total = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating (rank {accelerator.process_index})", disable=not accelerator.is_local_main_process):
            student_inputs = batch["student_inputs"]
            labels = batch["labels"]
            image_paths = batch["image_paths"]

            outputs = model(**student_inputs, output_hidden_states=False, return_dict=True)
            cls_logits, _ = get_class_logits_probs(processor, outputs.logits[:, -1, :])
            preds = cls_logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_paths.extend(image_paths)
            total += len(labels)

            for b in range(cls_logits.size(0)):
                pred_class = preds[b].item()
                pred_logit = cls_logits[b, pred_class]
                other = torch.cat([cls_logits[b, :pred_class], cls_logits[b, pred_class + 1 :]])
                other_mean = other.mean() if len(other) > 0 else torch.tensor(0.0, device=cls_logits.device)
                logit_diff_sum += (pred_logit - other_mean).item()

    if all_preds:
        preds_tensor = torch.cat(all_preds).to(device)
        labels_tensor = torch.cat(all_labels).to(device)
    else:
        preds_tensor = torch.tensor([], dtype=torch.long, device=device)
        labels_tensor = torch.tensor([], dtype=torch.long, device=device)

    gathered_preds = accelerator.gather(preds_tensor.unsqueeze(0) if preds_tensor.numel() else torch.zeros(1, 0, device=device, dtype=torch.long))
    gathered_labels = accelerator.gather(labels_tensor.unsqueeze(0) if labels_tensor.numel() else torch.zeros(1, 0, device=device, dtype=torch.long))
    gathered_diff = accelerator.gather(torch.tensor([logit_diff_sum], device=device, dtype=torch.float32))
    gathered_total = accelerator.gather(torch.tensor([total], device=device, dtype=torch.long))

    if not accelerator.is_main_process:
        return {}, None, None

    preds_np = gathered_preds.flatten().cpu().numpy()
    labels_np = gathered_labels.flatten().cpu().numpy()
    metrics = compute_metrics(labels_np, preds_np, float(gathered_diff.sum().item()), int(gathered_total.sum().item()))
    confusion = build_confusion_matrix(labels_np, preds_np, num_classes=7)

    prediction_records = None
    if save_predictions:
        prediction_records = [
            {"image_path": path, "label": int(lbl) + 1, "pred": int(prd) + 1}
            for path, lbl, prd in zip(all_paths, labels_np, preds_np)
        ]

    return metrics, confusion, prediction_records


def main() -> None:
    setup_logging()
    args = get_eval_args()
    accelerator = Accelerator()
    device = accelerator.device

    eval_base = CholecFrameDataset(args.eval_path, args.csv_dir)
    eval_dataset = EvalFrameDataset(eval_base)

    processor_source = args.base_model_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        cache_dir=cache_dir,
        image_processor_kwargs={"do_resize": True, "size": {"shortest_edge": 224}},
    )
    processor.model_max_length = args.max_length

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=EvalCollator(processor),
        pin_memory=torch.cuda.is_available(),
    )

    model = _load_model(args, device)
    model, dataloader = accelerator.prepare(model, dataloader)

    metrics, confusion, predictions = run_evaluation(
        accelerator.unwrap_model(model), processor, dataloader, accelerator, args.save_predictions
    )

    if accelerator.is_main_process:
        logger.info(
            "Eval - Accuracy: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f | Jaccard: %.4f",
            metrics["accuracy"], metrics["recall"], metrics["precision"], metrics["f1"], metrics["jaccard"],
        )
        output_dir = args.output_dir or os.path.join(args.model_name_or_path, "eval_results")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if args.save_confusion_matrix and confusion is not None:
            save_confusion_matrix_artifacts(output_dir, 0, confusion)
        if args.save_predictions and predictions is not None:
            with open(os.path.join(output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
                for row in predictions:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
