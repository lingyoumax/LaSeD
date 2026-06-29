"""Load teacher/student models, build dataloaders, and run LaSeD training."""

import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoConfig, BitsAndBytesConfig

from .cli import get_args
from .data import (
    CholecFrameDataset,
    DistillationDataCollator,
    DistillationDataset,
    build_label_weights_tensor,
    stratified_phase_subset_indices,
)
from .defaults import cache_dir
from .logging_utils import setup_logging
from .teacher_precompute import precompute_teacher_features
from .trainer import train

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry: parse args, precompute teacher targets, train student with Accelerate."""
    setup_logging()
    args = get_args()

    if args.use_qlora and not args.use_lora:
        args.use_lora = True
        logger.info("--use_qlora is set, automatically enabling --use_lora (QLoRA requires LoRA)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    accelerator = Accelerator()
    logger.info("Initialized accelerator")
    device = accelerator.device

    quantization_config = None
    model_dtype = None
    if args.use_qlora and args.use_lora:
        compute_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        compute_dtype = compute_dtype_map.get(args.bnb_4bit_compute_dtype, torch.bfloat16)
        if args.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization (QLoRA)")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(
                "Using 4-bit quantization (QLoRA) (%s, compute_dtype=%s)",
                args.bnb_4bit_quant_type,
                args.bnb_4bit_compute_dtype,
            )
    else:
        if args.bf16:
            model_dtype = torch.bfloat16
        elif args.fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32

    attn_implementation = None
    if args.use_flash_attention:
        try:
            from flash_attn import flash_attn_func  # noqa: F401

            attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 enabled and verified")
        except Exception as e:
            logger.warning("Flash Attention requested but unavailable: %s", e)
            attn_implementation = None

    def _build_config(model_path: str) -> AutoConfig:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "max_position_embeddings") and getattr(cfg, "max_position_embeddings", None):
            original_max_len = cfg.max_position_embeddings
            if args.max_length > original_max_len:
                rope_scaling = {
                    "type": "yarn",
                    "factor": args.max_length / original_max_len,
                    "original_max_position_embeddings": original_max_len,
                }
                cfg.rope_scaling = rope_scaling
                cfg.max_position_embeddings = args.max_length
                logger.info("Applied rope scaling for %s: %s", model_path, rope_scaling)
        return cfg

    def _build_model_kwargs(config: AutoConfig, quant_cfg, dtype):
        kwargs = {
            "config": config,
            "quantization_config": quant_cfg,
            "dtype": dtype,
            "device_map": None,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        return kwargs

    train_set = CholecFrameDataset(args.train_path, args.csv_dir)
    eval_set = None
    if args.eval_path and os.path.exists(args.eval_path):
        eval_set = CholecFrameDataset(args.eval_path, args.csv_dir)
        if args.max_eval_samples > 0:
            eval_size = min(args.max_eval_samples, len(eval_set))
            sampled_indices = stratified_phase_subset_indices(
                eval_set,
                max_samples=eval_size,
                seed=args.seed,
            )
            phase_total: Dict[int, int] = {}
            phase_sampled: Dict[int, int] = {}
            for _, label in eval_set.samples:
                phase_total[label] = phase_total.get(label, 0) + 1
            sampled_samples = [eval_set.samples[i] for i in sampled_indices]
            for _, label in sampled_samples:
                phase_sampled[label] = phase_sampled.get(label, 0) + 1
            eval_set.samples = sampled_samples
            logger.info(
                "Using stratified eval subset: %d / %d samples (seed=%d)",
                len(sampled_samples),
                sum(phase_total.values()),
                args.seed,
            )
            logger.info(
                "Stratified phase counts (sampled/total): %s",
                ", ".join(
                    f"{phase}:{phase_sampled.get(phase, 0)}/{phase_total[phase]}"
                    for phase in sorted(phase_total.keys())
                ),
            )

    if accelerator.is_main_process:
        logger.info("Train samples: %s, Eval samples: %s", len(train_set), len(eval_set) if eval_set else 0)

    label_weights: Optional[torch.Tensor] = None
    if args.use_label_weight:
        label_weights = build_label_weights_tensor(train_set)
        if accelerator.is_main_process:
            counts = torch.zeros(7, dtype=torch.float64)
            for _, lb in train_set.samples:
                if 1 <= lb <= 7:
                    counts[lb - 1] += 1.0
            mean_d = counts.mean().item()
            logger.info(
                "use_label_weight: phase counts 1..7 = %s, mean(count)= %.2f",
                [int(x) for x in counts.tolist()],
                mean_d,
            )
            logger.info(
                "use_label_weight: weights D/mean(D) per phase 1..7 = %s",
                [round(x, 4) for x in label_weights.tolist()],
            )

    if accelerator.is_main_process:
        logger.info("Precomputing teacher features (using the teacher model)...")

    teacher_processor = AutoProcessor.from_pretrained(
        args.teacher_model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        image_processor_kwargs={"do_resize": True, "size": {"shortest_edge": 224}},
    )
    teacher_processor.model_max_length = args.max_length
    teacher_config = _build_config(args.teacher_model_name_or_path)
    teacher_dtype = torch.float16 if args.teacher_fp16 else model_dtype
    teacher_model_kwargs = _build_model_kwargs(teacher_config, quant_cfg=None, dtype=teacher_dtype)
    teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.teacher_model_name_or_path, **teacher_model_kwargs
    )
    teacher_model.config.use_cache = False
    teacher_model = teacher_model.to(device)

    teacher_feats = precompute_teacher_features(
        teacher_model,
        teacher_processor,
        train_set,
        batch_size=args.per_device_train_batch_size,
        device=device,
        use_amp=args.teacher_fp16,
        accelerator=accelerator,
    )

    del teacher_model
    del teacher_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logger.info("Teacher feature precomputation completed, and the teacher model has been released.")

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info("Loading student tokenizer from checkpoint: %s", args.resume_from_checkpoint)
        student_processor = AutoProcessor.from_pretrained(args.resume_from_checkpoint, trust_remote_code=True)
    else:
        student_processor = AutoProcessor.from_pretrained(
            args.student_model_name_or_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            image_processor_kwargs={"do_resize": True, "size": {"shortest_edge": 224}},
        )
    student_processor.model_max_length = args.max_length

    student_model_source = (
        args.resume_from_checkpoint
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)
        else args.student_model_name_or_path
    )
    student_config = _build_config(student_model_source)
    student_model_kwargs = _build_model_kwargs(student_config, quantization_config, model_dtype)

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info("Loading student model from checkpoint: %s", args.resume_from_checkpoint)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.resume_from_checkpoint, **student_model_kwargs
        )
        if args.use_qlora and args.use_lora:
            model = prepare_model_for_kbit_training(model)
        if args.use_lora:
            model = PeftModel.from_pretrained(model, args.resume_from_checkpoint)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.student_model_name_or_path, **student_model_kwargs
        )
        if args.use_qlora and args.use_lora:
            model = prepare_model_for_kbit_training(model)
        if args.use_lora:
            target_modules = args.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            logger.info(
                "Applied LoRA with r=%s, alpha=%s, target_modules=%s",
                args.lora_r,
                args.lora_alpha,
                target_modules,
            )
            model.print_trainable_parameters()

    if args.train_visual_only:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = 0
        for name, param in model.named_parameters():
            if not name.startswith("model.visual"):
                param.requires_grad = False
            else:
                trainable_params += param.numel()
        logger.info(
            "Total parameters: %s, trainable parameters: %s (%.2f%%)",
            f"{total_params:,}",
            f"{trainable_params:,}",
            trainable_params / total_params * 100,
        )
    elif not args.use_lora:
        logger.info("Training all model parameters (LoRA disabled)")

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_dataset = DistillationDataset(train_set)
    data_collator = DistillationDataCollator(
        train_set,
        student_processor,
        teacher_feats,
        lambda_ce=args.lambda_ce,
        lambda_feat=args.lambda_feat,
        label_weights=label_weights,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    eval_dataloader = None
    if eval_set is not None:
        eval_dataset = DistillationDataset(eval_set)
        eval_data_collator = DistillationDataCollator(
            eval_set,
            student_processor,
            label_weights=None,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            collate_fn=eval_data_collator,
            pin_memory=torch.cuda.is_available(),
        )

    os.makedirs(args.output_dir, exist_ok=True)

    train(
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        processor=student_processor,
        num_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lambda_ce=args.lambda_ce,
        lambda_feat=args.lambda_feat,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        resume_from_checkpoint=args.resume_from_checkpoint,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        best_model_metric=args.best_model_metric,
        kd_debug=args.kd_debug,
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
