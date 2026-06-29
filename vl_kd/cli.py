"""Command-line argument parser for distillation training."""

import argparse

from .defaults import CSV_DIR, MODEL_NAME, TEST_DIR, TRAIN_DIR


def get_args():
    """Parse CLI arguments for LaSeD training on Qwen3-VL (Accelerate + optional QLoRA)."""
    parser = argparse.ArgumentParser(
        description="LaSeD: Label-Semantic Self-Distillation for visual-only surgical phase recognition"
    )

    # --- Core training ---
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        default=MODEL_NAME,
        help="Teacher model path (used to precompute teacher targets)",
    )
    parser.add_argument(
        "--student_model_name_or_path",
        type=str,
        default=MODEL_NAME,
        help="Student model path (used for training)",
    )
    parser.add_argument("--train_path", type=str, default=TRAIN_DIR, help="Training data directory")
    parser.add_argument("--eval_path", type=str, default=TEST_DIR, help="Evaluation data directory")
    parser.add_argument("--csv_dir", type=str, default=CSV_DIR, help="CSV label directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/student_distilled", help="Output directory")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")

    # --- Loss weights ---
    parser.add_argument("--lambda_ce", type=float, default=1.0, help="Cross-entropy loss weight")
    parser.add_argument("--lambda_feat", type=float, default=1.0, help="Feature MSE loss weight")

    # --- Evaluation / logging ---
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save interval in steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation interval in steps")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging interval in steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=0,
        help="Maximum number of evaluation samples; 0 means use the full eval set",
    )

    # --- Precision / device ---
    parser.add_argument("--bf16", action="store_true", help="Use bf16")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    # --- Data loading ---
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Teacher precomputation ---
    parser.add_argument("--teacher_fp16", action="store_true", help="Use fp16 during teacher precomputation")

    # --- Model config ---
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use_flash_attention", action="store_true", help="Enable Flash Attention")
    parser.add_argument("--train_visual_only", action="store_true", help="Train only visual module parameters")
    parser.add_argument(
        "--train_all_modules",
        action="store_false",
        dest="train_visual_only",
        help="Train all model parameters (default)",
    )

    # --- LoRA / QLoRA ---
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="LoRA target modules")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="4-bit compute dtype")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="4-bit quantization type")

    # --- Best checkpoint selection ---
    parser.add_argument(
        "--best_model_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "recall", "precision", "jaccard", "logit_diff"],
        help="Metric used to select the best model",
    )

    # --- Misc ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--kd_debug",
        action="store_true",
        help=(
            "Print an extra [Train] debug line on optimizer steps aligned with logging_steps "
            "(CE, feature MSE, total loss, and batch accuracy)"
        ),
    )
    parser.add_argument(
        "--use_label_weight",
        action="store_true",
        help=(
            "Weight samples by phase frequency in the training set: "
            "w_k = count_k / mean(count_1..count_7), applied per sample's CSV label"
        ),
    )

    return parser.parse_args()
