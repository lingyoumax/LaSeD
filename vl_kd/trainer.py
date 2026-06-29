"""Main training loop: optimization, periodic eval, checkpointing, TensorBoard."""

import logging
import os
from datetime import datetime
from typing import Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .checkpoints import cleanup_old_checkpoints
from .evaluation import evaluate_student, save_confusion_matrix_artifacts
from .loss import compute_distillation_loss

logger = logging.getLogger(__name__)


def train(
    accelerator: Accelerator,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    processor,
    *,
    num_epochs: int,
    learning_rate: float,
    lambda_ce: float,
    lambda_feat: float,
    output_dir: str,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    resume_from_checkpoint: Optional[str],
    warmup_ratio: float,
    weight_decay: float,
    save_total_limit: Optional[int],
    best_model_metric: str,
    kd_debug: bool = False,
) -> None:
    """
    Distillation training with AdamW, cosine schedule, gradient accumulation, and Accelerate.

    Main process writes TensorBoard scalars, text logs, confusion matrices, and checkpoints.
    """
    is_main = accelerator.is_main_process

    log_dir = None
    writer = None
    log_file_path = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(output_dir, "runs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logs will be saved to: %s", log_dir)

        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        log_file_path = os.path.join(logs_dir, f"{timestamp}.log")
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        logger.info("Log file will be saved to: %s", log_file_path)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    global_step = 0
    best_metric = 0.0
    start_epoch = 0
    eval_count = 0

    valid_metrics = ["accuracy", "f1", "recall", "precision", "jaccard", "logit_diff"]
    if best_model_metric not in valid_metrics:
        raise ValueError(f"best_model_metric must be one of {valid_metrics}, got {best_model_metric}")

    training_state = None
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        logger.info("Loading checkpoint from %s", resume_from_checkpoint)
        training_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu")
            global_step = training_state.get("global_step", 0)
            best_metric = training_state.get("best_metric", 0.0)
            start_epoch = training_state.get("epoch", 0)
            logger.info(
                "Resumed from step %s, epoch %s, best_%s=%.4f",
                global_step,
                start_epoch,
                best_model_metric,
                best_metric,
            )

            optimizer_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                logger.info("Loaded optimizer state")
        else:
            logger.warning("Training state not found at %s, starting from scratch", training_state_path)

    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_ratio_clamped = max(0.0, min(1.0, warmup_ratio))
    num_warmup_steps = int(warmup_ratio_clamped * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    if training_state is not None and resume_from_checkpoint is not None:
        scheduler_path = os.path.join(resume_from_checkpoint, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            logger.info("Loaded scheduler state")

    if is_main:
        logger.info(
            "Train config | lr=%.2e | epochs=%s | lambda_ce=%s | lambda_feat=%s | grad_acc=%s",
            learning_rate,
            num_epochs,
            lambda_ce,
            lambda_feat,
            gradient_accumulation_steps,
        )
        logger.info(
            "best_model_metric=%s | Total training steps: %s, Warmup steps: %s (ratio: %.2f%%)",
            best_model_metric,
            total_steps,
            num_warmup_steps,
            warmup_ratio_clamped * 100,
        )
        if kd_debug:
            logger.info(
                "kd_debug=True: whenever global_step is a multiple of logging_steps, "
                "the corresponding batch prints an extra [Train] batch dbg line "
                "(CE, feature MSE, total loss, and batch accuracy)."
            )

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    for epoch in range(start_epoch, num_epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        if is_main:
            logger.info("Epoch %s/%s", epoch + 1, num_epochs)

        epoch_ce_loss = 0.0
        epoch_feat_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}") if is_main else train_dataloader
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            student_inputs = batch["student_inputs"]
            teacher_visual_feats = batch["teacher_visual_feats"]
            batch_labels = batch["labels"]
            lambda_ce_batch = batch.get("lambda_ce", lambda_ce)
            lambda_feat_batch = batch.get("lambda_feat", lambda_feat)

            accum_done = (step + 1) % gradient_accumulation_steps == 0
            next_gs = global_step + 1 if accum_done else None
            log_kd_batch = (
                is_main
                and kd_debug
                and accum_done
                and next_gs is not None
                and next_gs % logging_steps == 0
            )

            loss, ce_loss, feat_loss = compute_distillation_loss(
                model,
                processor,
                student_inputs,
                teacher_visual_feats,
                batch_labels,
                lambda_ce_batch,
                lambda_feat_batch,
                log_kd_batch=log_kd_batch,
                sample_weights=batch.get("sample_weights"),
            )

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_ce_loss += ce_loss.item()
            epoch_feat_loss += feat_loss.item()
            num_batches += 1

            del loss, ce_loss, feat_loss

            if (step + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if is_main and global_step % logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_ce_loss = epoch_ce_loss / num_batches
                    avg_feat_loss = epoch_feat_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "Step %s/%s | Loss: %.4f | CE: %.4f | Feat: %.4f | LR: %.2e",
                        global_step,
                        total_steps,
                        avg_loss,
                        avg_ce_loss,
                        avg_feat_loss,
                        current_lr,
                    )
                    if writer:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/ce_loss", avg_ce_loss, global_step)
                        writer.add_scalar("train/feat_loss", avg_feat_loss, global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)

                if eval_dataloader is not None and global_step % eval_steps == 0:
                    if is_main:
                        logger.info("Running evaluation...")
                    eval_count += 1
                    unwrapped_model = accelerator.unwrap_model(model)
                    metrics, confusion_matrix = evaluate_student(
                        unwrapped_model, processor, eval_dataloader, accelerator
                    )
                    if is_main:
                        if confusion_matrix is not None:
                            save_confusion_matrix_artifacts(output_dir, eval_count, confusion_matrix)
                        logger.info(
                            "Eval - Accuracy: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f, Jaccard: %.4f, Logit_Diff: %.4f",
                            metrics["accuracy"],
                            metrics["recall"],
                            metrics["precision"],
                            metrics["f1"],
                            metrics["jaccard"],
                            metrics["logit_diff"],
                        )
                        if writer:
                            writer.add_scalar("eval/accuracy", metrics["accuracy"], global_step)
                            writer.add_scalar("eval/recall", metrics["recall"], global_step)
                            writer.add_scalar("eval/precision", metrics["precision"], global_step)
                            writer.add_scalar("eval/f1", metrics["f1"], global_step)
                            writer.add_scalar("eval/jaccard", metrics["jaccard"], global_step)
                            writer.add_scalar("eval/logit_diff", metrics["logit_diff"], global_step)

                        current_metric_value = metrics[best_model_metric]
                        if current_metric_value > best_metric:
                            best_metric = current_metric_value
                            save_path = os.path.join(output_dir, "best_model")
                            os.makedirs(save_path, exist_ok=True)
                            unwrapped_model.save_pretrained(save_path)
                            processor.save_pretrained(save_path)
                            logger.info(
                                "Saved best model (%s=%.4f) to %s",
                                best_model_metric,
                                best_metric,
                                save_path,
                            )
                            if writer:
                                writer.add_scalar(f"eval/best_{best_model_metric}", best_metric, global_step)

                    model.train()

                if is_main and global_step % save_steps == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_path)
                    processor.save_pretrained(checkpoint_path)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_metric": best_metric,
                    }
                    torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
                    logger.info("Saved checkpoint to %s", checkpoint_path)
                    if save_total_limit is not None:
                        cleanup_old_checkpoints(output_dir, save_total_limit)

        if is_main:
            avg_loss = epoch_loss / max(1, num_batches)
            logger.info("Epoch %s completed. Average loss: %.4f", epoch + 1, avg_loss)
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)

    accelerator.wait_for_everyone()

    if eval_dataloader is not None:
        if is_main:
            logger.info("Running final evaluation after training...")
        unwrapped_model = accelerator.unwrap_model(model)
        eval_count += 1
        final_metrics, confusion_matrix = evaluate_student(
            unwrapped_model, processor, eval_dataloader, accelerator
        )
        if is_main:
            if confusion_matrix is not None:
                save_confusion_matrix_artifacts(output_dir, eval_count, confusion_matrix)
            logger.info(
                "Final Eval - Accuracy: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f, Jaccard: %.4f, Logit_Diff: %.4f",
                final_metrics["accuracy"],
                final_metrics["recall"],
                final_metrics["precision"],
                final_metrics["f1"],
                final_metrics["jaccard"],
                final_metrics["logit_diff"],
            )
            if writer:
                writer.add_scalar("eval/final_accuracy", final_metrics["accuracy"], global_step)
                writer.add_scalar("eval/final_recall", final_metrics["recall"], global_step)
                writer.add_scalar("eval/final_precision", final_metrics["precision"], global_step)
                writer.add_scalar("eval/final_f1", final_metrics["f1"], global_step)
                writer.add_scalar("eval/final_jaccard", final_metrics["jaccard"], global_step)
                writer.add_scalar("eval/final_logit_diff", final_metrics["logit_diff"], global_step)

            final_metric_value = final_metrics[best_model_metric]
            if final_metric_value > best_metric:
                logger.info(
                    "Final evaluation %s (%.4f) is better than previous best %s (%.4f)",
                    best_model_metric,
                    final_metric_value,
                    best_model_metric,
                    best_metric,
                )

    if is_main:
        final_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_path)
        processor.save_pretrained(final_path)
        logger.info("Saved final model to %s", final_path)
        if writer:
            writer.close()
            logger.info("TensorBoard logs saved to: %s", log_dir)

        if log_file_path is not None:
            handlers = logger.handlers[:]
            for handler in handlers:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(
                    log_file_path
                ):
                    handler.close()
                    logger.removeHandler(handler)
                    logger.info("Log file saved to: %s", log_file_path)
