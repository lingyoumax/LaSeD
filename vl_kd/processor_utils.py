"""Chat templates, batching for the processor, and logits over digit tokens (phases 1-7)."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from modelscope import AutoProcessor

from .prompts import PHASE_NAMES, SYSTEM_PROMPT, USER_PROMPT_TEACHER


def build_batch_inputs(processor: AutoProcessor, batch_msgs: List[List[dict]]) -> dict:
    """
    Tokenize a batch of chat messages and pad into a single model input dict.

    Applies ``apply_chat_template`` per sample, pads text fields, then stacks
    ``pixel_values`` and optional ``image_grid_thw`` for vision inputs.
    """
    encoded = [
        processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        for msgs in batch_msgs
    ]

    batch_dicts = []
    for e in encoded:
        sample_dict = {}
        if "input_ids" in e:
            input_ids = e["input_ids"]
            if input_ids.dim() == 2 and input_ids.shape[0] == 1:
                sample_dict["input_ids"] = input_ids[0]
            else:
                sample_dict["input_ids"] = input_ids
        if "attention_mask" in e:
            attn_mask = e["attention_mask"]
            if attn_mask.dim() == 2 and attn_mask.shape[0] == 1:
                sample_dict["attention_mask"] = attn_mask[0]
            else:
                sample_dict["attention_mask"] = attn_mask
        batch_dicts.append(sample_dict)

    text_padded = processor.tokenizer.pad(batch_dicts, padding=True, return_tensors="pt")

    pixel_values_list = []
    for e in encoded:
        pv = e["pixel_values"]
        if pv.dim() == 4 and pv.shape[0] == 1:
            pixel_values_list.append(pv[0])
        else:
            pixel_values_list.append(pv)
    text_padded["pixel_values"] = torch.stack(pixel_values_list)

    if "image_grid_thw" in encoded[0]:
        grid_thw_list = []
        for e in encoded:
            grid_thw = e["image_grid_thw"]
            if grid_thw.dim() == 2 and grid_thw.shape[0] == 1:
                grid_thw = grid_thw[0]
            elif grid_thw.dim() == 1:
                pass
            else:
                grid_thw = grid_thw.flatten()[:3]
            grid_thw_list.append(grid_thw)
        text_padded["image_grid_thw"] = torch.stack(grid_thw_list)

    return text_padded


def get_class_logits_probs(
    processor: AutoProcessor, logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract logits and softmax probabilities for classes 1-7 from the last position.

    Uses single-digit token ids ``"1"`` through ``"7"``; unknown tokens map to a large negative logit.
    """
    token_ids = [processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 8)]
    cls_logits = []
    for tid in token_ids:
        if tid == processor.tokenizer.unk_token_id:
            cls_logits.append(torch.full_like(logits[:, 0], -1e4))
        else:
            cls_logits.append(logits[:, tid])
    cls_logits = torch.stack(cls_logits, dim=-1)
    probs = F.softmax(cls_logits, dim=-1)
    return cls_logits, probs


def extract_visual_interaction_features(outputs, hidden_state_index: int = -1) -> torch.Tensor:
    """
    Take the last hidden state slice used as a visual-language interaction embedding.

    Returns shape ``(batch, hidden_dim)`` from the final sequence position.
    """
    last_hidden_state = outputs.hidden_states[hidden_state_index]  # (B, L, H)
    return last_hidden_state[:, -1, :]


def make_teacher_messages(img_path: str, true_label: int) -> List[dict]:
    """Build chat messages for the teacher: image + text hint with the true phase name."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"{USER_PROMPT_TEACHER}{PHASE_NAMES[true_label]}"},
            ],
        },
    ]


def make_student_messages(img_path: str) -> List[dict]:
    """Build chat messages for the student: image only (no phase name hint)."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": img_path}],
        },
    ]
