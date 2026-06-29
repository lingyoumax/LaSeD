"""Rotate periodic checkpoints on disk, keeping the N most recent step folders."""

import glob
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_old_checkpoints(output_dir: str, save_total_limit: Optional[int]) -> None:
    """
    Delete oldest ``checkpoint-*`` directories under ``output_dir`` beyond ``save_total_limit``.

    Does not remove ``best_model`` or ``final_model``.
    """
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)

    if len(checkpoint_dirs) <= save_total_limit:
        return

    def get_step(ckpt_dir: str) -> int:
        try:
            return int(os.path.basename(ckpt_dir).split("-")[1])
        except Exception:
            return 0

    checkpoint_dirs.sort(key=get_step, reverse=True)

    for old_ckpt in checkpoint_dirs[save_total_limit:]:
        if os.path.isdir(old_ckpt):
            shutil.rmtree(old_ckpt)
            logger.info("Removed old checkpoint: %s", old_ckpt)
