"""Central logging configuration for console (and optional file handlers added in trainer)."""

import logging


def setup_logging() -> None:
    """
    Configure the root logger once with a stream handler.

    File logging for training is attached separately in ``trainer.train`` on the main process.
    """
    if logging.root.handlers:
        return
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
