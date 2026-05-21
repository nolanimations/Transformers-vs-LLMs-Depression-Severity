"""
Shared utilities: seeding, logging setup.

    from src.utils import set_seed, get_logger
"""

import logging
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.
    Call this at the very start of every training script and notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (small performance cost — worth it for science)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes timestamped messages to stdout.

        log = get_logger(__name__)
        log.info("Training epoch 1/4")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
