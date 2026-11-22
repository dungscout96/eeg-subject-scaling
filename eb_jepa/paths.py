"""Where preprocessed HBN data lives: explicit config > ``HBN_PREPROCESS_DIR``."""

import os
from pathlib import Path

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


# Optional extra locations checked for auto-detection (none by default).
PREPROCESSED_DIRS: list[Path] = []


def resolve_preprocessed_dir(configured: str | None) -> Path | None:
    """Return preprocessed_dir: explicit config > ``HBN_PREPROCESS_DIR`` > auto-detect."""
    if configured:
        return Path(configured)
    env = os.environ.get("HBN_PREPROCESS_DIR")
    if env:
        logger.info("preprocessed_dir from HBN_PREPROCESS_DIR: %s", env)
        return Path(env)
    for p in PREPROCESSED_DIRS:
        if p.exists():
            logger.info("Auto-detected preprocessed_dir: %s", p)
            return p
    return None
