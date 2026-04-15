"""FORGE: Python agent framework for the FORGE simulation project."""

from __future__ import annotations

import contextlib

from forge.utils.logging_config import setup_logging

# Expose training loggers at the forge package level so that scripts can do:
#   from forge import ForgeLogger, WandbLogger
with contextlib.suppress(ImportError):
    from forge.training.loggers import (
        CompositeLogger,
        ForgeLogger,
        MLflowLogger,
        TensorBoardLogger,
        WandbLogger,
        make_logger,
    )
