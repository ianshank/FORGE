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

# `__all__` makes the conditional re-exports explicit so static analyzers
# (ruff F401, pylint) recognise them as the package's public surface even when
# the optional logger backends fail to import on a stripped-down install.
__all__ = [
    "CompositeLogger",
    "ForgeLogger",
    "MLflowLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "make_logger",
    "setup_logging",
]
