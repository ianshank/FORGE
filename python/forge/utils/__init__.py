"""FORGE utils module."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def dataclass_from_dict(cls: type[_T], data: dict[str, Any]) -> _T:
    """Instantiate a dataclass from a dict, ignoring unknown fields.

    Unknown keys are logged at DEBUG level for backwards-compatibility
    diagnostics. Only fields declared in the dataclass are passed through.
    """
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(data.keys()) - known
    if unknown:
        _logger.debug("Ignoring unknown fields for %s: %s", cls.__name__, unknown)
    return cls(**{k: v for k, v in data.items() if k in known})
