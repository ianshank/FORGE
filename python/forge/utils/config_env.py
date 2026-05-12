"""Shared environment-variable override helper for config dataclasses.

Used by ``forge.config.ForgeConfig`` and
``forge.mangomas.config.MangoMASBridgeConfig`` so the override semantics
stay identical across the codebase. Values are read from
``<prefix>_<SECTION>_<FIELD>`` style environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import fields
from typing import Any

logger = logging.getLogger(__name__)


_SUPPORTED_SCALAR_TYPES: tuple[Any, ...] = (
    bool, "bool", int, "int", float, "float", str, "str",
)


class UnsupportedFieldType(TypeError):
    """Raised when a dataclass field's type isn't a primitive scalar.

    The shared env-override helper only knows how to cast string env vars
    into ``bool / int / float / str``. Complex annotations such as
    ``list[str]`` or ``dict[str, Any]`` would silently take the raw env
    value, hiding configuration bugs — so we refuse them up front.
    """


def _parse_value(field_type: Any, val: str) -> Any:
    """Best-effort cast ``val`` to the declared dataclass field type.

    Raises :class:`UnsupportedFieldType` for non-scalar annotations so
    callers can surface a config-design issue instead of silently
    assigning a string to (e.g.) a ``list[str]`` field.
    """
    if field_type in (bool, "bool"):
        return val.lower() in ("1", "true", "yes", "on")
    if field_type in (int, "int"):
        return int(val)
    if field_type in (float, "float"):
        return float(val)
    if field_type in (str, "str"):
        return val
    msg = f"unsupported field type for env-override: {field_type!r}"
    raise UnsupportedFieldType(msg)


def apply_env_overrides(obj: Any, section: str, *, prefix: str = "FORGE_") -> None:
    """Apply ``<prefix><SECTION>_<FIELD>`` env overrides to ``obj`` in place.

    Only fields declared on the dataclass are considered. Unknown keys are
    ignored. Invalid scalar values are logged and skipped.
    Unsupported field types (collections, optionals) raise a warning and
    are skipped — extending the helper to cast those should be a
    deliberate, audited change.
    """
    for f in fields(obj):
        key = f"{prefix}{section.upper()}_{f.name.upper()}"
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            parsed = _parse_value(f.type, val)
        except UnsupportedFieldType as exc:
            logger.warning(
                "Skipping env override %s=%s for field %s: %s",
                key, val, f.name, exc,
            )
            continue
        except (ValueError, TypeError) as exc:
            logger.warning("Invalid env override %s=%s: %s", key, val, exc)
            continue
        setattr(obj, f.name, parsed)
        logger.debug("Env override applied: %s = %s", key, parsed)
