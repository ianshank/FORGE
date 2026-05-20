"""Python twin of the Rust-side ``schema_id`` computation.

Mirrors ``forge_env_mc::action_map::ActionMap::canonical_sha256``,
``forge_env_mc::reward_config::RewardConfig::canonical_sha256``, and
``forge_env_mc::reward_config::combined_schema_id`` byte-for-byte so a
Python caller (e.g. ``mc_self_play.sh`` orchestrator, the trainer's
bootstrap step, or the muzero_mc test suite) can derive the canonical
``schema_id`` without spinning up the Rust binary.

Cross-language pin: a known-good fixture hash is asserted in BOTH
``tests/python/training/test_muzero_mc_schema_id.py::test_matches_rust_known_good``
AND ``crates/forge-env-mc/src/<…>::xlang_*_pinned_to_known_good``.
Drift on either side fails both suites simultaneously.

Canonicalisation rules (must match the Rust reference exactly):

- **Action map**: entries sorted by ``id`` (ascending). Each entry
  emits ``{"id": <int>, "kind": <str>, <kind-specific fields in
  Rust enum declaration order>}``. ``json.dumps`` with
  ``separators=(",", ":")`` and the field-emit order above produces
  the byte-identical canonical string Rust ``serde_json::to_string``
  produces over ``Vec<ActionEntry>``.
- **Rewards**: each top-level ``[[reward]]`` table converted to JSON
  recursively. Whole-number TOML floats normalised to integers
  (matches JS ``JSON.stringify(100.0) === "100"``). Table keys
  emitted in alphabetical order (Rust's ``toml::Table`` is
  ``BTreeMap``-backed).
- **Combined**: ``sha256(action_map_hash + ":" + rewards_hash)``,
  hex-encoded.
"""

from __future__ import annotations

__all__ = [
    "ACTION_KIND_FIELD_ORDER",
    "action_map_canonical_sha256",
    "combined_schema_id",
    "compute_schema_id_from_paths",
    "rewards_canonical_sha256",
]

import hashlib
import json
import sys
from collections.abc import Mapping  # noqa: TC003 — used at runtime by isinstance/type hints
from pathlib import Path
from typing import Any, Final

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — py3.9/3.10 fallback
    import tomli as tomllib

#: Canonical field order per ``ActionKind`` variant. MUST mirror the
#: Rust enum declaration order in
#: ``crates/forge-env-mc/src/action_map.rs::ActionKind``. ``id`` and
#: ``kind`` are always emitted first; the lists below carry ONLY the
#: kind-specific extras. Variants with no extra fields map to an empty
#: list. Mirrors ``mc-bot/src/schema_id.js::CANONICAL_FIELD_ORDER``.
ACTION_KIND_FIELD_ORDER: Final[Mapping[str, tuple[str, ...]]] = {
    "noop": ("ticks",),
    "move": ("direction", "ticks"),
    "jump": (),
    "attack": (),
    "use": (),
    "place": ("hotbar_slot",),
    "select_slot": ("hotbar_slot",),
    "look": ("yaw_deg", "pitch_deg"),
}


def _canonical_action_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ordered dict that serialises to the same JSON Rust's
    ``serde_json::to_string(&ActionEntry)`` produces.

    ``json.dumps`` preserves dict insertion order in CPython 3.7+, so
    the resulting string is byte-identical to Rust + JS.
    """
    kind = entry.get("kind")
    if not isinstance(kind, str):
        raise ValueError(f"action entry missing string 'kind': {entry!r}")
    out: dict[str, Any] = {"id": entry["id"], "kind": kind}
    order = ACTION_KIND_FIELD_ORDER.get(kind)
    if order is not None:
        for field in order:
            if field in entry:
                out[field] = entry[field]
    else:
        # Forward-compat: unknown kind — emit remaining keys in source
        # dict order. Rust will reject on its side if truly unknown,
        # so the hash drift surfaces clearly.
        for k, v in entry.items():
            if k in ("id", "kind"):
                continue
            out[k] = v
    return out


def action_map_canonical_sha256(action_map: Mapping[str, Any]) -> str:
    """Compute the canonical SHA256 of an action map's entries.

    Mirrors ``forge_env_mc::action_map::ActionMap::canonical_sha256``.

    Args:
        action_map: Parsed TOML mapping with key ``"action"`` holding
            a list of entries.

    Returns:
        Lowercase-hex SHA256 of the canonical serialisation.
    """
    entries = list(action_map.get("action", []))
    entries.sort(key=lambda e: e["id"])
    canonical = [_canonical_action_entry(e) for e in entries]
    text = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _toml_to_canonical_json(value: Any) -> Any:
    """Mirror Rust's ``toml_to_canonical_json``:

    - whole-number floats → integers (matches JS
      ``JSON.stringify(100.0) === "100"``);
    - dicts → JSON objects with alphabetically-sorted keys
      (Rust ``toml::Table`` is ``BTreeMap``-backed);
    - lists recurse;
    - primitives pass through.
    """
    if isinstance(value, dict):
        return {k: _toml_to_canonical_json(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_toml_to_canonical_json(v) for v in value]
    if isinstance(value, bool):
        # bool BEFORE int since ``isinstance(True, int)`` is True.
        return value
    if isinstance(value, float):
        if value == int(value) and abs(value) < float(2**63 - 1):
            return int(value)
        return value
    return value


def rewards_canonical_sha256(rewards: Mapping[str, Any]) -> str:
    """Compute the canonical SHA256 of a rewards config.

    Mirrors ``forge_env_mc::reward_config::RewardConfig::canonical_sha256``.

    Args:
        rewards: Parsed TOML mapping with key ``"reward"`` holding a
            list of reward entries (file-order preserved at the top
            level — only nested table keys are alphabetically sorted).

    Returns:
        Lowercase-hex SHA256 of the canonical serialisation.
    """
    entries = list(rewards.get("reward", []))
    normalised = [_toml_to_canonical_json(e) for e in entries]
    text = json.dumps(normalised, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combined_schema_id(action_map_hash: str, rewards_hash: str) -> str:
    """Fold the two component hashes into the canonical ``schema_id``.

    Order is fixed: ``sha256(action_map_hash + ":" + rewards_hash)``.
    Mirrors ``forge_env_mc::reward_config::combined_schema_id``.
    """
    combined = f"{action_map_hash}:{rewards_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def compute_schema_id_from_paths(
    action_map_path: str | Path,
    rewards_path: str | Path,
) -> str:
    """End-to-end helper: load the two TOML files and return the
    combined ``schema_id``. Used by the ``compute-schema-id`` CLI
    subcommand + ``scripts/mc_self_play.sh``.
    """
    with Path(action_map_path).open("rb") as f:
        action_map = tomllib.load(f)
    with Path(rewards_path).open("rb") as f:
        rewards = tomllib.load(f)
    return combined_schema_id(
        action_map_canonical_sha256(action_map),
        rewards_canonical_sha256(rewards),
    )
