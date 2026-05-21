"""Cross-language pin tests for the Python ``schema_id`` twin.

These tests assert that
``python/forge/training/muzero_mc/schema_id.py`` produces the
byte-identical canonical SHA256 the Rust + JS sides produce for the
same fixture inputs. Drift on any side fails both this test AND its
Rust counterparts:

- ``crates/forge-env-mc/src/action_map.rs::xlang_schema_id_pinned_to_known_good``
- ``crates/forge-env-mc/src/reward_config.rs::xlang_rewards_schema_id_pinned_to_known_good``
- ``mc-bot/test/schema_id.test.js`` (action map JS twin)
- ``mc-bot/test/reward_config.test.js`` (rewards JS twin)

If you change ANY of: ``ACTION_KIND_FIELD_ORDER`` (Python /
``CANONICAL_FIELD_ORDER`` in JS), the Rust ``ActionKind`` enum
declaration order, the TOML→JSON normalisation rules in
``_toml_to_canonical_json``, OR the ``combined_schema_id`` formula
— bump the pinned hashes in EVERY suite simultaneously.
"""

from __future__ import annotations

import hashlib

import pytest

from forge.training.muzero_mc.schema_id import (
    ACTION_KIND_FIELD_ORDER,
    action_map_canonical_sha256,
    combined_schema_id,
    rewards_canonical_sha256,
)

# --- Action map xlang pin ---------------------------------------------


def test_action_map_canonical_sha256_matches_rust_known_good() -> None:
    """Pinned against `xlang_schema_id_pinned_to_known_good` in
    `crates/forge-env-mc/src/action_map.rs`. Drift on either side
    fails BOTH tests with the same expected hash.
    """
    action_map = {
        "schema_version": 1,
        "action": [
            {"id": 0, "kind": "noop", "ticks": 1},
            {"id": 1, "kind": "move", "direction": "forward", "ticks": 4},
            {"id": 2, "kind": "jump"},
        ],
    }
    assert (
        action_map_canonical_sha256(action_map)
        == "587b13077b8c7cd90503f9ee5e1bae1bb92bdf738c8abc51d2ff6deb1908224f"
    )


def test_action_map_canonical_sha256_invariant_under_reorder() -> None:
    """Entries are sorted by id before hashing, so reordering the
    input list MUST NOT change the hash. Mirrors
    `canonical_sha256_invariant_under_reorder` on the Rust side.
    """
    base = {
        "action": [
            {"id": 0, "kind": "noop", "ticks": 1},
            {"id": 1, "kind": "jump"},
        ],
    }
    reversed_ = {"action": list(reversed(base["action"]))}
    assert action_map_canonical_sha256(base) == action_map_canonical_sha256(reversed_)


def test_action_map_canonical_kind_field_order_covers_rust_variants() -> None:
    """Pins the variant set against the Rust `ActionKind` enum
    declaration. If a new variant is added on the Rust side, the
    `ACTION_KIND_FIELD_ORDER` mapping MUST gain a matching entry —
    otherwise the canonicaliser falls back to "unknown kind" and the
    hash drifts silently.
    """
    expected_variants = {
        "noop",
        "move",
        "jump",
        "attack",
        "use",
        "place",
        "select_slot",
        "look",
    }
    assert set(ACTION_KIND_FIELD_ORDER) == expected_variants


# --- Rewards xlang pin ------------------------------------------------


def test_rewards_canonical_sha256_matches_rust_known_good() -> None:
    """Pinned against
    `crates/forge-env-mc/src/reward_config.rs::xlang_rewards_schema_id_pinned_to_known_good`.
    Uses the same fixture (`sample_toml`) so any drift surfaces in
    both suites.
    """
    rewards = {
        "schema_version": 1,
        "reward": [
            {"kind": "survival", "value": 0.01},
            {
                "kind": "distance_to_goal",
                "clip": 100.0,
                "target": {"x": 0, "y": 64, "z": 0},
            },
        ],
    }
    assert (
        rewards_canonical_sha256(rewards)
        == "451b10f995371924a374633e5c42deab35c137fbbc65bc8f551bf2bd7844b478"
    )


def test_rewards_canonical_sha256_normalises_whole_floats_to_ints() -> None:
    """Rust + JS treat `100.0` and `100` as the same canonical form.
    Python's `json.dumps(100.0)` emits `"100.0"` by default, so the
    canonicaliser MUST coerce whole floats to ints first. Without this
    normalisation, the hash would diverge from Rust + JS.
    """
    a = rewards_canonical_sha256({"reward": [{"kind": "x", "v": 100.0}]})
    b = rewards_canonical_sha256({"reward": [{"kind": "x", "v": 100}]})
    assert a == b


def test_rewards_canonical_sha256_sorts_nested_table_keys() -> None:
    """Rust's `toml::Table` is BTreeMap-backed; nested table keys MUST
    serialise in alphabetical order. Verify by reordering the nested
    dict and asserting hash invariance.
    """
    a = rewards_canonical_sha256({"reward": [{"kind": "x", "target": {"x": 1, "y": 2, "z": 3}}]})
    b = rewards_canonical_sha256({"reward": [{"kind": "x", "target": {"z": 3, "y": 2, "x": 1}}]})
    assert a == b


# --- Combined hash ----------------------------------------------------


def test_combined_schema_id_is_deterministic_and_order_sensitive() -> None:
    """Mirrors
    `crates/forge-env-mc/src/reward_config.rs::combined_schema_id_is_deterministic_and_unique`.
    """
    c1 = combined_schema_id("aaa", "bbb")
    c2 = combined_schema_id("aaa", "bbb")
    c3 = combined_schema_id("aaa", "ccc")
    c4 = combined_schema_id("bbb", "aaa")  # order swap
    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert len(c1) == 64  # hex sha256


def test_combined_schema_id_matches_documented_formula() -> None:
    """Pins the exact formula:
    ``sha256(action_map_hash + ":" + rewards_hash)``.
    """
    am_hash = "a" * 64
    rw_hash = "b" * 64
    expected = hashlib.sha256(f"{am_hash}:{rw_hash}".encode()).hexdigest()
    assert combined_schema_id(am_hash, rw_hash) == expected


# --- End-to-end via TOML files ----------------------------------------


def test_compute_schema_id_from_paths_loads_repo_default_configs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Loads the SHIPPED `configs/minecraft/{action_map,rewards}.toml`
    and asserts the combined hash is non-empty 64-hex. The exact
    value isn't pinned (it's not a fixture) but the path-loading
    plumbing is.
    """
    from pathlib import Path

    from forge.training.muzero_mc.schema_id import compute_schema_id_from_paths

    # Locate the repo root via the package layout — same discipline
    # PR #58's integration conftest uses.
    repo_root = Path(__file__).resolve().parents[3]
    action_map_path = repo_root / "configs" / "minecraft" / "action_map.toml"
    rewards_path = repo_root / "configs" / "minecraft" / "rewards.toml"
    if not action_map_path.exists() or not rewards_path.exists():
        pytest.skip(
            f"shipped configs missing — action_map={action_map_path.exists()}, "
            f"rewards={rewards_path.exists()}"
        )
    h = compute_schema_id_from_paths(action_map_path, rewards_path)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
