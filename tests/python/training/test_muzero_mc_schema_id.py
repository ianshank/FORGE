"""Cross-language pin tests for the Python ``schema_id`` twin.

These tests assert that
``python/forge/training/muzero_mc/schema_id.py`` produces the
byte-identical canonical SHA256 the Rust + JS sides produce for the
same fixture inputs. Drift on any side fails both this test AND its
Rust counterparts:

- ``crates/forge-env-mc/src/action_map.rs::xlang_schema_id_pinned_to_known_good``
- ``crates/forge-env-mc/src/reward_config.rs::xlang_rewards_schema_id_pinned_to_known_good``
- ``crates/forge-env-mc/src/reward_config.rs::xlang_shipped_rewards_schema_id_folds_nested_files``
- ``crates/forge-env-mc/src/block_embeddings.rs::xlang_block_embeddings_pinned_to_known_good``
- ``mc-bot/test/schema_id.test.ts`` (action map JS twin)
- ``mc-bot/test/reward_config.test.ts`` (rewards JS twin)
- ``mc-bot/test/block_embeddings.test.ts`` (obs-layout JS twin)

If you change ANY of: ``ACTION_KIND_FIELD_ORDER`` (Python /
``CANONICAL_FIELD_ORDER`` in JS), the Rust ``ActionKind`` enum
declaration order, the TOML→JSON normalisation rules in
``_toml_to_canonical_json``, OR the ``combined_schema_id`` formula
— bump the pinned hashes in EVERY suite simultaneously.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from forge.training.muzero_mc.schema_id import (
    ACTION_KIND_FIELD_ORDER,
    NESTED_REWARD_PATH_KEY_CONFIG,
    NESTED_REWARD_PATH_KEY_CRAFTING,
    action_map_canonical_sha256,
    block_embeddings_canonical_sha256,
    block_embeddings_vocab_size,
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
        "eat",
        "craft_planks",
        "craft_sticks",
        "craft_crafting_table",
        "place_crafting_table",
        "craft_wooden_pickaxe",
        "mine_stone",
        "craft_stone_pickaxe",
        "craft_furnace",
        "place_furnace",
        "smelt_iron",
        "craft_iron_pickaxe",
        "equip_pickaxe",
        "sprint",
        "sneak",
        "swim_up",
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


def test_shipped_rewards_canonical_sha256_folds_nested_files() -> None:
    """Pinned against
    ``xlang_shipped_rewards_schema_id_folds_nested_files`` in
    ``crates/forge-env-mc/src/reward_config.rs``. Nested milestone /
    crafting file **contents** must fold into the hash.
    """
    repo_root = Path(__file__).resolve().parents[3]
    rewards_path = repo_root / "configs" / "minecraft" / "rewards.toml"
    if not rewards_path.is_file():
        pytest.skip("shipped rewards.toml missing")
    with rewards_path.open("rb") as handle:
        rewards = tomllib.load(handle)
    assert (
        rewards_canonical_sha256(rewards, source_path=rewards_path)
        == "78f96c103767f3db7280175e92b8564937bcb5d505e4d75e8aab0c570d237f4b"
    )


def test_nested_path_missing_file_fails_closed(tmp_path: Path) -> None:
    """A set ``config_path`` whose file is missing must not hash as if
    the bot's hardcoded milestone defaults applied.
    """
    rewards_path = tmp_path / "rewards.toml"
    rewards_path.write_text(
        '[[reward]]\nkind = "milestone"\nconfig_path = "missing.toml"\n',
        encoding="utf-8",
    )
    rewards = {"reward": [{"kind": "milestone", "config_path": "missing.toml"}]}
    with pytest.raises(FileNotFoundError, match="nested reward file not found"):
        rewards_canonical_sha256(rewards, source_path=rewards_path)


def test_nested_rename_without_content_change_does_not_bump_hash(tmp_path: Path) -> None:
    """Path-string rename with identical nested content must not bump."""
    nested_body = "[milestones]\nfirst_wood = { reward = 10.0, once = true }\n"
    (tmp_path / "mil_a.toml").write_text(nested_body, encoding="utf-8")
    (tmp_path / "mil_b.toml").write_text(nested_body, encoding="utf-8")
    rewards_a = tmp_path / "a.toml"
    rewards_b = tmp_path / "b.toml"
    rewards_a.write_text(
        '[[reward]]\nkind = "milestone"\nconfig_path = "mil_a.toml"\n',
        encoding="utf-8",
    )
    rewards_b.write_text(
        '[[reward]]\nkind = "milestone"\nconfig_path = "mil_b.toml"\n',
        encoding="utf-8",
    )
    with rewards_a.open("rb") as handle:
        data_a = tomllib.load(handle)
    with rewards_b.open("rb") as handle:
        data_b = tomllib.load(handle)
    assert rewards_canonical_sha256(data_a, source_path=rewards_a) == rewards_canonical_sha256(
        data_b, source_path=rewards_b
    )


def test_block_embeddings_shipped_pin() -> None:
    """Pinned against ``xlang_block_embeddings_pinned_to_known_good``."""
    from forge.models.muzero_config import (
        DEFAULT_NUM_BLOCK_EMBEDDINGS,
        MuZeroConfig,
        load_num_block_embeddings,
    )

    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "configs" / "minecraft" / "block_embeddings.toml"
    if not path.is_file():
        pytest.skip("shipped block_embeddings.toml missing")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert block_embeddings_vocab_size(data) == DEFAULT_NUM_BLOCK_EMBEDDINGS
    assert load_num_block_embeddings(path) == DEFAULT_NUM_BLOCK_EMBEDDINGS
    assert MuZeroConfig().num_block_embeddings == DEFAULT_NUM_BLOCK_EMBEDDINGS
    assert (
        block_embeddings_canonical_sha256(data)
        == "b5aef9f434474c17ffbdee7fe894ae93ada0f4e6477cb51b0a8cf4fc0d7a7a7e"
    )


def test_empty_nested_path_fails_closed(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rewards.toml"
    rewards_path.write_text(
        f'[[reward]]\nkind = "milestone"\n{NESTED_REWARD_PATH_KEY_CONFIG} = ""\n',
        encoding="utf-8",
    )
    rewards = {"reward": [{"kind": "milestone", NESTED_REWARD_PATH_KEY_CONFIG: ""}]}
    with pytest.raises(ValueError, match="non-empty string"):
        rewards_canonical_sha256(rewards, source_path=rewards_path)


def test_non_string_nested_path_fails_closed(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rewards.toml"
    rewards_path.write_text(
        f'[[reward]]\nkind = "milestone"\n{NESTED_REWARD_PATH_KEY_CONFIG} = 1\n',
        encoding="utf-8",
    )
    rewards = {"reward": [{"kind": "milestone", NESTED_REWARD_PATH_KEY_CONFIG: 1}]}
    with pytest.raises(ValueError, match="non-empty string"):
        rewards_canonical_sha256(rewards, source_path=rewards_path)


def test_missing_crafting_config_path_fails_closed(tmp_path: Path) -> None:
    rewards_path = tmp_path / "rewards.toml"
    rewards_path.write_text(
        f'[[reward]]\nkind = "milestone"\n{NESTED_REWARD_PATH_KEY_CRAFTING} = "missing.toml"\n',
        encoding="utf-8",
    )
    rewards = {"reward": [{"kind": "milestone", NESTED_REWARD_PATH_KEY_CRAFTING: "missing.toml"}]}
    with pytest.raises(FileNotFoundError, match="nested reward file not found"):
        rewards_canonical_sha256(rewards, source_path=rewards_path)


def test_without_source_path_hashes_path_strings() -> None:
    """Fixture pin: omitted source_path must not try to load nested files."""
    rewards = {
        "reward": [
            {
                "kind": "milestone",
                NESTED_REWARD_PATH_KEY_CONFIG: "configs/minecraft/does-not-exist.toml",
            }
        ]
    }
    digest = rewards_canonical_sha256(rewards)
    assert len(digest) == 64


def test_repo_style_nested_path_prefers_sibling_over_cwd_relative(tmp_path: Path) -> None:
    unique = "[milestones]\nsibling_only = { reward = 99.0, once = true }\n"
    (tmp_path / "milestone_rewards.toml").write_text(unique, encoding="utf-8")
    rewards_path = tmp_path / "rewards.toml"
    rewards_path.write_text(
        f'[[reward]]\nkind = "milestone"\n{NESTED_REWARD_PATH_KEY_CONFIG} = '
        '"configs/minecraft/milestone_rewards.toml"\n',
        encoding="utf-8",
    )
    with rewards_path.open("rb") as handle:
        data = tomllib.load(handle)
    from_repo_style = rewards_canonical_sha256(data, source_path=rewards_path)

    dir2 = tmp_path / "basename"
    dir2.mkdir()
    (dir2 / "milestone_rewards.toml").write_text(unique, encoding="utf-8")
    rewards2 = dir2 / "rewards.toml"
    rewards2.write_text(
        f'[[reward]]\nkind = "milestone"\n{NESTED_REWARD_PATH_KEY_CONFIG} = '
        '"milestone_rewards.toml"\n',
        encoding="utf-8",
    )
    with rewards2.open("rb") as handle:
        data2 = tomllib.load(handle)
    from_basename = rewards_canonical_sha256(data2, source_path=rewards2)
    assert from_repo_style == from_basename


def test_load_num_block_embeddings_missing_file_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from forge.models.muzero_config import (
        DEFAULT_NUM_BLOCK_EMBEDDINGS,
        load_num_block_embeddings,
    )

    missing = tmp_path / "no_such_embeddings.toml"
    with caplog.at_level("WARNING", logger="forge.models.muzero_config"):
        assert load_num_block_embeddings(missing) == DEFAULT_NUM_BLOCK_EMBEDDINGS
    assert "missing" in caplog.text.lower()


def test_load_num_block_embeddings_corrupt_toml_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from forge.models.muzero_config import (
        DEFAULT_NUM_BLOCK_EMBEDDINGS,
        load_num_block_embeddings,
    )

    corrupt = tmp_path / "bad.toml"
    corrupt.write_text("not = valid toml [", encoding="utf-8")
    with caplog.at_level("WARNING", logger="forge.models.muzero_config"):
        assert load_num_block_embeddings(corrupt) == DEFAULT_NUM_BLOCK_EMBEDDINGS
    assert "Failed to parse" in caplog.text


def test_muzero_config_grid_depth_uses_named_default() -> None:
    from forge.models.muzero_config import DEFAULT_GRID_DEPTH, MuZeroConfig

    assert MuZeroConfig().grid_depth == DEFAULT_GRID_DEPTH
