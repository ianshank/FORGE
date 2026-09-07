"""Evidence integrity gate.

Enforces that:
1. Every snapshot under ``docs/results/`` is indexed in ``docs/results/INDEX.toml``
   with its exact SHA-256 digest.
2. Any snapshot with zero evidential episodes carries an explicit
   ``<snapshot>.declaration`` audit note recording what it is and why it is
   retained.
3. No snapshot with evidential episodes carries a declaration (preventing
   laundering).
4. Every published Markdown table citing a committed snapshot agrees with it
   on episode count and any cited per-episode values.
5. All failure messages state the remedy, not only the mismatch.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import tomllib

from scripts.mc_plot_baseline import _is_evidential

REPO_ROOT = Path(__file__).resolve().parents[2]

ROW_LINK_PATTERN = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*.*?\[.*?\]\(([^)]+\.json)\)",
    re.MULTILINE,
)
REWARD_CLAIM_PATTERN = re.compile(r"reward[=:]\s*([\u2212\-+]?\d+(?:\.\d+)?)")


def _sha256(path: Path) -> str:
    """Compute hex SHA-256 of the file at `path`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_index(root: Path, index_path: Path, errors: list[str]) -> set[Path] | None:
    """Validate INDEX.toml existence and all [[snapshots]] entries."""
    if not index_path.is_file():
        errors.append(
            f"Evidence index missing at {index_path.relative_to(root)}. "
            "Remedy: create docs/results/INDEX.toml listing every snapshot under "
            "docs/results/ with its sha256 digest."
        )
        return None

    try:
        index_data = tomllib.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(
            f"Malformed INDEX.toml at {index_path.relative_to(root)}: {exc}. "
            "Remedy: format docs/results/INDEX.toml as valid TOML with [[snapshots]] tables."
        )
        return None

    entries = index_data.get("snapshots", [])
    if not isinstance(entries, list):
        errors.append(
            f"INDEX.toml at {index_path.relative_to(root)} has invalid 'snapshots' key "
            f"(expected list, got {type(entries).__name__}). "
            "Remedy: define [[snapshots]] array in docs/results/INDEX.toml."
        )
        return None

    indexed_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            errors.append(
                f"INDEX.toml contains invalid [[snapshots]] entry: {entry}. "
                "Remedy: supply both 'path' and 'sha256' in each [[snapshots]] entry."
            )
            continue

        rel_str = entry["path"]
        expected_sha = entry["sha256"]
        snap_path = root / rel_str
        indexed_paths.add(snap_path.resolve())

        if not snap_path.is_file():
            errors.append(
                f"Indexed snapshot missing from disk: {rel_str}. "
                f"Remedy: restore {rel_str} or remove its entry from {index_path.relative_to(root)}."
            )
            continue

        actual_sha = _sha256(snap_path)
        if actual_sha != expected_sha:
            errors.append(
                f"Snapshot digest mismatch for {rel_str}: expected {expected_sha}, "
                f"got {actual_sha}. "
                f"Remedy: if the snapshot change was intentional, update its sha256 in "
                f"{index_path.relative_to(root)}."
            )

    return indexed_paths


def _validate_snapshot_json(
    root: Path, json_file: Path, errors: list[str]
) -> tuple[dict[str, Any] | None, int]:
    """Parse snapshot JSON and return the data dict and count of evidential episodes."""
    rel_json = json_file.relative_to(root)
    try:
        content = json_file.read_text(encoding="utf-8")
        if not content.strip():
            errors.append(
                f"Snapshot {rel_json} is empty. "
                f"Remedy: populate {rel_json} with valid snapshot JSON or remove it."
            )
            return None, 0
        data = json.loads(content)
    except Exception as exc:
        errors.append(
            f"Snapshot {rel_json} contains malformed JSON: {exc}. "
            f"Remedy: fix JSON syntax in {rel_json}."
        )
        return None, 0

    if not isinstance(data, dict):
        errors.append(
            f"Snapshot {rel_json} root is not a JSON object. "
            f"Remedy: ensure {rel_json} top-level structure is a JSON object."
        )
        return None, 0

    per_episode = data.get("per_episode")
    if not isinstance(per_episode, list):
        errors.append(
            f"Snapshot {rel_json} has non-list 'per_episode' field: {type(per_episode).__name__}. "
            f"Remedy: set 'per_episode' to a JSON list of episode records in {rel_json}."
        )
        return None, 0

    hello = data.get("hello")
    hello_obs_dim = hello.get("obs_dim") if isinstance(hello, dict) else None

    evidential_count = sum(
        1 for rec in per_episode if isinstance(rec, dict) and _is_evidential(rec, hello_obs_dim)
    )
    return data, evidential_count


def _validate_declaration(
    root: Path, json_file: Path, evidential_count: int, errors: list[str]
) -> None:
    """Validate snapshot declaration rules and syntax."""
    rel_json = json_file.relative_to(root)
    decl_file = json_file.with_name(f"{json_file.name}.declaration")
    has_decl = decl_file.is_file()

    if evidential_count == 0:
        if not has_decl:
            errors.append(
                f"Snapshot {rel_json} has 0 evidential records and no declaration file. "
                f"Remedy: create {decl_file.relative_to(root)} recording what the snapshot is "
                "and why it is retained."
            )
            return

        try:
            decl_data = tomllib.loads(decl_file.read_text(encoding="utf-8"))
            decl_table = decl_data.get("declaration", {})
            if not isinstance(decl_table, dict) or not decl_table.get("rationale"):
                errors.append(
                    f"Declaration {decl_file.relative_to(root)} missing [declaration] table "
                    "or non-empty 'rationale' field. "
                    f"Remedy: populate 'rationale' under [declaration] in {decl_file.relative_to(root)}."
                )
            superseded_by = decl_table.get("superseded_by")
            if superseded_by:
                sup_parent = (decl_file.parent / superseded_by).resolve()
                sup_root = (root / superseded_by).resolve()
                if not sup_parent.is_file() and not sup_root.is_file():
                    errors.append(
                        f"Declaration {decl_file.relative_to(root)} names nonexistent "
                        f"supersession '{superseded_by}'. "
                        f"Remedy: update 'superseded_by' in {decl_file.relative_to(root)} "
                        "to point to an existing snapshot."
                    )
        except Exception as exc:
            errors.append(
                f"Malformed declaration file {decl_file.relative_to(root)}: {exc}. "
                f"Remedy: ensure {decl_file.relative_to(root)} is valid TOML."
            )
    elif has_decl:
        errors.append(
            f"Laundering detected: snapshot {rel_json} has {evidential_count} evidential "
            f"record(s) but carries declaration {decl_file.relative_to(root)}. "
            f"Remedy: remove declaration {decl_file.relative_to(root)} because declarations "
            "are only permitted for snapshots with zero evidential records."
        )


def _validate_snapshots_and_declarations(
    root: Path, results_dir: Path, indexed_paths: set[Path], errors: list[str]
) -> None:
    """Verify all snapshots are indexed, evidential or declared, and valid."""
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.resolve() not in indexed_paths:
            errors.append(
                f"Unindexed snapshot on disk: {json_file.relative_to(root)}. "
                f"Remedy: add {json_file.relative_to(root)} and its sha256 digest to "
                f"docs/results/INDEX.toml."
            )

        data, evidential_count = _validate_snapshot_json(root, json_file, errors)
        if data is not None:
            _validate_declaration(root, json_file, evidential_count, errors)


def _validate_markdown_tables(root: Path, results_dir: Path, errors: list[str]) -> None:
    """Verify table claims in docs/results/*.md agree with cited snapshots."""
    for md_file in sorted(results_dir.glob("*.md")):
        rel_md = md_file.relative_to(root)
        text = md_file.read_text(encoding="utf-8")
        for match in ROW_LINK_PATTERN.finditer(text):
            run_name = match.group(1).strip()
            row_episodes = int(match.group(2).strip())
            rollout_info = match.group(3).strip()
            cited_link = match.group(4).strip()

            target_path = (md_file.parent / cited_link).resolve()
            if not target_path.is_file():
                errors.append(
                    f"Table in {rel_md} cites nonexistent snapshot: '{cited_link}'. "
                    f"Remedy: correct the snapshot link in {rel_md}."
                )
                continue

            try:
                snap_data = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            snap_episodes = snap_data.get("episodes_observed")
            if snap_episodes is None:
                per_ep = snap_data.get("per_episode")
                snap_episodes = len(per_ep) if isinstance(per_ep, list) else 0

            if row_episodes != snap_episodes:
                errors.append(
                    f"Table in {rel_md} row '{run_name}' claims {row_episodes} episode(s), but cited "
                    f"snapshot {target_path.relative_to(root)} records {snap_episodes} episode(s). "
                    f"Remedy: update the episode count in {rel_md} or point to the matching snapshot."
                )

            reward_match = REWARD_CLAIM_PATTERN.search(rollout_info)
            if reward_match:
                claimed_str = reward_match.group(1).replace("\u2212", "-")
                try:
                    claimed_reward = float(claimed_str)
                    per_ep = snap_data.get("per_episode", [])
                    rewards = [
                        ep.get("total_reward")
                        for ep in per_ep
                        if isinstance(ep, dict) and ep.get("total_reward") is not None
                    ]
                    if not any(abs(r - claimed_reward) < 0.05 for r in rewards):
                        errors.append(
                            f"Table in {rel_md} row '{run_name}' cites reward {claimed_str}, but "
                            f"no episode in {target_path.relative_to(root)} has a matching reward. "
                            f"Remedy: update the reward claim in {rel_md} to match {target_path.name}."
                        )
                except ValueError:
                    pass


def check_evidence_integrity(root: Path) -> list[str]:
    """Check evidence integrity for the repository at `root`.

    Returns a list of error strings; an empty list indicates all checks passed.
    Every error string includes an actionable 'Remedy:' instruction.
    """
    errors: list[str] = []
    results_dir = root / "docs" / "results"

    if not results_dir.is_dir():
        return errors

    indexed_paths = _validate_index(root, results_dir / "INDEX.toml", errors)
    if indexed_paths is None:
        return errors

    _validate_snapshots_and_declarations(root, results_dir, indexed_paths, errors)
    _validate_markdown_tables(root, results_dir, errors)

    return errors


# ==============================================================================
# Gate test over the committed repository
# ==============================================================================


def test_committed_evidence_integrity() -> None:
    """Every table row citing a snapshot agrees with it, and non-evidential snapshots are declared."""
    errors = check_evidence_integrity(REPO_ROOT)
    assert not errors, (
        f"Evidence integrity check failed ({len(errors)} error(s)):\n"
        + "\n".join(f"- {e}" for e in errors)
    )


# ==============================================================================
# Negative unit tests against synthetic repository trees
# ==============================================================================


def _make_snapshot(
    episodes: int,
    *,
    evidential: bool = False,
    first_reward: float = 0.0,
    first_steps: int = 1,
) -> dict[str, Any]:
    per_episode: list[dict[str, Any]] = []
    for i in range(episodes):
        if i == 0 and evidential:
            per_episode.append(
                {
                    "episode_id": "ep-000001",
                    "steps": first_steps,
                    "total_reward": first_reward,
                    "protocol_errors": 0,
                    "obs_dim": 920,
                    "terminated": True,
                    "truncated": False,
                }
            )
        elif i == 0:
            per_episode.append(
                {
                    "episode_id": "ep-000001",
                    "steps": first_steps,
                    "total_reward": first_reward,
                    "protocol_errors": 1,
                    "obs_dim": 0,
                    "terminated": False,
                    "truncated": True,
                }
            )
        else:
            per_episode.append(
                {
                    "episode_id": f"ep-{i+1:06d}",
                    "steps": 1,
                    "total_reward": 0.0,
                    "protocol_errors": 1,
                    "obs_dim": 0,
                    "terminated": False,
                    "truncated": True,
                }
            )
    return {
        "episodes_observed": episodes,
        "episodes_target": episodes,
        "hello": {"obs_dim": 920},
        "per_episode": per_episode,
    }


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def test_negative_laundering_by_declaring_everything(tmp_path: Path) -> None:
    """Declaring an evidential snapshot is rejected as laundering."""
    snap = _make_snapshot(5, evidential=True, first_reward=10.0, first_steps=10)
    snap_json = json.dumps(snap)
    sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()

    _write_tree(
        tmp_path,
        {
            "docs/results/sample.json": snap_json,
            "docs/results/sample.json.declaration": (
                '[declaration]\nsnapshot = "sample.json"\nrationale = "Trying to declare evidential snapshot"\n'
            ),
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/sample.json"\nsha256 = "{sha}"\n'
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("Laundering detected" in e for e in errors), errors
    assert any("Remedy: remove declaration" in e for e in errors), errors


def test_negative_declaration_naming_nonexistent_supersession(tmp_path: Path) -> None:
    """A declaration naming a nonexistent supersession fails."""
    snap = _make_snapshot(3, evidential=False)
    snap_json = json.dumps(snap)
    sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()

    _write_tree(
        tmp_path,
        {
            "docs/results/sample.json": snap_json,
            "docs/results/sample.json.declaration": (
                '[declaration]\nsnapshot = "sample.json"\n'
                'rationale = "Old run"\nsuperseded_by = "nonexistent.json"\n'
            ),
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/sample.json"\nsha256 = "{sha}"\n'
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("nonexistent supersession 'nonexistent.json'" in e for e in errors), errors
    assert any("Remedy:" in e for e in errors), errors


def test_negative_row_disagreeing_on_episode_count(tmp_path: Path) -> None:
    """A Markdown table row claiming a different episode count than cited snapshot fails."""
    snap = _make_snapshot(30, evidential=False)
    snap_json = json.dumps(snap)
    sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()

    _write_tree(
        tmp_path,
        {
            "docs/results/sample.json": snap_json,
            "docs/results/sample.json.declaration": (
                '[declaration]\nsnapshot = "sample.json"\nrationale = "Retained historical run"\n'
            ),
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/sample.json"\nsha256 = "{sha}"\n'
            ),
            "docs/results/report.md": (
                "| Run | Episodes attempted | Rollout | Snapshot file |\n"
                "|---|---|---|---|\n"
                "| Mismatched | 4 | 1 rollout | [`sample.json`](sample.json) |\n"
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any(
        "claims 4 episode(s), but cited snapshot" in e and "records 30" in e for e in errors
    ), errors
    assert any("Remedy: update the episode count" in e for e in errors), errors


def test_negative_malformed_and_empty_snapshots(tmp_path: Path) -> None:
    """Empty files and malformed JSON snapshots fail."""
    empty_content = "   \n"
    malformed_content = "{ not valid json "
    empty_sha = hashlib.sha256(empty_content.encode("utf-8")).hexdigest()
    malformed_sha = hashlib.sha256(malformed_content.encode("utf-8")).hexdigest()

    _write_tree(
        tmp_path,
        {
            "docs/results/empty.json": empty_content,
            "docs/results/empty.json.declaration": '[declaration]\nrationale = "Empty"\n',
            "docs/results/malformed.json": malformed_content,
            "docs/results/malformed.json.declaration": '[declaration]\nrationale = "Malformed"\n',
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/empty.json"\nsha256 = "{empty_sha}"\n'
                f'[[snapshots]]\npath = "docs/results/malformed.json"\nsha256 = "{malformed_sha}"\n'
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("Snapshot docs/results/empty.json is empty" in e for e in errors), errors
    assert any("contains malformed JSON" in e for e in errors), errors
    assert all("Remedy:" in e for e in errors), errors


def test_negative_non_list_record_collection(tmp_path: Path) -> None:
    """A snapshot whose 'per_episode' is not a list fails."""
    bad_snap = {"episodes_observed": 1, "per_episode": "not-a-list"}
    snap_json = json.dumps(bad_snap)
    sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()

    _write_tree(
        tmp_path,
        {
            "docs/results/bad.json": snap_json,
            "docs/results/bad.json.declaration": '[declaration]\nrationale = "Invalid per_episode"\n',
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/bad.json"\nsha256 = "{sha}"\n'
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("non-list 'per_episode' field" in e for e in errors), errors
    assert any("Remedy: set 'per_episode' to a JSON list" in e for e in errors), errors


def test_negative_undeclared_non_evidential_snapshot(tmp_path: Path) -> None:
    """A snapshot with 0 evidential records and no declaration fails."""
    snap = _make_snapshot(10, evidential=False)
    snap_json = json.dumps(snap)
    sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()

    _write_tree(
        tmp_path,
        {
            "docs/results/sample.json": snap_json,
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/sample.json"\nsha256 = "{sha}"\n'
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("has 0 evidential records and no declaration file" in e for e in errors), errors
    assert any("Remedy: create" in e for e in errors), errors


def test_negative_index_missing_and_digest_mismatch(tmp_path: Path) -> None:
    """INDEX.toml listing missing paths and SHA-256 mismatches fail."""
    snap = _make_snapshot(10, evidential=False)
    snap_json = json.dumps(snap)
    wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    _write_tree(
        tmp_path,
        {
            "docs/results/sample.json": snap_json,
            "docs/results/sample.json.declaration": '[declaration]\nrationale = "Test"\n',
            "docs/results/INDEX.toml": (
                f'[[snapshots]]\npath = "docs/results/sample.json"\nsha256 = "{wrong_sha}"\n'
                '[[snapshots]]\npath = "docs/results/ghost.json"\nsha256 = "1234"\n'
            ),
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("Snapshot digest mismatch" in e for e in errors), errors
    assert any("missing from disk: docs/results/ghost.json" in e for e in errors), errors
    assert all("Remedy:" in e for e in errors), errors


def test_negative_unindexed_snapshot_fails(tmp_path: Path) -> None:
    """A snapshot on disk that is omitted from INDEX.toml fails."""
    snap = _make_snapshot(10, evidential=False)
    snap_json = json.dumps(snap)

    _write_tree(
        tmp_path,
        {
            "docs/results/sample.json": snap_json,
            "docs/results/sample.json.declaration": '[declaration]\nrationale = "Test"\n',
            "docs/results/INDEX.toml": "",
        },
    )

    errors = check_evidence_integrity(tmp_path)
    assert any("Unindexed snapshot on disk: docs/results/sample.json" in e for e in errors), errors
    assert any("Remedy: add docs/results/sample.json" in e for e in errors), errors
