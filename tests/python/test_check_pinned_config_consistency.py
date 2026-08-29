"""Tests for ``scripts/check_pinned_config_consistency.py``.

The script's module-level ``REPO_ROOT`` and its dependent-file path tuples
(``RUST_TOOLCHAIN_WORKFLOWS``, ``RUST_TOOLCHAIN_DOCKERFILES``,
``LMSTUDIO_WORKFLOWS``) are all resolved relative to the real FORGE repo,
so these tests never mutate real repository files (even temporarily) --
that would be a real blast-radius risk in a test that could crash mid-run.
Instead, ``monkeypatch.setattr(module, "REPO_ROOT", tmp_path)`` redirects
every file access the module makes to a throwaway directory populated
with the same *relative* paths the module's tuples expect, so the full
check functions run against fully isolated, controlled fixtures.

Covers: a clean/consistent fixture passing on all four checks (Rust
toolchain, ONNX Runtime, wasm-pack version, LM Studio endpoint), a
deliberately-introduced drift in each of the four, a missing dependent
occurrence, and the comment-blindness fix (a commented-out stale pin
must not be treated as a live occurrence -- confirmed a real
false-positive bug before fixing, not a hypothetical one; see the
module's own ``_strip_comment`` docstring).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_HELPER: Path = Path(__file__).resolve().parents[2] / "scripts" / "check_pinned_config_consistency.py"

_CONSISTENT_TOOLCHAIN = "1.94.1"
_CONSISTENT_ONNXRUNTIME = "1.23.2"
_CONSISTENT_WASM_PACK = "0.15.0"
_CONSISTENT_LMSTUDIO_URL = "http://localhost:1234/v1"
_CONSISTENT_LMSTUDIO_PORT = "1234"


@pytest.fixture(scope="module")
def helper_module() -> ModuleType:
    """Load ``check_pinned_config_consistency.py`` by file path."""
    assert _HELPER.is_file(), f"helper not found at {_HELPER}"
    spec = importlib.util.spec_from_file_location("check_pinned_config_consistency", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_pinned_config_consistency"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_consistent_repo(
    root: Path,
    *,
    toolchain: str = _CONSISTENT_TOOLCHAIN,
    onnxruntime: str = _CONSISTENT_ONNXRUNTIME,
    wasm_pack: str = _CONSISTENT_WASM_PACK,
    lmstudio_url: str = _CONSISTENT_LMSTUDIO_URL,
    lmstudio_port: str = _CONSISTENT_LMSTUDIO_PORT,
) -> None:
    """Populate `root` with a minimal, fully-consistent fixture tree
    covering every path the module's RUST_TOOLCHAIN_WORKFLOWS /
    RUST_TOOLCHAIN_DOCKERFILES / LMSTUDIO_WORKFLOWS tuples expect.
    """
    _write(root / "rust-toolchain.toml", f'[toolchain]\nchannel = "{toolchain}"\n')

    workflow_toolchain_line = f'          toolchain: "{toolchain}"\n'
    _write(
        root / ".github" / "workflows" / "ci.yml",
        "jobs:\n  fmt:\n    steps:\n      - uses: dtolnay/rust-toolchain@stable\n        with:\n"
        + workflow_toolchain_line
        + f'  onnx-features:\n    steps:\n      - run: ORT_VERSION={onnxruntime}\n'
        + f'      - run: LMSTUDIO_PORT: "{lmstudio_port}"\n'
        + f'      - run: LMSTUDIO_BASE_URL: "{lmstudio_url}"\n',
    )
    for rel in ("e2e-long.yml", "gh-pages.yml", "hf-dataset.yml", "hf-space.yml"):
        extra = f'\n          LMSTUDIO_PORT: "{lmstudio_port}"\n' if rel == "e2e-long.yml" else ""
        if rel in ("gh-pages.yml", "hf-space.yml"):
            extra += f'\n          WASM_PACK_VERSION: "{wasm_pack}"\n'
        _write(
            root / ".github" / "workflows" / rel,
            "jobs:\n  build:\n    steps:\n      - uses: dtolnay/rust-toolchain@stable\n        with:\n"
            + workflow_toolchain_line
            + extra,
        )

    for rel in ("Dockerfile", "mc-runner.Dockerfile"):
        _write(root / "docker" / rel, f"ARG RUST_IMAGE_TAG={toolchain}-bookworm\nFROM rust:${{RUST_IMAGE_TAG}}\n")
    # mc-runner.Dockerfile also carries the canonical ONNXRUNTIME_VERSION.
    _write(
        root / "docker" / "mc-runner.Dockerfile",
        f"ARG RUST_IMAGE_TAG={toolchain}-bookworm\n"
        f"FROM rust:${{RUST_IMAGE_TAG}}\n"
        f"ARG ONNXRUNTIME_VERSION={onnxruntime}\n",
    )

    _write(
        root / "python" / "forge" / "cognitive" / "providers.py",
        f'DEFAULT_LMSTUDIO_BASE_URL: str = "{lmstudio_url}"\n',
    )


def _run_main(module: ModuleType) -> int:
    return int(module.main())


class TestStripComment:
    def test_no_comment_unchanged(self, helper_module: ModuleType) -> None:
        assert helper_module._strip_comment('toolchain: "1.94.1"') == 'toolchain: "1.94.1"'

    def test_trailing_comment_stripped(self, helper_module: ModuleType) -> None:
        assert helper_module._strip_comment('toolchain: "1.94.1"  # keep in sync').strip() == 'toolchain: "1.94.1"'

    def test_whole_line_comment_becomes_empty(self, helper_module: ModuleType) -> None:
        assert helper_module._strip_comment('# toolchain: "1.60.0"').strip() == ""


class TestConsistentFixture:
    def test_all_three_checks_pass(self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _build_consistent_repo(tmp_path)
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_OK


class TestRustToolchainDrift:
    def test_workflow_toolchain_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_consistent_repo(tmp_path)
        # Drift ci.yml's own toolchain pin away from the canonical value.
        ci_yml = tmp_path / ".github" / "workflows" / "ci.yml"
        ci_yml.write_text(ci_yml.read_text().replace(_CONSISTENT_TOOLCHAIN, "1.60.0", 1), encoding="utf-8")
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_dockerfile_image_tag_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_consistent_repo(tmp_path)
        docker = tmp_path / "docker" / "Dockerfile"
        docker.write_text("ARG RUST_IMAGE_TAG=1.60.0-bookworm\nFROM rust:${RUST_IMAGE_TAG}\n", encoding="utf-8")
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_missing_toolchain_pin_in_a_tracked_workflow_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_consistent_repo(tmp_path)
        # gh-pages.yml exists but has no toolchain: line at all. Keeps its
        # WASM_PACK_VERSION line so this stays isolated to the toolchain
        # check -- gh-pages.yml is also the wasm-pack check's canonical
        # source, and dropping that line too would fail on an unrelated
        # SystemExit(EXIT_INPUT_ERROR) instead of the toolchain mismatch
        # this test exists to verify.
        _write(
            tmp_path / ".github" / "workflows" / "gh-pages.yml",
            f'jobs:\n  build:\n    steps:\n      - run: WASM_PACK_VERSION: "{_CONSISTENT_WASM_PACK}"\n',
        )
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_commented_out_stale_pin_is_ignored(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression test: a commented-out stale toolchain line used to be
        # parsed as a live occurrence (comment-blind regex), which would
        # have failed this exact fixture before the fix.
        _build_consistent_repo(tmp_path)
        ci_yml = tmp_path / ".github" / "workflows" / "ci.yml"
        ci_yml.write_text(
            '          # toolchain: "1.60.0"  (stale, commented out)\n' + ci_yml.read_text(), encoding="utf-8"
        )
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_OK


class TestOnnxRuntimeDrift:
    def test_ci_ort_version_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_consistent_repo(tmp_path)
        ci_yml = tmp_path / ".github" / "workflows" / "ci.yml"
        ci_yml.write_text(ci_yml.read_text().replace(_CONSISTENT_ONNXRUNTIME, "1.18.0", 1), encoding="utf-8")
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_trainer_dockerfile_onnxruntime_is_never_checked(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # docker/trainer.Dockerfile pins a DIFFERENT artifact (the Python
        # onnxruntime wheel) on its own independent cadence -- must not be
        # coupled to the C++ redistributable's version.
        _build_consistent_repo(tmp_path)
        _write(tmp_path / "docker" / "trainer.Dockerfile", "ARG ONNXRUNTIME_VERSION=1.20.0\n")
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_OK


class TestWasmPackVersionDrift:
    def test_hf_space_version_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # gh-pages.yml is canonical; hf-space.yml's copy drifts.
        _build_consistent_repo(tmp_path)
        hf_space = tmp_path / ".github" / "workflows" / "hf-space.yml"
        hf_space.write_text(
            hf_space.read_text().replace(_CONSISTENT_WASM_PACK, "0.13.1", 1), encoding="utf-8"
        )
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_missing_hf_space_pin_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # hf-space.yml exists but its WASM_PACK_VERSION step env is gone
        # entirely (e.g. someone re-adds the old jetli/wasm-pack-action
        # without the pinned-version env this check now expects).
        _build_consistent_repo(tmp_path)
        _write(
            tmp_path / ".github" / "workflows" / "hf-space.yml",
            "jobs:\n  build:\n    steps:\n      - uses: dtolnay/rust-toolchain@stable\n        with:\n"
            f'          toolchain: "{_CONSISTENT_TOOLCHAIN}"\n',
        )
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_gh_pages_canonical_change_propagates(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bumping the canonical source AND its one dependent together
        # must still pass -- this is the "update both" remedy the
        # mismatch message points to, exercised end-to-end.
        _build_consistent_repo(tmp_path, wasm_pack="0.16.0")
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_OK


class TestLmStudioEndpointDrift:
    def test_ci_port_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_consistent_repo(tmp_path)
        ci_yml = tmp_path / ".github" / "workflows" / "ci.yml"
        ci_yml.write_text(
            ci_yml.read_text().replace(f'"{_CONSISTENT_LMSTUDIO_PORT}"', '"9999"', 1), encoding="utf-8"
        )
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_e2e_long_port_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # e2e-long.yml's copy had no cross-reference comment at all before
        # this script existed -- the least-guarded of the three copies.
        _build_consistent_repo(tmp_path)
        e2e = tmp_path / ".github" / "workflows" / "e2e-long.yml"
        e2e.write_text(e2e.read_text().replace(f'"{_CONSISTENT_LMSTUDIO_PORT}"', '"9999"', 1), encoding="utf-8")
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH

    def test_base_url_mismatch_fails(
        self, helper_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _build_consistent_repo(tmp_path)
        ci_yml = tmp_path / ".github" / "workflows" / "ci.yml"
        ci_yml.write_text(
            ci_yml.read_text().replace(_CONSISTENT_LMSTUDIO_URL, "http://localhost:9999/v1", 1),
            encoding="utf-8",
        )
        monkeypatch.setattr(helper_module, "REPO_ROOT", tmp_path)
        assert _run_main(helper_module) == helper_module.EXIT_MISMATCH


class TestAgainstTheRealRepo:
    """A single integration-style check against the actual repo files
    (read-only -- confirms the real, currently-committed state is
    consistent, complementing the isolated fixture tests above rather
    than replacing them)."""

    def test_real_repo_is_currently_consistent(self, helper_module: ModuleType) -> None:
        assert _run_main(helper_module) == helper_module.EXIT_OK
