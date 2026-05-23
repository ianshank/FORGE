"""Tests for `forge.training.muzero_mc.bootstrap` + the v0.5 shape contract.

These tests are deliberately torch-free so they run in the
`python-test-no-native` CI job. Anything that touches the actual ONNX
export path (which requires torch) lives in
`test_muzero_mc_cli.py::test_bootstrap_rejects_invalid_obs_dim` and is
covered by the `python-test-maturin` job instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator  # noqa: TC003 — runtime use in pytest fixture yield
from typing import Any

import pytest

from forge.models.muzero_config import MuZeroConfig


def test_bootstrap_default_obs_dim_matches_muzero_shape() -> None:
    """The bot's default `Hello` `obs_dim` must equal `MuZeroConfig()`'s
    derived `obs_dim` — otherwise the runner's handshake cross-check
    rejects the first real episode.

    With the shipped `configs/minecraft/env.toml` block-grid defaults
    (`grid_radius=5, grid_height_radius=0, grid_channels=7,
    flat_vector_dim=73`), the bot emits `11*11*1*7 + 73 = 920` floats.
    `MuZeroConfig()` with its `DEFAULT_GRID_*` + `DEFAULT_VECTOR_DIM`
    constants derives the same total via `grid_flat_dim + vector_dim`.
    """
    cfg = MuZeroConfig()
    # MuZeroConfig.__post_init__ auto-computes `obs_dim` from
    # `grid_flat_dim + vector_dim` when `obs_dim==0` (the default).
    assert cfg.obs_dim == cfg.grid_flat_dim + cfg.vector_dim
    assert cfg.obs_dim == 11 * 11 * 7 + 73 == 920


def test_muzero_config_legacy_31_float_obs_validates() -> None:
    """Backwards-compat pin: setting `include_block_grid=false` on the
    bot side downgrades the obs to the legacy 31-float surface, and the
    matching `MuZeroConfig(obs_dim=31, grid_height=0, grid_width=0,
    vector_dim=31)` must construct cleanly (no `__post_init__` warning).

    The bot's `flat_vector_dim` knob can also be unset in env.toml when
    grid is off, in which case the raw 31-float surface is what's
    emitted — this test pins that the trainer-side config accepts it.
    """
    cfg = MuZeroConfig(
        obs_dim=31,
        action_dim=12,
        grid_height=0,
        grid_width=0,
        vector_dim=31,
    )
    assert cfg.obs_dim == 31
    assert cfg.grid_flat_dim == 0
    assert cfg.vector_dim == 31


def test_muzero_config_warns_on_obs_dim_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`MuZeroConfig` warns (does NOT error) when `obs_dim` diverges
    from `grid_flat_dim + vector_dim`. The downstream handshake gate
    in the Rust runner is what hard-fails — keeping this a warning
    here lets unit tests construct asymmetric configs without
    boilerplate, while production startup still catches the regression
    via the WS handshake.
    """
    caplog.set_level(logging.WARNING, logger="forge.models.muzero_config")
    cfg = MuZeroConfig(
        obs_dim=999,
        action_dim=12,
    )
    assert cfg.obs_dim == 999  # respects caller's value rather than auto-deriving
    warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("obs_dim=999" in r.getMessage() for r in warning_lines), (
        f"expected a WARN about obs_dim drift, got: {[r.getMessage() for r in warning_lines]}"
    )


@pytest.fixture
def stub_torch_and_world_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The bootstrap module lazy-imports torch + MuZeroWorldModel
    inside `bootstrap()`. Stub both so a torch-free CI job can still
    exercise the logging + config-validation path.

    The stub world-model deliberately omits any `eval`-style hook;
    bootstrap calls `getattr(model, "eval", None)` and short-circuits
    when absent, so the stub doesn't need to define it.
    """
    import sys
    import types

    fake_torch = types.ModuleType("torch")

    def manual_seed(_seed: int) -> None:
        return None

    fake_torch.manual_seed = manual_seed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_world_model = types.ModuleType("forge.models.muzero_world_model")

    class _StubModel:
        def __init__(self, _cfg: Any) -> None:
            return None

    fake_world_model.MuZeroWorldModel = _StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "forge.models.muzero_world_model", fake_world_model)
    yield


def test_bootstrap_logs_shape_summary_at_entry(
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    stub_torch_and_world_model: None,
) -> None:
    """The bootstrap entry-point logs the resolved (obs_dim, grid,
    vector_dim, schema_id) tuple at INFO so a first-real-run operator
    can diagnose handshake mismatches without grepping multiple
    container logs. Pin the log shape against regression.
    """
    import sys
    import types
    from pathlib import Path

    fake_exporter_module = types.ModuleType("forge.models.muzero_export")

    class _StubExporter:
        def __init__(self, _model: Any) -> None:
            return None

        def export_onnx(self, out_dir: Path, opset_version: int) -> list[Path]:
            _ = opset_version
            paths: list[Path] = []
            for name in ("representation.onnx", "dynamics.onnx", "prediction.onnx"):
                p = Path(out_dir) / name
                p.write_bytes(b"stub")
                paths.append(p)
            return paths

    fake_exporter_module.MuZeroExporter = _StubExporter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "forge.models.muzero_export", fake_exporter_module)

    from forge.training.muzero_mc.bootstrap import BootstrapConfig, bootstrap

    caplog.set_level(logging.INFO, logger="forge.training.muzero_mc.bootstrap")
    cfg = BootstrapConfig(
        obs_dim=920,
        action_dim=12,
        schema_id="a" * 64,
        output_dir=tmp_path,
    )
    bootstrap(cfg)

    entry_messages = [r.getMessage() for r in caplog.records if "bootstrap start" in r.getMessage()]
    assert entry_messages, (
        "expected an INFO line tagged 'bootstrap start' from "
        f"forge.training.muzero_mc.bootstrap; got {[r.getMessage() for r in caplog.records]}"
    )
    msg = entry_messages[0]
    assert "obs_dim=920" in msg
    assert "action_dim=12" in msg
    assert "grid=" in msg
    assert "vector_dim=" in msg


def test_bootstrap_from_hf(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test bootstrapping from a HuggingFace checkpoint.

    This mocks hf_hub_download to simulate pulling ONNX files
    and verifies that a valid manifest and versioned subdir are created.
    """
    import sys
    import types
    from pathlib import Path

    # Mock huggingface_hub
    fake_hf = types.ModuleType("huggingface_hub")
    download_calls = []

    def mock_hf_hub_download(
        repo_id: str,
        filename: str,
        subfolder: str | None = None,
        local_dir: str | Path | None = None,
        local_dir_use_symlinks: bool = False,
    ) -> str:
        download_calls.append({
            "repo_id": repo_id,
            "filename": filename,
            "subfolder": subfolder,
            "local_dir": Path(local_dir) if local_dir else None,
        })
        # Simulate writing the downloaded file
        assert local_dir is not None
        file_path = Path(local_dir) / filename
        file_path.write_bytes(b"mocked_onnx_content")
        return str(file_path)

    fake_hf.hf_hub_download = mock_hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    from forge.training.muzero_mc.bootstrap import BootstrapConfig, bootstrap

    cfg = BootstrapConfig(
        obs_dim=920,
        action_dim=12,
        schema_id="mock_schema_id_hash",
        output_dir=tmp_path,
        from_hf="mock-user/mock-repo",
        subfolder="models/v1",
    )

    result = bootstrap(cfg)

    # Verify download calls
    assert len(download_calls) == 3
    filenames = {call["filename"] for call in download_calls}
    assert filenames == {"representation.onnx", "dynamics.onnx", "prediction.onnx"}
    for call in download_calls:
        assert call["repo_id"] == "mock-user/mock-repo"
        assert call["subfolder"] == "models/v1"
        assert call["local_dir"] == tmp_path / "v00000001"

    # Verify output structure
    versioned_dir = tmp_path / "v00000001"
    assert versioned_dir.exists()
    assert (versioned_dir / "representation.onnx").read_bytes() == b"mocked_onnx_content"

    manifest_path = tmp_path / "model_manifest.json"
    assert manifest_path.exists()

    from forge.training.muzero_mc.manifest import load_manifest
    manifest = load_manifest(manifest_path)
    assert manifest.version == 1
    assert manifest.schema_id == "mock_schema_id_hash"
    assert manifest.files.representation.path == "v00000001/representation.onnx"
    assert manifest.files.dynamics.path == "v00000001/dynamics.onnx"
    assert manifest.files.prediction.path == "v00000001/prediction.onnx"

    assert result.manifest_path == manifest_path
    assert len(result.onnx_paths) == 3
    assert result.onnx_paths["representation"] == versioned_dir / "representation.onnx"

