"""Tests for forge_env.feature_extractors — ForgeGridCnnExtractor, ForgeObsExtractor."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if torch / SB3 are not installed
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="PyTorch required for feature extractor tests")
pytest.importorskip("stable_baselines3", reason="SB3 required for feature extractor tests")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from torch import nn  # noqa: E402

from forge_env.feature_extractors import (  # noqa: E402
    ForgeGridCnnExtractor,
    ForgeObsExtractor,
    _build_cnn,
    _cnn_output_dim,
    _scalar_obs_dim,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal Dict observation spaces
# ---------------------------------------------------------------------------

_GRID_H, _GRID_W, _GRID_C = 11, 11, 7
_INV_CAPACITY = 10


def _make_dict_space(
    include_grid: bool = True,
    include_scalars: bool = True,
    include_messages: bool = False,
) -> gym.spaces.Dict:
    """Build a gymnasium Dict space matching the FORGE observation layout."""
    spaces: dict[str, gym.spaces.Space] = {}
    if include_grid:
        spaces["grid_view"] = gym.spaces.Box(
            low=0, high=255, shape=(_GRID_H, _GRID_W, _GRID_C), dtype=np.uint8
        )
    if include_scalars:
        spaces["health"] = gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        spaces["stamina"] = gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        spaces["position"] = gym.spaces.Box(low=0, high=65535, shape=(2,), dtype=np.uint16)
        spaces["inventory"] = gym.spaces.Box(
            low=0, high=65535, shape=(_INV_CAPACITY, 2), dtype=np.uint16
        )
        spaces["day_phase"] = gym.spaces.Discrete(4)
    if include_messages:
        spaces["messages"] = gym.spaces.Box(low=0, high=65535, shape=(0,), dtype=np.uint16)
    return gym.spaces.Dict(spaces)


def _make_obs_batch(
    obs_space: gym.spaces.Dict,
    batch_size: int = 2,
) -> dict[str, torch.Tensor]:
    """Sample a batch of observations and convert to tensors."""
    obs: dict[str, torch.Tensor] = {}
    for key, space in obs_space.spaces.items():
        if isinstance(space, gym.spaces.Discrete):
            arr = np.zeros((batch_size,), dtype=np.int64)
        else:
            arr = np.zeros((batch_size, *space.shape), dtype=np.float32)
        obs[key] = torch.tensor(arr)
    return obs


# ---------------------------------------------------------------------------
# _build_cnn
# ---------------------------------------------------------------------------


class TestBuildCnn:
    """Tests for the _build_cnn helper."""

    def test_returns_sequential(self) -> None:
        cnn = _build_cnn(7, (32, 64), (3, 3), (1, 1))
        assert isinstance(cnn, nn.Sequential)

    def test_output_contains_flatten(self) -> None:
        cnn = _build_cnn(7, (32,), (3,), (1,))
        assert any(isinstance(m, nn.Flatten) for m in cnn)

    def test_single_layer(self) -> None:
        cnn = _build_cnn(4, (16,), (3,), (1,))
        # Conv2d + ReLU + Flatten = 3 modules
        assert len(list(cnn.children())) == 3

    def test_two_layers(self) -> None:
        cnn = _build_cnn(7, (32, 64), (3, 3), (1, 1))
        # 2 * (Conv2d + ReLU) + Flatten = 5 modules
        assert len(list(cnn.children())) == 5


# ---------------------------------------------------------------------------
# _cnn_output_dim
# ---------------------------------------------------------------------------


class TestCnnOutputDim:
    """Tests for _cnn_output_dim."""

    def test_computes_without_error(self) -> None:
        cnn = _build_cnn(7, (32,), (3,), (1,))
        dim = _cnn_output_dim(cnn, _GRID_H, _GRID_W, _GRID_C, torch.device("cpu"))
        assert isinstance(dim, int)
        assert dim > 0

    def test_different_channels_differ(self) -> None:
        cnn1 = _build_cnn(7, (16,), (3,), (1,))
        cnn2 = _build_cnn(7, (64,), (3,), (1,))
        dim1 = _cnn_output_dim(cnn1, _GRID_H, _GRID_W, _GRID_C, torch.device("cpu"))
        dim2 = _cnn_output_dim(cnn2, _GRID_H, _GRID_W, _GRID_C, torch.device("cpu"))
        assert dim1 != dim2


# ---------------------------------------------------------------------------
# _scalar_obs_dim
# ---------------------------------------------------------------------------


class TestScalarObsDim:
    """Tests for _scalar_obs_dim."""

    def test_no_scalars_returns_zero(self) -> None:
        space = _make_dict_space(include_grid=True, include_scalars=False)
        assert _scalar_obs_dim(space) == 0

    def test_grid_excluded_from_scalar_dim(self) -> None:
        space_with_grid = _make_dict_space(include_grid=True, include_scalars=True)
        space_without_grid = _make_dict_space(include_grid=False, include_scalars=True)
        dim_with = _scalar_obs_dim(space_with_grid)
        dim_without = _scalar_obs_dim(space_without_grid)
        assert dim_with == dim_without  # grid_view is excluded in both cases

    def test_messages_excluded(self) -> None:
        space_with = _make_dict_space(include_scalars=True, include_messages=True)
        space_without = _make_dict_space(include_scalars=True, include_messages=False)
        assert _scalar_obs_dim(space_with) == _scalar_obs_dim(space_without)

    def test_varied_shapes_are_counted_correctly(self) -> None:
        space = gym.spaces.Dict(
            {
                "scalar_1d": gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
                "scalar_0d": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
                "scalar_2d": gym.spaces.Box(low=0, high=1, shape=(3, 4), dtype=np.float32),
            }
        )
        assert _scalar_obs_dim(space) == 18


# ---------------------------------------------------------------------------
# ForgeGridCnnExtractor
# ---------------------------------------------------------------------------


class TestForgeGridCnnExtractor:
    """Tests for ForgeGridCnnExtractor."""

    def _make(
        self,
        features_dim: int = 64,
        cnn_channels: tuple[int, ...] = (16,),
        cnn_kernel_sizes: tuple[int, ...] = (3,),
        cnn_strides: tuple[int, ...] = (1,),
    ) -> ForgeGridCnnExtractor:
        space = _make_dict_space()
        return ForgeGridCnnExtractor(
            space,
            features_dim=features_dim,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
        )

    def test_features_dim_attribute(self) -> None:
        extractor = self._make(features_dim=128)
        assert extractor.features_dim == 128

    def test_output_shape(self) -> None:
        extractor = self._make(features_dim=64)
        obs_space = _make_dict_space()
        obs = _make_obs_batch(obs_space, batch_size=4)
        out = extractor(obs)
        assert out.shape == (4, 64)

    def test_different_features_dims(self) -> None:
        for dim in (32, 64, 256):
            extractor = self._make(features_dim=dim)
            obs = _make_obs_batch(_make_dict_space(), batch_size=2)
            assert extractor(obs).shape == (2, dim)

    def test_configurable_channels(self) -> None:
        """Different cnn_channels produce different parameter counts."""
        e1 = self._make(cnn_channels=(8,))
        e2 = self._make(cnn_channels=(64, 64))
        params1 = sum(p.numel() for p in e1.parameters())
        params2 = sum(p.numel() for p in e2.parameters())
        assert params1 != params2

    def test_missing_grid_view_raises(self) -> None:
        space = _make_dict_space(include_grid=False)
        with pytest.raises(KeyError, match="grid_view"):
            ForgeGridCnnExtractor(space)

    def test_no_hard_coded_channel_count(self) -> None:
        """Extractor with 3 CNN layers should have more params than one with 1."""
        e_shallow = self._make(cnn_channels=(16,))
        e_deep = self._make(cnn_channels=(16, 32, 64))
        params_shallow = sum(p.numel() for p in e_shallow.parameters())
        params_deep = sum(p.numel() for p in e_deep.parameters())
        assert params_deep > params_shallow


# ---------------------------------------------------------------------------
# ForgeObsExtractor
# ---------------------------------------------------------------------------


class TestForgeObsExtractor:
    """Tests for ForgeObsExtractor."""

    def _make(
        self,
        cnn_out_dim: int = 64,
        cnn_channels: tuple[int, ...] = (16,),
        mlp_hidden_sizes: tuple[int, ...] = (64,),
    ) -> ForgeObsExtractor:
        space = _make_dict_space()
        return ForgeObsExtractor(
            space,
            cnn_out_dim=cnn_out_dim,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=mlp_hidden_sizes,
        )

    def test_features_dim_is_sum_of_branches(self) -> None:
        cnn_out = 64
        mlp_out = 32
        extractor = ForgeObsExtractor(
            _make_dict_space(),
            cnn_out_dim=cnn_out,
            cnn_channels=(16,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(mlp_out,),
        )
        assert extractor.features_dim == cnn_out + mlp_out

    def test_output_shape(self) -> None:
        extractor = self._make(cnn_out_dim=32, mlp_hidden_sizes=(16,))
        obs = _make_obs_batch(_make_dict_space(), batch_size=3)
        out = extractor(obs)
        assert out.shape == (3, 32 + 16)

    def test_no_grid_key_still_works(self) -> None:
        """Extractor should work when there is no grid_view in the space."""
        space = _make_dict_space(include_grid=False, include_scalars=True)
        extractor = ForgeObsExtractor(
            space,
            cnn_out_dim=0,
            cnn_channels=(16,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(32,),
        )
        obs = _make_obs_batch(space, batch_size=2)
        out = extractor(obs)
        assert out.shape[0] == 2  # batch dim preserved

    def test_messages_key_skipped(self) -> None:
        """Messages key should not appear in the scalar branch."""
        space_with_msg = _make_dict_space(include_messages=True)
        space_without_msg = _make_dict_space(include_messages=False)
        extractor_with = ForgeObsExtractor(
            space_with_msg,
            cnn_out_dim=32,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(16,),
        )
        extractor_without = ForgeObsExtractor(
            space_without_msg,
            cnn_out_dim=32,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(16,),
        )
        assert extractor_with.features_dim == extractor_without.features_dim

    def test_dynamic_features_dim_no_hard_coding(self) -> None:
        """features_dim should change when mlp_hidden_sizes changes."""
        e1 = ForgeObsExtractor(
            _make_dict_space(),
            cnn_out_dim=32,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(16,),
        )
        e2 = ForgeObsExtractor(
            _make_dict_space(),
            cnn_out_dim=32,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(128,),
        )
        assert e1.features_dim != e2.features_dim

    def test_missing_scalar_key_is_zero_filled(self) -> None:
        extractor = self._make(cnn_out_dim=32, mlp_hidden_sizes=(16,))
        obs = _make_obs_batch(_make_dict_space(), batch_size=2)
        del obs["inventory"]
        out = extractor(obs)
        assert out.shape == (2, 32 + 16)

    def test_no_parts_returns_zero_tensor(self) -> None:
        space = _make_dict_space(include_grid=False, include_scalars=False, include_messages=True)
        extractor = ForgeObsExtractor(
            space,
            cnn_out_dim=0,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
            mlp_hidden_sizes=(),
        )
        obs = _make_obs_batch(space, batch_size=3)
        out = extractor(obs)
        assert out.shape == (3, 0)

    def test_grid_extractor_preserves_batch_size_for_multiple_batches(self) -> None:
        extractor = ForgeGridCnnExtractor(_make_dict_space(), features_dim=32)
        for batch_size in (1, 4, 8):
            obs = _make_obs_batch(_make_dict_space(), batch_size=batch_size)
            out = extractor(obs)
            assert out.shape == (batch_size, 32)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


def test_import_error_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """_require_torch_sb3 raises ImportError when torch is absent."""
    import forge_env.feature_extractors as fe_mod  # noqa: PLC0415

    monkeypatch.setattr(fe_mod, "HAS_TORCH", False)
    with pytest.raises(ImportError, match="PyTorch"):
        fe_mod._require_torch_sb3()


def test_import_error_without_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    """_require_torch_sb3 raises ImportError when SB3 is absent."""
    import forge_env.feature_extractors as fe_mod  # noqa: PLC0415

    monkeypatch.setattr(fe_mod, "HAS_SB3", False)
    with pytest.raises(ImportError, match="Stable Baselines"):
        fe_mod._require_torch_sb3()
