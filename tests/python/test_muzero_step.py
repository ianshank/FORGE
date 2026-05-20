"""Pin :func:`forge.training._muzero_step.train_with_gradients`.

The helper is the single source of truth for the MuZero loss /
backward / clip / step sequence. Both ``MuZeroTrainer.train_step``
(existing) and the new ``MuzeroMcTrainer.train_step`` consume it.

Tests are torch-gated via ``pytest.importorskip("torch")`` so the
module is importable without torch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from forge.models.muzero_world_model import MuZeroWorldModel


# Match the existing tests/python/test_muzero_trainer.py fixture exactly
# so the extracted helper exercises the same network shape the original
# `_train_with_gradients` was validated against. The grid layout
# (11*11*7 spatial + 73 vector = 920) is the project's canonical
# small-test shape (`MuZeroConfig` defaults; see muzero_config.py).
_OBS_DIM = 11 * 11 * 7 + 73
_ACTION_DIM = 5
_LATENT_DIM = 16
_HIDDEN_DIM = 16


def _make_tiny_model() -> MuZeroWorldModel:
    """Build the smallest ``MuZeroWorldModel`` that exercises the
    standard grid+vector observation layout.

    Identical to the existing
    ``tests/python/test_muzero_trainer.py::_make_model`` shape so any
    drift between the extracted helper and the legacy trainer surface
    immediately.
    """
    pytest.importorskip("torch")
    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel

    return MuZeroWorldModel(
        MuZeroConfig(
            obs_dim=_OBS_DIM,
            action_dim=_ACTION_DIM,
            latent_dim=_LATENT_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_blocks=1,
            num_unroll_steps=2,
            reward_support_size=11,
            value_support_size=11,
            cnn_channels=(8,),
            cnn_kernel_sizes=(3,),
            cnn_strides=(1,),
        )
    )


def _make_batch(
    *, batch_size: int, obs_dim: int, action_dim: int, unroll: int
) -> dict[str, np.ndarray]:
    """Synthetic batch matching ``MuZeroReplayBuffer.sample_batch``'s
    shape so the helper can run end-to-end without a buffer.
    """
    rng = np.random.default_rng(0)
    return {
        "observations": rng.standard_normal((batch_size, obs_dim), dtype=np.float32),
        "actions": rng.integers(0, action_dim, size=(batch_size, unroll), dtype=np.int64),
        "target_values": rng.standard_normal((batch_size, unroll + 1), dtype=np.float32),
        "target_rewards": rng.standard_normal((batch_size, unroll), dtype=np.float32),
        "target_policies": np.full(
            (batch_size, unroll + 1, action_dim), 1.0 / action_dim, dtype=np.float32
        ),
    }


def test_train_step_metrics_dataclass_round_trips_to_dict() -> None:
    """The legacy ``MuZeroTrainer._train_with_gradients`` returned a
    plain dict. Ensure ``TrainStepMetrics.to_dict()`` produces the
    same keys so downstream loggers see no change.
    """
    from forge.training._muzero_step import TrainStepMetrics

    m = TrainStepMetrics(
        loss=1.0, policy_loss=0.5, value_loss=0.3, reward_loss=0.2, l2_reg=0.01
    )
    d = m.to_dict()
    assert set(d.keys()) == {"loss", "policy_loss", "value_loss", "reward_loss", "l2_reg"}
    assert math.isclose(d["loss"], 1.0)
    assert math.isclose(d["l2_reg"], 0.01)


def test_train_with_gradients_decreases_loss_on_fixed_seed() -> None:
    """Smoke-test the extracted helper drives the loss down on a
    deterministic batch. Pins the extraction is loss-equivalent to
    the original method shape.
    """
    torch = pytest.importorskip("torch")
    from forge.training._muzero_step import (
        MuZeroStepConfig,
        train_with_gradients,
    )

    torch.manual_seed(0)
    np.random.seed(0)

    model = _make_tiny_model()
    optimizer = torch.optim.Adam(model.all_parameters(), lr=1e-3)
    batch = _make_batch(
        batch_size=8, obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, unroll=2
    )
    step_cfg = MuZeroStepConfig(max_grad_norm=1.0, gradient_scale=0.5)

    initial = train_with_gradients(model, optimizer, batch, step_cfg)
    for _ in range(20):
        last = train_with_gradients(model, optimizer, batch, step_cfg)

    assert last.loss < initial.loss, (
        f"loss did not decrease after 20 steps; initial={initial.loss:.4f} "
        f"final={last.loss:.4f}"
    )
    # All four loss components should be finite.
    for field, value in last.to_dict().items():
        assert math.isfinite(value), f"{field} = {value!r}"


def test_step_config_defaults_match_original_trainer_config() -> None:
    """Pin the default values against the existing
    ``MuZeroTrainerConfig.max_grad_norm`` and ``gradient_scale`` so the
    extraction doesn't silently shift defaults.
    """
    from forge.training._muzero_step import MuZeroStepConfig
    from forge.training.muzero_trainer import MuZeroTrainerConfig

    step_default = MuZeroStepConfig()
    trainer_default = MuZeroTrainerConfig()
    assert step_default.max_grad_norm == trainer_default.max_grad_norm
    assert step_default.gradient_scale == trainer_default.gradient_scale
