"""Tests for ``BCTrainer`` and ``BCDataset``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from forge.mangomas.bc_trainer import (
    DEFAULT_BC_KL_WEIGHT,
    DEFAULT_BC_SEED,
    BCDataset,
    BCTrainer,
    BCTrainerConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def toy_actor_critic_factory():
    """Factory `(state_dim, num_actions) -> _ToyActorCritic`.

    Consolidates the inline `_ToyActorCritic` class previously duplicated by
    each torch-path test. Skips immediately if torch isn't installed.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    def _factory(state_dim: int, num_actions: int):
        class _ToyActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = nn.Linear(state_dim, num_actions)
                self.critic = nn.Linear(state_dim, 1)

            def forward(self, x):
                return self.actor(x), self.critic(x)

        return _ToyActorCritic()

    return _factory


def _synthetic_episodes(
    n_episodes: int, steps_per_ep: int, state_dim: int, num_actions: int, *, seed: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    obs_episodes: list[np.ndarray] = []
    action_episodes: list[np.ndarray] = []
    # Build a separable dataset: action = argmax of obs over the first num_actions dims.
    for _ in range(n_episodes):
        obs = rng.normal(size=(steps_per_ep, state_dim)).astype(np.float32)
        actions = obs[:, :num_actions].argmax(axis=1).astype(np.int64)
        obs_episodes.append(obs)
        action_episodes.append(actions)
    return obs_episodes, action_episodes


def test_build_dataset_shapes() -> None:
    obs, actions = _synthetic_episodes(2, 5, state_dim=4, num_actions=3)
    dataset = BCTrainer.build_dataset(obs, actions, num_actions=3)
    assert dataset.observations.shape == (10, 4)
    assert dataset.teacher_action_ids.shape == (10,)
    assert dataset.teacher_top_k_probs is None


def test_build_dataset_with_top_k_normalises_rows() -> None:
    obs = [np.zeros((1, 3), dtype=np.float32)]
    actions = [np.zeros((1,), dtype=np.int64)]
    top_k = [[[{"action_id": 0, "prob": 0.5}, {"action_id": 1, "prob": 0.25}]]]
    dataset = BCTrainer.build_dataset(obs, actions, top_k_probs=top_k, num_actions=4)
    assert dataset.teacher_top_k_probs is not None
    row = dataset.teacher_top_k_probs[0]
    assert row.shape == (4,)
    assert pytest.approx(float(row.sum())) == 1.0


def test_numpy_loss_decreases_over_epochs() -> None:
    obs, actions = _synthetic_episodes(8, 20, state_dim=6, num_actions=4)
    dataset = BCTrainer.build_dataset(obs, actions, num_actions=4)
    trainer = BCTrainer(
        BCTrainerConfig(num_epochs=20, learning_rate=0.05, batch_size=16, seed=1)
    )
    result = trainer.train(dataset)
    assert result.epochs_run == 20
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.accuracy_history[-1] > result.accuracy_history[0]


def test_kl_distillation_when_top_k_probs_supplied() -> None:
    obs, actions = _synthetic_episodes(4, 5, state_dim=3, num_actions=3, seed=2)
    top_k = [
        [
            [{"action_id": int(a), "prob": 0.9}, {"action_id": (int(a) + 1) % 3, "prob": 0.1}]
            for a in ep
        ]
        for ep in actions
    ]
    dataset = BCTrainer.build_dataset(obs, actions, top_k_probs=top_k, num_actions=3)
    trainer = BCTrainer(
        BCTrainerConfig(num_epochs=5, learning_rate=0.05, kl_weight=1.0)
    )
    result = trainer.train(dataset)
    assert result.final_loss > 0.0
    assert dataset.teacher_top_k_probs is not None


def test_export_weights_npz_keys(tmp_path: Path) -> None:
    obs, actions = _synthetic_episodes(2, 4, state_dim=3, num_actions=2)
    dataset = BCTrainer.build_dataset(obs, actions, num_actions=2)
    trainer = BCTrainer(BCTrainerConfig(num_epochs=2))
    trainer.train(dataset)
    out_path = tmp_path / "bc.npz"
    trainer.export_weights(out_path)
    loaded = np.load(out_path)
    assert set(loaded.files) == {"actor_w", "actor_b"}


def test_export_weights_without_train_raises(tmp_path: Path) -> None:
    trainer = BCTrainer()
    with pytest.raises(RuntimeError, match="no exportable weights"):
        trainer.export_weights(tmp_path / "x.npz")


def test_build_dataset_empty_returns_empty() -> None:
    dataset = BCTrainer.build_dataset([], [], num_actions=3)
    assert isinstance(dataset, BCDataset)
    assert dataset.num_samples == 0


def test_train_on_empty_dataset_warns_and_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty datasets are a no-op so we don't silently report 0/0 success."""
    import logging

    dataset = BCTrainer.build_dataset([], [], num_actions=3)
    trainer = BCTrainer(BCTrainerConfig(num_epochs=10))
    caplog.set_level(logging.WARNING, logger="forge.mangomas.bc_trainer")
    result = trainer.train(dataset)
    assert result.epochs_run == 0
    assert result.final_loss == 0.0
    assert any("empty dataset" in r.message for r in caplog.records)
    # No weights are produced, so export must surface the missing-weights
    # error rather than silently writing an empty file.
    with pytest.raises(RuntimeError, match="no exportable weights"):
        trainer.export_weights("/tmp/forge-bc-empty-skip.npz")


def test_torch_path_skipped_when_torch_missing(toy_actor_critic_factory) -> None:
    """Skip when torch isn't installed; runs and updates weights when present."""
    import torch

    obs, actions = _synthetic_episodes(4, 8, state_dim=3, num_actions=2)
    dataset = BCTrainer.build_dataset(obs, actions, num_actions=2)
    net = toy_actor_critic_factory(3, 2)
    before = net.actor.weight.detach().clone()
    trainer = BCTrainer(
        BCTrainerConfig(num_epochs=3, learning_rate=0.05, seed=7)
    )
    trainer.train(dataset, actor_critic=net)
    after = net.actor.weight.detach()
    assert not torch.allclose(before, after)


def test_torch_path_uses_value_loss_when_value_hats_supplied(toy_actor_critic_factory) -> None:
    """Covers bc_trainer.py:330-341 — value-loss term only fires when both
    teacher_value_hats are present AND value_loss_weight > 0."""
    import torch

    obs, actions = _synthetic_episodes(4, 8, state_dim=3, num_actions=2)
    teacher_values: list[list[float]] = [
        [0.0] * int(arr.shape[0]) for arr in actions
    ]
    dataset = BCTrainer.build_dataset(
        obs, actions, num_actions=2, value_hats=teacher_values
    )
    assert dataset.teacher_value_hats is not None
    net = toy_actor_critic_factory(3, 2)
    critic_before = net.critic.weight.detach().clone()
    trainer = BCTrainer(
        BCTrainerConfig(
            num_epochs=3,
            learning_rate=0.05,
            value_loss_weight=0.5,
            seed=11,
        )
    )
    trainer.train(dataset, actor_critic=net)
    critic_after = net.critic.weight.detach()
    # value_loss_weight > 0 must produce gradient flow into the critic head.
    assert not torch.allclose(critic_before, critic_after)


@pytest.mark.parametrize(
    "kl_weight,expect_kl_in_loss",
    [(DEFAULT_BC_KL_WEIGHT, True), (0.0, False)],
    ids=["kl_active", "kl_disabled"],
)
def test_torch_path_kl_only_branch(
    toy_actor_critic_factory, kl_weight: float, expect_kl_in_loss: bool
) -> None:
    """Covers bc_trainer.py:336-339 — KL branch fires when teacher_top_k_probs
    is supplied AND teacher_value_hats is None AND kl_weight > 0.

    The KL term contributes positively to the loss, so the active run's final
    loss must exceed the CE-only baseline (same initial weights, same seed)
    by at least a small float-noise tolerance.
    """
    import torch

    rng = np.random.default_rng(DEFAULT_BC_SEED)
    n, state_dim, num_actions = 32, 8, 4
    x = rng.standard_normal((n, state_dim)).astype(np.float32)
    y = rng.integers(0, num_actions, size=n).astype(np.int64)
    topk = rng.dirichlet(np.ones(num_actions), size=n).astype(np.float32)

    dataset = BCDataset(
        observations=x,
        teacher_action_ids=y,
        teacher_top_k_probs=topk,
        teacher_value_hats=None,  # forces KL-only branch
    )
    cfg = BCTrainerConfig(
        num_epochs=3,
        kl_weight=kl_weight,
        value_loss_weight=0.0,
        seed=DEFAULT_BC_SEED,
    )
    trainer = BCTrainer(cfg)
    net = toy_actor_critic_factory(state_dim, num_actions)

    # Snapshot PRE-training weights so the CE-only baseline starts from the
    # same initialisation. Without this, net2.load_state_dict(net.state_dict())
    # after `trainer.train(...)` would copy POST-training weights into net2.
    initial_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    weight_before = net.actor.weight.detach().clone()

    result = trainer.train(dataset, actor_critic=net)

    assert result.epochs_run == 3
    assert not torch.equal(weight_before, net.actor.weight.detach()), (
        "weights must update under torch path"
    )

    if expect_kl_in_loss:
        ce_only_trainer = BCTrainer(
            BCTrainerConfig(
                num_epochs=3,
                kl_weight=0.0,
                value_loss_weight=0.0,
                seed=DEFAULT_BC_SEED,
            )
        )
        net2 = toy_actor_critic_factory(state_dim, num_actions)
        net2.load_state_dict(initial_state)
        ce_only_dataset = BCDataset(
            observations=x,
            teacher_action_ids=y,
            teacher_top_k_probs=None,
            teacher_value_hats=None,
        )
        ce_only = ce_only_trainer.train(ce_only_dataset, actor_critic=net2)
        assert result.final_loss > ce_only.final_loss + 1e-6, (
            f"KL term did not contribute: kl_active_loss={result.final_loss:.6f} "
            f"ce_only_loss={ce_only.final_loss:.6f}"
        )


def test_resolve_num_actions_falls_back_when_topk_has_zero_columns() -> None:
    """Covers bc_trainer.py:_resolve_num_actions — topk array exists but has
    shape (N, 0) (degenerate teacher), so we must fall back to
    teacher_action_ids.max()+1 rather than returning 0."""
    obs = [np.zeros((4, 3), dtype=np.float32)]
    actions = [np.array([0, 1, 2, 1], dtype=np.int64)]
    # num_actions=3 is what build_dataset bakes into the topk matrix; we
    # then mutate that matrix to shape (N, 0) below to simulate a degenerate
    # teacher that produced no top-k probabilities at all.
    dataset = BCTrainer.build_dataset(obs, actions, num_actions=3)
    dataset.teacher_top_k_probs = np.zeros((dataset.num_samples, 0), dtype=np.float32)
    trainer = BCTrainer(BCTrainerConfig(num_actions=0))
    resolved = trainer._resolve_num_actions(dataset)
    assert resolved == 3  # max action_id (2) + 1
