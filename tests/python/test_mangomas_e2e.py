"""End-to-end smoke test for the MangoMAS training pipeline.

Wires together all MangoMAS bridge components (config, adapters,
BDI pre-trainer, constitutional pre-trainer, RSSM pre-trainer,
curiosity optimizer, curriculum controller, sweep runner) in a
minimal 3-episode loop and validates the full stage chain
produces correct, deterministic outputs without the native Rust
extension.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from forge.mangomas.adapters import ActionSpaceAdapter, ObservationAdapter
from forge.mangomas.bdi_trainer import BDIDataset, BDIPreTrainer
from forge.mangomas.config import (
    BDITrainerConfig,
    ConstitutionalTrainerConfig,
    CuriosityOptimizerConfig,
    CurriculumConfig,
    MangoMASBridgeConfig,
    RSSMPreTrainConfig,
    SweepConfig,
)
from forge.mangomas.constitutional_trainer import (
    ConstitutionalDataset,
    ConstitutionalPreTrainer,
)
from forge.mangomas.curiosity_optimizer import CuriosityWeightOptimizer, CuriosityWeights
from forge.mangomas.curriculum_controller import PlatformCurriculumController, TierStatus
from forge.mangomas.rssm_pretrainer import RSSMPreTrainer
from forge.mangomas.sweep_runner import MCTSSweepRunner, SweepReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_obs(
    *,
    rng: np.random.Generator,
    step: int,
    include_drone: bool = True,
) -> dict[str, Any]:
    """Build a mock FORGE observation dict suitable for the adapters."""
    return {
        "grid_view": rng.random((5, 5, 11), dtype=np.float32).tolist(),
        "health": float(rng.uniform(0.5, 1.0)),
        "stamina": float(rng.uniform(0.5, 1.0)),
        "position": [float(rng.uniform(0, 16)), float(rng.uniform(0, 16))],
        "day_phase": float(step % 4),
        "inventory": {"wood": int(rng.integers(0, 5)), "stone": int(rng.integers(0, 3))},
        "altitude": float(rng.uniform(0.0, 0.8)) if include_drone else 0.0,
        "battery": float(rng.uniform(0.3, 1.0)),
        "morphology": 2.0 if include_drone else 0.0,
        "heading": float(rng.uniform(0.0, 6.28)),
    }


def _raw_obs_from_mock(obs: dict[str, Any]) -> dict[str, float]:
    """Extract the constitutional constraint fields from a mock observation."""
    return {
        "battery": float(obs.get("battery", 1.0)),
        "altitude": float(obs.get("altitude", 0.0)),
        "stamina_inverse": 1.0 - float(obs.get("stamina", 1.0)),
        "boundary_distance": min(
            float(obs["position"][0]),
            float(obs["position"][1]),
            16.0 - float(obs["position"][0]),
            16.0 - float(obs["position"][1]),
        )
        / 8.0,
        "threat_proximity": 0.5,
    }


ACTION_NAMES = [
    "Move", "PickUp", "Craft", "Push", "Communicate",
    "Scan", "Noop", "MoveUp", "MoveDown", "Drop",
]


def _collect_mock_episodes(
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    platform: str = "drone",
) -> dict[str, Any]:
    """Simulate collecting episode data without the native FORGE env.

    Returns a dict with the same structure that the pipeline stages consume.
    """
    rng = np.random.default_rng(seed)
    obs_adapter = ObservationAdapter(platform=platform)

    all_observations: list[np.ndarray] = []
    all_action_names: list[list[str]] = []
    all_action_ids: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_dones: list[np.ndarray] = []
    all_raw_observations: list[list[dict[str, float]]] = []

    for _ep in range(num_episodes):
        ep_obs_vecs: list[np.ndarray] = []
        ep_action_names: list[str] = []
        ep_action_ids: list[int] = []
        ep_rewards: list[float] = []
        ep_dones: list[float] = []
        ep_raw_obs: list[dict[str, float]] = []

        for step in range(steps_per_episode + 1):
            mock_obs = _make_mock_obs(rng=rng, step=step, include_drone=(platform == "drone"))
            ep_obs_vecs.append(obs_adapter.adapt(mock_obs))

            if step < steps_per_episode:
                action_idx = int(rng.integers(0, len(ACTION_NAMES)))
                ep_action_names.append(ACTION_NAMES[action_idx])
                ep_action_ids.append(int(rng.integers(0, 75)))
                ep_rewards.append(float(rng.uniform(-1.0, 1.0)))
                ep_dones.append(1.0 if step == steps_per_episode - 1 else 0.0)
                ep_raw_obs.append(_raw_obs_from_mock(mock_obs))

        all_observations.append(np.array(ep_obs_vecs, dtype=np.float32))
        all_action_names.append(ep_action_names)
        all_action_ids.append(np.array(ep_action_ids, dtype=np.int64))
        all_rewards.append(np.array(ep_rewards, dtype=np.float32))
        all_dones.append(np.array(ep_dones, dtype=np.float32))
        all_raw_observations.append(ep_raw_obs)

    return {
        "observations": all_observations,
        "action_names": all_action_names,
        "action_ids": all_action_ids,
        "rewards": all_rewards,
        "dones": all_dones,
        "raw_observations": all_raw_observations,
    }


@pytest.fixture()
def bridge_config() -> MangoMASBridgeConfig:
    """Minimal MangoMASBridgeConfig with fast training settings."""
    return MangoMASBridgeConfig(
        platform="drone",
        bdi_trainer=BDITrainerConfig(num_epochs=3, batch_size=16, log_interval=1),
        constitutional_trainer=ConstitutionalTrainerConfig(
            num_epochs=3, batch_size=16, log_interval=1, seed=42
        ),
        rssm_pretrain=RSSMPreTrainConfig(
            num_epochs=3, batch_size=8, sequence_length=5, log_interval=1
        ),
        curiosity_optimizer=CuriosityOptimizerConfig(
            population_size=4, seed=42, log_interval=1
        ),
        sweep=SweepConfig(
            c_puct_steps=2,
            sim_budget_steps=2,
            depth_steps=2,
            discount_steps=2,
            episodes_per_config=2,
            seed=42,
        ),
        curriculum=CurriculumConfig(seed=42, warmup_episodes=5),
    )


@pytest.fixture()
def mock_episode_data() -> dict[str, Any]:
    """Three mock episodes of 10 steps each."""
    return _collect_mock_episodes(
        num_episodes=3, steps_per_episode=10, seed=42, platform="drone"
    )


# ---------------------------------------------------------------------------
# E2E smoke test
# ---------------------------------------------------------------------------


class TestMangoMASEndToEnd:
    """End-to-end smoke test wiring all MangoMAS stages together."""

    def test_full_stage_chain(
        self,
        bridge_config: MangoMASBridgeConfig,
        mock_episode_data: dict[str, Any],
        tmp_path: Any,
    ) -> None:
        """Run BDI -> Constitutional -> RSSM -> Curiosity -> Sweep -> Curriculum."""
        data = mock_episode_data

        # --- Stage 1: BDI pre-training ---
        bdi_trainer = BDIPreTrainer(config=bridge_config.bdi_trainer)
        bdi_dataset = bdi_trainer.build_dataset(
            data["observations"],
            data["action_names"],
            [r.tolist() for r in data["rewards"]],
        )
        assert bdi_dataset.num_samples > 0
        bdi_result = bdi_trainer.train(bdi_dataset)
        assert bdi_result.epochs_run == 3
        assert bdi_result.final_loss >= 0.0
        assert 0.0 <= bdi_result.final_accuracy <= 1.0
        bdi_weights_path = tmp_path / "bdi_weights.npz"
        bdi_trainer.export_weights(bdi_weights_path)
        assert bdi_weights_path.exists()

        # --- Stage 2: Constitutional pre-training ---
        const_trainer = ConstitutionalPreTrainer(
            config=bridge_config.constitutional_trainer
        )
        flat_obs = np.concatenate(
            [ep[:-1] for ep in data["observations"]], axis=0
        ).astype(np.float32)
        flat_actions = np.concatenate(data["action_ids"]).astype(np.int64)
        flat_rewards = np.concatenate(data["rewards"]).astype(np.float32)
        flat_raw_obs: list[dict[str, float]] = []
        for ep_raw in data["raw_observations"]:
            flat_raw_obs.extend(ep_raw)

        n = min(len(flat_obs), len(flat_actions), len(flat_rewards), len(flat_raw_obs))
        const_dataset = const_trainer.build_dataset(
            flat_obs[:n], flat_actions[:n], flat_rewards[:n], flat_raw_obs[:n]
        )
        assert const_dataset.num_samples == n
        const_result = const_trainer.train(const_dataset)
        assert const_result.epochs_run == 3
        assert len(const_result.loss_history) == 3
        const_weights_path = tmp_path / "constitutional_weights.npz"
        const_trainer.export_weights(const_weights_path)
        assert const_weights_path.exists()

        # --- Stage 3: RSSM pre-training ---
        rssm_trainer = RSSMPreTrainer(config=bridge_config.rssm_pretrain)
        rssm_dataset = rssm_trainer.build_sequences(
            data["observations"],
            data["action_ids"],
            data["rewards"],
            data["dones"],
        )
        assert rssm_dataset.num_sequences > 0
        rssm_result = rssm_trainer.train(rssm_dataset)
        assert rssm_result.epochs_run == 3
        rssm_weights_path = tmp_path / "rssm_weights.npz"
        rssm_trainer.export_all(rssm_weights_path)
        assert rssm_weights_path.exists()

        # --- Stage 4: Curiosity optimization ---
        curiosity_optimizer = CuriosityWeightOptimizer(
            config=bridge_config.curiosity_optimizer
        )

        def _curiosity_evaluate(weights: dict[str, float]) -> float:
            return sum(weights.values())

        curiosity_result = curiosity_optimizer.optimize(_curiosity_evaluate, num_iterations=3)
        assert isinstance(curiosity_result, CuriosityWeights)
        assert len(curiosity_result.weights) == 4
        assert abs(sum(curiosity_result.weights.values()) - 1.0) < 0.05

        # --- Stage 5: MCTS sweep ---
        sweep_runner = MCTSSweepRunner(config=bridge_config.sweep)
        grid = sweep_runner.generate_grid()
        assert len(grid) > 0

        def _sweep_evaluate(
            config: dict[str, Any], episodes: int
        ) -> tuple[float, float, float]:
            return (float(config["c_puct"]) * 0.5, 0.1, 100.0)

        report = sweep_runner.run_sweep(_sweep_evaluate, param_grid=grid[:4])
        assert isinstance(report, SweepReport)
        assert report.best is not None
        assert report.best.mean_reward > 0.0
        optimal_path = tmp_path / "optimal_mcts.json"
        sweep_runner.export_optimal_config(report, optimal_path)
        assert optimal_path.exists()

        # --- Stage 6: Curriculum controller ---
        curriculum = PlatformCurriculumController(
            platform="drone", config=bridge_config.curriculum
        )
        assert curriculum.current_tier == 1
        for outcome in [True, True, False, True, True, True, True, False, True, True]:
            curriculum.record_outcome(outcome)
        assert curriculum.current_tier >= 1
        status = curriculum.status()
        assert isinstance(status, list)
        assert all(isinstance(s, TierStatus) for s in status)
        curriculum_path = tmp_path / "curriculum_state.json"
        curriculum.export_state(curriculum_path)
        assert curriculum_path.exists()

    def test_adapter_roundtrip_in_pipeline(
        self,
        bridge_config: MangoMASBridgeConfig,
    ) -> None:
        """Verify action adapter roundtrip and observation adapter output dims."""
        action_adapter = ActionSpaceAdapter(
            config=bridge_config.action_adapter, platform="drone"
        )
        obs_adapter = ObservationAdapter(
            config=bridge_config.observation_adapter, platform="drone"
        )

        rng = np.random.default_rng(42)
        continuous = rng.uniform(-1, 1, size=4).astype(np.float32)
        discrete = action_adapter.continuous_to_discrete(continuous)
        recovered = action_adapter.discrete_to_continuous(discrete)
        assert recovered.shape == (4,)
        assert np.all(recovered >= -1.0)
        assert np.all(recovered <= 1.0)

        mock_obs = _make_mock_obs(rng=rng, step=0, include_drone=True)
        state_vec = obs_adapter.adapt(mock_obs)
        assert state_vec.ndim == 1
        assert state_vec.shape[0] == obs_adapter.output_dim
        assert state_vec.dtype == np.float32

    def test_stage_outputs_not_none(
        self,
        bridge_config: MangoMASBridgeConfig,
        mock_episode_data: dict[str, Any],
    ) -> None:
        """Every stage output must be a non-None object of expected type."""
        data = mock_episode_data

        bdi = BDIPreTrainer(config=bridge_config.bdi_trainer)
        bdi_ds = bdi.build_dataset(
            data["observations"],
            data["action_names"],
            [r.tolist() for r in data["rewards"]],
        )
        bdi_result = bdi.train(bdi_ds)
        assert bdi_result is not None
        assert isinstance(bdi_result.loss_history, list)
        assert isinstance(bdi_result.accuracy_history, list)

        const = ConstitutionalPreTrainer(config=bridge_config.constitutional_trainer)
        flat_obs = np.concatenate([ep[:-1] for ep in data["observations"]], axis=0)
        flat_acts = np.concatenate(data["action_ids"])
        flat_rew = np.concatenate(data["rewards"])
        raw: list[dict[str, float]] = []
        for ep in data["raw_observations"]:
            raw.extend(ep)
        n = min(len(flat_obs), len(flat_acts), len(flat_rew), len(raw))
        const_ds = const.build_dataset(flat_obs[:n], flat_acts[:n], flat_rew[:n], raw[:n])
        const_result = const.train(const_ds)
        assert const_result is not None
        assert isinstance(const_result.loss_history, list)

        rssm = RSSMPreTrainer(config=bridge_config.rssm_pretrain)
        rssm_ds = rssm.build_sequences(
            data["observations"], data["action_ids"],
            data["rewards"], data["dones"],
        )
        rssm_result = rssm.train(rssm_ds)
        assert rssm_result is not None
        assert isinstance(rssm_result.loss_history, list)


class TestMangoMASDeterminism:
    """Verify same seed produces identical outputs across all stages."""

    def test_bdi_determinism(self) -> None:
        """Same seed + data -> identical BDI training results."""
        for _ in range(2):
            data = _collect_mock_episodes(2, 8, seed=99)
            trainer = BDIPreTrainer(
                config=BDITrainerConfig(num_epochs=3, batch_size=8)
            )
            ds = trainer.build_dataset(
                data["observations"],
                data["action_names"],
                [r.tolist() for r in data["rewards"]],
            )
            result = trainer.train(ds)
            # Capture first run
            if _ == 0:
                first_loss = result.loss_history[:]
                first_acc = result.accuracy_history[:]
            else:
                assert result.loss_history == pytest.approx(first_loss, abs=1e-6)
                assert result.accuracy_history == pytest.approx(first_acc, abs=1e-6)

    def test_constitutional_determinism(self) -> None:
        """Same seed + data -> identical constitutional training results."""
        results: list[list[float]] = []
        for _ in range(2):
            data = _collect_mock_episodes(2, 8, seed=99)
            trainer = ConstitutionalPreTrainer(
                config=ConstitutionalTrainerConfig(
                    num_epochs=3, batch_size=8, seed=42
                )
            )
            flat_obs = np.concatenate([ep[:-1] for ep in data["observations"]], axis=0)
            flat_acts = np.concatenate(data["action_ids"])
            flat_rew = np.concatenate(data["rewards"])
            raw: list[dict[str, float]] = []
            for ep in data["raw_observations"]:
                raw.extend(ep)
            n = min(len(flat_obs), len(flat_acts), len(flat_rew), len(raw))
            ds = trainer.build_dataset(flat_obs[:n], flat_acts[:n], flat_rew[:n], raw[:n])
            result = trainer.train(ds)
            results.append(result.loss_history[:])
        assert results[0] == pytest.approx(results[1], abs=1e-6)

    def test_curiosity_determinism(self) -> None:
        """Same seed -> identical curiosity optimization results."""
        results: list[dict[str, float]] = []
        for _ in range(2):
            optimizer = CuriosityWeightOptimizer(
                config=CuriosityOptimizerConfig(population_size=4, seed=42)
            )
            result = optimizer.optimize(
                lambda w: sum(v**2 for v in w.values()), num_iterations=3
            )
            results.append(dict(result.weights))
        for key in results[0]:
            assert results[0][key] == pytest.approx(results[1][key], abs=1e-6)

    def test_sweep_determinism(self) -> None:
        """Same seed -> identical sweep grid generation."""
        grids: list[list[dict[str, Any]]] = []
        for _ in range(2):
            runner = MCTSSweepRunner(
                config=SweepConfig(
                    c_puct_steps=2, sim_budget_steps=2,
                    depth_steps=2, discount_steps=2, seed=42,
                )
            )
            grids.append(runner.generate_grid())
        assert len(grids[0]) == len(grids[1])
        for a, b in zip(grids[0], grids[1]):
            for key in a:
                assert a[key] == pytest.approx(b[key], abs=1e-9)

    def test_mock_collection_determinism(self) -> None:
        """Same seed -> identical mock episode data."""
        d1 = _collect_mock_episodes(2, 5, seed=77)
        d2 = _collect_mock_episodes(2, 5, seed=77)
        for i in range(2):
            np.testing.assert_array_equal(d1["observations"][i], d2["observations"][i])
            assert d1["action_names"][i] == d2["action_names"][i]
            np.testing.assert_array_equal(d1["action_ids"][i], d2["action_ids"][i])
            np.testing.assert_array_equal(d1["rewards"][i], d2["rewards"][i])


class TestMangoMASConfigIntegration:
    """Verify config dataclass defaults produce valid component instances."""

    def test_default_config_creates_all_components(self) -> None:
        """MangoMASBridgeConfig defaults wire into every component without error."""
        config = MangoMASBridgeConfig()

        action_adapter = ActionSpaceAdapter(
            config=config.action_adapter, platform=config.platform
        )
        assert action_adapter.total_action_space > 0

        obs_adapter = ObservationAdapter(
            config=config.observation_adapter, platform=config.platform
        )
        assert obs_adapter.output_dim > 0

        bdi = BDIPreTrainer(config=config.bdi_trainer)
        assert bdi.config.num_intentions == 8

        const = ConstitutionalPreTrainer(config=config.constitutional_trainer)
        assert len(const.constraints) == 5

        rssm = RSSMPreTrainer(config=config.rssm_pretrain)
        assert rssm.config.latent_dim == 30

        curiosity = CuriosityWeightOptimizer(config=config.curiosity_optimizer)
        assert len(curiosity.channels) == 4

        sweep = MCTSSweepRunner(config=config.sweep)
        grid = sweep.generate_grid()
        assert len(grid) > 0

        curriculum = PlatformCurriculumController(
            platform=config.platform, config=config.curriculum
        )
        assert curriculum.current_tier == 1

    def test_car_platform_config(self) -> None:
        """Car platform variant wires correctly through all components."""
        config = MangoMASBridgeConfig(platform="car")
        action_adapter = ActionSpaceAdapter(
            config=config.action_adapter, platform="car"
        )
        assert action_adapter.continuous_dims == 2

        obs_adapter = ObservationAdapter(
            config=config.observation_adapter, platform="car"
        )
        rng = np.random.default_rng(42)
        mock_obs = _make_mock_obs(rng=rng, step=0, include_drone=False)
        state_vec = obs_adapter.adapt(mock_obs)
        assert state_vec.ndim == 1
        assert state_vec.dtype == np.float32

        curriculum = PlatformCurriculumController(
            platform="car", config=config.curriculum
        )
        tiers = curriculum.status()
        assert len(tiers) == 5
        assert tiers[0].name == "Straight Line"
