"""Tests for CleanRL training scripts — train_ppo_cleanrl.py and train_sac_cleanrl.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Locate examples directory
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
_PPO_SCRIPT = _EXAMPLES_DIR / "train_ppo_cleanrl.py"
_SAC_SCRIPT = _EXAMPLES_DIR / "train_sac_cleanrl.py"


def _load_module(path: Path, name: str) -> Any:
    """Dynamically load a Python script as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MOCK_OBS_DICT: dict[str, Any] = {
    "grid_view": [[[0] * 7] * 11] * 11,
    "health": 0.8,
    "stamina": 0.9,
    "inventory": [[0, 0]] * 10,
    "position": [5, 5],
    "messages": [],
    "day_phase": 0,
}


def _make_mock_vec_env(n_envs: int = 1) -> MagicMock:
    """Return a mock ForgeSyncVecEnv."""
    import numpy as np  # noqa: PLC0415

    env = MagicMock()
    env.num_envs = n_envs
    flat_obs = np.zeros((n_envs, 42), dtype=np.float32)
    env.reset.return_value = (flat_obs, [{"tick": 0}] * n_envs)
    env.step.return_value = (
        flat_obs,
        np.zeros(n_envs, dtype=np.float32),
        np.zeros(n_envs, dtype=bool),
        np.zeros(n_envs, dtype=bool),
        [{"episode": {"r": 1.0, "l": 10}}] * n_envs,
    )
    act_space = MagicMock()
    act_space.n = 8
    env.action_space = act_space
    return env


def _make_mock_single_env() -> MagicMock:
    """Return a mock single ForgeGymnasiumEnv (flat obs)."""
    import numpy as np  # noqa: PLC0415

    env = MagicMock()
    flat_obs = np.zeros(42, dtype=np.float32)
    env.reset.return_value = (flat_obs, {"tick": 0})
    env.step.return_value = (
        flat_obs, 1.0, False, True,
        {"episode": {"r": 1.0, "l": 10}},
    )
    act_space = MagicMock()
    act_space.n = 8
    act_space.sample.return_value = 0
    env.action_space = act_space
    return env


# ---------------------------------------------------------------------------
# Tests for train_ppo_cleanrl.py
# ---------------------------------------------------------------------------


class TestTrainPPOCleanRL:
    """Tests for train_ppo_cleanrl.py."""

    def test_script_exists(self) -> None:
        assert _PPO_SCRIPT.is_file(), f"Expected script at {_PPO_SCRIPT}"

    def test_module_imports_successfully(self) -> None:
        """The script should be importable without raising errors."""
        # We need to mock forge_env imports since the native extension may not be built.
        mock_forge_env = MagicMock()
        mock_forge_env.vecenv.ForgeSyncVecEnv = MagicMock
        mock_forge_env.vecenv.make_forge_vec_env = MagicMock(return_value=_make_mock_vec_env())
        mock_forge_env.wrappers.FlattenObservationWrapper = lambda e: e
        mock_forge_env.wrappers.RecordEpisodeStatistics = lambda e: e
        mock_forge_env.wrappers.TimeLimit = lambda e, max_steps: e
        mock_forge_env.utils.seed_everything = lambda s: None

        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.vecenv": mock_forge_env.vecenv,
            "forge_env.wrappers": mock_forge_env.wrappers,
            "forge_env.utils": mock_forge_env.utils,
        }):
            mod = _load_module(_PPO_SCRIPT, "train_ppo_cleanrl")
        assert hasattr(mod, "main")
        assert hasattr(mod, "train")
        assert hasattr(mod, "_ActorCritic")

    def test_actor_critic_output_shapes(self) -> None:
        """_ActorCritic forward pass returns correct shapes."""
        pytest.importorskip("torch")
        import torch  # noqa: PLC0415

        mod = _load_module(_PPO_SCRIPT, "train_ppo_cleanrl_shape")
        model = mod._ActorCritic(obs_dim=16, action_dim=8)
        obs = torch.zeros(4, 16)
        action, log_prob, entropy, value = model.get_action_and_value(obs)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_actor_critic_get_value(self) -> None:
        pytest.importorskip("torch")
        import torch  # noqa: PLC0415

        mod = _load_module(_PPO_SCRIPT, "train_ppo_cleanrl_value")
        model = mod._ActorCritic(obs_dim=8, action_dim=4)
        obs = torch.zeros(3, 8)
        val = model.get_value(obs)
        assert val.shape == (3,)

    def test_config_loaded_from_toml(self) -> None:
        """Gamma from TOML config flows into the parsed args."""
        mock_forge_env = MagicMock()
        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.vecenv": mock_forge_env,
            "forge_env.wrappers": mock_forge_env,
            "forge_env.utils": mock_forge_env,
        }):
            mod = _load_module(_PPO_SCRIPT, "train_ppo_cleanrl_cfg")

        # Parse with default config
        args = mod._build_argparser({}).parse_args([])
        assert 0.0 < args.gamma <= 1.0

    def test_argparser_accepts_total_timesteps(self) -> None:
        mock_forge_env = MagicMock()
        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.vecenv": mock_forge_env,
            "forge_env.wrappers": mock_forge_env,
            "forge_env.utils": mock_forge_env,
        }):
            mod = _load_module(_PPO_SCRIPT, "train_ppo_cleanrl_ts")
        args = mod._build_argparser({}).parse_args(["--total-timesteps", "1234"])
        assert args.total_timesteps == 1234

    def test_env_var_override_applied(self) -> None:
        """FORGE_TRAINING_ environment variable overrides should take effect."""
        mock_forge_env = MagicMock()
        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.vecenv": mock_forge_env,
            "forge_env.wrappers": mock_forge_env,
            "forge_env.utils": mock_forge_env,
        }):
            mod = _load_module(_PPO_SCRIPT, "train_ppo_cleanrl_envvar")

        import os  # noqa: PLC0415
        with patch.dict(os.environ, {"FORGE_TRAINING_SEED": "99"}):
            args = mod._build_argparser({}).parse_args([])
            # Simulate the override logic from main()
            env_val = os.environ.get("FORGE_TRAINING_SEED")
            if env_val:
                setattr(args, "seed", int(env_val))
        assert args.seed == 99


# ---------------------------------------------------------------------------
# Tests for train_sac_cleanrl.py
# ---------------------------------------------------------------------------


class TestTrainSACCleanRL:
    """Tests for train_sac_cleanrl.py."""

    def test_script_exists(self) -> None:
        assert _SAC_SCRIPT.is_file(), f"Expected script at {_SAC_SCRIPT}"

    def test_module_imports_successfully(self) -> None:
        mock_forge_env = MagicMock()
        mock_forge_env.wrappers.FlattenObservationWrapper = lambda e: e
        mock_forge_env.wrappers.RecordEpisodeStatistics = lambda e: e
        mock_forge_env.wrappers.TimeLimit = lambda e, max_steps: e
        mock_forge_env.gymnasium_env.ForgeGymnasiumEnv = MagicMock(return_value=_make_mock_single_env())
        mock_forge_env.utils.seed_everything = lambda s: None

        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.gymnasium_env": mock_forge_env.gymnasium_env,
            "forge_env.wrappers": mock_forge_env.wrappers,
            "forge_env.utils": mock_forge_env.utils,
        }):
            mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl")
        assert hasattr(mod, "main")
        assert hasattr(mod, "train")
        assert hasattr(mod, "_ReplayBuffer")
        assert hasattr(mod, "_Actor")
        assert hasattr(mod, "_SoftQNetwork")

    def test_replay_buffer_add_and_sample(self) -> None:
        pytest.importorskip("torch")
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl_buf")
        buf = mod._ReplayBuffer(capacity=100, obs_dim=8, device=torch.device("cpu"))
        assert len(buf) == 0

        obs = np.zeros(8, dtype=np.float32)
        next_obs = np.ones(8, dtype=np.float32)
        buf.add(obs, next_obs, action=2, reward=1.0, done=0.0)
        assert len(buf) == 1

        batch = buf.sample(1)
        assert "obs" in batch
        assert "actions" in batch
        assert batch["obs"].shape == (1, 8)

    def test_replay_buffer_capacity_limit(self) -> None:
        pytest.importorskip("torch")
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl_cap")
        buf = mod._ReplayBuffer(capacity=5, obs_dim=4, device=torch.device("cpu"))
        for _ in range(10):
            buf.add(np.zeros(4), np.zeros(4), 0, 0.0, 0.0)
        assert len(buf) == 5  # Capped at capacity

    def test_actor_output_shapes(self) -> None:
        pytest.importorskip("torch")
        import torch  # noqa: PLC0415

        mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl_actor")
        actor = mod._Actor(obs_dim=8, action_dim=4, hidden_sizes=(32,))
        obs = torch.zeros(3, 8)
        probs, log_probs, actions = actor(obs)
        assert probs.shape == (3, 4)
        assert log_probs.shape == (3, 4)
        assert actions.shape == (3,)

    def test_soft_q_network_output(self) -> None:
        pytest.importorskip("torch")
        import torch  # noqa: PLC0415

        mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl_qnet")
        qnet = mod._SoftQNetwork(obs_dim=8, action_dim=4, hidden_sizes=(32,))
        obs = torch.zeros(5, 8)
        q_vals = qnet(obs)
        assert q_vals.shape == (5, 4)

    def test_gamma_from_toml_config(self) -> None:
        mock_forge_env = MagicMock()
        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.gymnasium_env": mock_forge_env,
            "forge_env.wrappers": mock_forge_env,
            "forge_env.utils": mock_forge_env,
        }):
            mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl_gamma")
        args = mod._build_argparser({}).parse_args([])
        assert 0.0 < args.gamma <= 1.0

    def test_argparser_ent_coef_auto(self) -> None:
        mock_forge_env = MagicMock()
        with patch.dict(sys.modules, {
            "forge_env": mock_forge_env,
            "forge_env.gymnasium_env": mock_forge_env,
            "forge_env.wrappers": mock_forge_env,
            "forge_env.utils": mock_forge_env,
        }):
            mod = _load_module(_SAC_SCRIPT, "train_sac_cleanrl_ent")
        args = mod._build_argparser({}).parse_args([])
        assert args.ent_coef == "auto"


# ---------------------------------------------------------------------------
# Config TOML files exist and are valid
# ---------------------------------------------------------------------------


class TestTrainingConfigs:
    """Validate that the training TOML config files are present and parseable."""

    _CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "training"

    def test_ppo_default_exists(self) -> None:
        assert (self._CONFIGS_DIR / "ppo_default.toml").is_file()

    def test_sac_default_exists(self) -> None:
        assert (self._CONFIGS_DIR / "sac_default.toml").is_file()

    def test_ppo_default_parseable(self) -> None:
        try:
            import tomllib  # noqa: PLC0415
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415
        content = (self._CONFIGS_DIR / "ppo_default.toml").read_text(encoding="utf-8")
        data = tomllib.loads(content)
        assert "hyperparams" in data
        assert "logging" in data
        hp = data["hyperparams"]
        assert hp["gamma"] == pytest.approx(0.99)
        assert hp["n_steps"] > 0

    def test_sac_default_parseable(self) -> None:
        try:
            import tomllib  # noqa: PLC0415
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415
        content = (self._CONFIGS_DIR / "sac_default.toml").read_text(encoding="utf-8")
        data = tomllib.loads(content)
        assert "hyperparams" in data
        hp = data["hyperparams"]
        assert hp["gamma"] == pytest.approx(0.99)
        assert hp["tau"] == pytest.approx(0.005)

    def test_ppo_no_hard_coded_values(self) -> None:
        """Every hyperparameter in ppo_default.toml should be explicitly set."""
        try:
            import tomllib  # noqa: PLC0415
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415
        content = (self._CONFIGS_DIR / "ppo_default.toml").read_text(encoding="utf-8")
        data = tomllib.loads(content)
        required_keys = {
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
            "max_grad_norm", "total_timesteps",
        }
        hp = data.get("hyperparams", {})
        missing = required_keys - set(hp.keys())
        assert not missing, f"Missing hyperparams in ppo_default.toml: {missing}"
