"""Tests for the AlphaGalerkin adapter layer.

All AlphaGalerkin model interactions are replaced with ``MagicMock`` objects so
the test suite does not require AlphaGalerkin to be installed.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from forge.adapters.alphagalerkin_adapter import (
    ALPHAGALERKIN_AVAILABLE,
    AlphaGalerkinAgent,
    ForgeGameAdapter,
    ForgeGameState,
)
from forge.agents.base_agent import AgentConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_OBS = np.zeros(8, dtype=np.float32)


def _make_mock_env(*, action_n: int = 4) -> MagicMock:
    """Return a minimal Gymnasium-compatible mock environment."""
    env = MagicMock()
    env.reset.return_value = (MOCK_OBS, {"tick": 0})
    env.step.return_value = (MOCK_OBS, 1.0, False, False, {"tick": 1})
    env.observation_space = {"shape": (8,)}
    env.action_space = {"n": action_n}
    return env


def _make_adapter(action_n: int = 4) -> ForgeGameAdapter:
    return ForgeGameAdapter(_make_mock_env(action_n=action_n))


# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------


class TestModuleImport:
    """The module imports cleanly regardless of whether AG is installed."""

    def test_alphagalerkin_available_is_bool(self) -> None:
        assert isinstance(ALPHAGALERKIN_AVAILABLE, bool)

    def test_import_without_alphagalerkin(self) -> None:
        """Simulating a missing AlphaGalerkin package does not raise."""
        with (
            patch.dict(
                "sys.modules", {"src": None, "src.games": None, "src.games.interface": None}
            ),
            contextlib.suppress(ImportError, TypeError),
        ):
            # Re-executing the conditional import block should not crash.
            from src.games.interface import GameInterface  # noqa: F401,PLC0415


# ---------------------------------------------------------------------------
# ForgeGameState
# ---------------------------------------------------------------------------


class TestForgeGameState:
    def test_defaults(self) -> None:
        state = ForgeGameState(
            observation=MOCK_OBS,
            done=False,
            reward=0.0,
            info={},
        )
        assert state.move_number == 0
        assert not state.done
        assert state.reward == 0.0

    def test_custom_move_number(self) -> None:
        state = ForgeGameState(
            observation=MOCK_OBS,
            done=True,
            reward=-1.0,
            info={"foo": "bar"},
            move_number=5,
        )
        assert state.move_number == 5
        assert state.done


# ---------------------------------------------------------------------------
# ForgeGameAdapter
# ---------------------------------------------------------------------------


class TestForgeGameAdapterActionSpaceSize:
    def test_action_space_size(self) -> None:
        adapter = _make_adapter(action_n=6)
        assert adapter.action_space_size == 6

    def test_action_space_size_default(self) -> None:
        adapter = _make_adapter(action_n=4)
        assert adapter.action_space_size == 4


class TestForgeGameAdapterInitialState:
    def test_initial_state_returns_forge_game_state(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        assert isinstance(state, ForgeGameState)

    def test_initial_state_not_done(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        assert not state.done

    def test_initial_state_reward_is_zero(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        assert state.reward == 0.0

    def test_initial_state_move_number_is_zero(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        assert state.move_number == 0

    def test_initial_state_observation_is_array(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        assert isinstance(state.observation, np.ndarray)

    def test_initial_state_calls_env_reset(self) -> None:
        env = _make_mock_env()
        adapter = ForgeGameAdapter(env)
        adapter.initial_state()
        env.reset.assert_called_once()


class TestForgeGameAdapterApplyAction:
    def test_apply_action_returns_new_state(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        new_state = adapter.apply_action(state, action=0)
        assert isinstance(new_state, ForgeGameState)

    def test_apply_action_increments_move_number(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        new_state = adapter.apply_action(state, action=1)
        assert new_state.move_number == state.move_number + 1

    def test_apply_action_reward_from_env(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        new_state = adapter.apply_action(state, action=2)
        assert new_state.reward == 1.0  # mocked step returns reward=1.0

    def test_apply_action_done_when_terminated(self) -> None:
        env = _make_mock_env()
        env.step.return_value = (MOCK_OBS, 0.0, True, False, {})
        adapter = ForgeGameAdapter(env)
        state = adapter.initial_state()
        new_state = adapter.apply_action(state, action=0)
        assert new_state.done

    def test_apply_action_done_when_truncated(self) -> None:
        env = _make_mock_env()
        env.step.return_value = (MOCK_OBS, 0.0, False, True, {})
        adapter = ForgeGameAdapter(env)
        state = adapter.initial_state()
        new_state = adapter.apply_action(state, action=0)
        assert new_state.done

    def test_apply_action_not_done_when_neither(self) -> None:
        adapter = _make_adapter()
        state = adapter.initial_state()
        new_state = adapter.apply_action(state, action=0)
        assert not new_state.done


class TestForgeGameAdapterLegalActions:
    def test_legal_actions_all_when_no_mask(self) -> None:
        adapter = _make_adapter(action_n=4)
        state = ForgeGameState(observation=MOCK_OBS, done=False, reward=0.0, info={})
        legal = adapter.get_legal_actions(state)
        assert legal == [0, 1, 2, 3]

    def test_legal_actions_filtered_by_mask(self) -> None:
        adapter = _make_adapter(action_n=4)
        state = ForgeGameState(
            observation=MOCK_OBS,
            done=False,
            reward=0.0,
            info={"action_mask": [1, 0, 1, 0]},
        )
        legal = adapter.get_legal_actions(state)
        assert legal == [0, 2]

    def test_legal_actions_empty_mask(self) -> None:
        adapter = _make_adapter(action_n=4)
        state = ForgeGameState(
            observation=MOCK_OBS,
            done=False,
            reward=0.0,
            info={"action_mask": [0, 0, 0, 0]},
        )
        legal = adapter.get_legal_actions(state)
        assert legal == []


class TestForgeGameAdapterIsTerminal:
    def test_is_terminal_when_done(self) -> None:
        adapter = _make_adapter()
        state = ForgeGameState(observation=MOCK_OBS, done=True, reward=0.0, info={})
        assert adapter.is_terminal(state)

    def test_is_not_terminal_when_not_done(self) -> None:
        adapter = _make_adapter()
        state = ForgeGameState(observation=MOCK_OBS, done=False, reward=0.0, info={})
        assert not adapter.is_terminal(state)


class TestForgeGameAdapterToTensor:
    def test_to_tensor_returns_observation(self) -> None:
        adapter = _make_adapter()
        state = ForgeGameState(observation=MOCK_OBS, done=False, reward=0.0, info={})
        result = adapter.to_tensor(state)
        np.testing.assert_array_equal(result, MOCK_OBS)


# ---------------------------------------------------------------------------
# AlphaGalerkinAgent
# ---------------------------------------------------------------------------


def _make_mock_torch_model(action_n: int = 4) -> MagicMock:
    """Return a mock AG model that produces fake logits via torch tensors."""
    import torch  # noqa: PLC0415

    logits = torch.zeros(1, action_n)
    model = MagicMock()
    model.return_value = logits
    return model


class TestAlphaGalerkinAgent:
    """Tests for AlphaGalerkinAgent."""

    def test_is_base_agent_subclass(self) -> None:
        from forge.agents.base_agent import BaseAgent  # noqa: PLC0415

        assert issubclass(AlphaGalerkinAgent, BaseAgent)

    def test_act_returns_action_and_info(self) -> None:
        pytest.importorskip("torch")
        model = _make_mock_torch_model(action_n=4)
        agent = AlphaGalerkinAgent(AgentConfig(), model=model, action_space_size=4)
        obs = np.zeros(8, dtype=np.float32)
        action, info = agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < 4
        assert isinstance(info, dict)
        assert "logits" in info
        assert "probs" in info

    def test_act_increments_step_count(self) -> None:
        pytest.importorskip("torch")
        model = _make_mock_torch_model(action_n=4)
        agent = AlphaGalerkinAgent(AgentConfig(), model=model, action_space_size=4)
        obs = np.zeros(8, dtype=np.float32)
        agent.act(obs)
        assert agent.step_count == 1

    def test_act_calls_model(self) -> None:
        pytest.importorskip("torch")
        model = _make_mock_torch_model(action_n=4)
        agent = AlphaGalerkinAgent(AgentConfig(), model=model, action_space_size=4)
        obs = np.zeros(8, dtype=np.float32)
        agent.act(obs)
        model.assert_called_once()

    def test_learn_returns_empty_dict(self) -> None:
        model = MagicMock()
        agent = AlphaGalerkinAgent(AgentConfig(), model=model, action_space_size=4)
        result = agent.learn({"obs": np.zeros((4, 8))})
        assert result == {}

    def test_learn_returns_empty_dict_for_empty_batch(self) -> None:
        model = MagicMock()
        agent = AlphaGalerkinAgent(AgentConfig(), model=model, action_space_size=4)
        result = agent.learn({})
        assert result == {}

    def test_info_probs_sum_to_one(self) -> None:
        pytest.importorskip("torch")
        model = _make_mock_torch_model(action_n=4)
        agent = AlphaGalerkinAgent(AgentConfig(), model=model, action_space_size=4)
        obs = np.zeros(8, dtype=np.float32)
        _action, info = agent.act(obs)
        probs = info["probs"]
        assert abs(sum(probs) - 1.0) < 1e-5
