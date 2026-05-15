"""Tests for forge.models.process_reward module."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from forge.models.process_reward import ConstantRewardModel, ProcessRewardModel

logger = logging.getLogger(__name__)


class TestConstantRewardModel:
    """Tests for the ConstantRewardModel stub."""

    @pytest.fixture()
    def model(self) -> ConstantRewardModel:
        """Return a default ConstantRewardModel."""
        return ConstantRewardModel()

    def test_constant_reward_model_score(self, model: ConstantRewardModel) -> None:
        """Default score is 1.0."""
        trace = [{"agent_id": 0, "tick": 1, "intent": "hold"}]
        assert model.score_trace(trace) == pytest.approx(1.0)

    def test_constant_reward_model_custom_score(self) -> None:
        """Custom score value is used."""
        model = ConstantRewardModel(score=0.5)
        trace = [{"agent_id": 0}]

        assert model.score_trace(trace) == pytest.approx(0.5)

    def test_constant_reward_model_train_step(self, model: ConstantRewardModel) -> None:
        """train_step() returns an empty dict."""
        batch = {"states": np.array([1.0])}
        result = model.train_step(batch)

        assert result == {}

    def test_constant_reward_model_save_load(
        self, model: ConstantRewardModel, tmp_path: pytest.TempPathFactory
    ) -> None:
        """save() and load() execute without error."""
        path = str(tmp_path / "reward_model.pt")
        model.save(path)
        model.load(path)

    def test_constant_reward_model_empty_trace(self, model: ConstantRewardModel) -> None:
        """Scoring an empty trace still returns the constant."""
        assert model.score_trace([]) == pytest.approx(1.0)


class TestProcessRewardModelABC:
    """Tests for the ProcessRewardModel abstract base class."""

    def test_process_reward_model_is_abstract(self) -> None:
        """ProcessRewardModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            ProcessRewardModel()
