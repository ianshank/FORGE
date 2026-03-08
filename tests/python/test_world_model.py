"""Tests for forge.models.world_model module."""
from __future__ import annotations

import logging

import numpy as np
import pytest
from forge.models.world_model import IdentityWorldModel, WorldModel

logger = logging.getLogger(__name__)


class TestIdentityWorldModel:
    """Tests for the IdentityWorldModel stub."""

    @pytest.fixture()
    def model(self) -> IdentityWorldModel:
        """Return an IdentityWorldModel instance."""
        return IdentityWorldModel()

    def test_identity_world_model_predict(self, model: IdentityWorldModel) -> None:
        """predict() returns a copy of the input state."""
        state = np.array([1.0, 2.0, 3.0])
        result = model.predict(state, action=0)

        np.testing.assert_array_equal(result, state)
        # Verify it is a copy, not the same object
        assert result is not state

    def test_identity_world_model_train_step(
        self, model: IdentityWorldModel
    ) -> None:
        """train_step() returns an empty dict."""
        batch = {"states": np.array([1.0, 2.0])}
        result = model.train_step(batch)

        assert result == {}

    def test_identity_world_model_save_load(
        self, model: IdentityWorldModel, tmp_path: pytest.TempPathFactory
    ) -> None:
        """save() and load() execute without error."""
        path = str(tmp_path / "model.pt")
        model.save(path)
        model.load(path)

    def test_identity_world_model_predict_preserves_dtype(
        self, model: IdentityWorldModel
    ) -> None:
        """predict() preserves the dtype of the input state."""
        state = np.array([1, 2, 3], dtype=np.int32)
        result = model.predict(state, action=5)

        assert result.dtype == state.dtype


class TestWorldModelABC:
    """Tests for the WorldModel abstract base class."""

    def test_world_model_is_abstract(self) -> None:
        """WorldModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            WorldModel()  # type: ignore[abstract]
