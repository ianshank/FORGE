"""Tests for MangoMAS curriculum controller."""
from __future__ import annotations

from typing import Any

import pytest
from forge.mangomas.config import CurriculumConfig
from forge.mangomas.curriculum_controller import (
    CAR_TIERS,
    DRONE_TIERS,
    PlatformCurriculumController,
)


class TestPlatformCurriculumController:
    """Tests for PlatformCurriculumController."""

    def test_initial_state_drone(self) -> None:
        ctrl = PlatformCurriculumController(platform="drone")
        assert ctrl.current_tier == 1
        assert ctrl.max_unlocked_tier == 1

    def test_initial_state_car(self) -> None:
        ctrl = PlatformCurriculumController(platform="car")
        assert ctrl.current_tier == 1

    def test_drone_has_5_tiers(self) -> None:
        assert len(DRONE_TIERS) == 5

    def test_car_has_5_tiers(self) -> None:
        assert len(CAR_TIERS) == 5

    def test_sample_scenario(self) -> None:
        ctrl = PlatformCurriculumController(platform="drone")
        scenario = ctrl.sample_scenario()
        assert scenario["tier"] == 1
        assert scenario["forge_scenario"] == "patrol"
        assert "name" in scenario

    def test_success_rate_starts_zero(self) -> None:
        ctrl = PlatformCurriculumController()
        assert ctrl.success_rate() == 0.0

    def test_record_outcomes_updates_rate(self) -> None:
        ctrl = PlatformCurriculumController()
        ctrl.record_outcome(True)
        ctrl.record_outcome(True)
        ctrl.record_outcome(False)
        assert ctrl.success_rate() == pytest.approx(2 / 3)

    def test_promotion_after_warmup(self) -> None:
        config = CurriculumConfig(warmup_episodes=5, window_size=10)
        ctrl = PlatformCurriculumController(platform="drone", config=config)

        # Record enough successes to promote past tier 1 (threshold 0.7)
        for _ in range(20):
            ctrl.record_outcome(True)

        assert ctrl.current_tier > 1
        assert ctrl.max_unlocked_tier > 1

    def test_no_promotion_before_warmup(self) -> None:
        config = CurriculumConfig(warmup_episodes=100)
        ctrl = PlatformCurriculumController(platform="drone", config=config)

        for _ in range(50):
            ctrl.record_outcome(True)

        assert ctrl.current_tier == 1  # Still warming up

    def test_demotion(self) -> None:
        config = CurriculumConfig(warmup_episodes=5, window_size=10, adjustment_rate=0.1)
        ctrl = PlatformCurriculumController(platform="drone", config=config)

        # Promote first
        for _ in range(20):
            ctrl.record_outcome(True)
        promoted_tier = ctrl.current_tier
        assert promoted_tier > 1

        # Now fail many times
        for _ in range(20):
            ctrl.record_outcome(False)

        assert ctrl.current_tier < promoted_tier

    def test_tier_info(self) -> None:
        ctrl = PlatformCurriculumController(platform="drone")
        info = ctrl.tier_info(1)
        assert info["name"] == "Hover and Altitude"
        assert info["forge_scenario"] == "patrol"

    def test_tier_info_invalid(self) -> None:
        ctrl = PlatformCurriculumController()
        with pytest.raises(ValueError, match="Unknown tier"):
            ctrl.tier_info(99)

    def test_status(self) -> None:
        ctrl = PlatformCurriculumController(platform="drone")
        ctrl.record_outcome(True)
        status = ctrl.status()
        assert len(status) == 5
        assert status[0].is_unlocked
        assert status[0].episodes_completed == 1

    def test_export_state(self, tmp_path: Any) -> None:
        ctrl = PlatformCurriculumController(platform="drone")
        ctrl.record_outcome(True)
        path = tmp_path / "curriculum.json"
        ctrl.export_state(path)
        assert path.exists()

    def test_max_tier_5_cap(self) -> None:
        config = CurriculumConfig(warmup_episodes=2, window_size=5)
        ctrl = PlatformCurriculumController(platform="drone", config=config)

        # Many successes — should cap at tier 5
        for _ in range(200):
            ctrl.record_outcome(True)

        assert ctrl.current_tier <= 5

    def test_custom_tiers_parameter(self) -> None:
        custom = [
            {"tier": 1, "name": "Custom", "forge_scenario": "patrol", "success_threshold": 0.5},
        ]
        ctrl = PlatformCurriculumController(tiers=custom)
        assert ctrl.tier_info(1)["name"] == "Custom"

    def test_config_tiers_parameter(self) -> None:
        tiers = [
            {"tier": 1, "name": "FromConfig", "forge_scenario": "patrol", "success_threshold": 0.6},
        ]
        config = CurriculumConfig(tiers=tiers)
        ctrl = PlatformCurriculumController(config=config)
        assert ctrl.tier_info(1)["name"] == "FromConfig"

    def test_record_outcome_with_metrics(self) -> None:
        ctrl = PlatformCurriculumController()
        ctrl.record_outcome(True, metrics={"reward": 10.0})
