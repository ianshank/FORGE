"""Tests for the trust tracker module."""
from __future__ import annotations

from forge.social.trust_tracker import SocialConfig, TrustTracker


class TestSocialConfig:
    """Tests for SocialConfig dataclass."""

    def test_defaults(self) -> None:
        config = SocialConfig()
        assert config.trust_initial == 0.5
        assert config.trust_update_rate == 0.1
        assert config.cooperation_reward_weight == 0.3
        assert config.betrayal_penalty == -0.5
        assert config.alliance_threshold == 0.7


class TestTrustTracker:
    """Tests for TrustTracker."""

    def test_creation(self) -> None:
        tracker = TrustTracker(num_agents=4)
        assert tracker.num_agents == 4
        assert tracker.trust_matrix.shape == (4, 4)

    def test_initial_trust(self) -> None:
        config = SocialConfig(trust_initial=0.3)
        tracker = TrustTracker(num_agents=3, config=config)
        assert tracker.trust_matrix[0, 1] == 0.3

    def test_cooperation_increases_trust(self) -> None:
        tracker = TrustTracker(num_agents=3)
        initial = float(tracker.trust_matrix[0, 1])
        tracker.record_cooperation(0, 1)
        assert tracker.trust_matrix[0, 1] > initial
        assert tracker.trust_matrix[1, 0] > initial  # symmetric

    def test_hostility_decreases_trust(self) -> None:
        tracker = TrustTracker(num_agents=3)
        initial = float(tracker.trust_matrix[0, 1])
        tracker.record_hostility(0, 1)
        assert tracker.trust_matrix[0, 1] < initial

    def test_trust_bounded_above(self) -> None:
        config = SocialConfig(trust_initial=0.95, trust_update_rate=0.1)
        tracker = TrustTracker(num_agents=2, config=config)
        tracker.record_cooperation(0, 1)
        assert tracker.trust_matrix[0, 1] <= 1.0

    def test_trust_bounded_below(self) -> None:
        config = SocialConfig(trust_initial=0.05, trust_update_rate=0.1)
        tracker = TrustTracker(num_agents=2, config=config)
        tracker.record_hostility(0, 1)
        assert tracker.trust_matrix[0, 1] >= 0.0

    def test_interaction_count(self) -> None:
        tracker = TrustTracker(num_agents=3)
        tracker.record_cooperation(0, 1)
        tracker.record_cooperation(0, 1)
        tracker.record_hostility(0, 1)
        assert tracker.interaction_count[0, 1] == 3
        assert tracker.interaction_count[1, 0] == 3

    def test_reputation_cooperation(self) -> None:
        tracker = TrustTracker(num_agents=3)
        tracker.record_cooperation(0, 1)
        # Agent 0 cooperated: 1 coop / 1 total = 1.0
        assert tracker.reputation[0] == 1.0

    def test_reputation_hostility(self) -> None:
        tracker = TrustTracker(num_agents=3)
        tracker.record_hostility(0, 1)
        assert tracker.reputation[0] == -1.0

    def test_reputation_mixed(self) -> None:
        tracker = TrustTracker(num_agents=3)
        tracker.record_cooperation(0, 1)
        tracker.record_cooperation(0, 1)
        tracker.record_hostility(0, 1)
        # (2 - 1) / 3 ≈ 0.333
        assert abs(tracker.reputation[0] - 1.0 / 3.0) < 0.01

    def test_reputation_bounds(self) -> None:
        tracker = TrustTracker(num_agents=5)
        for _ in range(50):
            tracker.record_cooperation(0, 1)
        for _ in range(50):
            tracker.record_hostility(1, 2)
        assert -1.0 <= tracker.reputation[0] <= 1.0
        assert -1.0 <= tracker.reputation[1] <= 1.0


class TestSocialRewards:
    """Tests for social reward computation."""

    def test_neutral_rewards_near_zero(self) -> None:
        tracker = TrustTracker(num_agents=3)
        rewards = tracker.compute_social_rewards()
        assert rewards.shape == (3,)
        for r in rewards:
            assert abs(r) < 0.01

    def test_cooperation_gives_positive_reward(self) -> None:
        tracker = TrustTracker(num_agents=3)
        tracker.record_cooperation(1, 0)
        tracker.record_cooperation(2, 0)
        rewards = tracker.compute_social_rewards()
        assert rewards[0] > 0.0  # agent 0 is trusted

    def test_hostility_gives_negative_components(self) -> None:
        tracker = TrustTracker(num_agents=3)
        for _ in range(5):
            tracker.record_hostility(0, 1)
        rewards = tracker.compute_social_rewards()
        # Agent 0 has negative reputation and lower trust
        assert rewards[0] < 0.0

    def test_single_agent(self) -> None:
        tracker = TrustTracker(num_agents=1)
        rewards = tracker.compute_social_rewards()
        assert rewards.shape == (1,)
        # Single agent: mean_trust=0 (no others), trust_delta=-0.5, reputation=0
        # reward = -0.5 * 0.3 + 0.0 * 0.7 = -0.15
        assert abs(float(rewards[0]) - (-0.15)) < 0.01
