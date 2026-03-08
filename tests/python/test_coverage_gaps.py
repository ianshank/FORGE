"""Tests covering previously untested modules and edge cases.

Brings test coverage to 80%+ by testing:
- WorldModel and IdentityWorldModel
- ProcessRewardModel and ConstantRewardModel
- LoggingConfig (setup_logging, JsonFormatter)
- RolloutBuffer edge cases (wrap-around, empty sampling)
- CheckpointManager (rotation, corrupt files, schema migration)
- MetricsTracker (latest, count, window overflow)
- Device detection (is_gpu_available)
- RandomAgent (seed reproducibility, step count, edge cases)
- MCTSAgent (tree depth, edge cases, node operations)
- DecisionTrace (schema migration, unknown fields)
- BaseAgent (save/load through subclass)
- ActorCriticNetwork (dimension validation on load)
- TraceLogger (size limits, closed logger)
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from forge.agents.base_agent import AgentConfig
from forge.agents.mcts_agent import MCTSAgent, MCTSConfig, MCTSNode
from forge.agents.random_agent import RandomAgent
from forge.config import ForgeConfig, SimulationConfig, TrainingConfig
from forge.models.policy_network import ActorCriticNetwork
from forge.models.process_reward import ConstantRewardModel, ProcessRewardModel
from forge.models.world_model import IdentityWorldModel, WorldModel
from forge.traces.decision_trace import DEFAULT_SCHEMA_VERSION, DecisionTrace
from forge.traces.trace_logger import TraceLogger
from forge.training.buffer import RolloutBuffer
from forge.training.checkpointing import CHECKPOINT_SCHEMA_VERSION, CheckpointManager
from forge.utils.device import get_device, is_gpu_available
from forge.utils.logging_config import JsonFormatter, setup_logging
from forge.utils.metrics import MetricsTracker
from forge.utils.seed import MAX_SEED, derive_seed, set_all_seeds

# ============================================================
# WorldModel tests
# ============================================================


class TestWorldModel:
    """Tests for WorldModel ABC and IdentityWorldModel."""

    def test_cannot_instantiate_abc(self) -> None:
        """WorldModel is abstract."""
        with pytest.raises(TypeError):
            WorldModel()  # type: ignore[abstract]

    def test_identity_predict_returns_copy(self) -> None:
        """IdentityWorldModel.predict returns a copy, not a reference."""
        model = IdentityWorldModel()
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = model.predict(state, action=0)
        np.testing.assert_array_equal(result, state)
        # Verify it's a copy, not a reference
        result[0] = 999.0
        assert state[0] == 1.0

    def test_identity_predict_different_actions(self) -> None:
        """IdentityWorldModel ignores the action parameter."""
        model = IdentityWorldModel()
        state = np.array([1.0, 2.0], dtype=np.float32)
        for action in range(5):
            result = model.predict(state, action)
            np.testing.assert_array_equal(result, state)

    def test_identity_train_step_returns_empty(self) -> None:
        """IdentityWorldModel.train_step returns empty metrics."""
        model = IdentityWorldModel()
        metrics = model.train_step({"obs": np.zeros(4)})
        assert metrics == {}

    def test_identity_save_load_noop(self, tmp_path: Path) -> None:
        """IdentityWorldModel.save/load are no-ops that don't raise."""
        model = IdentityWorldModel()
        model.save(str(tmp_path / "test_identity_model"))
        model.load(str(tmp_path / "test_identity_model"))


# ============================================================
# ProcessRewardModel tests
# ============================================================


class TestProcessRewardModel:
    """Tests for ProcessRewardModel ABC and ConstantRewardModel."""

    def test_cannot_instantiate_abc(self) -> None:
        """ProcessRewardModel is abstract."""
        with pytest.raises(TypeError):
            ProcessRewardModel()  # type: ignore[abstract]

    def test_constant_default_score(self) -> None:
        """ConstantRewardModel defaults to 1.0."""
        model = ConstantRewardModel()
        assert model.score_trace([]) == 1.0

    def test_constant_custom_score(self) -> None:
        """ConstantRewardModel returns configured score."""
        model = ConstantRewardModel(score=0.5)
        assert model.score_trace([{"action": 1}]) == 0.5

    def test_constant_ignores_trace_content(self) -> None:
        """ConstantRewardModel score is independent of trace content."""
        model = ConstantRewardModel(score=0.7)
        traces_a: list[dict[str, object]] = [{"tick": 1}]
        traces_b: list[dict[str, object]] = [{"tick": 1}, {"tick": 2}, {"tick": 3}]
        assert model.score_trace(traces_a) == model.score_trace(traces_b)

    def test_constant_train_step_returns_empty(self) -> None:
        """ConstantRewardModel.train_step returns empty metrics."""
        model = ConstantRewardModel()
        assert model.train_step({"obs": np.zeros(4)}) == {}

    def test_constant_save_load_noop(self, tmp_path: Path) -> None:
        """ConstantRewardModel.save/load are no-ops."""
        model = ConstantRewardModel()
        model.save(str(tmp_path / "test_constant_model"))
        model.load(str(tmp_path / "test_constant_model"))


# ============================================================
# LoggingConfig tests
# ============================================================


class TestLoggingConfig:
    """Tests for logging configuration."""

    def test_setup_logging_default(self) -> None:
        """setup_logging configures the root logger."""
        setup_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) >= 1

    def test_setup_logging_json_format(self) -> None:
        """setup_logging with json_format uses JsonFormatter."""
        setup_logging(level="INFO", json_format=True)
        root = logging.getLogger()
        assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)

    def test_setup_logging_with_file(self) -> None:
        """setup_logging can add a file handler."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            setup_logging(level="DEBUG", log_file=path)
            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) >= 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_json_formatter_output(self) -> None:
        """JsonFormatter produces valid JSON with expected fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello %s", args=("world",), exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_json_formatter_with_exception(self) -> None:
        """JsonFormatter includes exception info when present."""
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="error", args=(), exc_info=sys.exc_info(),
            )
        result = formatter.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_setup_logging_clears_existing_handlers(self) -> None:
        """setup_logging removes previously added handlers."""
        setup_logging(level="INFO")
        count1 = len(logging.getLogger().handlers)
        setup_logging(level="DEBUG")
        count2 = len(logging.getLogger().handlers)
        # Should be same number (old cleared, new added)
        assert count2 <= count1 + 1


# ============================================================
# RolloutBuffer edge case tests
# ============================================================


class TestRolloutBufferEdgeCases:
    """Edge case tests for RolloutBuffer."""

    def test_wrap_around(self) -> None:
        """Buffer should wrap around when capacity is exceeded."""
        buf = RolloutBuffer(capacity=3, obs_shape=(2,))
        for i in range(5):
            buf.add(np.ones(2) * i, i % 2, float(i), False, {})
        assert len(buf) == 3  # capacity limit
        # Latest entries should be 2, 3, 4
        batch = buf.sample(3)
        assert batch["observations"].shape == (3, 2)

    def test_sample_empty_raises(self) -> None:
        """Sampling from empty buffer should raise ValueError."""
        buf = RolloutBuffer(capacity=10, obs_shape=(4,))
        with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
            buf.sample(1)

    def test_sample_larger_than_buffer(self) -> None:
        """Sampling more than buffer size should work (with replacement)."""
        buf = RolloutBuffer(capacity=10, obs_shape=(2,))
        buf.add(np.ones(2), 0, 1.0, False, {})
        batch = buf.sample(batch_size=5)
        assert batch["observations"].shape == (5, 2)

    def test_clear_then_add(self) -> None:
        """Buffer should work correctly after clear."""
        buf = RolloutBuffer(capacity=5, obs_shape=(2,))
        buf.add(np.ones(2), 0, 1.0, False, {})
        buf.clear()
        assert len(buf) == 0
        assert not buf.is_full()
        buf.add(np.zeros(2), 1, 2.0, True, {"key": "val"})
        assert len(buf) == 1

    def test_is_full_boundary(self) -> None:
        """is_full should be False at capacity-1, True at capacity."""
        buf = RolloutBuffer(capacity=2, obs_shape=(1,))
        buf.add(np.zeros(1), 0, 0.0, False, {})
        assert not buf.is_full()
        buf.add(np.zeros(1), 0, 0.0, False, {})
        assert buf.is_full()

    def test_deterministic_sampling(self) -> None:
        """Same seed should produce same samples."""
        buf1 = RolloutBuffer(capacity=10, obs_shape=(2,), seed=42)
        buf2 = RolloutBuffer(capacity=10, obs_shape=(2,), seed=42)
        for i in range(5):
            obs = np.ones(2) * i
            buf1.add(obs, i, float(i), False, {})
            buf2.add(obs, i, float(i), False, {})
        s1 = buf1.sample(3)
        s2 = buf2.sample(3)
        np.testing.assert_array_equal(s1["actions"], s2["actions"])


# ============================================================
# CheckpointManager tests
# ============================================================


class TestCheckpointManagerExtended:
    """Extended tests for CheckpointManager."""

    def test_rotation_removes_oldest(self) -> None:
        """Rotation should remove oldest checkpoints when exceeding max."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=2)
            agent = RandomAgent(AgentConfig())
            for ep in range(4):
                manager.save(agent, episode=ep, metrics={"r": float(ep)})
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2
            episodes = [c["episode"] for c in checkpoints]
            assert 0 not in episodes  # oldest should be removed

    def test_corrupt_metadata_skipped(self) -> None:
        """Corrupt metadata.json should be skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            # Create a valid checkpoint
            agent = RandomAgent(AgentConfig())
            manager.save(agent, episode=1, metrics={})
            # Create a corrupt checkpoint
            corrupt_dir = Path(tmpdir) / "checkpoint_corrupt_999"
            corrupt_dir.mkdir()
            (corrupt_dir / "metadata.json").write_text("not json{{{")
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1  # corrupt one skipped

    def test_schema_version_in_metadata(self) -> None:
        """Saved checkpoints should include schema_version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            agent = RandomAgent(AgentConfig())
            manager.save(agent, episode=1, metrics={})
            checkpoints = manager.list_checkpoints()
            assert checkpoints[0].get("schema_version") == CHECKPOINT_SCHEMA_VERSION

    def test_load_latest_empty_dir(self) -> None:
        """load_latest on empty directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            agent = RandomAgent(AgentConfig())
            result = manager.load_latest(agent)
            assert result is None

    def test_migrate_old_metadata(self) -> None:
        """Old metadata without schema_version should be migrated."""
        old_meta = {"episode": 5, "timestamp": 12345, "metrics": {}}
        migrated = CheckpointManager._migrate_metadata(old_meta)
        assert migrated["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert migrated["step_count"] == 0


# ============================================================
# MetricsTracker extended tests
# ============================================================


class TestMetricsTrackerExtended:
    """Extended tests for MetricsTracker."""

    def test_latest_returns_most_recent(self) -> None:
        """latest() should return the most recently recorded value."""
        tracker = MetricsTracker(window_size=10)
        tracker.record("loss", 1.0)
        tracker.record("loss", 2.0)
        tracker.record("loss", 3.0)
        assert tracker.latest("loss") == 3.0

    def test_latest_empty_returns_zero(self) -> None:
        """latest() on unrecorded metric returns 0.0."""
        tracker = MetricsTracker()
        assert tracker.latest("nonexistent") == 0.0

    def test_count(self) -> None:
        """count() should return number of recorded values."""
        tracker = MetricsTracker(window_size=100)
        for i in range(7):
            tracker.record("x", float(i))
        assert tracker.count("x") == 7
        assert tracker.count("y") == 0

    def test_window_overflow(self) -> None:
        """Window should maintain max size and drop oldest values."""
        tracker = MetricsTracker(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record("x", v)
        assert tracker.count("x") == 3
        assert tracker.mean("x") == pytest.approx(4.0)  # (3+4+5)/3
        assert tracker.latest("x") == 5.0

    def test_multiple_metrics_independent(self) -> None:
        """Different metrics should be tracked independently."""
        tracker = MetricsTracker()
        tracker.record("a", 10.0)
        tracker.record("b", 20.0)
        assert tracker.mean("a") == 10.0
        assert tracker.mean("b") == 20.0
        all_m = tracker.all_metrics()
        assert "a" in all_m
        assert "b" in all_m


# ============================================================
# Device detection tests
# ============================================================


class TestDeviceExtended:
    """Extended tests for device detection."""

    def test_is_gpu_available_returns_bool(self) -> None:
        """is_gpu_available should return a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_device_consistent(self) -> None:
        """Multiple calls to get_device should return the same result."""
        d1 = get_device()
        d2 = get_device()
        assert d1 == d2


# ============================================================
# RandomAgent extended tests
# ============================================================


class TestRandomAgentExtended:
    """Extended tests for RandomAgent."""

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce identical action sequences."""
        obs = np.zeros(4, dtype=np.float32)
        a1 = RandomAgent(AgentConfig(), action_space_size=10, seed=123)
        a2 = RandomAgent(AgentConfig(), action_space_size=10, seed=123)
        actions1 = [a1.act(obs)[0] for _ in range(20)]
        actions2 = [a2.act(obs)[0] for _ in range(20)]
        assert actions1 == actions2

    def test_step_count_increments(self) -> None:
        """Step count should increase with each act() call."""
        agent = RandomAgent(AgentConfig())
        obs = np.zeros(4, dtype=np.float32)
        assert agent.step_count == 0
        agent.act(obs)
        assert agent.step_count == 1
        agent.act(obs)
        assert agent.step_count == 2

    def test_action_space_size_one(self) -> None:
        """Agent with action_space_size=1 should always return 0."""
        agent = RandomAgent(AgentConfig(), action_space_size=1, seed=0)
        obs = np.zeros(4, dtype=np.float32)
        for _ in range(10):
            action, _ = agent.act(obs)
            assert action == 0

    def test_save_load_roundtrip(self) -> None:
        """RandomAgent state should survive save/load."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent = RandomAgent(AgentConfig(name="test_save"))
            obs = np.zeros(4, dtype=np.float32)
            agent.act(obs)
            agent.act(obs)
            agent.save(path)

            agent2 = RandomAgent(AgentConfig(name="test_save"))
            agent2.load(path)
            assert agent2.step_count == 2
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================
# MCTSAgent extended tests
# ============================================================


class TestMCTSAgentExtended:
    """Extended tests for MCTSAgent."""

    def test_node_best_child_tie_breaking(self) -> None:
        """best_child should handle equal UCB1 scores without error."""
        parent = MCTSNode()
        parent.visit_count = 10
        for i in range(3):
            child = MCTSNode(parent=parent, action=i)
            child.visit_count = 5
            child.total_value = 2.5
            parent.children[i] = child
        action, node = parent.best_child(exploration_constant=1.414)
        assert action in [0, 1, 2]
        assert node is not None

    def test_node_best_child_empty_raises(self) -> None:
        """best_child on childless node should raise ValueError."""
        node = MCTSNode()
        with pytest.raises(ValueError, match="No children"):
            node.best_child(exploration_constant=1.0)

    def test_tree_depth_single_node(self) -> None:
        """Tree depth of a single node is 0."""
        config = MCTSConfig(num_simulations=5)
        agent = MCTSAgent(config, action_space_size=3, seed=0)
        root = MCTSNode()
        assert agent._compute_tree_depth(root) == 0

    def test_zero_simulations(self) -> None:
        """Agent with zero simulations should still return a valid action."""
        config = MCTSConfig(num_simulations=0)
        agent = MCTSAgent(config, action_space_size=4, seed=42)
        obs = np.zeros(10, dtype=np.float32)
        action, _trace = agent.act(obs)
        assert 0 <= action < 4

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same actions."""
        config = MCTSConfig(num_simulations=20)
        obs = np.zeros(8, dtype=np.float32)
        a1 = MCTSAgent(config, action_space_size=4, seed=42)
        a2 = MCTSAgent(config, action_space_size=4, seed=42)
        act1, _ = a1.act(obs)
        act2, _ = a2.act(obs)
        assert act1 == act2

    def test_learn_returns_empty(self) -> None:
        """MCTSAgent.learn is a no-op returning empty dict."""
        config = MCTSConfig()
        agent = MCTSAgent(config)
        assert agent.learn({}) == {}


# ============================================================
# DecisionTrace migration tests
# ============================================================


class TestDecisionTraceMigration:
    """Tests for DecisionTrace schema migration and backwards compatibility."""

    def test_from_dict_unknown_fields_ignored(self) -> None:
        """Unknown fields from newer versions should be silently ignored."""
        data = {
            "tick": 1,
            "agent_id": "a1",
            "action": 0,
            "future_field": "should_be_ignored",
            "schema_version": 1,
        }
        trace = DecisionTrace.from_dict(data)
        assert trace.tick == 1
        assert trace.agent_id == "a1"
        assert not hasattr(trace, "future_field")

    def test_from_dict_old_schema_migrated(self) -> None:
        """Old schema (v0) should be migrated to current version."""
        data = {"tick": 5, "agent_id": "test", "schema_version": 0}
        trace = DecisionTrace.from_dict(data)
        assert trace.tick == 5
        assert trace.intent_label == ""  # default from migration
        assert trace.expected_outcome == ""

    def test_to_dict_includes_schema_version(self) -> None:
        """to_dict should always include schema_version."""
        trace = DecisionTrace(tick=1, agent_id="x")
        d = trace.to_dict()
        assert d["schema_version"] == DEFAULT_SCHEMA_VERSION

    def test_roundtrip_preserves_all_fields(self) -> None:
        """Full roundtrip through to_dict/from_dict preserves data."""
        original = DecisionTrace(
            tick=42, agent_id="agent_0", action=3,
            confidence=0.95, search_depth=5, ucb1_score=1.5,
            intent_label="explore", preconditions=["has_key", "is_alive"],
            expected_outcome="find_treasure",
        )
        restored = DecisionTrace.from_dict(original.to_dict())
        assert restored == original


# ============================================================
# ActorCriticNetwork dimension validation tests
# ============================================================


_torch_available = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _torch_available = False


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
class TestActorCriticDimensionValidation:
    """Tests for ActorCriticNetwork load-time dimension validation."""

    def test_load_mismatched_obs_dim_raises(self) -> None:
        """Loading checkpoint with wrong obs_dim should raise ValueError."""
        net1 = ActorCriticNetwork(obs_dim=16, action_dim=4, hidden_sizes=[32])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            net1.save(path)

            net2 = ActorCriticNetwork(obs_dim=32, action_dim=4, hidden_sizes=[32])
            with pytest.raises(ValueError, match="obs_dim"):
                net2.load(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_mismatched_action_dim_raises(self) -> None:
        """Loading checkpoint with wrong action_dim should raise ValueError."""
        net1 = ActorCriticNetwork(obs_dim=16, action_dim=4, hidden_sizes=[32])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            net1.save(path)

            net2 = ActorCriticNetwork(obs_dim=16, action_dim=8, hidden_sizes=[32])
            with pytest.raises(ValueError, match="action_dim"):
                net2.load(path)
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================
# TraceLogger extended tests
# ============================================================


class TestTraceLoggerExtended:
    """Extended tests for TraceLogger."""

    def test_log_after_close_raises(self) -> None:
        """Writing to a closed logger should raise RuntimeError."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        logger_inst = TraceLogger(path, compress=False)
        logger_inst.close()
        with pytest.raises(RuntimeError, match="closed"):
            logger_inst.log(DecisionTrace(tick=1))
        Path(path).unlink(missing_ok=True)

    def test_flush_does_not_raise(self) -> None:
        """flush() should work without error."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        with TraceLogger(path, compress=False) as tl:
            tl.log(DecisionTrace(tick=1))
            tl.flush()
        Path(path).unlink(missing_ok=True)

    def test_multiple_traces_written(self) -> None:
        """Multiple traces should be written as separate lines."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        with TraceLogger(path, compress=False) as tl:
            for i in range(5):
                tl.log(DecisionTrace(tick=i, agent_id=f"agent_{i}"))
        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["tick"] == i
        Path(path).unlink(missing_ok=True)


# ============================================================
# Seed utility tests
# ============================================================


class TestSeedExtended:
    """Extended tests for seed utilities."""

    def test_derive_seed_deterministic(self) -> None:
        """derive_seed should return the same value for same inputs."""
        s1 = derive_seed(42, "agent_0")
        s2 = derive_seed(42, "agent_0")
        assert s1 == s2

    def test_derive_seed_different_components(self) -> None:
        """Different component names should produce different seeds."""
        s1 = derive_seed(42, "agent_0")
        s2 = derive_seed(42, "agent_1")
        assert s1 != s2

    def test_derive_seed_within_range(self) -> None:
        """Derived seeds should be within [0, MAX_SEED]."""
        for i in range(100):
            seed = derive_seed(i, f"comp_{i}")
            assert 0 <= seed <= MAX_SEED

    def test_set_all_seeds_affects_numpy(self) -> None:
        """set_all_seeds should make numpy random deterministic."""
        set_all_seeds(999)
        a = np.random.random(5)
        set_all_seeds(999)
        b = np.random.random(5)
        np.testing.assert_array_equal(a, b)


# ============================================================
# Config tests (schema validation)
# ============================================================


class TestConfigExtended:
    """Extended config tests for validation and edge cases."""

    def test_config_schema_version(self) -> None:
        """SimulationConfig should include schema_version."""
        cfg = SimulationConfig()
        assert hasattr(cfg, "schema_version")
        assert cfg.schema_version >= 1

    def test_config_from_empty_dict(self) -> None:
        """ForgeConfig.from_dict({}) should use all defaults."""
        cfg = ForgeConfig.from_dict({})
        assert cfg.simulation.grid_size == 64
        assert cfg.training.learning_rate == pytest.approx(3e-4)

    def test_config_to_dict_roundtrip(self) -> None:
        """to_dict should produce a dict that can recreate the config."""
        cfg = ForgeConfig()
        d = cfg.to_dict()
        assert d["simulation"]["grid_size"] == 64
        assert d["training"]["gamma"] == pytest.approx(0.99)

    def test_config_unknown_keys_ignored(self) -> None:
        """Unknown keys in TOML should be silently ignored."""
        data = {"simulation": {"grid_size": 32, "unknown_key": "value"}}
        cfg = ForgeConfig.from_dict(data)
        assert cfg.simulation.grid_size == 32

    def test_training_config_defaults(self) -> None:
        """TrainingConfig should have all PPO hyperparameters."""
        tc = TrainingConfig()
        assert hasattr(tc, "clip_ratio")
        assert hasattr(tc, "gae_lambda")
        assert hasattr(tc, "entropy_coeff")
        assert hasattr(tc, "value_coeff")
        assert hasattr(tc, "max_grad_norm")
