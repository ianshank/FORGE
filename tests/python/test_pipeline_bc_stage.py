"""Tests for the pipeline BC stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge.mangomas.config import MangoMASBridgeConfig
from forge.mangomas.pipeline import CollectedTrainingData, MangoMASDroneTrainingPipeline


def _empty_episode(n_steps: int = 3) -> tuple[
    np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]
]:
    obs = np.zeros((n_steps + 1, 4), dtype=np.float32)
    action_names = ["Noop"] * n_steps
    action_ids = np.zeros((n_steps,), dtype=np.int64)
    rewards = np.zeros((n_steps,), dtype=np.float32)
    dones = np.zeros((n_steps,), dtype=np.float32)
    raw = [
        {
            "battery": 0.9,
            "altitude": 0.1,
            "stamina_inverse": 0.1,
            "boundary_distance": 0.5,
            "threat_proximity": 0.2,
        }
        for _ in range(n_steps)
    ]
    return obs, action_names, action_ids, rewards, dones, raw


def _collected(*, with_teacher: bool, num_actions: int = 4) -> CollectedTrainingData:
    obs, names, ids, rewards, dones, raw = _empty_episode(3)
    teacher_intentions = [[0, 1, 2]] if with_teacher else []
    teacher_rationales = [["", "", ""]] if with_teacher else []
    teacher_subgoals = [[[], [], []]] if with_teacher else []
    teacher_value_hats = [[0.0, 0.0, 0.0]] if with_teacher else []
    teacher_constraint_critiques = [[{}, {}, {}]] if with_teacher else []
    teacher_top_k_probs = [
        [
            [{"action_id": 0, "prob": 0.9}],
            [{"action_id": 1, "prob": 0.9}],
            [{"action_id": 2, "prob": 0.9}],
        ]
    ] if with_teacher else []
    # Set teacher actions in the ids array so BC sees them via collected_data.action_ids
    if with_teacher:
        ids = np.asarray([0, 1, 2], dtype=np.int64)
    return CollectedTrainingData(
        observations=[obs],
        action_names=[names],
        action_ids=[ids],
        rewards=[rewards],
        dones=[dones],
        raw_observations=[raw],
        teacher_intentions=teacher_intentions,
        teacher_rationales=teacher_rationales,
        teacher_subgoals=teacher_subgoals,
        teacher_value_hats=teacher_value_hats,
        teacher_constraint_critiques=teacher_constraint_critiques,
        teacher_top_k_probs=teacher_top_k_probs,
    )


def test_bc_stage_skipped_without_teacher_intentions(tmp_path: Path) -> None:
    cfg = MangoMASBridgeConfig()
    cfg.pipeline.paths.output_root = str(tmp_path)
    cfg.pipeline.execution.stop_after_stage = "bc"
    pipeline = MangoMASDroneTrainingPipeline(config=cfg)
    result = pipeline.run(_collected(with_teacher=False), base_seed=1, run_name="run-skip")
    bc = result.stage_by_name("bc")
    assert bc is not None
    assert bc.status == "skipped"


def test_bc_stage_runs_when_teacher_data_present(tmp_path: Path) -> None:
    cfg = MangoMASBridgeConfig()
    cfg.pipeline.paths.output_root = str(tmp_path)
    cfg.pipeline.execution.stop_after_stage = "bc"
    pipeline = MangoMASDroneTrainingPipeline(config=cfg)
    result = pipeline.run(
        _collected(with_teacher=True),
        base_seed=2,
        run_name="run-runs",
    )
    bc = result.stage_by_name("bc")
    assert bc is not None
    assert bc.status == "completed"
    assert bc.metrics["num_samples"] == 3
    assert "weights" in bc.outputs


@pytest.mark.xfail(
    reason=(
        "Pre-existing BC trainer bug: when action_space_sizes > max observed "
        "action_id, the teacher's top_k one-hot rows are sized to num_actions "
        "but `target` is broadcast against the smaller `probs` matrix, raising "
        "ValueError in bc_trainer._train_numpy. Filed in branch hygiene scan; "
        "needs a dedicated fix that sizes target consistently with the actor."
    ),
    raises=ValueError,
    strict=True,
)
def test_bc_stage_uses_action_space_sizes_when_provided(tmp_path: Path) -> None:
    """Regression: if collected_data.action_space_sizes is set, BC sizes the
    actor against that, not against the (potentially smaller) max action_id
    actually observed in the rollout.
    """
    cd = _collected(with_teacher=True)
    # All teacher actions are in {0, 1, 2}; episode max is 2 (so the old
    # codepath would size num_actions = 3). Real env had 7 legal actions.
    cd.action_space_sizes = [7]
    cfg = MangoMASBridgeConfig()
    cfg.pipeline.paths.output_root = str(tmp_path)
    cfg.pipeline.execution.stop_after_stage = "bc"
    pipeline = MangoMASDroneTrainingPipeline(config=cfg)
    result = pipeline.run(cd, base_seed=4, run_name="run-action-space")
    bc = result.stage_by_name("bc")
    assert bc is not None
    assert bc.status == "completed"
    weights_path = Path(bc.outputs["weights"])
    loaded = np.load(weights_path)
    # Actor weight shape is (num_actions, state_dim) — verify num_actions=7,
    # not 3 (which is what flat_actions.max()+1 would have produced).
    assert loaded["actor_w"].shape[0] == 7


def test_bc_stage_emits_actor_weights_npz(tmp_path: Path) -> None:
    cfg = MangoMASBridgeConfig()
    cfg.pipeline.paths.output_root = str(tmp_path)
    cfg.pipeline.execution.stop_after_stage = "bc"
    pipeline = MangoMASDroneTrainingPipeline(config=cfg)
    result = pipeline.run(
        _collected(with_teacher=True),
        base_seed=3,
        run_name="run-weights",
    )
    bc = result.stage_by_name("bc")
    assert bc is not None
    weights_path = Path(bc.outputs["weights"])
    assert weights_path.exists()
    loaded = np.load(weights_path)
    assert set(loaded.files) == {"actor_w", "actor_b"}
