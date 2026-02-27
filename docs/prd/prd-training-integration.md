# PRD — E8/E9: Training Integration (SB3 + CleanRL)

**Epic Slug**: `training-integration`  
**Priority**: P0 (SB3) / P1 (CleanRL)  
**Sprint**: 4–5  
**Size**: M each

---

## User Story

> **As an** RL researcher evaluating FORGE as a training environment,  
> **I want to** run a working PPO training example with Stable Baselines 3 and CleanRL out of the box,  
> **So that** I can benchmark FORGE against established environments and validate my training pipeline before writing custom code.

---

## Problem Statement

`examples/train_ppo.py` exists but is not CI-validated. There are no learning curves, logged metrics, or verified convergence. Researchers evaluating FORGE cannot trust examples that are never run.

---

## Acceptance Criteria

### E8 — Stable Baselines 3 PPO

| # | Given | When | Then |
|---|---|---|---|
| AC1 | User installs `pip install forge-env[sb3]` | Environment installs | `stable-baselines3`, `torch`, and `forge-env` are all available |
| AC2 | User runs `python examples/train_ppo.py --timesteps 10000` | After 10K steps | Script completes without error; prints mean episode reward |
| AC3 | CI runs `train_ppo.py --timesteps 1000 --no-render` | On every PR to `main` | Smoke test passes in < 60 seconds |
| AC4 | Training run completes | With W&B disabled | Metrics logged to a local `runs/` CSV by default |

### E9 — CleanRL PPO

| # | Given | When | Then |
|---|---|---|---|
| AC5 | User runs `python examples/train_ppo_cleanrl.py --total-timesteps 10000` | With default config | Script completes without error |
| AC6 | CI runs CleanRL smoke test | On every PR | Passes in < 90 seconds |

### E10 — W&B / MLflow Hooks (P2 extension)

| # | Given | When | Then |
|---|---|---|---|
| AC7 | `WANDB_API_KEY` env var set | Running `train_ppo.py --wandb` | Metrics appear in W&B dashboard |
| AC8 | `MLFLOW_TRACKING_URI` set | Running with `--mlflow` flag | Runs logged to MLflow server |

---

## Out of Scope

- Pre-trained model checkpoints (E15)
- Hyperparameter tuning (post-beta)
- Distributed training

---

## Success Metrics

- `train_ppo.py` smoke test consistently passes in CI < 60s
- Mean reward > 0 after 10K steps (environment is learnable)
- Example scripts score ≥ 80% coverage in the test suite

---

## Open Questions

1. Should SB3 and CleanRL be separate files or configurable via a `--backend` flag?
2. What is the minimum timestep count for a "meaningful" smoke test? (recommend 1000)
3. Should we vendor a tiny test environment config or use full 64×64 world?

---

## Implementation Notes

- Add `[sb3]` and `[cleanrl]` extras to `pyproject.toml`
- Create `examples/train_ppo.py` with argparse (`--timesteps`, `--seed`, `--no-render`, `--wandb`)
- Add smoke test: `pytest tests/integration/test_training_smoke.py`
- Wrapper: `RecordEpisodeStatistics` already provides episode metrics; add a CSV writer callback
