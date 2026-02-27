"""CleanRL-style PPO implementation for FORGE environments.

Single-file, research-friendly PPO following the CleanRL convention:
https://github.com/vwxyzjn/cleanrl

Every hyperparameter is exposed via CLI.  Metrics are written to TensorBoard
and optionally to W&B / CSV.

Usage::

    python examples/train_ppo_cleanrl.py --total-timesteps 100000 --seed 42
    python examples/train_ppo_cleanrl.py --total-timesteps 1000 --no-render  # CI smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import nn, optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment]
    TB_AVAILABLE = False

try:
    from forge_env.callbacks import CompositeCallback, ConsoleCallback, CsvCallback, EpisodeStats
    from forge_env.gymnasium_env import ForgeGymnasiumEnv
    from forge_env.wrappers import FlattenObservationWrapper, RecordEpisodeStatistics, TimeLimit

    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False
    logger.warning("forge_env not importable — run `maturin develop`")


# ---------------------------------------------------------------------------
# Hyperparameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """All CleanRL PPO hyperparameters (no hardcoded defaults in training code)."""

    # Environment
    world_size: int = 32
    num_agents: int = 1
    max_ep_steps: int = 500
    seed: int = 42

    # Training duration
    total_timesteps: int = 100_000
    num_envs: int = 1

    # PPO
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    """Number of steps per rollout per env."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("runs/cleanrl"))
    run_name: str | None = None
    no_render: bool = False
    log_every: int = 10

    # Logging
    wandb: bool = False
    wandb_project: str = "forge-cleanrl"
    csv: bool = True

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------


def make_network(obs_dim: int, act_dim: int) -> tuple[Any, Any]:
    """Build shared-backbone actor-critic network.

    Returns
    -------
    (actor, critic) as two separate ``nn.Sequential`` modules.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for CleanRL PPO — pip install 'forge-env[sb3]'")

    def layer_init(layer: Any, std: float = np.sqrt(2), bias: float = 0.0) -> Any:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
        return layer

    backbone = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
    )

    actor = nn.Sequential(backbone, layer_init(nn.Linear(256, act_dim), std=0.01))
    critic = nn.Sequential(backbone, layer_init(nn.Linear(256, 1), std=1.0))
    return actor, critic


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(args: Args) -> Any:
    if not FORGE_AVAILABLE:
        logger.error("forge_env not available; cannot create env.")
        return None

    config: dict[str, Any] = {
        "world": {"width": args.world_size, "height": args.world_size},
        "agents": {"num_agents": args.num_agents},
    }
    env = ForgeGymnasiumEnv(config=config, seed=args.seed)
    env = TimeLimit(env, max_steps=args.max_ep_steps)
    env = FlattenObservationWrapper(env)
    env = RecordEpisodeStatistics(env)
    return env


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: Args) -> None:
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for CleanRL PPO. Install: pip install 'forge-env[sb3]'")
        sys.exit(1)

    run_name = args.run_name or f"cleanrl_seed{args.seed}_{int(time.time())}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks: list[Any] = [ConsoleCallback(log_every=args.log_every)]
    if args.csv:
        callbacks.append(CsvCallback(output_path=args.output_dir / "metrics.csv"))
    composite = CompositeCallback(callbacks)
    composite.on_training_start()

    writer = SummaryWriter(str(args.output_dir / "tb")) if TB_AVAILABLE else None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(args)
    if env is None:
        sys.exit(1)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    actor, critic = make_network(obs_dim, act_dim)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    # Storage buffers
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim))
    acts_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs))
    rewards_buf = torch.zeros((args.num_steps, args.num_envs))
    dones_buf = torch.zeros((args.num_steps, args.num_envs))
    values_buf = torch.zeros((args.num_steps, args.num_envs))

    global_step = 0
    episode_index = 0
    num_updates = args.total_timesteps // args.batch_size
    start_time = time.monotonic()

    curr_obs, _ = env.reset(seed=args.seed)
    curr_obs_t = torch.tensor(curr_obs, dtype=torch.float32).unsqueeze(0)
    curr_done = torch.zeros(1)
    ep_return = 0.0
    ep_len = 0
    ep_start = time.monotonic()

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        frac = 1.0 - (update - 1) / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ── Rollout ──────────────────────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += 1
            obs_buf[step] = curr_obs_t
            dones_buf[step] = curr_done

            with torch.no_grad():
                logits = actor(curr_obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = critic(curr_obs_t).flatten()

            acts_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            ep_return += float(reward)
            ep_len += 1
            rewards_buf[step] = torch.tensor([reward], dtype=torch.float32)

            done = terminated or truncated
            curr_done = torch.tensor([float(done)])
            curr_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

            if done:
                eps = time.monotonic() - ep_start or 1e-6
                fps = ep_len / eps
                stats = EpisodeStats(
                    episode=episode_index,
                    total_steps=global_step,
                    episode_length=ep_len,
                    episode_return=ep_return,
                    fps=fps,
                )
                composite.on_episode_end(stats)
                if writer:
                    writer.add_scalar("train/ep_return", ep_return, global_step)
                    writer.add_scalar("train/ep_len", ep_len, global_step)
                    writer.add_scalar("train/fps", fps, global_step)

                episode_index += 1
                ep_return = 0.0
                ep_len = 0
                ep_start = time.monotonic()
                curr_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        # ── Advantage estimation (GAE) ───────────────────────────────────────
        with torch.no_grad():
            next_value = critic(curr_obs_t).flatten()
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - curr_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = (
                    rewards_buf[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                lastgaelam = float(
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        # ── PPO Update ───────────────────────────────────────────────────────
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_acts = acts_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advs = advantages.reshape(-1)
        b_rets = returns.reshape(-1)
        b_vals = values_buf.reshape(-1)

        pg_losses, v_losses, ent_losses = [], [], []

        inds = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = inds[start : start + args.minibatch_size]
                mb_obs = b_obs[mb_inds]
                mb_acts = b_acts[mb_inds]
                mb_advs = b_advs[mb_inds]
                if args.norm_adv:
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                logits = actor(mb_obs)
                dist = Categorical(logits=logits)
                logprob = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                logratio = logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                pg_loss = -torch.min(
                    ratio * mb_advs,
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advs,
                ).mean()

                new_val = critic(mb_obs).flatten()
                if args.clip_vloss:
                    v_clipped = b_vals[mb_inds] + torch.clamp(
                        new_val - b_vals[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (new_val - b_rets[mb_inds]) ** 2,
                        (v_clipped - b_rets[mb_inds]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * ((new_val - b_rets[mb_inds]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy.item())

        sps = int(global_step / (time.monotonic() - start_time + 1e-9))
        if not args.no_render and update % 10 == 0:
            logger.info(
                "update=%d  steps=%d  sps=%d  pg_loss=%.4f  v_loss=%.4f",
                update, global_step, sps,
                np.mean(pg_losses), np.mean(v_losses),
            )
        if writer:
            writer.add_scalar("train/policy_loss", np.mean(pg_losses), global_step)
            writer.add_scalar("train/value_loss", np.mean(v_losses), global_step)
            writer.add_scalar("train/entropy", np.mean(ent_losses), global_step)
            writer.add_scalar("train/sps", sps, global_step)

    # Save model
    ckpt = args.output_dir / "cleanrl_ppo.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, str(ckpt))
    logger.info("Checkpoint saved to %s", ckpt)

    env.close()
    composite.on_training_end()
    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--max-ep-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cleanrl"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="forge-cleanrl")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    ns = parser.parse_args(argv)

    return Args(
        world_size=ns.world_size,
        num_agents=ns.num_agents,
        max_ep_steps=ns.max_ep_steps,
        seed=ns.seed,
        total_timesteps=ns.total_timesteps,
        learning_rate=ns.learning_rate,
        num_steps=ns.num_steps,
        gamma=ns.gamma,
        gae_lambda=ns.gae_lambda,
        num_minibatches=ns.num_minibatches,
        update_epochs=ns.update_epochs,
        clip_coef=ns.clip_coef,
        ent_coef=ns.ent_coef,
        vf_coef=ns.vf_coef,
        output_dir=ns.output_dir,
        run_name=ns.run_name,
        no_render=ns.no_render,
        log_every=ns.log_every,
        wandb=ns.wandb,
        wandb_project=ns.wandb_project,
        csv=not ns.no_csv,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.DEBUG if False else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
