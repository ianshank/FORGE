# Minecraft RL Quickstart

End-to-end walkthrough for the FORGE Minecraft integration: a real
Minecraft server, the `mc-bot` mineflayer bridge, the Rust
`forge-mc-runner` driving latent-MCTS episodes, and (v0.4+) the
**Python trainer continuously consuming runner-emitted trajectories +
bumping the manifest the runner hot-reloads** — all wired together
via `docker compose`.

## v0.4 happy path: one-command self-play

If you just want the **self-improving loop** (runner playing + trainer
training simultaneously), skip steps 2-3 and use:

```sh
# Accept Mojang's EULA (see §1 below)
cp docker/compose.minecraft.env.example docker/compose.minecraft.env
# edit and set MC_EULA=TRUE

# Bring up the full self-play stack (CPU)
scripts/mc_self_play.sh --detach

# CUDA host with nvidia-container-toolkit:
scripts/mc_self_play.sh --gpu --detach

# Tear down:
scripts/mc_self_play.sh --down
```

`mc_self_play.sh` automates §2 (bootstrap) + §3 (compose-up) for you
via the `trainer-bootstrap` one-shot container — your host needs
**only Docker Compose v2.20+**, no local Python / torch install.

The longer-form sections below are for operators who want to drive
the runner without the trainer (v0.3-pre flow) or who need to override
specific steps.

## Prerequisites

- Docker 24+ with the `compose` plugin (`docker compose version`).
- Linux/macOS/Windows with WSL2.
- ~4 GB of free disk for the Minecraft server image + world data.
- For local Python tooling (bootstrap / validate-manifest / `train`):
  Python 3.11+ with this repo installed in editable mode and the
  `minecraft` extras:

  ```sh
  pip install -e ".[minecraft]"
  ```

  v0.4: NOT needed for `mc_self_play.sh` — the orchestrator runs
  bootstrap inside a container.

## 1. Accept the Minecraft EULA

The compose stack **will not start** until you accept Mojang's EULA.
Read it at <https://www.minecraft.net/en-us/eula>, then:

```sh
cp docker/compose.minecraft.env.example docker/compose.minecraft.env
# Edit docker/compose.minecraft.env and set:
#   MC_EULA=TRUE
```

The env file also lets you override the Minecraft version
(`MC_VERSION=1.20.4`), memory (`MC_MEMORY=2G`), and ports without
editing the compose YAML.

## 2. Bootstrap a model bundle

The Rust runner refuses to start without a `model_manifest.json` and
the three ONNX files it references. The bootstrap CLI writes a
random-init bundle. v0.4 emits the **atomic versioned layout**:

```
models/
├── model_manifest.json     (points at v00000001)
└── v00000001/
    ├── representation.onnx
    ├── dynamics.onnx
    └── prediction.onnx
```

```sh
# v0.4: compute schema_id directly from the shipped configs
# (no need to scrape the bot's startup log):
SCHEMA_ID=$(python -m forge.training.muzero_mc.cli compute-schema-id \
    --action-map configs/minecraft/action_map.toml \
    --rewards configs/minecraft/rewards.toml \
    --quiet)

python -m forge.training.muzero_mc.cli bootstrap \
    --obs-dim 920 \
    --action-dim 12 \
    --schema-id "$SCHEMA_ID" \
    --out models/
```

`--schema-id` must equal the sha256 the mc-bot advertises in its
`Hello` handshake (computed from `action_map.toml` + `rewards.toml`).
`compute-schema-id --quiet` derives it deterministically from the
TOMLs so the runner's startup cross-check passes on the first try.

Validate the bundle before bringing the stack up:

```sh
python -m forge.training.muzero_mc.cli validate-manifest models/
```

Exit code `0` means the Rust runner will accept it.

## 3. Bring the stack up

Foreground (logs stream to your terminal; Ctrl-C runs `compose down`):

```sh
scripts/mc_run.sh --build
```

Detached:

```sh
scripts/mc_run.sh --detach --build
scripts/mc_run.sh --down   # tear down
```

`--dry-run` prints the resolved `docker compose` invocation without
touching the host, useful for CI debugging:

```sh
scripts/mc_run.sh --dry-run
```

## 4. Watch the bot

prismarine-viewer renders the bot's first-person view in your browser:

<http://localhost:3007>

(Port comes from `MC_BOT_VIEWER_PORT` in the env file.)

## 5. Inspect outputs

The runner writes one trajectory per episode under `trajectories/`:

```sh
ls trajectories/
# ep-000001.json  ep-000002.json  ...
```

Each file is a `TrajectoryV2` JSON document — the same shape the
Python `forge.training.muzero_mc.replay.TrajectoryReader` consumes.
A quick sanity check:

```sh
python -c "
from forge.training.muzero_mc.replay import TrajectoryReader
r = TrajectoryReader('trajectories', batch_size=128)
total = sum(len(b) for b in r)
print(f'{total} steps across {len(list(r.episode_paths()))} episodes')
"
```

## 6. Iterating on the model

When you want to swap in fresh weights, write a new manifest into
`models/model_manifest.json` with a **strictly greater** `version` than
the one the runner last saw. The `HotReloadWatcher` polls between
episodes and applies the swap atomically. v0.4 layout: each bundle
version lives in its own `v{NNNNNNNN}/` subdir; the manifest's per-
role `path` field carries the prefix and the runner's
`config_from_manifest` resolves it transparently against `bundle_dir`.

The **full trainer loop** (load `TrajectoryV2` batches → train
rep/dyn/pred nets → export atomic versioned ONNX bundle → bump
manifest) ships in `python/forge/training/muzero_mc/trainer.py` as
of **v0.3-pre** (`MuzeroMcTrainer.train` for fixed-iter mode) and
**v0.4** (`MuzeroMcTrainer.train_continuous` for the self-improving
loop). Use the `train` CLI subcommand:

```sh
python -m forge.training.muzero_mc.cli train \
    --input trajectories/ \
    --out models/ \
    --schema-id "$SCHEMA_ID" \
    --obs-dim 920 --action-dim 12 \
    --continuous \
    --round-iters 10 \
    --round-poll-sleep 5 \
    --max-trajectories 200 \
    --max-bundle-versions 5 \
    --device cpu
```

Or — recommended — use `scripts/mc_self_play.sh` which wraps all of
this in a single command (see top of file).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `minecraft` service exits with EULA message | `MC_EULA=FALSE` in env file |
| `mc-bot` healthcheck fails | Bot can't reach `minecraft:25565` — wait for server to finish world-gen (90s+ on first boot) |
| Runner exits with schema_id mismatch | `--schema-id` passed to bootstrap doesn't match what mc-bot advertises in `Hello` |
| Runner refuses to start ("manifest not found") | `models/model_manifest.json` missing — re-run step 2 |
| Viewer at :3007 shows nothing | Bot not yet logged in to MC; check `docker logs forge-mc-bot` |

## Where things live

| Path | Owner | Purpose |
|---|---|---|
| `docker/compose.minecraft.yml` | this PR | Compose definition (Phase 6) |
| `docker/mc-bot.Dockerfile` | this PR | mc-bot container image (Phase 6) |
| `docker/compose.minecraft.env.example` | this PR | Sample env values |
| `scripts/mc_run.sh` | this PR | Idempotent orchestration entry point |
| `mc-bot/src/index.js` | PR #56 | Mineflayer bridge + WS server (Phase 3) |
| `crates/forge-env-mc/src/mc_env.rs` | PR #56 | Rust WS client + `Env` impl (Phase 3) |
| `crates/forge-mc-runner/src/runner.rs` | preceding commit | Episode loop (Phase 4) |
| `python/forge/training/muzero_mc/bootstrap.py` | preceding commit | Random-init bundle generator (Phase 5) |
| `configs/minecraft/*.toml` | PR #56 | Action map, reward config, env settings |
