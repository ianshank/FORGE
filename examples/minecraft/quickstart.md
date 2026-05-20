# Minecraft RL Quickstart

End-to-end walkthrough for the FORGE Minecraft integration: a real
Minecraft server, the `mc-bot` mineflayer bridge, and the Rust
`forge-mc-runner` driving latent-MCTS episodes — all wired together
via `docker compose`.

## Prerequisites

- Docker 24+ with the `compose` plugin (`docker compose version`).
- Linux/macOS/Windows with WSL2.
- ~4 GB of free disk for the Minecraft server image + world data.
- For local Python tooling (bootstrap / validate-manifest): Python 3.11+
  with this repo installed in editable mode and the `minecraft` extras:

  ```sh
  pip install -e ".[minecraft]"
  ```

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
random-init bundle:

```sh
python -m forge.training.muzero_mc.cli bootstrap \
    --obs-dim 920 \
    --action-dim 12 \
    --schema-id "$(cat configs/minecraft/schema_id.txt)" \
    --out models/
```

`--schema-id` must equal the sha256 the mc-bot advertises in its
`Hello` handshake (computed from `action_map.toml` + `rewards.toml` +
`obs_dim` + `action_count`). The bot prints it on startup and pins it
in `configs/minecraft/schema_id.txt` after the first run.

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
episodes and applies the swap atomically.

The full trainer loop (load `TrajectoryV2` batches → train rep/dyn/pred
nets → export ONNX → bump manifest) ships in a follow-up to
`forge.training.muzero_mc`; the bootstrap CLI above produces a
v1 bundle that demonstrates the full hot-reload path end-to-end.

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
