---
license: apache-2.0
library_name: onnx
pipeline_tag: reinforcement-learning
tags:
  - forge
  - muzero
  - minecraft
  - onnx
  - model-based-rl
---

# FORGE MuZero (Minecraft) — ONNX bundle

__TRAINED_WARNING__

MuZero world-model bundle for the
[FORGE](https://github.com/ianshank/FORGE) self-improving Minecraft loop:
a Rust episode runner drives a live Minecraft environment over a
WebSocket bridge, records flat-tensor trajectories, trains this model in
Python, and hot-reloads the exported ONNX back into the runner's latent
MCTS between episodes.

## Files

The three MuZero networks are exported as separate ONNX graphs (opset
__ONNX_OPSET__) so the Rust runner can load them independently. The Rust
runner binds inputs/outputs **by name**:

| File | Inputs | Outputs |
|---|---|---|
| `representation.onnx` | `observation` `[B, obs_dim]` | `latent_state` `[B, latent_dim]` |
| `dynamics.onnx` | `latent_action` `[B, latent_dim + action_dim]` | `next_latent`, `reward_logits` |
| `prediction.onnx` | `latent_state` `[B, latent_dim]` | `policy_logits`, `value_logits` |

`model_manifest.json` records per-file SHA-256s, the bundle version, and
the environment `schema_id` (manifest schema_version __MANIFEST_SCHEMA_VERSION__ —
byte-compatible with the Rust runner's `ModelManifest`).

## Contract

- **schema_id**: `__SCHEMA_ID__`
  (sha256 over the canonical action_map + rewards configs; the runner
  refuses bundles whose schema_id mismatches its env handshake)
- **obs_dim**: __OBS_DIM__ · **action_dim**: __ACTION_DIM__
- **Bundle version**: __VERSION__ · exported __CREATED_AT__

### Checksums

| Role | SHA-256 |
|---|---|
| representation | `__REPR_SHA256__` |
| dynamics | `__DYN_SHA256__` |
| prediction | `__PRED_SHA256__` |

## Usage — warm-start a FORGE bundle

```bash
pip install -e ".[minecraft]"
python -m forge.training.muzero_mc.cli bootstrap \
    --from-hf __REPO_ID__ \
    --obs-dim __OBS_DIM__ --action-dim __ACTION_DIM__ \
    --schema-id __SCHEMA_ID__ \
    --out models/
```

The runner's `HotReloadWatcher` picks up the bundle between episodes; see
the repository's `docs/hf/README.md` for the full pipeline.

## Training configuration

Defaults from `python/forge/models/muzero_config.py`: latent_dim 256,
hidden_dim 256, 4 residual blocks, reward/value support 31,
discount 0.997, 5 unroll steps, TD-10, lr 3e-4. Observation layout:
11×11×1×7 block grid + 73-dim state vector → obs_dim 920.

Published by `scripts/hf_publish_model.py` from
[ianshank/FORGE](https://github.com/ianshank/FORGE).
