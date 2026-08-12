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

{{TRAINED_WARNING}}

MuZero world-model bundle for the
[FORGE](https://github.com/ianshank/FORGE) self-improving Minecraft loop:
a Rust episode runner drives a live Minecraft environment over a
WebSocket bridge, records flat-tensor trajectories, trains this model in
Python, and hot-reloads the exported ONNX back into the runner's latent
MCTS between episodes.

## Files

The three MuZero networks are exported as separate ONNX graphs (opset
{{ONNX_OPSET}}) so the Rust runner can load them independently. The Rust
runner binds inputs/outputs **by name**:

| File | Inputs | Outputs |
|---|---|---|
| `representation.onnx` | `observation` `[B, obs_dim]` | `latent_state` `[B, latent_dim]` |
| `dynamics.onnx` | `latent_action` `[B, latent_dim + action_dim]` | `next_latent`, `reward_logits` |
| `prediction.onnx` | `latent_state` `[B, latent_dim]` | `policy_logits`, `value_logits` |

`model_manifest.json` records per-file SHA-256s, the bundle version, and
the environment `schema_id` (manifest schema_version {{MANIFEST_SCHEMA_VERSION}} —
byte-compatible with the Rust runner's `ModelManifest`).

## Contract

- **schema_id**: `{{SCHEMA_ID}}`
  (sha256 over the canonical action_map + rewards configs; the runner
  refuses bundles whose schema_id mismatches its env handshake)
- **obs_dim**: {{OBS_DIM}} · **action_dim**: {{ACTION_DIM}}
- **Bundle version**: {{VERSION}} · exported {{CREATED_AT}}

### Checksums

| Role | SHA-256 |
|---|---|
| representation | `{{REPR_SHA256}}` |
| dynamics | `{{DYN_SHA256}}` |
| prediction | `{{PRED_SHA256}}` |

## Usage — warm-start a FORGE bundle

```bash
pip install -e ".[minecraft]"
python -m forge.training.muzero_mc.cli bootstrap \
    --from-hf {{REPO_ID}} \
    --obs-dim {{OBS_DIM}} --action-dim {{ACTION_DIM}} \
    --schema-id {{SCHEMA_ID}} \
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
