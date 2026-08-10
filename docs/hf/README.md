# Hugging Face publication pipelines

FORGE publishes three artifact classes to the Hugging Face Hub under the
`ianshank` namespace. All three pipelines run as GitHub Actions and share
one repository secret.

## One-time setup

Add a **write-scoped** Hugging Face token as the `HF_TOKEN` secret:
GitHub → Settings → Secrets and variables → Actions → New repository
secret. Every workflow below fails fast with a clear error when the
secret is missing.

## 1. Live demo Space — `ianshank/forge-wasm-demo`

`.github/workflows/hf-space.yml` builds `crates/forge-wasm` with
wasm-pack and mirrors the static demo (`web/index.html`, `web/app.js`,
`web/pkg/`, plus the Space card `web/space/README.md`) to a static-SDK
Space. Runs automatically on pushes to `main` that touch the demo or the
sim crates (same paths filter as `gh-pages.yml`), or manually via
workflow dispatch.

The demo is fully client-side (no server, no external requests); the
`.wasm` artifact is ~450 KB, far below any LFS threshold.

## 2. Trajectories dataset — `ianshank/forge-gridworld-trajectories`

`.github/workflows/hf-dataset.yml` (manual dispatch) generates the
dataset with the `forge-gen-dataset` bin and uploads it:

```bash
# Local equivalent:
cargo run --release -p forge-data --features hf --bin forge-gen-dataset -- \
    --out ./_hf_dataset --episodes-per-cell 185
```

- 54 configuration cells (world size × agents × tier × {mcts, random}),
  disjoint seed blocks, one procedurally generated task per episode.
- Output: Parquet shards + `dataset_info.json` + `export_manifest.json`,
  loadable via `datasets.load_dataset`.
- The Hub README is rendered from `docs/hf/dataset-card.md` with
  provenance (git SHA, row count) substituted at publish time.
- The workflow ends with a `load_dataset` round-trip that asserts the
  Hub row count matches the generated `dataset_info.json`.

The `hf` cargo feature (arrow/parquet) is off by default; the `hf-export`
CI job keeps it compiling and smoke-tested on every PR.

## 3. MuZero model bundle — `ianshank/forge-muzero-minecraft`

`.github/workflows/hf-model.yml` (manual dispatch) validates the model
release path: computes the canonical schema_id, runs
`python -m forge.training.muzero_mc.cli bootstrap` for a bundle, publishes
it with `scripts/hf_publish_model.py`, and round-trips it back through
`bootstrap --from-hf` to prove consumers can warm-start from the repo.

```bash
# Local equivalent (against any bundle dir with a model_manifest.json):
python scripts/hf_publish_model.py \
    --bundle-dir models/ --repo-id ianshank/forge-muzero-minecraft \
    --obs-dim 920 --action-dim 12 --private --dry-run
```

- Staged layout is exactly what `checkpoint_loader.load_from_hf`
  consumes: the three ONNX files **flat at the repo root** under their
  canonical names, plus a rebuilt `model_manifest.json` and a card
  rendered from `docs/hf/model-card.md`.
- Per-file SHA-256s are re-verified before staging; a corrupted bundle
  cannot be published.
- **The repo stays private and the card carries a random-init warning
  until real training lands** (v0.5 Phase 2). The public trained release
  is the same dispatch with `trained: true` + `private: false` against a
  trained bundle.
