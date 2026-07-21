# Contributing to FORGE

Thanks for your interest in FORGE. This guide covers the practical entrypoints;
read [`docs/CHARTER.md`](docs/CHARTER.md) first for the mission, scope boundaries,
and the Seven Core Invariants every change must preserve, and
[`CLAUDE.md`](CLAUDE.md) for day-to-day build/test conventions.

## Prerequisites

- **Rust** — the toolchain is pinned via `rust-toolchain.toml` (channel `stable`
  with the `rustfmt`, `clippy` components and the `wasm32-unknown-unknown` target).
  The declared MSRV is in `Cargo.toml` (`rust-version`).
- **Python** 3.11 recommended (`requires-python >= 3.9`), with `maturin` to build
  the native `forge_env` extension.
- **Node** 22+ for `mc-bot/` and `dashboard/`.

## Build & test (the CI gates, run them locally)

```sh
# Rust
cargo build --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets --features forge-cloud/gcs -- -D warnings
cargo test --workspace --features forge-cloud/gcs

# Rust coverage (85% workspace floor; install a tarpaulin that builds under the pin)
cargo tarpaulin --workspace --exclude forge-python --exclude forge-wasm \
  --features forge-cloud/gcs --skip-clean --fail-under 85

# Supply chain (advisory)
cargo deny check

# Python (build the native ext first with `maturin develop`)
ruff check
mypy --config-file pyproject.toml
pytest tests/python -m 'not lmstudio and not e2e_long and not minecraft_e2e'

# mc-bot
cd mc-bot && npm ci && npm run typecheck && npm run lint && npm test
```

More task-specific commands (benchmarks, the visualization server, the Minecraft
self-play stack) are listed in [`CLAUDE.md`](CLAUDE.md).

## Coverage gates by runtime

| Runtime | Floor |
|---|---|
| Rust (tarpaulin, excl. forge-python/forge-wasm) | 85% |
| Python (pytest-cov) | 85% |
| dashboard (Vitest) | 85% |
| mc-bot (c8) | being introduced |
| demo_ui | ungated (known gap) |

## Conventions

- Run `cargo fmt` before committing; all public items need doc comments; errors use
  `thiserror`; log via `tracing`, not `println!`.
- **No hard-coded values** — constants flow through config structs / `Default` impls
  / env vars (see [`docs/hardcoded-values-audit.md`](docs/hardcoded-values-audit.md)).
- Add **property tests** (`proptest`) for invariants alongside unit tests.
- JS/TS: ESM modules, Node 22+, `node:test`, Biome for lint/format, no `eval`.
- Cross-language config (e.g. `configs/minecraft/*.toml`) must keep the paired
  `schema_id` tests (Rust ↔ JS ↔ Python) in agreement.

## Branch & PR workflow

1. Branch from the default branch (feature branches under a `claude/**` or
   `feature/**` prefix are picked up by CI automatically).
2. Keep PRs focused; separate a CI/infra hotfix from a large refactor.
3. Fill in the pull-request template, including the verification checklist.
4. CI must be green. Open as a draft while iterating.
5. If a change crosses a scope boundary or bends an invariant, say so in the PR and
   amend the charter deliberately (add a Deliberate Exception with rationale) — do
   not silently rewrite scope.

## Security

Report vulnerabilities privately — see [`SECURITY.md`](SECURITY.md). Never commit
secrets; `.env*` files are git-ignored.
