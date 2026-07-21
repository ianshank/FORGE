# FORGE — Coverage Policy & How to Read It

> This file previously pinned a hand-picked 6-crate snapshot ("91.13%") that was
> **not** the number CI enforces and drifted from reality. It has been replaced
> with an honest description of the actual gates. **Live coverage numbers come
> from the CI `coverage` job's uploaded artifact, not from a committed file.**

## The gates that actually run (source of truth = CI, not this doc)

| Runtime | Gate | Where |
|---|---|---|
| Rust | `cargo tarpaulin --workspace --exclude forge-python --exclude forge-wasm --features forge-cloud/gcs --fail-under 85` | `.github/workflows/ci.yml` (`coverage` job) |
| Python | `pytest --cov=forge --cov=forge_env --cov-fail-under=85` | `ci.yml` (`python-test`), `.coveragerc` |
| Dashboard (Vitest) | 85% statements/branches/functions/lines | `dashboard/vite.config.ts` |
| mc-bot | see note ¹ | `mc-bot/package.json` |
| demo_ui | see note ² | `ci.yml` (`demo-ui`) |

¹ mc-bot: a `c8`-based coverage floor is being introduced (it owns first-class
  wire-contract logic — `schema_id.ts`, `protocol.ts`, `bot_manager.ts`).
² demo_ui: its pytest job runs without a `--cov-fail-under` floor; this is a
  known, deliberate gap pending a decision — not an accident.

## What the Rust gate does and does not measure

- **Excluded** from the tarpaulin run: `forge-python` (needs a Python runtime) and
  `forge-wasm` (wasm target). Everything else in the workspace is measured as a
  single aggregate against the 85% floor.
- The number is an **aggregate over the measured crates**, so per-crate figures are
  not enforced individually. Removing a well-tested crate can *lower* the aggregate
  and a poorly-tested one can *raise* it — treat the aggregate accordingly.

## Getting the current numbers locally

Mirror the CI invocation exactly (install a tarpaulin version that builds under the
current toolchain first):

```sh
cargo install cargo-tarpaulin --locked
cargo tarpaulin --workspace --exclude forge-python --exclude forge-wasm \
  --features forge-cloud/gcs --skip-clean --fail-under 85 --print-summary
```

The HTML/XML reports (`tarpaulin-report.html`, `cobertura.xml`) are build
side-effects and are **git-ignored** — download them from the CI run's
`coverage-report` artifact rather than committing them.
