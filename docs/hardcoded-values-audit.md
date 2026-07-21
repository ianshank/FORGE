# FORGE — Hard-Coded Values Audit

Evidence for the "Configuration-driven operation — no hard-coded values" invariant
(CHARTER Invariant 5 / CLAUDE.md "No hard-coded values"). This is a point-in-time
inventory produced by a repo-wide sweep; regenerate it when the config surface
changes materially.

## Method

Swept Rust (`crates/*/src`), Python (`python/forge`), and TypeScript (`mc-bot/src`,
`dashboard/src`) for the categories most prone to hard-coding: network ports,
hosts/URLs, filesystem paths, timeouts/intervals, retention/limit sizes, and
observation/action dimensions. Each hit was triaged into:

- **Compliant** — the literal lives in a `Default` impl, a `#[serde(default=…)]`
  function, a named `const`/`Final`, or a frozen defaults object (the sanctioned
  config homes); or it is inside `#[cfg(test)]` / a doc example.
- **Stray** — a literal inlined in runtime logic with no config seam.

## Config homes (where values are supposed to live)

| Surface | Home |
|---|---|
| Rust shared constants | `crates/forge-types/src/constants.rs` (185 named `pub const`s) |
| Rust server | `crates/forge-server/src/config.rs` (`FORGE_SERVER_*` ladder) |
| Rust runner | `crates/forge-mc-runner/src/config.rs` (`RunnerConfig`, `FORGE_MC_*`) |
| Rust MC env | `crates/forge-env-mc/src/config.rs` (serde-default fns) |
| Python | module-level `Final` constants (e.g. `capture_baseline.py:DEFAULT_METRICS_URL`, `providers.py:DEFAULT_LMSTUDIO_BASE_URL`) |
| TS mc-bot | `mc-bot/src/config.ts` `DEFAULT_ENV_CONFIG` + `bot_manager.ts` `DEFAULT_*` constants |

## Findings

**The invariant is well honored.** Effectively every port/host/path/dim/timeout in
runtime code resolves to one of the homes above. Representative confirmations:

- Ports `8080/8765/9090/25565/3007` — all named consts (`forge-types/constants.rs`,
  `forge-cloud/constants.rs`) or `Default`/serde-default values
  (`forge-mc-runner/config.rs:287` `metrics_port: 9090` in `Default`;
  `forge-env-mc/config.rs:54` `ws://127.0.0.1:8765` in `default_ws_url()`).
- Observation dims (`920`, `73`, `7`, `11`) appear only in tests/doc-examples;
  production `obs_dim` is derived from config.
- mc-bot reconnect backoff/timeouts live in `DEFAULT_RECONNECT_CONFIG`; Python
  URLs/metrics endpoints are `Final` constants; Python sleeps are config-driven.

### Strays and their disposition

| Location | Value | Status |
|---|---|---|
| `mc-bot/src/bot_manager.ts` spawn timeout | `30000` ms, inlined in logic **and** the error string | **Fixed** — moved to `bot.spawn_timeout_ms` (default via `DEFAULT_SPAWN_TIMEOUT_MS`); error string built from the value. |
| LM Studio port/URL | duplicated between `.github/workflows/ci.yml` env and the Python provider default | **Open (minor)** — cross-file duplication of a value, not an in-logic hardcode. Candidate to source both from one place; low priority. |
| `python/forge/utils/dashboard_client.py` | `http://localhost:8080` in a module docstring/example | **Benign** — documentation only, not a runtime default. |

## Bottom line

No in-logic hard-coded runtime values remain after the mc-bot spawn-timeout fix.
The one open item is a cross-file duplication (CI env ↔ Python default), tracked as
low priority. Re-run the sweep and update this table when adding new config-bearing
literals.
