# Contributing to FORGE

Thanks for your interest in FORGE. This guide covers the practical entrypoints;
read [`docs/CHARTER.md`](docs/CHARTER.md) first for the mission, scope boundaries,
and the Seven Core Invariants every change must preserve, and
[`CLAUDE.md`](CLAUDE.md) for day-to-day build/test conventions.

## Prerequisites

- **Rust** — CI installs the `stable` toolchain via `dtolnay/rust-toolchain@stable`,
  adding the `rustfmt`/`clippy` components (and the `wasm32-unknown-unknown` target
  for the WASM demo) per workflow. Locally, install `stable` with those components.
  The declared MSRV floor is in `Cargo.toml` (`rust-version`): the workspace needs
  cargo/rustc **≥ 1.85** because of edition-2024 dependencies.
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

# WebAssembly target (crates/forge-wasm -> the in-browser demo in web/).
# `rust-toolchain.toml` lists wasm32-unknown-unknown, so rustup installs it
# for you. wasm-test additionally needs wasm-pack (scripts/install_wasm_pack.sh).
make wasm-check   # clippy on wasm32 -- part of `make verify`
make wasm-test    # #[wasm_bindgen_test]s under Node -- NOT part of `make verify`
make web-e2e      # unit + Playwright E2E for web/ -- also NOT part of `make verify` (needs a Chromium download)

# Rust coverage (85% workspace floor; install a tarpaulin that builds under the pin)
cargo tarpaulin --workspace --exclude forge-python --exclude forge-wasm \
  --features forge-cloud/gcs --skip-clean --fail-under 85

# ONNX feature surface (onnx/onnx-reload/mc-live-bundled; not covered by the
# workspace-wide commands above, which only build --features forge-cloud/gcs).
# Needs a real ONNX Runtime >=1.23.2 .so + ORT_DYLIB_PATH set -- see the
# `onnx-features` CI job in .github/workflows/ci.yml for the exact fetch steps.
cargo clippy -p forge-agent --all-targets --features onnx-bundled -- -D warnings
cargo test -p forge-agent --features onnx-bundled
cargo test -p forge-mc-runner --features mc-live-bundled

# Supply chain (advisory)
cargo deny check
gitleaks git --redact -v .   # secret scanning; needs the gitleaks binary on PATH

# Python (build the native ext first with `maturin develop`)
ruff check
mypy python/ scripts/ --config-file pyproject.toml
pytest tests/python -m 'not lmstudio and not e2e_long and not minecraft_e2e'

# mc-bot
cd mc-bot && npm ci && npm run typecheck && npm run lint && npm test && npm run test:coverage

# dashboard
cd dashboard && npm ci && npm run build && npm run lint && npm run test:coverage

# Claude Code tooling (hooks/skills self-checks; see "Claude Code tooling" below)
python3 -m unittest discover -s .claude/hooks -p 'test_*.py' -v

# Pinned-config consistency (Rust toolchain / ONNX Runtime / LM Studio endpoint, duplicated across workflows/Dockerfiles/Python)
python3 scripts/check_pinned_config_consistency.py
```

More task-specific commands (benchmarks, the visualization server, the Minecraft
self-play stack) are listed in [`CLAUDE.md`](CLAUDE.md). Or run `make verify` to
execute this whole sequence at once (see the root `Makefile`).

## Coverage gates by runtime

| Runtime | Floor |
|---|---|
| Rust (tarpaulin, excl. forge-python/forge-wasm) | 85% |
| Python (pytest-cov) | 85% |
| dashboard (Vitest) | 85% |
| mc-bot (c8) | report-only (baseline ~88%; no fail-under yet) |
| demo_ui (pytest-cov) | 70% (`demo_ui/pytest.ini` `--cov-fail-under=70`; baseline ~90% on `demo_ui/backend`) |

## Claude Code tooling

Repo-checked-in, applies to any Claude Code session opened here (`.claude/`,
no per-user setup needed):

- **`/forge-verify` skill** (`.claude/skills/forge-verify/SKILL.md`) — runs
  `make verify` / `make verify-full` and reports a per-category pass/fail
  summary instead of a single opaque result.
- **`forge-docs-audit` skill** (`.claude/skills/forge-docs-audit/SKILL.md`)
  — re-verifies factual claims in `docs/next_steps.md`, `CHANGELOG.md`,
  `README.md`, `docs/architecture.md`, `Agent.md`, and `CLAUDE.md`
  against the actual codebase (counts, file:line references, named
  tests, status markers) and corrects drift in the house style already
  established in those files, rather than each pass re-deriving the same
  evidence by hand. Packages the single most-repeated pattern in this
  repo's own history — multiple `docs: fix stale ...` commits and several
  Technical Debt rows that turned out to be false when re-checked.
- **`forge-pr-review` skill** (`.claude/skills/forge-pr-review/SKILL.md`)
  — dispatches five parallel adversarial-review subagents against a PR or
  diff (hardcoded values/modularity, dead/redundant code, branch-coverage
  and mutation-style test quality, independent GitHub CI/review-status
  re-verification, correctness/security), triages every finding as fixed
  or declined-with-reason, then pushes and updates the PR body. Run once
  `forge-verify` is already green — it hunts for what still runs *wrong*,
  not whether it runs at all.
- **Tracked-file deletion guard** (`.claude/hooks/guard_tracked_deletion.py`,
  wired via `.claude/settings.json`'s `PreToolUse` hook) — blocks a `Bash`
  `rm`/`find -delete` command whose glob pattern matches a *git-tracked*
  file, not just the generated/ignored ones it was presumably aimed at
  (cross-checked against `git ls-files`, so it needs no hand-maintained
  path list and can't drift). Fails open on any parse or git error — it
  must never be the reason a legitimate command can't run. Added after a
  real incident where a `find … -name ".coverage*" -delete` cleanup swept
  up the tracked `.coveragerc` along with coverage.py's temp files.
  Self-tests: `make hooks-test` (also runs as part of `make verify` and in
  CI's `python-lint` job).
- **Staged-secret guard** (`.claude/hooks/guard_staged_secrets.py`, same
  `PreToolUse`/`Bash` wiring) — runs `gitleaks protect --staged` before a
  `git commit` actually happens, blocking on a real finding. Closes a gap
  the security workflow's own comments admit: its `gitleaks` job scans
  *history* after a push and is explicitly report-only (`|| true`) —
  nothing previously scanned *before* a commit. Not a replacement for that
  job (still the full-history backstop), just an earlier, cheaper
  checkpoint. Fails open if `gitleaks` isn't installed locally — this
  hook is best-effort, not a hard requirement to commit. Self-tests:
  `python3 .claude/hooks/test_guard_staged_secrets.py -v` (the two cases
  needing a real `gitleaks` binary skip themselves without one, so `make
  hooks-test` still exercises the rest everywhere).
- **Line-ending drift guard** (`.claude/hooks/guard_line_ending_drift.py`,
  same `PreToolUse`/`Bash` wiring) — blocks a `git commit` that flips more
  than half of an already-tracked file's lines between CRLF and LF (or
  back), the signature of a whole-file EOL rewrite rather than a real
  content edit. Added after a real incident on this repo's own WASM-E2E
  branch: a Python `open(p).read()` / `open(p, 'w').write(s)` round-trip
  silently flattened `CHANGELOG.md` from CRLF to LF, turning a ~40-line
  edit into a ~2000-line rewrite that nothing caught until an abnormally
  large `git diff --stat` after the fact. Compares the CRLF-line fraction
  of the committed blob (`HEAD`) against the staged blob (the index);
  skips brand-new files (nothing to drift from) and any path with an
  explicit `.gitattributes` line-ending declaration (`-text` or `eol=`),
  which records a deliberate choice rather than drift. Fails open on any
  git or parse error. Self-tests:
  `python3 .claude/hooks/test_guard_line_ending_drift.py -v` (stdlib +
  `git` only, no external binary — always runs as part of `make
  hooks-test`).
- **Minecraft schema_id pin reminder** (`.claude/hooks/guard_schema_id_pins.py`,
  same `PreToolUse`/`Bash` wiring) — advisory (always exit 0). When a
  `git commit` stages `configs/minecraft/{action_map,rewards,milestone_rewards,crafting_rewards,block_embeddings}.toml`,
  prints a reminder to bump the Rust/JS/Python xlang pins together.
  Self-tests: `python3 .claude/hooks/test_guard_schema_id_pins.py -v`.
- **CompactReplay golden reminder** (`.claude/hooks/guard_golden_replay.py`,
  same `PreToolUse`/`Bash` wiring) — advisory (always exit 0). When a
  `git commit` stages `configs/scenarios/*.toml` or
  `crates/forge-types/src/{config,constants}.rs` without also staging
  `tests/golden/replays/` and `docs/results/replay_flip_log.md`, prints a
  reminder to regenerate the v2 corpus.
  Self-tests: `python3 .claude/hooks/test_guard_golden_replay.py -v`.

## Conventions

- Run `cargo fmt` before committing; all public items need doc comments; errors use
  `thiserror`; log via `tracing`, not `println!`.
- **No hard-coded values** — constants flow through config structs / `Default` impls
  / env vars (see [`docs/hardcoded-values-audit.md`](docs/hardcoded-values-audit.md)).
- Add **property tests** (`proptest`) for invariants alongside unit tests.
- JS/TS: ESM modules, Node 22+, `node:test`, Biome for lint/format, no `eval`.
- Cross-language config (e.g. `configs/minecraft/*.toml`) must keep the paired
  `schema_id` tests (Rust ↔ JS ↔ Python) in agreement. Nested
  `config_path` / `crafting_config_path` **contents** fold into the shipped
  rewards pin (`xlang_shipped_rewards_schema_id_folds_nested_files`);
  `block_embeddings.toml` is a separate obs-layout pin. Changing those
  TOML files without bumping all three language pins fails CI. An advisory
  PreToolUse hook (`.claude/hooks/guard_schema_id_pins.py`, wired in
  `.claude/settings.json`) prints a reminder on `git commit` when they are
  staged — it does **not** block; `make hooks-test` covers it.

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
