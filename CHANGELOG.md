# Changelog

All notable changes to FORGE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Enterprise Codebase Optimization & Architectural Hardening (2026-09)

Completed full implementation of the 5-phase optimization and enterprise hardening master plan:

- **Strict Tier L0–L5 Layering Enforced in CI**:
  - Formalized all 26 workspace crates into 6 acyclic architectural tiers in `docs/architecture.md` §4.2.
  - Enforced unidirectional dependency constraints in `deny.toml` via `[bans].deny` with explicit `wrappers = [...]` rules, verified on every push via `cargo deny --all-features check`.
- **God File Elimination & Modularization**:
  - Extracted 3,900+ lines of in-file unit/proptest suites out of `world.rs`, `physics.rs`, `config.rs`, and `action.rs` into dedicated `tests.rs` submodules, reducing core production files by 50%–65%.
  - Modularized `crates/forge-core::world` into cohesive submodules (`state.rs`, `step.rs`, `reset.rs`, `observation.rs`, `serialize.rs`, `debug.rs`) while strictly maintaining the zero-allocation hot-path contract.
  - Extracted in-file tests from `crates/forge-core/src/systems.rs` into `systems/tests.rs`.
  - Extracted in-file tests from `crates/forge-eval/src/harness.rs`, `crates/forge-eval/src/exporters/mlflow_http.rs`, `crates/forge-mc-runner/src/runner.rs`, and `crates/forge-task/src/predicate.rs` via the same `#[path]` pattern (production files no longer host 600–770 line test modules).
  - Centralized canonical discrete action constants and decoders in root-level `python/forge/actions.py`, decoupling circular import loops between `forge.agents` and `forge.mangomas`.
  - Added shared configuration-driven skill catalog (`configs/agents/skills_default.toml`) mirrored identically across Rust (`forge-types::skill::SkillsConfig`, `forge-agent::skills::HierarchicalSkillAgent`) and Python (`forge.agents.skills::SkillCatalog`, `forge.agents.skills::HierarchicalSkillPolicy`).
  - Added centralized Python policy names (`forge.policy_names`) and wired `--collection-policy skill` across the MangoMAS training and collection pipeline.
  - Decomposed `python/forge/mangomas/collector.py` (1,476 LOC monolith) into `python/forge/mangomas/collector/` sub-packages (`types`, `action_decoder`, `scenario`, `writer`, `sync_rollout`, `async_rollout`).
  - Decomposed `mc-bot/src/index.ts` (616 LOC monolith) into focused ESM modules (`auth.ts`, `connection.ts`, `server.ts`, and a minimal `index.ts` bootstrap <100 LOC).
- **Test Coverage & Quality Gate Ratcheting**:
  - Established a strict 70% coverage floor for `demo_ui/backend` in `demo_ui/pytest.ini` (`--cov-fail-under=70`) and `demo_ui/.coveragerc` (current baseline: 90.38%).
  - Pinned derive-only false positives in `[package.metadata.cargo-machete]` across 21 crates and promoted `cargo-machete` from advisory to a blocking CI gate.
  - Promoted `pip-audit` to a blocking check in `.github/workflows/security.yml`.
  - Fixed Docker ABI mismatch in `docker/Dockerfile` by aligning runtime stage to Python 3.11.
- **Config Hardening & Zero-Drift**:
  - Added `#[serde(default, deny_unknown_fields)]` across `ForgeConfig` and all 11 child structs in `crates/forge-types/src/config.rs` to prevent silent config drift.
  - Added `FORGE_MC_METRICS_URL` environment variable override for Prometheus endpoint in `capture_baseline.py` and CLI.
  - Added `FORGE_PLOTLY_JS_URL` environment variable support in `forge-eval::mlflow_payload` for air-gapped evaluation artifact generation.
  - Authored comprehensive configuration index in `docs/config-catalog.md`.
- **Research Stack Governance & Benchmarks**:
  - Added standardized `README.md` files to all 26 crates across `crates/`.
  - Added explicit operational maturity badges (`[Production]`, `[Research]`, `[Experimental]`) across research stack crates.
  - Wired `forge-env` and `forge-env-forge` into `forge-bench` allocation audit and throughput benchmark suites to verify zero-allocation contracts across generic `Env` implementations.
  - Formalized Tiered API Stability Policy and the Bincode 2.x Wire-Format Migration Roadmap in `docs/CHARTER.md`.
- **Hierarchical Skills over Reusable Actions**:
  - Added `forge_types::skill::{SkillCategory, SkillSpec, SkillsConfig}` (options/HRL catalog, `deny_unknown_fields`, env overrides).
  - Wired `forge_agent::skills::HierarchicalSkillAgent` and Python `HierarchicalSkillPolicy` to `configs/agents/skills_default.toml`.
  - Collector policy `"skill"` plus action-decoder skill-family mapping for traces.
  - Wired `--collection-policy skill` through `python/forge/policy_names.py` so CLI, collector, and tests share identifiers.
  - Derived `ACTION_BASE_COUNT` from slot-width constants; Python `FORGE_BASE_ACTIONS` is computed the same way.

---

## [0.5.0] - 2026-09-05

A standards audit found that the controls were configured but could not
fail, and that the test guarding Invariant 6 checked five fields out of
roughly twenty-five. This lands the code half of the remediation.

- **Determinism is verified against full state.** `step_determinism`
  compared `tick` plus four per-agent fields through an unguarded `.zip()`
  (differing agent counts truncated and passed) against a hard-coded
  two-action sequence. It now generates the action sequence across all 22
  `Action` variants, asserts collection lengths before any zip, and
  compares serialized state including the RNG stream plus the 15 fields
  `SerializableWorldState` omits. The obvious fix was itself vacuous:
  `to_bytes()` ends in `unwrap_or_default()`, so two failed
  serializations compare equal as empty vectors — hence the new fallible
  `try_to_bytes()`. Added golden state hashes over three seeds and four
  reward-value tests; nothing in the repo previously asserted a reward
  *value* out of `step`.
- **`overflow-checks = true` in `[profile.release]`.** Integer overflow
  panicked under `cargo test` and wrapped silently in the shipped binary,
  so the tested and deployed builds had different arithmetic semantics.
- **Model bundle digests are verified.** `ModelManifest::validate`
  checked only that `sha256` was non-empty, and `sha2` was not a
  dependency of the crate, so the runner could not verify despite a doc
  comment saying it could. Path resolution also accepted absolute paths
  and `..`. New `integrity` module enforces containment and re-hashes at
  a single choke point both the initial load and the hot-reload pass
  through.
- **Unauthenticated control planes closed.** `forge-server` defaulted to
  `0.0.0.0:8080` with `CorsLayer` as its only middleware — browser-only,
  irrelevant to `curl`. mc-bot's WebSocket had no auth, no frame cap, an
  unbounded queue, and a single-client guard released only on `close`.
  Both now bind loopback by default with optional tokens, config-driven
  limits, and idle reclaim.
- **The mc-bot container could not start.** Its entrypoint pointed at
  `src/index.js`; the TypeScript migration deleted every `.js` file and
  the Dockerfile was never updated, blocking the whole v0.5 stack through
  `service_healthy`. Now builds and runs compiled output.
- **`cargo deny` had never actually run.** The CI invocation was
  malformed (`--all-features` is a global option) and `|| true` hid the
  exit-2 argument error. Corrected, it found `crossbeam-epoch` and `rand`
  advisories (bumped, not suppressed) and `bincode` unmaintained (a
  dated, reviewed exception). `cargo-deny` and `gitleaks` are now
  blocking, with a new `.gitleaks.toml` whose exceptions are scoped so a
  real credential in an allowlisted file is still caught.
- **`seed_everything` never seeded torch** despite documenting that it
  did — it called only `cuda.manual_seed_all` inside a CUDA guard, so on
  CPU-only hosts, which is every CI runner, torch was entirely unseeded.
- **Coverage exclusions were unanchored regexes.** `pass` matched 31
  lines that are not `pass` statements; the determinism checker was
  excluded outright because a local variable is named `passed`. Anchoring
  them *raised* coverage to 93.58% — the exclusions were hiding
  well-tested code.
- Compose publishes every port through `BIND_HOST` (default
  `127.0.0.1`), Grafana fails closed instead of defaulting to
  `admin`/`admin`, and `.gitignore` now covers the key and credential
  patterns `SECURITY.md` already claimed were ignored.

### Added — WASM demo verified end to end

Nothing on a PR had ever compiled `crates/forge-wasm` for
`wasm32-unknown-unknown`, nothing had ever executed the compiled module, and
nothing had ever loaded the demo page. An audit of the Actions history also
found the demo had never actually published: `gh-pages.yml` failed all 7 of its
runs (Pages not enabled on the repo) and `hf-space.yml` all 3 (`HF_TOKEN`
unset). Both need one-time repository settings — see
`docs/next_steps.md` §6.

- **Blocking `wasm` CI job** — clippy for `wasm32-unknown-unknown` with
  `-D warnings`, plus `#[wasm_bindgen_test]`s executed under Node. The wasm
  tests include a determinism check on the target the browser demo ships to,
  which nothing previously verified (wasm32 has a 32-bit `usize`, a different
  float ABI, and trap-based panics).
- **Non-blocking `wasm-e2e` CI job** — Playwright drives the real `web/` page in
  Chromium against the wasm-pack build, with no mocks.
- **`ForgeWasmEnv::new` returns `Result<_, JsError>`** and installs
  `console_error_panic_hook`, so an invalid config throws a readable JS `Error`
  rather than `RuntimeError: unreachable executed`. `try_new` is the Rust-side
  equivalent.
- **Seed control in the demo.** `reset` takes `Option<u64>`, which wasm-bindgen
  lowers to an `i64` wasm parameter — so seeds cross as `BigInt`. The call
  documented in `README.md` (`env.reset(42)`) threw, and `web/app.js` never
  passed a seed at all, leaving the demo unable to demonstrate the
  reproducibility it advertises. Both fixed, and pinned by an E2E spec.
- **Post-publish smoke in `gh-pages.yml` and `hf-space.yml`**, matching the
  convention `hf-dataset.yml` and `hf-model.yml` already followed: fetch the
  published artifact back and assert on it, including that the `.wasm` is
  served as `application/wasm`.
- **Shared `scripts/install_wasm_pack.sh` / `build_wasm_demo.sh` /
  `wasm_test_node.sh`.** The build script always passes an absolute
  `--out-dir` (wasm-pack resolves a relative one against the crate directory);
  the test wrapper fails on `wasm-pack test`'s vacuous exit-0 when a crate has
  no wasm tests.
- `deny.toml` gains the `wasm32-unknown-unknown` triple, so cargo-deny now
  evaluates the wasm dependency graph at all.

### Fixed — WASM E2E adversarial review pass (PR #134)

A five-lens adversarial review of the work above (hardcoded values, dead
code, test coverage, independent CI re-verification, correctness/security)
found several real gaps the original pass missed:

- `WasmEnvError::InvalidConfig` now wraps the structured
  `forge_types::ForgeError` instead of flattening it to a string, so a Rust
  caller (`try_new`'s documented audience) can inspect which config field
  failed and why, not just read a pre-rendered message.
- `web/app.js`'s `parseSeed` no longer throws for a digit string too long
  for `BigInt` to represent — V8 enforces its own internal size cap
  independent of `MAX_SEED` — it now returns `null` like every other
  unparseable seed. `app.js`'s bottom-of-file `main()` call is guarded
  behind a `document` check so the module can be imported from a
  non-browser test without running the whole app as a side effect.
- `tests/web-e2e/playwright.config.ts`'s `HOST` now reads `WEB_E2E_HOST`
  like `serve.mjs` already did, instead of a second, independently
  hardcoded `127.0.0.1` that could silently disagree with it.
- The post-publish smoke retry cadence in `gh-pages.yml` and `hf-space.yml`
  is now named (`RETRY_ATTEMPTS` / `RETRY_DELAY_SECONDS` env vars) instead
  of a magic `5`/`10` inlined in loop syntax, once in bash and once in
  Python.
- `tests/web-e2e/preflight.mjs` now fails fast with a clear message if
  `FORGE_WASM_OUT_DIR` is set, rather than silently building to a path the
  E2E harness's `serve.mjs` never looks at and then failing a confusing
  "no wasm bundle" check.
- `crates/forge-wasm/Agent.md`'s tools table dropped a stale, unpinned
  `wasm-pack build` row left over from before this PR's own `make wasm`
  wrapper existed. `web/README.md`'s local build instructions now match
  root `README.md`'s (pinned wasm-pack via `scripts/install_wasm_pack.sh`,
  `make wasm`, `tests/web-e2e/serve.mjs`) instead of an unpinned
  `cargo install wasm-pack` and `python3 -m http.server`, which would have
  served the `.wasm` with the wrong content type.
- New test coverage: a Playwright spec for the seed-above-`MAX_SEED`
  fallback path and its status message, and
  `tests/web-e2e/unit/app.test.mjs` (`node:test`, no browser — the
  BigInt-overflow case needs a ~350-million-digit string, impractical to
  drive through a real DOM input) for `parseSeed`'s boundary and
  overflow-guard behaviour in isolation.
- **Line-ending drift guard** (`.claude/hooks/guard_line_ending_drift.py`,
  documented in full under this same Unreleased section's hooks list) —
  added in the same pass, directly modeled on this changelog's own
  CRLF-corruption entry.

### Added — pinned-config consistency check (`scripts/check_pinned_config_consistency.py`)

Cross-checks three values that get duplicated across files which can't
share one source (a Dockerfile `ARG`, a GH Actions `env:`/`with:` entry,
and a Python/TOML source can't all read the same file without much more
invasive templating) — so instead of eliminating the duplication, the
script re-derives every copy and fails on drift (the wasm-pack pin now has
two dependent workflows, `hf-space.yml` and `ci.yml`):

- The Rust toolchain version (`rust-toolchain.toml`'s `channel`) against
  its 15 `dtolnay/rust-toolchain@stable` `toolchain:` copies (5 workflow
  files) and 2 Dockerfiles' `RUST_IMAGE_TAG`.
- The ONNX Runtime version `docker/mc-runner.Dockerfile` and `ci.yml`'s
  `onnx-features` job each pin independently. Deliberately excludes
  `docker/trainer.Dockerfile`'s own `ONNXRUNTIME_VERSION` (a different
  artifact — the Python wheel, not the C++ redistributable — on an
  independent release cadence).
- The LM Studio port/base-URL (`providers.py`'s `DEFAULT_LMSTUDIO_BASE_URL`)
  against `ci.yml`'s `LMSTUDIO_PORT`/`LMSTUDIO_BASE_URL` env (already
  comment-annotated "keep in lock-step" — now actually enforced) and
  `e2e-long.yml`'s own `LMSTUDIO_PORT`, which had no such comment at all
  and was the least-guarded of the three copies before this check existed.

Started as a Rust/ONNX-only script (`check_version_consistency.py`),
renamed once the same "duplicated pin, no single source possible" pattern
turned up a third time for the LM Studio endpoint rather than adding a
mismatched-scope check under the old name. An adversarial peer review
found its line-matching regexes were comment-blind: a commented-out stale
pin like `# toolchain: "1.60.0"` was parsed as a live occurrence, a real
false-positive risk (confirmed: reproduced the false CI failure, then
fixed by truncating each line at its first `#` before matching). Covered
by `tests/python/test_check_pinned_config_consistency.py` (14 cases,
using `monkeypatch` to redirect the script's `REPO_ROOT` to an isolated
fixture tree rather than ever mutating real repository files) in addition
to the ad hoc deliberate-mismatch/restore verification run directly
against the real repo. Wired into `make verify` (`pin-check` target) and
CI's `python-lint` job; documented in `docs/hardcoded-values-audit.md`.

### Added — Claude Code tooling (`.claude/`)

- **`/forge-verify` skill** (`.claude/skills/forge-verify/SKILL.md`): wraps
  the Makefile's `verify`/`verify-full` pre-PR gate sequence with a
  per-category pass/fail report instead of one opaque result.
- **`forge-docs-audit` skill** (`.claude/skills/forge-docs-audit/SKILL.md`):
  re-verifies factual claims (counts, file:line refs, named tests, status
  markers) in `docs/next_steps.md`/`CHANGELOG.md`/`README.md`/
  `docs/architecture.md`/`Agent.md`/`CLAUDE.md` against the actual
  codebase and corrects drift in the house style already established in
  those files. Motivated by a fresh skills/hooks re-survey pointing out
  this is the single most-repeated pattern in this repo's own git
  history — the several `docs: fix stale ...` commits and multiple
  Technical Debt rows resolved specifically because they were re-checked
  and found false, each requiring independently re-deriving evidence by
  hand.
- **Tracked-file deletion guard** (`.claude/hooks/guard_tracked_deletion.py`,
  registered via `.claude/settings.json` as a `PreToolUse` hook on `Bash`):
  blocks `rm`/`find -delete` commands whose glob pattern matches a
  git-tracked file rather than just the generated/ignored ones intended,
  by cross-checking against `git ls-files` (no hand-maintained path list,
  so it can't drift). Fails open on any parse/git error. Directly
  motivated by a real incident this pass where a `.coverage*` cleanup glob
  also matched and deleted the tracked `.coveragerc`.
  Went through two rounds of adversarial review before landing in its
  current form — an initial version detecting `rm` via an anchored regex
  (only recognized `rm` as the string's first token, or immediately after
  `;`/`&`/`|`) turned out to have real, reproducible bypasses: `find X |
  xargs rm -rf` (the single most common bulk-delete idiom), `sudo rm -rf`,
  `VAR=x rm -rf`, `(rm ...)`/`{ rm ...; }`, a bare leading space, a
  newline instead of `;`, plus `find`'s own `-iname`/`-path`/`-regex`
  selectors (and no selector at all) being silently unchecked, and
  `./`-prefixed/absolute-path/bare-directory targets not matching a
  tracked file's basename. Rewritten to tokenize the whole command once
  and check token membership rather than anchor a regex on what precedes
  `rm` — recall-favoring by design, since an over-liberal detection only
  costs one extra, cheap `git ls-files` cross-reference for a command
  that turns out to need no scrutiny; a `find | xargs rm` invocation with
  no literal delete target in its own text is resolved via `find`'s own
  `-name`/`-iname`/`-path`/`-ipath` selectors instead of the sibling `rm`
  extraction, since xargs supplies the actual argument at runtime. Every
  bypass above is now a named regression test (34 total), each proven
  non-vacuous by confirming it fails against the pre-fix code before
  passing post-fix. Covered by this stdlib-only self-test suite (`make
  hooks-test`, now `python3 -m unittest discover` over `.claude/hooks/`
  rather than a hardcoded filename, so a new hook's tests are picked up
  automatically), run in CI's `python-lint` job.
- **Staged-secret guard** (`.claude/hooks/guard_staged_secrets.py`, same
  `PreToolUse`/`Bash` wiring): runs `gitleaks protect --staged` before a
  `git commit` actually happens and blocks on a real finding. Closes a
  gap `security.yml`'s own comments admit exists: its `gitleaks` job
  scans git *history* after a push and is explicitly report-only
  (`|| true`) — nothing previously scanned *before* a commit landed.
  Not a replacement for that job (still the full-history backstop), an
  earlier and cheaper checkpoint. Fails open if `gitleaks` isn't
  installed. Test fixtures use a repo-local custom `.gitleaks.toml` rule
  rather than a well-known example secret (AWS's own published EXAMPLE
  key, tried first, is — confirmed empirically — excluded from gitleaks'
  default ruleset, almost certainly because it's such a common docs/
  tutorial placeholder), so the tests don't depend on the exact shape of
  gitleaks' bundled rules. The two cases needing the real binary skip
  themselves when it's absent; `security.yml`'s `gitleaks` job (which
  already fetches the binary for the history scan) now also adds it to
  `$GITHUB_PATH` and runs this hook's suite unskipped there.
- **Line-ending drift guard** (`.claude/hooks/guard_line_ending_drift.py`,
  same `PreToolUse`/`Bash` wiring): blocks a `git commit` that flips more
  than half of an already-tracked file's lines between CRLF and LF (or the
  reverse) — the signature of a whole-file EOL rewrite, never a legitimate
  content edit. Modeled directly on this same changelog entry's own
  incident: preparing this file's corruption fix, a Python
  `open(p).read()` / `open(p, 'w').write(s)` round-trip would have
  silently re-flattened it back to LF, and nothing would have caught that
  before the commit. Compares the CRLF-line fraction of the `HEAD` blob
  against the staged (index) blob; skips brand-new files, which have no
  prior convention to drift from, and any path with an explicit
  `.gitattributes` line-ending declaration (`-text` or `eol=`) — such as
  this file's own new entry above, which records a deliberate choice, not
  drift. Fails open on any git or parse error. Verified against this
  session's real git history: replaying the actual pre-corruption and
  corrupted `CHANGELOG.md` blobs through the guard's own comparison
  reproduces a full 0-to-1 CRLF-fraction swing, confirming it would have
  blocked the real incident had it existed at the time. Self-tests
  (stdlib + `git` only, no external binary): `python3
  .claude/hooks/test_guard_line_ending_drift.py -v`, picked up
  automatically by `make hooks-test`'s `unittest discover`.

### Added — Hugging Face publication pipelines (`docs/hf/README.md`)

- **Static Space sync** (`.github/workflows/hf-space.yml`): builds the
  `forge-wasm` demo and mirrors `web/` + a new Space card
  (`web/space/README.md`) to the `ianshank/forge-wasm-demo` static Space on
  pushes to `main`. Requires a write-scoped `HF_TOKEN` repository secret.
- **Trajectories dataset generator** (`forge-gen-dataset`, behind the new
  `forge-data/hf` feature): streams `ExpertDemoGenerator` episodes across a
  54-cell config cross-product (world size × agents × tier × {mcts, random})
  into `forge-replay`'s Parquet exporter — the first production caller of
  `write_parquet_shards`. Disjoint per-cell seed blocks make
  `(scenario_id, seed)` a reproducible episode key; a failed episode aborts
  the run instead of shipping a silently incomplete dataset. Published by
  `.github/workflows/hf-dataset.yml` with a card rendered from
  `docs/hf/dataset-card.md`; the new `hf-export` CI job keeps the feature
  compiling with a generation smoke.
- **`ExpertDemoConfig` gains `policy` (`mcts` | seeded `random`),
  `scenario_label`, and `tasks_per_episode`** (all serde-defaulted,
  backward-compatible). Episodes now assign procedurally generated tasks via
  `forge_task::generator::generate_active_task` — previously **no** Rust
  driver populated `WorldState.tasks`, so every generated episode carried
  all-zero rewards.
- **MuZero model publisher** (`scripts/hf_publish_model.py` +
  `.github/workflows/hf-model.yml`): verifies bundle sha256s, stages the
  three ONNX graphs flat at the repo root (the exact layout
  `checkpoint_loader.load_from_hf` consumes), renders
  `docs/hf/model-card.md`, uploads via `HfApi`, and round-trips the repo
  back through `bootstrap --from-hf`. Publishes **private** with a
  random-init warning until real training lands.

### Fixed

- **`forge-agent`'s ONNX feature surface (`onnx`/`onnx-reload`/`mc-live-bundled`)
  compiles and runs again**: a routine Dependabot bump (`ort` rc.12 → rc.13,
  `a7b078c`) had silently broken it, invisible because no CI job built this
  surface. Fix was 5 mechanical lines in `onnx_model.rs`
  (`try_extract_raw_tensor` → `try_extract_tensor`, a stray `?` removed ×3
  after `ort::inputs![...]` — it no longer returns a `Result` — and `&mut
  self` on the session-holding locals) plus enabling `ort/std` for
  `commit_from_file`. Separately, `docker/mc-runner.Dockerfile`'s
  `ONNXRUNTIME_VERSION` is now pinned `>=1.23.2`: earlier releases hit a
  known upstream `ort` rc.13 teardown segfault on process exit under
  `load-dynamic` (pykeio/ort#614, fixed in the runtime by pykeio/ort#610),
  reproduced and confirmed fixed locally end-to-end (bootstrap a real
  MuZero bundle → load via `OnnxMuZeroModel` → run real MCTS inference,
  zero crashes). A new `onnx-features` CI job now builds and tests this
  surface on every push.
- **`forge-cloud`'s `gcs` feature compiles again**: `object_store` 0.14 moved
  `put`/`get`/`delete`/`head` behind the `ObjectStoreExt` extension trait —
  one missing import broke the feature-gated build (and with it the
  `--features forge-cloud/gcs` CI jobs) since the 0.14 bump.
- `forge-server` compiles again under axum 0.8 (`Message::Text` takes
  `Utf8Bytes`; two send sites in `ws_handler.rs`).
- Dataset-generation hardening from branch peer review: seed blocks derive
  from each cell's position in the full unfiltered grid (so `--cells`
  regeneration reproduces published episodes exactly), duplicate axis values
  are rejected, `--episodes-per-cell 0` / zero-row runs abort before upload,
  failed episodes abort instead of shipping a silently incomplete dataset,
  and `DemoPolicy` has a single string form (serde/`Display`/`FromStr`).
- `hf_publish_model.py` hardening: refuses non-empty staging dirs, cleans up
  its ephemeral staging dir after upload, and malformed manifests exit
  cleanly via `PublishError` instead of raw tracebacks. Workflow pip
  installs are version-pinned; the dataset upload mirrors with
  `delete_patterns` so re-runs can't leave stale shards.
- `checkpoint_loader.load_from_hf`: removed the `local_dir_use_symlinks`
  kwarg (deleted in huggingface_hub 1.0 — the locked 1.8/1.16 clients raised
  `TypeError`), and `subfolder` downloads now normalize from the Hub
  client's returned path instead of recomputing it (previously
  `FileNotFoundError` in manifest hashing). The fake-hub test double now
  mirrors the real 1.x signature so neither bug can re-mask.
- `huggingface-hub>=0.20` added to the `minecraft` extra so
  `bootstrap --from-hf` works under `pip install -e ".[minecraft]"`.

### Changed — charter ↔ codebase alignment

- **Removed the unimplemented `live-test-stub` Cargo feature** from
  `crates/forge-mc-runner`. It was declared and promised a `--live-stub` mode
  plus a `MockMinecraftEnv` helper, but no `#[cfg(feature = …)]` site, symbol,
  test, or CI job ever referenced it. `mc-live`, `onnx-reload`, and
  `mc-live-bundled` are unaffected; no build that worked before can break.
- `docs/CHARTER.md`: Invariant 2 now describes the `schema_id` contract as
  three-language (Rust ↔ JS ↔ Python) with a per-value pin table, and names the
  third wire-version constant (`protocol::SCHEMA_VERSION`). Invariant 3 drops
  the dead feature and points `Runner<E, M>` at its defining module. Invariant 6
  recategorises `deny.toml` (a `security.yml` supply-chain policy, advisory
  today — not a `ci.yml` coverage/lint config) and names the non-blocking jobs
  it deliberately omits. Permanent non-goals are **ratified**.
- `docs/architecture.md`: container table extended from 11 to all 26 workspace
  crates, restoring its role as the declared source of truth for the crate list;
  removed the deleted `forge-procgen` from two ASCII diagrams that survived the
  `e5eca3d` sweep (the name was split across lines, defeating a grep); corrected
  `mc-live` to `["dep:forge-env-mc"]` per CHARTER Deliberate Exception 2.
- Doc reconciliation: `Agent.md` ("8 crates" → 26, `mc-live` description),
  `ANTIGRAVITY.md` (80% → 85% coverage target), `README.md` ("23 Rust crates" →
  26), Windows-absolute `file:///c:/…` links converted to repo-relative, and 20
  stale `mc-bot/**/*.js` doc-comment paths retargeted to `.ts` after the
  TypeScript migration.
- `benchmarks/baselines/README.md` records that `multi_agent_scaling.json` is
  uncommitted for both profiles, so the "130,000+ steps/second" headline rests
  on locally-run benchmark code rather than a checked-in measurement.

### Added — charter alignment guard

- `tests/python/test_charter_alignment.py`: fails CI when `docs/CHARTER.md`
  cites a path that no longer resolves, a Cargo feature that is declared but
  never gated on, or a CI job that no longer exists; when a workspace crate is
  missing from `docs/architecture.md`; or when either document names a crate
  that is no longer a workspace member. Stdlib + pytest only — no new CI
  dependency. Runs under the existing `python-test` gate.

### Fixed — follow-up hardening pass on the charter alignment guard

- **The guard's own feature-citation check was vacuously true.** A dead
  fallback branch in `test_charter_cited_features_are_declared` made
  `undeclared` provably empty for any charter text — a typo'd feature name in
  a `` `--features <name>` `` command citation (the shape `docs/CHARTER.md`
  actually uses) was silently invisible to the check. Fixed by extracting
  feature names from both bare-backtick and `--features` command citations;
  verified with a negative control (an injected typo now fails the test with
  a `file:line` pointer; reverted).
- **Corrected an inaccurate comment this same guard-adding change introduced**
  in `crates/forge-mc-runner/src/live.rs`: it claimed the ONNX hot-reload path
  "is exercised end-to-end by the compose-stack E2E," which doesn't hold —
  `--features onnx-reload` doesn't currently build (pre-existing
  `forge-agent`/`ort` incompatibility), the compose stack's default build uses
  `mc-live` only, and the actual E2E test never touches reload/manifest logic.
  The comment now states plainly that the reload path is unverified
  end-to-end, without claiming coverage that doesn't exist.
- Added 10 unit tests isolating `test_charter_alignment.py`'s own parser
  helpers from the live repo via `tmp_path` fixtures and module-global
  monkeypatching — closing previously-dead defensive branches (a workspace
  member with no `Cargo.toml`, a workflow with no `jobs:` key) and pinning
  `reconstruct_split_crate_names`'s column/box-art contract independently of
  whatever `docs/architecture.md` happens to contain.
- Added best-effort `file:line` hints to failure messages where the raw
  charter/doc text is already in scope; factored a `_crate_manifest()` helper
  and a `WORKFLOWS_DIR` constant to remove path duplication (and deleted
  `CI_WORKFLOW`, which had become fully unread once job discovery was
  generalised to scan every workflow); made the Determinism-invariant lookup
  match on heading text instead of a hardcoded invariant number.
- `README.md`: two more stale crate counts ("23-crate Rust workspace",
  "27 crates") corrected to 26, matching `docs/architecture.md`'s table,
  `CLAUDE.md`, and README's own already-corrected occurrence. (A prior pass in
  this same `[Unreleased]` cycle fixed only one of the three occurrences.)

---

Production-hardening & gap-closure track (PR #64): operational hardening
(security scanning, observability, deploy limits) plus closure of the
documented functional gaps (server history, dashboard wiring, mc-bot
reconnect). All additions are config/env-driven with `Default`s and
backwards-compatible.

### Added — `forge-observability` crate (shared tracing init)

- New lightweight crate `crates/forge-observability` exposing
  `init_tracing(TracingOptions)` / `try_init_tracing` — the single home for
  `tracing-subscriber` setup. Output format is env-driven via
  `FORGE_LOG_FORMAT` (`text` default, `json` for structured aggregation);
  `RUST_LOG` still controls the filter with a caller-supplied default.
- `forge-server` and `forge-mc-runner` now call it, removing the duplicated
  subscriber bootstrap and their direct `tracing-subscriber` dependency.

### Added — structured logging parity (Python + Node)

- `forge.utils.logging_config`: `setup_logging_from_env()` +
  `json_format_from_env()` honour the same `FORGE_LOG_FORMAT` / `FORGE_LOG_LEVEL`
  switch; `setup_logging()` gains an optional `stream` and `clear_existing`.
  `forge.training.muzero_mc.cli` uses it (preserving `--log-level`/`--quiet`).
- mc-bot `src/logger.ts`: `createLogger()` emits one JSON object per line when
  `FORGE_LOG_FORMAT=json` (else delegates to `console`); serializes `Error`s
  and guards system metadata from caller spoofing.

### Added — supply-chain & SAST scanning (advisory-first)

- `.github/workflows/security.yml`: report-only `cargo-deny`, `pip-audit`,
  `npm-audit` (mc-bot + dashboard), Trivy filesystem scan, and CodeQL (gated
  behind the `ENABLE_CODEQL` repo variable since SARIF upload needs GitHub code
  scanning). Non-blocking during rollout.
- `.github/dependabot.yml` (cargo/pip/npm×2/github-actions, weekly, grouped) +
  workspace `deny.toml` (advisories/licenses/bans/sources thresholds).

### Added — observability stack & deploy resource limits

- Opt-in `monitoring` Compose profile in `docker/compose.minecraft.yml`
  (Prometheus + Grafana) with `docker/monitoring/` scrape config, datasource +
  dashboard provisioning, and a starter dashboard for the eight `forge_mc_*`
  signals. Default `up` is unaffected.
- Env-driven `deploy.resources` (CPU/memory limits + reservations) on every
  service across `docker/docker-compose.yml`, `compose.minecraft.yml`,
  `docker-compose.distributed.yml`; documented in `compose.minecraft.env.example`.

### Added — `forge-server` persistent history + query endpoints

- `crates/forge-server/src/history.rs`: a `HistoryStore` trait with a
  file-backed `JsonlHistoryStore` (append-only, bounded retention) and an
  `InMemoryHistoryStore` for tests. `POST /api/training-metrics` and
  `/api/decision-traces` now persist (run id from `?runId=` →
  `X-Forge-Run-Id` → server-session id) in addition to broadcasting.
- New `GET /api/training-metrics/history`, `GET /api/decision-traces/history`,
  `GET /api/runs` (camelCase, `runId`/`limit` query params). New config
  `FORGE_SERVER_HISTORY_DIR` / `_RETENTION` / `_QUERY_LIMIT`.

### Added — dashboard Training/Runs/Live wiring

- `useTrainingHistory`, `useRuns`, `useDecisionTraces` hooks (mirroring
  `useMetrics`) poll the new endpoints; `TrainingPage`, `RunsPage`, `LivePage`
  now render real data (empty states preserved). New `VITE_TRAINING_HISTORY_INTERVAL`,
  `VITE_RUNS_INTERVAL`, `VITE_HISTORY_LIMIT`.

### Added — mc-bot background heartbeat

- `BotManager.startHeartbeat()/stopHeartbeat()` wire the previously-unused
  `EnvConfig.heartbeat_ms`: a monitor that triggers the existing coalesced
  `reconnect()` when the bot goes stale, closing the half-open-socket gap.

### Added — project charter + documentation governance (PR #89)

- `docs/CHARTER.md`: durable governance charter — mission, scope boundaries
  (Included / Deferred / Permanent non-goals / Deliberate Exceptions) and the
  Seven Core Invariants, each anchored to the code or CI that enforces it.
  Cross-linked from `README.md` and `CLAUDE.md`.
- `forge-civ` now carries `#![deny(missing_docs)]` — it was the only workspace
  crate without the lint; all public items were already documented, so this
  enforces the convention going forward with zero code churn.

### Changed

- Rust tracing initialization is centralized in `forge-observability` (was
  duplicated in both binaries).
- Dashboard slider (`components/ui/slider.tsx`) forwards `aria-label` to the
  Radix Thumb, fixing the axe `aria-input-field-name` violation on `/live`.
- `forge-cloud` in-memory worker registry recovers from a poisoned mutex via a
  shared `lock_recover` helper instead of panicking the whole pool — mirroring
  `forge-server`'s `JsonlHistoryStore` policy. Backwards-compatible; the
  recovered guard still exposes the consistent map/seed state (PR #89).
- Python typing hygiene: removed a version-fragile
  `# type: ignore[no-any-return]` in `social/trust_tracker.py` (via an
  ndarray-annotated local, keeping strict checking) and a movable `[arg-type]`
  ignore in `memory/memory_store.py` (PR #89).
- `forge-civ` pathfinding goal-walkability check simplified to the `?` operator,
  clearing a `clippy::question_mark` warning (PR #89).

### Tests

- Rust: `history.rs` unit tests + `tests/history_endpoints_integration.rs`
  (driven via `tower::oneshot`); `config.rs` history-env coverage;
  `forge-observability` unit/doctests.
- Dashboard: `historyHooks` + `historyPages` + `environment` additions
  (coverage ≥85%). mc-bot: `bot_manager` heartbeat + `logger` suites. Python:
  `test_logging_config` env-helper coverage.
- Python: `test_muzero_mc_checkpoint_loader` covers the HuggingFace warm-start
  loader end-to-end via a fake `huggingface_hub` — happy path, `filename_map` /
  `subfolder` pass-through, `str` output_dir, download-failure (no partial
  manifest), and structured-logging lines (0% → 100% line coverage). Rust:
  `forge-cloud` worker-registry poison-recovery regression test (PR #89).

## [0.5.0] — 2026-06-02

Release-hygiene cut plus three non-Minecraft feature tracks. The workspace
version is moved off the stale `0.1.0` placeholder to `0.5.0`, reflecting the
v0.2–v0.5 features already shipped.

### Added — Cooperative multi-agent MCTS (`forge-mangomas::swarm`)

Fills the former Phase-6 stub with a real Centralized-Training /
Decentralized-Execution (CTDE) planner that reuses FORGE's single-agent PUCT
search rather than reinventing it:

- `CooperativeMctsProtocol` implements the existing `SwarmProtocol` trait (a
  drop-in swap for `IndependentProtocol`) **and** `batch_runner::ActionPolicy`.
- Additive, non-breaking `SwarmProtocol::coordinate_stateful(world, obs, comm)`
  default method carries the `WorldState` the search needs; `IndependentProtocol`
  inherits the default unchanged.
- `JointMctsPlanner` with two config-selected strategies: `SequentialFactored`
  (per-agent search at the shared root — simultaneous best-response, linear in
  agent count) and `Sampled` (seeded joint-action sampling scored by the
  centralized critic). Determinism via `rand_pcg` seeding.
- `JointPolicyValue` critic abstraction: `CentralizedCritic` (CTDE value
  aggregation) and `IndependentCritic`, behind the `Critic` enum.
- New `MangoMasError::SwarmCoordination` variant. No hard-coded values — the
  action space is derived from the model's comm vocab.

### Added — REST API for the environment (`forge-server`)

- `POST /api/env/reset`, `POST /api/env/step`, `GET /api/env/render` over the
  existing axum 0.7 server, reusing `SimulationSnapshot`/`AgentSnapshot` DTOs.
- Session world isolated in `AppState` (`Arc<Mutex<Option<WorldState>>>`),
  decoupled from the live demo ticker — request/response semantics without
  racing the broadcast loop.
- New `ApiError` (`thiserror` + axum `IntoResponse`): Config→400, NotReset→409,
  InvalidAction→422, Internal→500.

### Added — WebAssembly GitHub Pages demo

- `.github/workflows/gh-pages.yml` builds `crates/forge-wasm` with `wasm-pack`
  and deploys a fully client-side demo (`web/`) — no server, shareable link.
  **Correction (2026-08):** the build shipped, the deploy never did. Every run
  of this workflow failed at `actions/deploy-pages` because GitHub Pages was
  not enabled on the repository, so no demo was ever published. See the
  Unreleased entry below.
- `crates/forge-wasm` gains the `getrandom/js` + wasm-target wiring needed for a
  browser build.

### Changed

- Workspace version `0.1.0` → `0.5.0` (`Cargo.toml`, `python/forge_env`,
  `dashboard`); `Cargo.lock` regenerated.
- Reconciled four stale `docs/next_steps.md` rows against verified source
  (false-positive panic!/unwrap debt, an already-fixed demo bug, and an
  untestable-here torch deprecation).

### Tests

- New cross-crate regression suite `tests/rust/integration_swarm.rs`
  (step-compatibility, determinism, baseline swappability, sampled strategy).
- New-code coverage: `forge-server` env/error at 100%; swarm modules 88.9–96%.

### Added — Minecraft RL Integration: v0.5 Phase 2 — Production Stability, TypeScript Migration & Hardening

Delivered complete structural hardening, connection resilience, and compile-time type safety for the Minecraft WebSocket integration:

- **100% TypeScript Migration (`mc-bot/`)**:
  - Converted all **16 source files** and **15 test files** to strict ESM TypeScript (`.ts`).
  - Added strict interfaces for `EnvConfig`, `ResetConfig`, `ConfigBundle`, `Snapshot`, `ActionMap`, `ActionEntry`, and WebSocket messages.
  - Enabled rigorous compiler flags (`strict: true`, `noImplicitAny: true`) in `tsconfig.json` with **0 compilation errors** in `npm run typecheck`.
  - Replaced ESLint with a unified Biome formatting and linting setup, reporting **0 linter/formatter violations** across all 41 package files.
- **Connection Auto-Reconnect (`BotManager`)**:
  - Implemented `BotManager` class in `mc-bot/src/bot_manager.ts` to actively monitor Mineflayer connection health via tick age tracking.
  - Added auto-teardown and dynamic reconstruction of Mineflayer instances on server-side exceptions (preventing connection lockouts), complete with exponential backoff and connection failure notifications.
- **ORT Toolchain Resolution**:
  - Pinned exact `ort` version `2.0.0-rc.9` in `crates/forge-agent/Cargo.toml` and refactored `build_session_from_path` in `onnx_model.rs` to leverage the stable `commit_from_file` API, bypassing all FFI ABI hazards and ureq dependency conflicts for trained-mode Docker builds.
- **Robust Security & Regression Coverage**:
  - Implemented dedicated network payload shape and prototype pollution validations inside `mc-bot/test/security.test.ts`.
  - Expanded unit test coverage in `actions.test.ts`, `bot_manager.test.ts`, and `config.test.ts` to **185 passing tests** under Node 22 native test runner via `tsx`.

### Added — Minecraft RL Integration: v0.5 Phase 1 - CLI Hardening & Test Coverage Boost

A comprehensive hardening and test coverage boost pass has been completed on the training CLI and trainer subcommands, achieving a package-level test coverage of **89.04%** (well exceeding the 85.0% global floor):

- **CLI Coverage Boost**:
  - Expanded unit test coverage in `test_muzero_mc_cli.py` to cover all subcommand entrypoints and error conditions.
  - Plumbed unit tests for `_run_bootstrap` success and error paths (`ValueError`, `OSError`).
  - Added unit tests for `_run_compute_schema_id` error paths (`ValueError`, `KeyError`).
  - Covered all execution branches of `_run_capture_baseline` including `--dry-run`, `ValueError`, `TimeoutError`, `OSError`, and successful run scenarios.
  - Added robust validation tests for `_run_train` path verification, optional dependency import failures (`ImportError` simulations for PyTorch/ONNX/ONNXRuntime), and execution anomalies (`ValueError`, `OSError`).
  - Plumbed `_drive_continuous_loop` success flow and signal handling, proving that `SIGINT` / `KeyboardInterrupt` triggers graceful shutdown, sets stop flags, and cleanly restores original signal handlers in the `finally` block.
- **WebSocket RFC 6455 Client Test Integration**:
  - Registered and tracked `tests/python/test_ws_client.py` validating the stdlib-only frame parser shared by the handshake probe and manual baseline scripts.
  - Added test coverage for `recv_text` happy paths, masked `send_text` payload frames, extended 16-bit/64-bit frame payload length encodings, socket EOF/close-frame exceptions, and security audit **HIGH-1** DoS mitigation (denying payload sizes exceeding 64 MiB).

### Added — Minecraft RL Integration: v0.5 Phase 1 post-T9 — hardening, docker runner image, first-real-run validation

Four commits landed after the original T1-T9 sweep, all preserved on
`feat/mc-v05-phase1-first-real-run` (PR #60):

- **Hardening pass 1** (`7d144a6`): folds in 16 peer-review findings —
  drops the `random-baseline` Cargo feature gate (no extra deps,
  no benefit); hoists Prometheus helpers to `forge.utils.metrics`;
  promotes `mc_capture_baseline.py` to the `forge.training.muzero_mc.cli
  capture-baseline` subcommand; renames `radius` → `grid_radius`
  with a serde alias for backwards-compat; per-tile `Number.isFinite`
  coercion against NaN gradients; per-variant `trajectories.<variant>/`
  dirs so the trainer's `_trim_replay_buffer` can't evict baseline
  files mid-capture; Rust-side `BLOCK_FEATURE_CHANNELS` xlang pin.
- **Lint pass** (`1797d71`): `cargo fmt + clippy -D warnings + ruff`
  sweep; `assert_eq!(x, true)` → `assert!(x)` everywhere; `PERF401`
  for-loop → comprehension; `PLR0915` build_parser extracted to a
  helper; TC003 `# noqa` annotations on pytest-fixture imports.
- **Docker infra** (`cf06bbf`): unblocks the runner image build by
  forward-porting `forge-agent/onnx_model.rs` from ort rc.9
  `commit_from_file` → rc.12 `commit_from_memory` API; refactors
  `live.rs` to split the trained-mode path into `run_live_trained()`
  feature-gated behind `onnx-reload`; new
  `docker/mc-runner.Dockerfile` (rust:1.93-bookworm builder, 135 MB
  debian:bookworm-slim runtime) building the `mc-live` variant
  cleanly; updates `docker/compose.minecraft.yml` to build the
  runner from source (the previous `ghcr.io/ianshank/forge-mc-runner:
  dev` reference was a placeholder that was never actually
  published); new `scripts/v05_handshake_probe.py` +
  `scripts/v05_manual_baseline.py` (stdlib-only WS clients) drive
  the first-ever real v0.5 episodes against a live `itzg/minecraft-
  server`; preserves the first-real-episode evidence as tracked
  JSONs in `docs/results/v0.5-first-real-run-baseline*.json`.
- **Hardening pass 2** (`5acb374`): folds in another peer-review
  pass against the docker commit — restores `configs/minecraft/
  env.toml` to local-dev `127.0.0.1` defaults (H1); new
  `configs/minecraft/env.docker.toml` overlay carries the docker-
  DNS hostnames (`mc-bot:8766`, `minecraft`) mounted on top of the
  dir mount in compose so docker runs get the override
  automatically without breaking local-dev; extracts the shared
  `scripts/_ws_client.py` (eliminates ~80 lines of duplication
  between the probe + baseline script; spec-mapped RFC 6455 frame
  parser); new `ships_default_runner_toml_parses_with_random_actions`
  Rust integration test (drift in the shipped runner.toml fails
  CI); `tracing::error!` events added on every error path in
  `build_session_from_path` + `run_live_trained` guard;
  `v05_manual_baseline.py` snapshot now schema-compat with
  `mc_plot_baseline.py`'s consumer (adds `summary_counters`,
  `summary_gauges`, `manifest_versions_seen`, `prometheus_snapshot`,
  `trajectory_dir` keys); `Final[...]` annotations in
  `v05_handshake_probe.py` replace the `920` + `8766` magic
  literals.

Security-audit follow-ups (one round of `security-auditor` agent
review against the post-T9 commits):

- **HIGH-1**: `_ws_client.recv_text` now caps inbound frame
  `payload_len` at `DEFAULT_MAX_FRAME_BYTES = 64 MiB` before any
  allocation — a hostile bot sending the u64-max payload_len
  header (~9 EiB) would previously crash the script via heap
  exhaustion.
- **MEDIUM-3**: `observation_grid.blockTypeName` now truncates
  block names to `MAX_BLOCK_NAME_LENGTH = 256` chars before
  hashing — a hostile MC server returning a megabyte-long block
  name would previously CPU-DoS the per-tile encoder.
- **LOW-2** (documented, not changed): `runner.toml`'s
  `metrics_bind = "0.0.0.0"` is container-internal only (compose
  doesn't publish the port to the host); the existing inline
  comment now spells this out.

### Added — Minecraft RL Integration: v0.5 Phase 1 — first real-end-to-end run readiness (block-grid obs + baseline capture)

Closes the v0.4 → real-run gap. Every test in PR #59 passed against
**stubs and mocks** — no one had actually brought the stack up against
a real `itzg/minecraft-server`. v0.5 Phase 1 closes the hidden
contract violation (mc-bot emitted 31 floats, MuZeroConfig required
920) and ships the operator-facing tooling needed to capture a
calibrated random-vs-trained baseline.

Branch `feat/mc-v05-phase1-first-real-run`.

9 tracks:

- **T1 (BLOCKER) — mc-bot block-grid encoder + `Hello.grid_shape`
  cross-check**: new `mc-bot/src/observation_grid.js` emits an
  11×11×1×7 ego-centric block grid (847 floats) + 73 flat features
  = 920 total, matching `MuZeroConfig`'s `grid + vector_dim` split.
  `BLOCK_FEATURE_CHANNELS` frozen const with a coordinated Rust-side
  pin (`crates/forge-env-mc/src/protocol.rs::BLOCK_FEATURE_CHANNELS`)
  + xlang test catches reorder regressions. `Hello` payload extended
  with `grid_shape: {h, w, depth, channels, vector_dim}`; the runner
  cross-checks against `config.observation.expected_grid_shape` and
  refuses to start on mismatch. Legacy 31-float path stays valid via
  `include_block_grid = false`.
- **T2 — `expected_dim = 920` flip across orchestration surfaces**:
  `configs/minecraft/env.toml` adds `expected_dim = 920` and the
  `[observation.expected_grid_shape]` sub-table; `mc_self_play.sh`
  and `docker/compose.minecraft.env.example` flip `OBS_DIM`/
  `TRAINER_OBS_DIM` defaults from 31 → 920. Bootstrap logs the
  resolved (obs_dim, grid_flat_dim, vector_dim, schema_id) tuple at
  INFO on entry. New `test_muzero_config_legacy_31_float_obs_validates`
  test pins backwards-compat for the `include_block_grid=false` path.
- **T3 — `--random-actions` runtime switch + `RandomLatentModel`
  baseline adapter**: new `crates/forge-mc-runner/src/random_baseline.rs`
  with `sample_random_action` + `RandomLatentModel` (no-op
  `LatentForwardModel` stub). Runner's planning step branches on
  `cfg.random_actions` to skip MCTS entirely (uniform-prior MCTS
  doesn't produce uniform action selection — peer-review noted). In
  live runs the random path also skips the ONNX bundle load entirely.
  χ²-test pins uniformity within α≈0.001.
- **T4 — `capture-baseline` CLI subcommand + Prometheus helper
  hoist**: `python -m forge.training.muzero_mc.cli capture-baseline
  --variant random|trained --episodes N --out PATH ...` drives N
  episodes against a running stack and writes a snapshot JSON.
  `scripts/mc_capture_baseline.py` is a 3-line shim. Metrics helpers
  (`fetch_prometheus_metrics`, `scrape_counter`, `scrape_gauge`)
  hoisted from `tests/python/integration/test_minecraft_e2e.py`
  into `forge.utils.metrics` so both the test and the subcommand
  consume one canonical impl. Per-variant trajectory dirs
  (`trajectories.<variant>/`) so the trainer's `_trim_replay_buffer`
  can't evict baseline files mid-capture.
- **T5 — `scripts/mc_plot_baseline.py` + Markdown report**:
  consumes the two snapshot JSONs and renders
  `docs/results/v0.5-first-real-run.md` with three matplotlib PNGs
  (reward curve, episode length histogram, reward histogram) and a
  per-variant summary table (mean / median / std / p95). Sources
  per-episode rewards from the trajectory JSON projection inside
  each snapshot (NOT from Prometheus, which only exposes
  aggregates — peer-review #14). `--no-plots` skips matplotlib for
  hosts without it; pytest tests use `importorskip` gracefully.
- **T6 — opt-in `python-test-minecraft-real-run` CI job**: new
  `workflow_dispatch` input + job that mirrors
  `python-test-minecraft-e2e`'s shape but exercises the full
  capture-baseline + plot flow with 5 episodes per variant.
- **T7 — cross-cutting logging + Rust-side channel-order pin**:
  mc-bot logs the resolved grid shape on every client connect;
  Rust runner logs `runner mode: trained|random` at startup;
  trainer rounds log version + iters + exports; capture subcommand
  logs per-episode + per-batch progress. Coordinated
  cross-language pins on `BLOCK_FEATURE_CHANNELS` (Rust + JS + the
  Python-side test in `test_muzero_mc_replay.py`).
- **T8 — `docs/results/v0.5-first-real-run.md` skeleton**:
  pre-shipped template with `{{TODO}}` placeholders the plotter
  auto-fills (summary table + PNGs) and operator-fill sections for
  the first-hour observations, anomalies, and Phase 2 outlook.
- **T9 — docs sweep**: CHANGELOG entry (this section), README +
  CLAUDE.md + Agent.md gain the new build commands,
  `examples/minecraft/quickstart.md` documents the baseline-capture
  flow, `mc-bot/README.md` documents the new `[observation]` knobs.

Known pre-existing infrastructure issue (NOT introduced by v0.5):
the `--features mc-live` build is broken on the v0.4 base branch
due to an `ort` 2.0.0-rc.9 → rc.12 API drift
(`commit_from_file` → `commit_from_memory`). Reproduces on the v0.4
worktree; tracked as a follow-up. The runner lib + binary builds
without the feature pass cleanly (85/85 unit tests).

Peer-review revisions folded in: gaps #1 (vec3 plumbing, dropped in
favor of plain `{x,y,z}` objects so `test:no-deps` still passes),
\#2 (OBS_DIM=31 defaults), #3 (Hello.grid_shape), #4
(`radius` → `grid_radius` rename), #5 (per-variant trajectory dirs
to avoid eviction), #7 (`finiteNumber` coercion), #8 (no Cargo
feature gate on random-baseline), #9 (drop `RandomAgent` reuse
claim), #10 (metrics helpers → `forge.utils.metrics`), #11
(capture-baseline as CLI subcommand, not standalone script), #12
(drop `--variant` shell flag in favor of env-var ladder), #13
(tests live flat under `tests/python/`), #14 (plot script sources
per-episode rewards from trajectory JSON), #15 (Rust-side
channel-order pin), #16 (legacy 31-float `MuZeroConfig` smoke).

### Added — Minecraft RL Integration: v0.4 self-improving training loop — live runner + continuous trainer + atomic versioned bundles (2026-05-20)

Closes the v0.3-pre BLOCKER (`ExitCode 64` "live runner wiring not
yet integrated") and delivers a **self-improving training loop**:
the runner plays Minecraft live while the trainer continuously
trains on emitted trajectories and bumps the manifest the runner
hot-reloads. Branch
`feat/mc-v04-self-improving-loop`; PR #59.

9 tracks (T1-T8 + T4a):

- **T1 — `compute-schema-id` CLI wrapper + Python schema_id twin**:
  `python -m forge.training.muzero_mc.cli compute-schema-id
  --action-map PATH --rewards PATH [--quiet]` computes the
  canonical 64-hex schema_id without spinning up the Rust binary.
  Python twin mirrors the Rust + JS canonicalisations byte-for-byte;
  cross-language fixture-hash test pins all three sides.
- **T2 — `MuZeroMcTrainerConfig.device` + GPU plumbing**:
  `device: str = "cpu"` validated cpu/cuda/auto; `--device` CLI
  flag; `MuZeroWorldModel.to` delegates to sub-networks.
  Backwards-compat: default stays `cpu`.
- **T3 — Live runner wiring (BLOCKER)**: replaces ExitCode 64
  else-branch with `run_live(cfg, metrics)` — sync function from
  `spawn_blocking`, mirrors `run_dry` shape. Loads
  `MinecraftEnvConfig` → connects bot → loads `ModelManifest` +
  `OnnxMuZeroModel` → installs reload-fn wrapper that calls
  `recorder.set_model_version(manifest.version)` on every successful
  hot-reload. New `--mc-config <TOML>` CLI flag +
  `FORGE_MC_SCHEMA_ID` env-var ladder on `RunnerConfig.schema_id`.
  New `mc-live` + `live-test-stub` Cargo features.
- **T4 — `train --continuous` + cold-start guard + replay-buffer
  hygiene**: new `train_continuous(round_iters, stop)` yields per-
  round summaries; polls `episode_paths()` until non-empty before
  `train_step` (cold-start guard). New `--continuous` /
  `--round-iters` / `--round-poll-sleep` / `--max-trajectories`
  CLI flags. `_trim_replay_buffer` deletes oldest-by-mtime files,
  always keeps newest `DEFAULT_TRIM_KEEP_NEWEST = 4`.
- **T4a — Atomic per-version ONNX bundle export (BLOCKER)**:
  `_export_bundle` writes to a NEW `v{NNNNNNNN}/` subdir; manifest
  atomically flips its pointer. Eliminates the v0.3-pre race
  where the runner could read a manifest pointing at a half-
  written bundle. `DEFAULT_MAX_BUNDLE_VERSIONS = 5` GC cap; old
  subdirs cleaned up AFTER the manifest swap so the runner has
  a safety window. Bootstrap emits the versioned layout too.
- **T5 — Compose `trainer` + `trainer-bootstrap` services + GPU
  overlay**: new `docker/trainer.Dockerfile` (parameterised
  `--build-arg TORCH_VARIANT={cpu,cu121}`); new services under
  `profiles: ["self-play"]` (existing `mc_run.sh` callers
  ignore them). New `docker/compose.minecraft.gpu.yml` overlay
  with `nvidia` device reservation. The `trainer-bootstrap`
  one-shot resolves the bootstrap chicken-and-egg.
- **T6 — `scripts/mc_self_play.sh` orchestrator**: one-command
  bring-up of the self-play stack. Preflights Compose v2 minimum,
  computes `schema_id` via the trainer-bootstrap container,
  exports `FORGE_MC_SCHEMA_ID`, runs `bootstrap` if needed, then
  dispatches to `mc_run.sh --profile self-play [--gpu] [--detach]`.
  `--dry-run` mode prints argv to STDERR. `mc_run.sh` extended
  with `--profile` / `--gpu` flag passthrough.
- **T7 — Self-improvement smoke (PR-CI gate)**: new
  `minecraft_e2e_smoke` marker that runs on every PR CI (NOT
  deselected by `addopts`). Two smoke tests drive
  `train_continuous` against a pre-seeded trajectory dir +
  assert atomic versioned-bundle layout + manifest bump within
  a 60s budget. Existing `minecraft_e2e` marker continues to
  cover the full docker compose stack via `workflow_dispatch`.
- **T8 — Docs sweep** (this commit): CHANGELOG, README,
  CLAUDE.md, Agent.md, docs/next_steps.md, docs/architecture.md
  §3.10.9-11, examples/minecraft/quickstart.md.

**Cross-cutting**:

- **No hard-coded values**: every numeric flows through a config
  struct or module-level `const` / `Final[…]` annotation.
  Cross-language constants pinned by xlang fixture-hash tests.
- **Backwards-compatible**: existing `--dry-run`, `train` fixed-
  iters mode, `bootstrap`, default `mc_run.sh` callers all
  continue to work unchanged. New `RunnerConfig` fields are
  `#[serde(default)]`; new trainer fields are dataclass defaults;
  new compose services are profile-gated.
- **TDD on every track**: each commit adds tests pinning the
  contract before / alongside the implementation.

**Validation gates** (full, post-T8):

- `cargo test -p forge-mc-runner --lib` → 79/79 pass (T3+T4a
  tests; the 3 env-var override tests consolidated into one
  `with_env_var_overrides_covers_all_scenarios` to avoid
  parallel-runner race).
- `cargo test -p forge-replay --lib` → 85/85 pass.
- `pytest tests/python/` → 1,455+ passed on PR-CI Linux (40+
  skipped on opt-in markers + missing optional extras); coverage
  92.01% (above 85% gate).
- `mypy python/ scripts/ --config-file pyproject.toml` → 0 issues
  across 102 source files.
- `ruff check python/ tests/python/ scripts/` → clean.
- `cargo clippy --workspace --all-targets -- -D warnings` → clean.
- `cargo fmt --all --check` → clean.
- `bash scripts/mc_self_play.sh --dry-run --gpu --detach` → exits 0
  with the expected docker-compose argv chain.

---

### Added — Minecraft RL Integration: v0.3-pre completion — ONNX reload + metrics + gzip + trainer + TS toolchain + opt-in E2E (2026-05-20)

Closes the five `## Deferred to follow-up PRs` items from PR #57.
Branch `feat/mc-completion-onnx-trainer-metrics-e2e-ts-gzip`.

Refactor (Track 0 — extracted before consumers landed so both old and
new call-sites share the same primitives):

- **`python/forge/training/_targets.py`** — `compute_n_step_return` lifted
  out of `MuZeroReplayBuffer._compute_n_step_return`. The original
  method is now a thin delegate; existing buffer test suite gates
  byte-stable behaviour.
- **`python/forge/training/_muzero_step.py`** — `train_with_gradients` +
  `TrainStepMetrics` + `MuZeroStepConfig` lifted out of
  `MuZeroTrainer._train_with_gradients`. Both the original trainer and
  the new `MuzeroMcTrainer` (Track 4) call into it; no duplication.
- Pin tests at `tests/python/test_targets.py` and
  `tests/python/test_muzero_step.py` lock the extracted formulas
  against hand-computed expected values.

ONNX hot-reload (Track 1):

- **`crates/forge-agent/src/latent_mcts/onnx_model.rs`** — adds
  `OnnxReloadError` (`MissingFile { path }` / `Ort(ort::Error)`),
  `validate_reload_paths`, and `OnnxMuZeroModel::reload(&mut self,
  new_config) -> Result<(), OnnxReloadError>`. Implementation is
  build-first-then-swap: all three new `Session` handles are
  constructed in stack locals BEFORE any mutex is acquired, so a
  partially-published bundle on disk never half-swaps the model. The
  `&mut self` receiver makes the borrow checker enforce sequencing
  against concurrent `&self` inference — a multi-threaded shared
  `Arc<OnnxMuZeroModel>` caller is explicitly out of scope and
  documented (would need an `ArcSwap<Sessions>` follow-up).
- **`crates/forge-mc-runner/src/onnx_reload.rs`** (new, behind
  `onnx-reload` feature) —
  `config_from_manifest(manifest, bundle_dir, action_space_size, latent_dim, num_threads) -> OnnxModelConfig`
  + `into_reload_fn(bundle_dir, action_space_size, latent_dim, num_threads) -> ReloadFn<OnnxMuZeroModel>`
  wrappers bridge the new method to the existing
  `Runner::with_reload_fn` builder hook. The runner instantiates
  the `OnnxMuZeroModel` itself; the reload callback only needs the
  bundle directory + invariants to construct the new
  `OnnxModelConfig` against the freshly-bumped manifest.
- **`crates/forge-mc-runner/Cargo.toml`** — declares the
  `onnx-reload = ["forge-agent/onnx"]` feature so the runner stays
  buildable on machines without ONNX Runtime.

Tokio + Prometheus metrics endpoint (Track 2):

- **`crates/forge-mc-runner/Cargo.toml`** — adds `prometheus = "0.13"`,
  `axum = { workspace = true }`, `tokio = { workspace = true,
  features = ["rt-multi-thread", "macros", "signal"] }`.
- **`crates/forge-mc-runner/src/metrics.rs`** (new) — `MetricsRecorder`
  exposing the v2-plan §3.6 five signals
  (`forge_mc_episode_total`, `forge_mc_episode_reward_sum`,
  `forge_mc_planning_latency_seconds`, `forge_mc_model_version`,
  `forge_mc_protocol_error_total`). Metric names live as `const &str`
  at the top of the module (single source of truth). `serve_metrics`
  binds an axum router on `{cfg.metrics_bind}:{cfg.metrics_port}` and
  returns the prometheus text-format body.
- **`crates/forge-mc-runner/src/runner.rs`** — adds
  `Runner::with_metrics(recorder)` builder. Per-step planning latency
  and per-episode reward sums are pushed into the recorder; if no
  recorder is installed, all calls are no-ops (zero-dep story
  preserved).
- **`crates/forge-mc-runner/src/main.rs`** — promoted to
  `#[tokio::main]` with a `tokio::select!` SIGINT shutdown that joins
  the runner loop (on `tokio::task::spawn_blocking`) and the metrics
  server (on `tokio::spawn`) on a single ctrl-c. `metrics_port = 0`
  in the config disables the server entirely (matches the existing
  `RunnerConfig::metrics_disabled` helper).
- **`crates/forge-mc-runner/src/config.rs`** — `metrics_bind: String`
  (default `127.0.0.1`) and `metrics_histogram_buckets: Vec<f64>`
  (default Prometheus latency buckets) added as `#[serde(default)]`
  fields. Existing configs continue to deserialise unchanged.

Opt-in trajectory gzip compression (Track 3):

- **`crates/forge-replay/Cargo.toml`** — adds `flate2 = "1"`.
- **`crates/forge-replay/src/v2.rs`** — adds `save_json_gz(path, level)`,
  extends `load_json` to auto-detect gzip by `.gz` extension, adds
  `TrajectoryGzipLevel` enum (`Named(Fastest|Default|Best)` /
  `Custom(0..=9)`) with validation, and clamps decompression at
  `MAX_DECOMPRESSED_TRAJECTORY_BYTES = 512 * 1024 * 1024` via
  `Read::take(...)` to defuse gzip bombs (a 1 KiB compressed bomb
  decompressing to 10 GiB now surfaces a clean `serde_json` EOF
  rather than an OOM).
- **`crates/forge-mc-runner/src/config.rs`** — adds
  `TrajectoryCompression { None, Gzip }` enum (default `None`) and
  `trajectory_gzip_level: TrajectoryGzipLevel` (default `Default`).
- **`crates/forge-mc-runner/src/trajectory.rs`** —
  `TrajectoryWriter::with_compression(codec, level)` builder; the
  finalize path emits either `<id>.json` or `<id>.json.gz` and the
  file extension is derived from the enum, not hard-coded at the
  call site.
- **`python/forge/training/muzero_mc/replay.py`** — mirrors the auto-
  detect + bomb-cap (`f.read(MAX + 1)` size-check) on the Python side,
  exposes `MAX_DECOMPRESSED_TRAJECTORY_BYTES` as a `Final[int]` (pinned
  to the Rust value by a cross-language test), and globs both
  `ep-*.json` and `ep-*.json.gz` in `TrajectoryReader`.

MuZero training loop (Track 4):

- **`python/forge/training/muzero_mc/trainer.py`** (new) —
  `MuzeroMcTrainer` consumes `TrajectoryReader` (which now handles
  `.json.gz` thanks to Track 3), calls `train_with_gradients` (from
  Track 0), and periodically exports the model + bumps the manifest
  the runner's reload (Track 1) picks up. `MuZeroMcTrainerConfig`
  carries trainer-only knobs (`train_iters`, `export_every_n_iters`,
  `log_every_n_iters`, `output_dir`); model hyperparameters flow
  through `MuZeroConfig`.
- **`python/forge/training/muzero_mc/cli.py`** — `train` subcommand
  added next to `bootstrap` / `validate-manifest`. Defaults from the
  config dataclass; exit codes (0/2/3/4) mirror the existing
  subcommands.

mc-bot TypeScript toolchain (Track 5 — hybrid):

- **`mc-bot/tsconfig.json`** (new) — ES2022 target with
  `allowJs: true`, `checkJs: true`, `strict: true`, `noEmit: true`.
  The toolchain typechecks the existing `.js` files in place so the
  full file-rename to `.ts` can land as a follow-up without breaking
  CI.
- **`mc-bot/package.json`** — adds `typescript`, `@types/node`,
  `@types/ws`, `tsx` devDeps + a `typecheck` npm script.
- **`.github/workflows/ci.yml`** — `mc-bot-test` job now runs
  `npm run typecheck` between `npm ci` and `npm test`, gating the
  build on `tsc --noEmit` success.

Opt-in pytest E2E (Track 6):

- **`tests/python/integration/`** (new package) — `_helpers.py`
  (polling helpers + container-state shims), `conftest.py` (the
  `compose_up_minecraft_stack` session fixture + a runner-health
  callback that surfaces a crashed container as `pytest.fail` with
  the last 50 log lines), `test_minecraft_e2e.py` (three tests
  covering the two-episode loop, the five §3.6 metrics signals, and
  `mc_run.sh --down` idempotency).
- **`pyproject.toml`** — registers the `minecraft_e2e` marker and
  extends `addopts` to deselect it by default.
- **`.github/workflows/ci.yml`** — `python-test-minecraft-e2e`
  workflow_dispatch-gated job + new `run_minecraft_e2e` boolean
  input that accepts Mojang's EULA on the runner only for the job's
  lifetime, runs the marker'd pytest, dumps each compose service's
  logs on failure, and tears the stack down with `mc_run.sh --down`.

Cross-cutting:

- All new constants flow through config structs / `Final[…]`
  annotations / module-level `const`s — no magic numbers at call
  sites.
- All new public APIs are additive — existing `OnnxMuZeroModel::load`,
  `TrajectoryV2::save_json`, `MuZeroReplayBuffer.sample_batch`,
  `RunnerConfig` parsers, and `TrajectoryReader.expected_obs_dim` all
  continue to work unchanged.
- mypy strict + ruff clean on every new Python file; zero new
  `# type: ignore` / `# noqa` lines added beyond the minimum for
  `from __future__ import annotations` typing.

Hardening + corrections (commits `19306ad`, `beb3536`, `6979009`,
plus the cross-file fixes folded into this PR's final docs sweep):

- **`thiserror` derive on `OnnxReloadError`** + compound `.json.gz`
  extension auto-detect on both sides (Rust `final_ext_is_gz &&
  stem_ext_is_json`; Python `UnicodeDecodeError` / `JSONDecodeError`
  wrap on the plain-JSON path so a stray `*.tar.gz` surfaces a
  clean `TrajectoryError` rather than gzip-bytes-fed-to-JSON
  confusion). Cross-language regression tests pin both sides.
- **`RunnerConfig::DryRunConfig`** lifts the dry-run binary's
  inline literals (`obs_dim = 8`, `action_count = 4`,
  `latent_dim = 16`, `max_episode_len = 8`) into a config struct
  with `Default` so `--dry-run` carries zero magic numbers.
- **`RunnerConfig::tokio_worker_threads`** (default
  `DEFAULT_TOKIO_WORKER_THREADS = 2`) replaces the literal
  `#[tokio::main(worker_threads = 2)]` attribute. `main` is now
  sync; the runtime is constructed explicitly via
  `tokio::runtime::Builder::new_multi_thread().worker_threads(
  cfg.tokio_worker_threads).enable_all().build()`.
- **`EPISODE_ID_PREFIX = "ep-"` + `EPISODE_ID_PAD_WIDTH = 6`** +
  `format_episode_id(seq: u64)` factored out and mirrored Python-
  side. `DEFAULT_TRAJECTORY_GLOB` / `GZIP_TRAJECTORY_GLOB` now
  derive from `EPISODE_ID_PREFIX`. Cross-language pin tests on
  both sides.
- **`MAX_DECOMPRESSED_TRAJECTORY_BYTES`** also pinned Rust-side
  (was Python-only before).
- **`_helpers.py` unit tests** (16 new) — exercise `wait_until` +
  the three docker shims via mocks, no `minecraft_e2e` marker, run
  on every PR CI invocation.
- **`train` CLI input-dir fast-fail** — `python -m
  forge.training.muzero_mc.cli train --input <missing>` now
  surfaces `EXIT_IO` BEFORE importing torch, so the CLI works in
  lint-only environments.
- **Crate-root re-exports** of the ONNX hot-reload surface from
  `forge_agent` (lets callers write `forge_agent::OnnxReloadError`
  instead of the three-segment path).
- **`MetricsError`** intentionally omits `From<MetricsError> for
  RunnerError`: the runner loop never produces / consumes
  `MetricsError`. Documented in `RunnerError`'s doc comment.
- **`organizeImports: false`** in `mc-bot/biome.json` — the Biome
  rule was tripping on pre-existing JS surfaces masked by the
  earlier typecheck failure. The `.js → .ts` rewrite (v0.4)
  re-enables both `organizeImports` and `formatter.enabled` in
  one cosmetic sweep.
- **`mc-bot/tsconfig.json` `checkJs: false`** — pure `tsc` project-
  structure gate against the legacy JS surface; `.ts` files added
  to the source tree ARE type-checked (the `strict: true` /
  `checkJs: false` combination only relaxes the JS half).

Validation gates (full):

- `cargo test -p forge-mc-runner --lib` 74 / 74; `--tests` 5 / 5.
- `cargo test -p forge-replay --lib` 85 / 85 (includes the new
  `max_decompressed_bytes_is_pinned_cross_language` + the
  `.tar.gz` regression).
- `pytest tests/python/` 1393+ passed, 33 skipped, 9 deselected;
  coverage 92.01% (>= 85% gate).
- `mypy python/ scripts/ --config-file pyproject.toml` 0 issues
  across 101 source files.
- `ruff check python/ tests/python/ scripts/` all checks passed.
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo fmt --all --check` clean.
- `cargo run -p forge-mc-runner -- --dry-run --episodes 1` exits 0.
- `cd mc-bot && npm run typecheck && npm run lint && npm test`
  green on Node 22.

---

### Added — Minecraft RL Integration: Phase 6 — Compose stack + mc-bot CI + Biome lint + quickstart (2026-05-20)

Lands the end-to-end orchestration layer for the Minecraft RL integration.
Branch `feat/mc-phase4-runner-loop` (commit `e872987`).

Orchestration:

- **`docker/mc-bot.Dockerfile`** — Multi-stage Node 22-slim image.
  Multi-arch via BuildKit (amd64 + arm64), rootless (`node` user). Every
  port + version flows through ARGs; no values are baked into the image.
- **`docker/compose.minecraft.yml`** — Three-service compose stack:
  `minecraft` (itzg/minecraft-server with EULA passed via env var, never
  baked into the image), `mc-bot` (the bridge with prismarine-viewer),
  and `runner` (the Phase-4 Rust binary). Every value is
  `${VAR:-default}` so CI / developers can override MC version, ports,
  image tags, and restart policies without editing the YAML.
- **`docker/compose.minecraft.env.example`** — Annotated example env
  file with every override exposed. EULA defaults FALSE; operators must
  opt in explicitly.
- **`scripts/mc_run.sh`** — Idempotent orchestration entry point with
  `--dry-run`, `--build`, `--detach`, `--down`, `--env-file PATH`,
  `--service NAME`, `--help`. Falls back to the example env file with a
  clear `WARN` when `compose.minecraft.env` is absent. SIGINT in
  foreground mode triggers a clean `compose down`. Zero hard-coded
  paths/values; everything derives from `$SCRIPT_DIR` or flags.
- **`examples/minecraft/quickstart.md`** — Step-by-step walkthrough:
  accept EULA, bootstrap a model bundle via the Phase-5 CLI, bring the
  stack up, watch the bot in prismarine-viewer at `:3007`, inspect
  trajectories, hot-reload by bumping the manifest version. Includes a
  troubleshooting table mapping common symptoms to causes.

mc-bot tooling:

- **`mc-bot/biome.json`** + **`mc-bot/package.json`** — Biome 1.9.4
  replaces ESLint (zero-dep, single binary, formats + lints together).
  Adds `lint`, `lint:fix`, `format` npm scripts and
  `@biomejs/biome ^1.9.4` as a devDependency. The existing 116
  `node:test` cases continue to run via `npm test`.

CI:

- **`.github/workflows/ci.yml`** — Two new CI jobs that previously did
  not exist:
  - **`mc-bot-test`** — `setup-node@v4` with Node 22, runs `npm ci` (or
    falls back to `npm install` when no lockfile is present), then
    `npm run lint` (Biome) and `npm test` (`node:test`). The 116 mc-bot
    tests now run on every CI build.
  - **`forge-mc-runner-bin`** — Builds the Phase-4 runner binary
    (`cargo build -p forge-mc-runner --bin forge-mc-runner`) and runs
    `--dry-run --episodes 1` as a smoke gate. Catches regressions in
    the env + search + writer composition without needing docker.

Gates (all green on branch tip):

- `scripts/mc_run.sh --dry-run` exits 0; tested with up (default),
  up `--build --detach`, and `--down`.
- `docker compose --env-file ... -f compose.minecraft.yml config` parses
  cleanly with the example env file.
- No literals in any new file; every port, image tag, filename, restart
  policy, EULA flag, and timeout flows through an env var.

Deferred (separate follow-up):

- Prometheus `/metrics` endpoint on `forge-mc-runner` (axum + counter /
  histogram setup).
- `tests/python/integration/test_minecraft_e2e.py` opt-in E2E test
  spinning up the compose stack from pytest.
- `mc-bot/` TypeScript migration (v2 plan §10 open decision).
- Replay storage compression (defer until output volume is measurable).

### Added — Minecraft RL Integration: Phase 5 — Python `muzero_mc` manifest mirror + replay reader + bootstrap CLI (2026-05-20)

Lands the Python side of the Phase-4/5 hot-reload loop. Branch
`feat/mc-phase4-runner-loop` (commit `4c31a7c`).

New package `python/forge/training/muzero_mc/` (5 modules):

- **`manifest.py`** — Python mirror of `forge_mc_runner::ModelManifest`.
  Pinned cross-language constants
  (`MANIFEST_SCHEMA_VERSION = 1`, `ONNX_OPSET_VERSION = 17`,
  `MANIFEST_FILENAME = "model_manifest.json"`,
  `DEFAULT_BUNDLE_FILENAMES`). `ModelManifest` /
  `ModelManifestFiles` / `ModelFileEntry` dataclasses with `validate()`
  matching the Rust invariants exactly. Atomic save (`.tmp-*.manifest`
  sibling + `os.replace`), JSON load with validation,
  `sha256_file(path)` stream-hashing helper, and a `build_manifest(...)`
  helper that assembles the manifest from three ONNX files on disk.
- **`replay.py`** — Streaming reader for `TrajectoryV2` JSONL files.
  `TrajectoryReader` yields `StepBatch` instances of configurable size;
  lazy `torch.tensor` conversion via `StepBatch.as_torch()` so module
  import does not require torch. Optional cross-checks against expected
  `obs_dim`, `action_count`, `schema_id` for fail-fast drift detection.
  Deterministic-shuffle support with seed.
- **`bootstrap.py`** — Reuses the existing
  `forge.models.muzero_export.MuZeroExporter` and
  `MuZeroWorldModel` to write a random-init ONNX bundle plus a
  versioned manifest. `BootstrapConfig` defaults flow back to
  `MuZeroConfig` for latent/hidden/blocks; the `schema_id` is
  caller-supplied (the env-handshake sha256, which bootstrap cannot
  invent). Idempotent + reproducible via `seed`.
- **`cli.py`** — `python -m forge.training.muzero_mc.cli`
  with two subcommands: `bootstrap` (writes ONNX bundle + manifest) and
  `validate-manifest` (load + validate; exit codes 0 / 3 / 4 for
  ok / validation-failed / io-error). All defaults trace back to the
  manifest module constants — no hard-coded values in the CLI.

Build:

- **`pyproject.toml`** — adds
  `[project.optional-dependencies] minecraft = ["torch>=2.0",
  "onnx>=1.16", "onnxruntime>=1.17"]` and folds the new packages into
  the `[all]` group. The `manifest` + `replay` modules import without
  torch installed (torch is only pulled in by `bootstrap` and by
  `StepBatch.as_torch`).

Tests (38 total, all passing):

- **`test_muzero_mc_manifest.py`** (18) — round-trip save / load,
  `build_manifest` from on-disk files, validation rejects schema-version
  drift / zero version / empty `schema_id` / empty per-role
  `path|sha256`, atomic-write leaves no orphan `.tmp` sibling,
  `sha256_file` matches a known reference, JSON top-level field names
  match the Rust struct exactly (catches drift without spawning a Rust
  subprocess), missing-file and malformed-JSON error shapes.
- **`test_muzero_mc_replay.py`** (13) — batch sizes / shapes,
  `format_version` rejection, expected `obs_dim` / `action_count` /
  `schema_id` cross-checks, sorted-by-default ordering, deterministic
  shuffle with seed, `batch_size=0` rejection, `StepBatch.as_torch`
  tensor shapes (gated on torch availability), missing-key rejection.
- **`test_muzero_mc_cli.py`** (7) — `validate-manifest` OK / dir-target
  / missing-file IO / invalid-JSON / schema-drift, parser introspection,
  `bootstrap` rejects invalid `--obs-dim` before reaching torch import.

Gates (all green):

- `ruff check python/forge/training/muzero_mc tests/python/training` —
  clean.
- `mypy python/forge/training/muzero_mc tests/python/training` —
  *Success: no issues found in 9 source files*.
- `pytest tests/python/training -q` — 38 passed in 0.20 s.

Backwards-compatible: this is a pure addition. The existing
`muzero_export` / `muzero_world_model` / `muzero_config` modules are
reused unchanged; manifest format is identical to the Rust runner's
expected shape.

The full MuZero training loop (loss + Adam stepping + periodic export +
manifest bump) is deferred to a follow-up. The bootstrap + manifest +
replay trio in this commit is sufficient for the Rust runner to start
end-to-end against a freshly-init bundle.

### Added — Minecraft RL Integration: Phase 4 — `Runner<E,M>` episode loop + binary (2026-05-20)

Closes the Phase-4 wire-up left over from PR #56's foundation. Branch
`feat/mc-phase4-runner-loop` (commit `b1cc7f8`).

New code:

- **`crates/forge-mc-runner/src/runner.rs`** —
  `Runner<E: FlatObsEnv, M: LatentForwardModel>` drives the full
  reset → plan → step → record → finalize loop, reusing the existing
  `LatentMctsSearch`, `TrajectoryWriter`, and `HotReloadWatcher`. Visit
  counts are normalised to a policy distribution; `root_value` becomes
  the `value_target`. Pre- and post-step obs buffers are swapped via
  `std::mem::swap` (no per-step allocation on the hot path).
- **Hot-reload** via opt-in `ReloadFn<M> = Box<dyn FnMut(&mut M,
  &ModelManifest) -> Result<(), RunnerError> + Send>` callback,
  installed through a `with_reload_fn(...)` builder. The watcher is
  polled strictly between episodes (plan §3.4); the model is borrowed
  mutably via the new `LatentMctsSearch::model_mut()` accessor.
  `prime_watcher_with(version)` lets the runner skip a spurious
  first-poll reload against an already-bootstrapped manifest.
- **`crates/forge-mc-runner/src/main.rs`** — clap CLI binary
  `forge-mc-runner` with `--config <TOML>`, `--episodes <n>`,
  `--dry-run`, `--log-level`. Live wiring against
  `forge-env-mc::MinecraftEnv` and `OnnxMuZeroModel` is the next
  follow-up; `--dry-run` exercises the loop with an in-process stub env
  + stub model so the CLI plumbing is verifiable without docker or a
  Minecraft server.
- **`crates/forge-agent/src/latent_mcts/search.rs`** —
  `LatentMctsSearch::model_mut()` and `model()` accessors (additive,
  no behavioural change).
- **`RunnerError`** gains three additive variants — `Env(String)`,
  `Planner(String)`, `Reload(String)` — so failures from the env
  trait, the planner, and the reload callback all flow through the
  same enum.

Tests added:

- **`crates/forge-mc-runner/src/runner.rs::tests`** (13) —
  `normalize_visits` uniform-on-zero and proportional-on-nonzero;
  single-episode recording with full step-count + reward + termination
  assertions and trajectory file readback; runner-side truncation at
  `max_steps`; multi-episode outcome accumulation; manifest-bump
  callback exactly-once semantics across three episodes;
  `prime_watcher_with` suppression; no-callback version recording;
  callback-error propagation as `RunnerError::Reload`; and the
  between-episode-poll contract.
- **`crates/forge-mc-runner/tests/runner_integration.rs`** (3) —
  end-to-end with manifest bump v1 → v2 and trajectory file inspection,
  reload-callback-error propagation, and zero-sim degenerate search
  uniform-policy fallback. Exercises only the re-exported public
  surface (no `#[cfg(test)]` internals).

Verified (branch tip):

- `cargo test --workspace --lib`: **27 crates, 2,546 lib tests passing**
  (+11 vs the pre-Phase-4 2,535 baseline).
- `cargo test -p forge-mc-runner`: 55 unit + 3 integration + 2
  foundation-integration tests, all green.
- `cargo clippy -p forge-mc-runner -p forge-agent --all-targets
  -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- `cargo run -p forge-mc-runner -- --dry-run --episodes 2`: end-to-end
  binary smoke pass — two trajectory files written + outcome printed.

No hard-coded values: every magic number flows through `RunnerConfig`
(buffer dims via the writer, episode caps, `action_repeat`,
`base_seed`, `planning_sims`, `metrics_port`).

Backwards-compatible: the four foundation modules' public APIs are
unchanged; the new Runner / `main.rs` / reload plumbing are purely
additive. `LatentMctsSearch::model()` and `model_mut()` are
non-breaking additions.

### Added — Branch audit follow-ups (2026-05-17)

Xlang protocol pin + viewer/actions/observation tests + runner
foundation-integration test.

Resolved tech-debt items surfaced by a full branch scan
(`claude/minecraft-phase3-wireup-runner-foundation`). The audit
flagged hard-coded constants, missing test files, untested branches
in `mc-bot/`, and one missing cross-language regression gate. None
of the audit findings were correctness bugs — the wire-format
schema_id contract is intact — but each gap is a future-rot vector.

Test additions:

- **`mc-bot/test/viewer.test.js` (new, 7 cases)** — covers
  `startViewer` end-to-end: the three ESM-interop resolution paths
  (`mineflayer` direct, `default.mineflayer`, `default` as fn), the
  not-enabled early-return, the precise error when the module
  exposes no callable, the verbatim pass-through of all viewer
  config knobs, and the no-dynamic-import guarantee when
  `viewerModule` is injected. Closes the "viewer.js has no test
  file" gap from the audit.
- **`mc-bot/test/actions.test.js` (+13 cases)** — covers `noop`,
  `attack` (both swing-arm fallback and `nearestEntity`+`attack`
  paths), `use`, `place`-missing-`activateItem`, `select_slot`,
  `look`-missing-`look`, unknown-direction, `move`-with-throwing
  `waitForTicks` (guard-fires-control-release invariant), the
  `delay`-based fallback when `bot.waitForTicks` is absent, and the
  bot/action `null`-validation paths. Forces the `actions.js`
  fallback chains and control-release-on-error finally blocks under
  test.
- **`mc-bot/test/observation.test.js` (+8 cases)** — covers the
  `bot.position`-when-no-entity fallback, the
  `bot.oxygen`-when-no-`oxygenLevel` fallback, the
  `bot.time.time`-when-no-`age` fallback, the `bot.tick`-when-no-
  `time` fallback, the `buildInventoryMap`-`slots.filter` fallback
  when `items()` is missing, the `features > 2` zero-padding loop,
  the `stableStringHash`-bad-modulus-default fallback, and the
  all-disabled observation-vector path.
- **`mc-bot/test/protocol.test.js`** — added
  `xlang_schema_version_matches_rust` regression test pinning
  `SCHEMA_VERSION` to the literal `1`, mirroring the new Rust-side
  counterpart so the constant drifts on both sides simultaneously.
- **`crates/forge-env-mc/src/protocol.rs`** — added
  `xlang_schema_version_pinned_to_known_good` Rust-side gate
  matching the JS-side test. Bumping the protocol now requires
  bumping both constants and both tests in the same PR (matches the
  existing pattern for `action_map_pinned_to_known_good` and
  `rewards_schema_id_pinned_to_known_good`).
- **`crates/forge-mc-runner/tests/foundation_integration.rs` (new)**
  — drives all four foundation modules through two episodes plus a
  between-episode manifest bump. Asserts: TOML config loads +
  validates; writer + watcher are lazy until first use; watcher
  emits on v1 with `previous=None`; same-version poll returns
  `None`; v2 poll emits with `previous=Some(1)`; trajectory files
  load round-trip through `TrajectoryV2::load_json`; no `.tmp`
  siblings remain after atomic saves; a second test exercises the
  schema_id drift detection path between `RunnerConfig` and
  `ModelManifest`. Promotes `ModelFileEntry` and
  `MANIFEST_SCHEMA_VERSION` to the crate-root re-export list.

Gates (all green):

- `mc-bot` `npm test` — 116/116 pass, 22 suites, 0 failed, 0 skipped.
- `cargo test --workspace --exclude forge-python --features
  forge-cloud/gcs` — 64/64 test-result lines OK, 0 failed.
- `cargo clippy --workspace --all-targets --features
  forge-cloud/gcs -- -D warnings` — 0 warnings.
- `cargo fmt --check` — clean.
- `cargo llvm-cov` per-file: every file modified or added on this
  branch is **>85%** line coverage (most 90-100%). Remaining
  sub-85% files (`forge-server/main.rs` 0%; `forge-cloud/{backend,
  storage, gcs_storage}.rs` 66.28–79.87%) are documented out-of-
  scope: binary entry points and network-bound code that requires a
  mock `object_store` backend.

Out-of-scope items the audit flagged but not addressed in this PR:

- `tick_ms = 50` hardcoded in `mc-bot/src/actions.js:4` — should
  flow through `EpisodeConfig.tick_ms`. Plumbing change touches
  `index.js` argument passing.
- Hotbar bounds `0..=8` duplicated in `actions.js:36` and
  `config.js:39` `inventory_slots: 9`. Move bounds derivation into
  one place.
- Chat-command effect duration/amplifier magic numbers in
  `mc-bot/src/reset.js:63-71` (`/effect give @s … 1 10`). Add a
  `reset.commands` config block.
- `DEFAULT_POSITION_SCALE`/`DEFAULT_HASH_MOD`/`DEFAULT_MAX_STACK_SIZE`
  triple-defined across `observation.js:1-6`, `config.js:39-43`,
  `env.toml:29-33`. Single-source via runtime config injection.
- `mc-bot/src/index.js` `createConnectionHandler` (43-136) is
  monolithic; should split into transport / episode-state-machine /
  message-dispatcher.
- Mineflayer-coupling refactor in `actions.js` + `observation.js`
  (introduce `ActionExecutor` + `BotSnapshotSource` interfaces).

Each is tracked in the audit report (see PR description) for
follow-up work.

### Added — Minecraft RL Integration: Phase 4 foundation (`forge-mc-runner`) + latent_mcts benchmark (2026-05-17)

Lays down the foundation pieces for Phase 4 of the Minecraft RL
integration without yet wiring up the full episode runner. Each piece
is independently tested and consumable by the eventual
`Runner<E: FlatObsEnv, M: LatentForwardModel>` loop.

New crate:

- **`forge-mc-runner`** — foundation modules:
  - `config::RunnerConfig` — episode loop knobs (counts, paths, ports,
    seeds, planning budget, action repeat) with `#[serde(default)]`
    field-wise overrides and a `validate()` invariant check.
  - `manifest::ModelManifest` + `ModelManifestFiles` — the
    `model_manifest.json` swap signal (schema version, monotonic
    `version`, `schema_id`, per-role file path + sha256). Atomic
    save (tmp-file + same-dir rename); load validates after parse.
    Pinned `MANIFEST_SCHEMA_VERSION = 1`; mismatch is a hard error.
  - `hot_reload::HotReloadWatcher` — polls the manifest for
    strictly-monotonic version bumps. Documented contract: callers
    poll only **between episodes** (per plan §3.4). Missing manifest
    returns `Ok(None)`, never an error. `prime_with(version)`
    suppresses the initial event after first-run bootstrap. Lower
    versions are silently ignored (no downgrade).
  - `trajectory::TrajectoryWriter` — episode-scoped wrapper over
    `forge_replay::v2::TrajectoryV2`. Lifecycle: `start_episode →
    record_step → finalize_and_save`. Forwards push-time validation
    (obs dim, policy dim, action range). Directory created on first
    save. Wrong-order calls return `RunnerError::WriterState`.
  - `error::RunnerError` — single `thiserror` enum spanning
    config / manifest / writer / IO / JSON / trajectory failures.
  - 42 unit tests covering happy paths, error paths, validation,
    atomic-write cleanup, and missing-manifest behaviour.

The full `Runner` loop, `LatentPlanner` adapter, and ONNX
`reload()` plumbing are deferred to a follow-up PR — this PR exists
so the runner / trainer wire formats stop diverging before the
live loop lands.

New benchmark:

- **`forge-bench::latent_mcts_inference`** (new Criterion bench) —
  measures per-decision latency of `LatentMctsSearch::search` at
  sim budgets `1 / 8 / 25 / 50 / 100 / 200` using `StubLatentModel`
  (no ONNX dep). Env-tunable via `FORGE_BENCH_MCTS_SIMS`,
  `FORGE_BENCH_MCTS_OBS_DIM`, `FORGE_BENCH_MCTS_ACTIONS`,
  `FORGE_BENCH_MCTS_LATENT_DIM`. Closes the missing-bench gap called
  out in the original audit and in `docs/next_steps.md` Phase 4.

### Added — Test coverage hardening for sub-85% files (2026-05-17)

Raised the per-file coverage floor toward 85% for the cheapest wins
identified by `cargo llvm-cov --workspace`. Workspace total stays at
~95.89% (well above the CI `--fail-under 85` gate); these additions
specifically target files that fell below the 85% per-file floor:

- **`forge-civ::grid_topology`** (was 54.55%) — added 8 dispatch
  tests exercising the `GridTopologyKind` enum dispatch arms for both
  `Square` and `Hex` variants (previously only square paths were
  exercised through the wrapper). Tests cover `num_directions`,
  `neighbor`, `neighbors`, `distance`, `line_of_sight`, `disk`,
  `Default`, and serde round-trip for both variants. `serde_json`
  added as a dev-dependency.
- **`forge-mangomas::transfer::export`** (was 81.03%) — added 4
  tests for `serialize_sweep_report`, `serialize_bdi_data`, a fully
  populated `ExportBundle` round-trip, and a non-default
  `WeightExportConfig`. Previously only the empty-bundle path was
  covered.
- **`forge-cloud::backend`** (was 43.28%) — added 2 tests covering
  the `Local` arm of `create_replay_transport` (the
  "not-implemented" diagnostic path) and the boxed-trait return path
  for `create_replay_store` / `create_model_store` with non-default
  paths. Uses `match` rather than `expect_err` because
  `Box<dyn ReplayTransport>` is not `Debug`.

Remaining sub-85% files documented as out-of-scope for this PR:

- `forge-server::main.rs` (0%) — binary entry point. Extraction of
  the bootstrap + simulation loop into a testable `lib::run` is
  tracked for a follow-up.
- `forge-cloud::{storage,gcs_storage}.rs` — network-bound code.
  Local-feature coverage further requires a mock `object_store`
  backend; live GCS smoke tests live in the opt-in `cloud-smoke`
  job, not the default test set.

### Added — Minecraft RL Integration: env-trait foundation + Node bridge (2026-05-17)

Introduces the env-agnostic abstraction layer that lets FORGE's
`latent_mcts` planner drive arbitrary environments — first concrete
new env: Minecraft (via a Node mc-bot speaking a versioned WebSocket
protocol). v1 trajectory format, classical `forge-agent::mcts`,
`forge-python::ForgeEnv`, and every existing FORGE workflow are
**untouched** — this is purely additive.

New crates / packages:

- **`forge-env`** (new crate) — generic `Env` and `FlatObsEnv` traits +
  `StepOutput`, `ObsSpec` / `ActionSpec` / `DType` space descriptors,
  and `EnvError`. `Env` requires buffer-filling `reset_into` and
  `step_into`; `reset` and `step` are allocating convenience wrappers.
  No `forge-types` dependency; consumable by any backend. Trait is
  dyn-compatible (`StepOutput<Obs, Info>` projection, not `Self`).
- **`forge-env-forge`** (new crate) — `WorldEnv` + `FlatForgeEnv`
  shims over `forge_core::WorldState`. `FlatForgeEnv` reuses caller
  buffers through `Env::step_into` for the zero-alloc hot path. 200-step
  lockstep parity test gates backwards compatibility
  (`tests/forge_env_parity.rs`).
- **`forge-env-mc`** (new crate) — sync `tungstenite`-backed
  WebSocket client to a Node mc-bot. Loads
  `configs/minecraft/{action_map,rewards}.toml`, performs a strict
  `Hello` handshake (schema_version + action_count + obs_dim +
  schema_id), and surfaces JSON `ServerMsg` errors as
  `McEnvError::Protocol`. Implements `Env::step_into` and reuses caller
  observation buffers; wire-bound I/O and JSON parsing remain the
  documented zero-alloc carve-out. Mock-server integration tests
  cover handshake mismatches, full short episodes, binary-frame
  rejection, server-close, and unexpected-Hello mid-episode.
- **`forge-replay::v2`** (additive module) — `TrajectoryV2` /
  `StepV2` carrying flat-tensor obs + MCTS `policy_target` +
  scalar `value_target`. `TRAJECTORY_FORMAT_VERSION = 2` pinned;
  readers fail fast on mismatch. v1 `Trajectory` untouched.
  `FromV1Options` + `from_v1()` migration converter takes a
  caller-supplied flatten closure (multi-agent-aware).
- **`mc-bot/`** (new Node package, ESM, Node 22+) — protocol builders
  + parser, `ActionMap` loader/validator with dense-id check,
  `RewardConfig` with the same canonical-hash contract, composable
  `RewardFn` registry (survival, inventory_acquired, distance_to_goal,
  health_delta, composite), teleport-based `applyReset`. Builtins are
  explicit factory exports (no auto-registration via side-effects) to
  dodge ESM TDZ traps in circular imports.

Cross-language regression gates (BLOCKER fix from peer review):

- `crates/forge-env-mc/src/action_map.rs::xlang_schema_id_pinned_to_known_good`
  ↔ `mc-bot/test/schema_id.test.js::"xlang schema_id matches Rust"`
  both pin `587b13077b8c7cd90503f9ee5e1bae1bb92bdf738c8abc51d2ff6deb1908224f`.
- `crates/forge-env-mc/src/reward_config.rs::xlang_rewards_schema_id_pinned_to_known_good`
  ↔ `mc-bot/test/reward_config.test.js::"hash matches Rust pinned constant"`
  both pin `451b10f995371924a374633e5c42deab35c137fbbc65bc8f551bf2bd7844b478`.
  Rust side coerces whole TOML floats to integers (`100.0 → 100`) so
  V8's `JSON.stringify` and Rust's serde produce byte-identical
  canonical strings.

Config files (no hard-coded values inline):

- `configs/minecraft/action_map.toml` — 12 default discrete actions
  (noop, 4 moves, jump, attack, use, 4 looks). Dense ids enforced.
- `configs/minecraft/rewards.toml` — default `composite` reward
  combining `distance_to_goal` (clip=100) and `survival` (value=0.01)
  with target `{0, 64, 0}`. Folded into the global `schema_id`.

Documentation:

- `docs/plans/minecraft_rl_integration_plan_v1.md` — initial six-phase
  plan (foundational design).
- `docs/plans/minecraft_rl_integration_plan_v2.md` — peer-review
  revisions: reward subsystem, episode reset, `Action: Copy` removed,
  binary-frame deferred, zero-alloc carve-out explicit,
  cross-runtime ONNX round-trip CI test, bootstrap ONNX exporter,
  `from_v1` multi-agent fix, `Cow<'_, str>` for `name()`, Phase 6
  narrowed to standalone viewer + Prometheus metrics.
- `README.md`, `Agent.md`, `CLAUDE.md`, `docs/architecture.md`,
  `docs/next_steps.md` updated for the env-trait abstraction.

Test counts:

- Rust new tests: **168** across `forge-env`, `forge-env-forge`,
  `forge-env-mc`, `forge-replay::v2`. Full workspace
  `cargo test --workspace`: 2473+ tests pass, zero failures.
- Node `mc-bot/`: **66 tests** via `node --test` (no install needed
  for the dep-free modules).
- Coverage (cargo-tarpaulin) on new Rust files: **97.2% / 523 of 538
  lines**; every file ≥ 93%.

### Changed — Teacher Pipeline Config Hoisting (2026-05-16)

Hoisted five inline numeric/string literals from the LM Studio teacher
pipeline into module-level `DEFAULT_*` constants + corresponding config
struct fields, per the project-wide "no hard-coded values" rule. All
defaults match the prior literals — behaviour is preserved.

- **`BCTrainerConfig.init_scale_numerator`** (default
  `DEFAULT_BC_INIT_SCALE_NUMERATOR = 6.0`) — Glorot-uniform weight init
  scale. Swap to `2.0` for He or `1.0` for unit-variance without
  forking the trainer. Replaces the inline `6.0` at
  `python/forge/mangomas/bc_trainer.py` `_train_numpy`.
- **`BCTrainerConfig.numerical_epsilon`** (default
  `DEFAULT_BC_NUMERICAL_EPSILON = 1e-8`) — shared additive epsilon
  inside `log()` for both the CE and KL terms. Replaces two duplicated
  `1e-8` literals.
- **`OpenAIProvider.retry_backoff_base`** (default
  `DEFAULT_LMSTUDIO_RETRY_BACKOFF_BASE = 2.0`) — multiplicative base for
  the exponential backoff (`delay = backoff_secs * base ** attempt`).
  Threaded through to `LMStudioProvider`. Replaces the inline `2**attempt`
  in both the sync and async retry loops.
- **`LLMAgentConfig.legacy_parse_keyword` / `.legacy_parse_strip_chars`**
  (defaults `DEFAULT_LEGACY_PARSE_KEYWORD = "action"`,
  `DEFAULT_LEGACY_PARSE_STRIP_CHARS = ":,. "`) — tokens used by the
  free-text fallback parser when the LLM ignores the structured
  `response_format`. Promotes the private module constants to public
  config fields so deployments that train models against alternative
  phrasings (e.g. `"move:"`) can override without forking.
- **`teacher_trace.DEFAULT_MAX_FILE_SIZE_MB`** (default `100`) — per-shard
  byte cap for `TeacherTraceWriter` before rotation. Was an inline default
  argument; now a module constant that callers can reference or override.
- **Single source of truth for the LM Studio base URL.**
  `forge.mangomas.config.DEFAULT_TEACHER_BASE_URL` now aliases
  `forge.cognitive.providers.DEFAULT_LMSTUDIO_BASE_URL` instead of
  redefining the same `"http://localhost:1234/v1"` literal. Four test
  files (`test_providers.py`, `test_teacher_config.py`,
  `test_gemma_teacher_preset.py`, `test_train_cli_llm_policy.py`) now
  import the constant and assert against it.

### Added — Regression Tests for Previously-Uncovered Branches (2026-05-16)

Six new tests targeting branches the coverage report flagged as
uncovered. Coverage moved from **93.62% → 93.77%** on a 7,146-statement
surface; the 85% gate at `pyproject.toml` `[tool.pytest.ini_options]
addopts --cov-fail-under=85` is unaffected.

- `test_providers.py::TestTruncate` — four cases covering
  `_truncate` (passthrough, exact-limit, ellipsis-marker, non-positive
  limit disables truncation).
- `test_structured_llm_agent.py::test_clip_value_handles_nan_and_non_float`
  — NaN must collapse to the configured floor (`0.0`), non-float values
  (dicts) must return `None`, `None` passthrough.
- `test_structured_llm_agent.py::test_parse_structured_non_int_action_with_validate_off`
  — non-integer `action_id` with `validate_action=False` logs a warning
  and falls back to `0` instead of raising, so a misbehaving LLM doesn't
  kill a rollout.
- `test_bc_trainer.py::test_torch_path_uses_value_loss_when_value_hats_supplied`
  — torch path's value-loss-weight branch (critic head must update when
  `teacher_value_hats` are present and `value_loss_weight > 0`).
- `test_bc_trainer.py::test_resolve_num_actions_falls_back_when_topk_has_zero_columns`
  — degenerate teacher (`teacher_top_k_probs` shape `(N, 0)`) falls back
  to `teacher_action_ids.max() + 1` instead of returning `0`.

### Added — Gemma 4 e4b LM Studio Teacher (2026-05-15)

- **`configs/cognitive/gemma_e4b_teacher.toml`**: LM Studio teacher preset for
  the `google/gemma-4-e4b` model, with companion template
  (`configs/cognitive/templates/gemma_teacher.txt`), action schema
  (`python/forge/cognitive/schemas/gemma_action.json`), and few-shot exemplars
  (`configs/cognitive/few_shots/gemma_teacher.jsonl`). Loadable via
  `MangoMASBridgeConfig.from_toml`.
- **`scripts/run_lmstudio_demo.py`**: reusable local helper. `--check` pings
  the configured LM Studio endpoint with the preset's model and logs
  `tokens_in/out/latency_ms`. No hard-coded model ids, URLs, or timeouts —
  every value flows through `TeacherConfig`. For real collection runs, use
  `scripts/train.py --collection-policy llm`.
- **`tests/python/test_lmstudio_url_invariant.py`**: regression test pinning
  the LM Studio base URL to `/v1` (and rejecting any `/api/...` path), so a
  future "fix" can never silently break the OpenAI-compatible request route.
- **`tests/python/snapshots/gemma_prompt_minimal.txt`**: byte-stable snapshot
  of the Gemma prompt rendering for a fixed observation. Catches silent
  template corruption that a determinism check alone would miss.
- **`tests/python/test_lmstudio_live_smoke.py`**: gated
  (`FORGE_LMSTUDIO_LIVE=1`) live smoke test that round-trips the Gemma preset
  against a running LM Studio instance.
- **`DEFAULT_LMSTUDIO_API_KEY`** constant in
  `python/forge/cognitive/providers.py` — removes the last inline literal in
  the LM Studio client and gives downstream tests a single symbol to override.

### Changed — Gemma 4 e4b LM Studio Teacher (2026-05-15)

- **`configs/cognitive/default.toml`** `[cognitive.lmstudio].model` now
  defaults to `google/gemma-4-e4b`; `[cognitive.structured]` paths point at
  the new Gemma assets. The Qwen preset
  (`configs/cognitive/qwen14b_teacher.toml`) remains the authoritative
  `MangoMASBridgeConfig`-shape file for callers that name it explicitly.
- **LM Studio provider tests** (`test_providers.py`,
  `test_lmstudio_provider_async.py`) are now parametrised across the qwen +
  gemma model ids via a shared `lmstudio_model_id` fixture in `conftest.py`,
  proving the provider is model-agnostic at the transport layer.
- **`scripts/train.py`** `--teacher-model` help string now headlines Gemma
  with Qwen retained as a sibling example.
- **`.github/workflows/ci.yml`** triggers on PRs and pushes targeting
  `v0.2/implementation` in addition to `main`/`master`/`develop`. Previously
  PRs against `v0.2/implementation` skipped every CI job.
- **`[tool.maturin]`** in `pyproject.toml` now includes JSON schemas under
  `python/forge/cognitive/schemas/` and TOML/prompt/few-shot files under
  `configs/cognitive/**` in sdist/wheel builds, so installed wheels carry
  the full LM Studio teacher surface.

### Added

#### LM Studio + Qwen 14B Teacher for Offline BC / SFT

- **`LMStudioProvider`** (`python/forge/cognitive/providers.py`): OpenAI-compatible
  subclass of `OpenAIProvider` with LM Studio defaults
  (`base_url=http://localhost:1234/v1`, longer timeout, light retry-with-backoff).
  Both sync (`complete`) and async (`acomplete` via `openai.AsyncOpenAI`) paths.
  Registered in `create_provider` as `"lmstudio"`.
- **`CognitiveProvider.acomplete`** (concrete, not abstract): default impl
  off-loads `complete` to a worker thread via `asyncio.to_thread`. Third-party
  subclasses gain async support without modification.
- **`CompletionConfig`** gains optional `response_format`, `seed`, `top_p`,
  `extra_body`, `timeout_secs` (all default `None`; only forwarded when set).
- **`PromptBuilder`** (`python/forge/cognitive/prompt_builder.py`):
  deterministic template-driven prompt rendering with optional JSONL
  few-shots; no Jinja dependency.
- **`StructuredLLMAgentConfig` / `LLMAgent.aact`**: JSON-mode teacher agent
  that emits rich `trace_info` (`intention`, `subgoals`, `rationale`,
  `value_hat`, `constraint_critique`, `top_k_probs`, token counts, latency).
  Malformed JSON falls back to the legacy integer extractor; invalid
  `action_id` raises when `validate_action=True`.
- **`TeacherConfig`** + `[teacher]` TOML section in `MangoMASBridgeConfig`;
  preset `configs/cognitive/qwen14b_teacher.toml`. Override any field via
  `FORGE_TEACHER_<UPPER_SNAKE>` env vars or new CLI flags
  (`--teacher-config`, `--teacher-model`, `--teacher-base-url`,
  `--teacher-concurrency`, `--teacher-output-root`).
- **`TeacherDecisionTrace` + `TeacherTraceWriter` / `TeacherTraceReader`**
  (`python/forge/mangomas/teacher_trace.py`): JSONL shards under
  `<output_root>/<scenario_id>/ep<episode:06d>-<shard:04d>.jsonl[.gz]`.
  Composes `forge.traces.trace_logger.TraceLogger`; `TraceLogger.log` widened
  to accept any `TraceRecord` Protocol (backwards-compatible with
  `DecisionTrace`).
- **Collector `policy_name="llm"`** branch with two paths:
  * **Sync** (`teacher.concurrency=1`): per-step traces streamed to a writer
    per `(scenario_id, episode_index)`.
  * **Async** (`teacher.concurrency>1`): episodes run concurrently under
    `asyncio.Semaphore(concurrency)`; trace shards written in
    `episode_index` order after `asyncio.gather` so on-disk bytes depend
    only on `(base_seed, scenario_id, episode_index)`, not coroutine
    completion order.
- **`BCTrainer` + `BCDataset` + `BCTrainResult` + `BCTrainerConfig`**
  (`python/forge/mangomas/bc_trainer.py`): pure behavioural-cloning trainer.
  NumPy path: linear softmax classifier with CE + optional KL on
  `top_k_probs`. Torch path (guarded import): fine-tunes
  `ActorCriticNetwork` actor head in-place. Scope is BC only — no DAgger /
  DPO.
- **`MangoMASPipeline._run_bc_stage`** prepended to `run()`. No-op when
  `CollectedTrainingData.teacher_intentions` is empty.
- **Teacher-aware build_dataset kwargs**:
  * `BDIPreTrainer.build_dataset(*, teacher_intentions=...)` — per-episode
    integer labels override the rule-based `DEFAULT_ACTION_INTENTION_MAP`.
  * `ConstitutionalPreTrainer.build_dataset(*, teacher_constraint_critiques=...,
    teacher_severity_default=...)` — teacher critiques OR-merged with the
    rule-derived violation matrix; penalty recomputed as
    `max(rule_penalty, teacher_flags * severity * penalty_weight)`.
- **`CollectedTrainingData`** gains optional `teacher_*` per-episode lists.
  `validate()` checks length parity only when fields are non-empty.
- **Shared env-override helper** `forge.utils.config_env.apply_env_overrides`,
  extracted from `forge.config._apply_env_overrides`. Reused by both
  `ForgeConfig` and `MangoMASBridgeConfig`.
- Documentation: README "LM Studio Teacher (offline behavioural cloning)"
  section under MangoMAS Collection.

### Fixed

- **`OpenAIProvider` silent token-capture bug**: `complete()` previously
  returned `input_tokens=0, output_tokens=0` regardless of what the API
  reported. Now reads `response.usage.prompt_tokens` /
  `completion_tokens` and propagates them into `CompletionResponse`. The
  existing dataclass-defaults test (`CompletionResponse()` → `(0, 0)`) is
  preserved unchanged — only API-returned values change.
- **Async teacher traces recorded wrong `legal_actions`** (post-review
  audit, commit `3a3eafc`): the asyncio path computed
  `legal_actions = range(action_ids.max() + 1)` which underestimated the
  action space whenever an episode never exercised the highest legal
  action. Now threads `action_space_size` through `_EpisodeRollout` from
  `env.action_space.n` and uses it directly. Regression test asserts a
  4-action env always produces `legal_actions=[0,1,2,3]` even when the
  teacher only selects `action_id=1`.
- **`BCTrainer.train` on an empty dataset silently reported success**
  (post-review audit, commit `3a3eafc`): the trainer would run zero-sample
  epochs and report `loss=0.0, accuracy=0.0`. Now logs a WARNING and
  returns `BCTrainResult(epochs_run=0)` early; subsequent
  `export_weights` raises rather than writing a no-op file.
- **`apply_env_overrides` silently miscast complex types** (post-review
  audit, commit `3a3eafc`): env vars for `list[…]` / `dict[…]` fields
  would silently take the raw string. The helper now raises
  `UnsupportedFieldType` (logged as a WARNING; affected fields are
  skipped). Today's `TeacherConfig` has only primitive fields, so this is
  a defensive hardening rather than a live bug fix.

### Removed

- Inert `--bc-train-after-collect` CLI flag (post-review audit). It was
  parsed but never read; the BC stage decision lives in
  `MangoMASPipeline._run_bc_stage` and keys off the presence of teacher
  data. See `docs/architecture.md` §3.9 and the README LM Studio Teacher
  subsection.

#### Fallible Action Encoders + Structured Scenario Errors (PR #39)

- Added **`ActionEncodingError`** in `crates/forge-types/src/error.rs` with four variants — `DroneActionRequiresFullEncoder`, `AgriActionUnsupported { drone_actions_enabled, agri_actions_enabled }`, `HexActionUnsupported { hex_actions_enabled }`, and `ParameterOutOfRange { action_name, value, max }` — wired into `ForgeError` via `#[from]`.
- Added **fallible action encoders** in `crates/forge-types/src/action.rs`: `Action::try_to_discrete()`, `Action::try_to_discrete_full(comm_vocab_size)`, and `Action::try_to_discrete_configured(comm_vocab_size, drone, agri, hex)`. The legacy panicking entrypoints (`to_discrete`, `to_discrete_configured`) now delegate to the fallible variants and `unwrap_or_else(panic!)`, so behavior is byte-identical for callers that already guarantee in-range inputs.
- Added **parameter-bounds validation** for every parameterized action variant via the new `param_check` helper. Out-of-range parameters that would otherwise silently produce a colliding discrete ID — `Action::Drop(slot >= 10)`, `Action::Use(slot >= 10)`, `Action::Craft(recipe >= 9)`, `Action::Communicate(token >= comm_vocab_size)`, `Action::DropPayload(slot >= 10)`, `Action::Spray(slot >= 10)` — now return `ActionEncodingError::ParameterOutOfRange`. Slot limits flow through new `ACTION_DROP_PAYLOAD_SLOTS` and `ACTION_SPRAY_SLOTS` constants in `forge-types::constants` (no more hard-coded `10` in encoder helper calls).
- Added **`ScenarioConfigError`** (`thiserror`-derived) in `crates/forge-scenario/src/config.rs`, replacing `Result<_, String>` on `ScenarioConfig::from_toml` / `to_toml`. Wraps the underlying `toml::de::Error` / `toml::ser::Error` so callers preserve span / line context for diagnostics.
- Added **22 new `forge-types::action::try_encoder_tests`** covering happy paths, every `ActionEncodingError` variant, parameter-bounds rejection, `ForgeError` conversion, Display formatting, the agri-gating-takes-precedence-over-bounds invariant, and `should_panic` regression guards proving the legacy entrypoints still panic identically.
- Added **1 proptest** (`try_and_panic_agree_on_happy_path`) that fuzzes 256 random `(vocab_size, drone, agri, hex, action_id)` layouts and asserts the fallible / panicking encoders never disagree on the happy path.
- Added **`docs/architecture.md` §4.7 Structured Error Types** documenting the new `ActionEncodingError` taxonomy and the relationship between fallible and panicking encoder entrypoints.

### Changed

#### Lint Surface (PR #39)

- **Ruff**: 94 → 0 errors across `python/ tests/python/ scripts/ demo_ui/ examples/`. Suppressed `PLC0415` globally with a documented justification (FORGE has ~50 deliberate lazy imports for optional ML deps); annotated four intentionally-linear functions with `# noqa: PLR09xx` (two example training loops, the MangoMAS config dispatcher, the pipeline stage runner); moved annotation-only `numpy` / `numpy.typing.NDArray` imports into `TYPE_CHECKING` blocks; added explicit `__all__` lists to `python/forge/__init__.py` and `python/forge/training/__init__.py`. Also fixed a real `F821` bug in `examples/train_ppo.py` (`Any` referenced without import).
- **Mypy**: 26 → 0 errors across `python/ scripts/`. Added `anthropic`, `openai`, `datasets`, `tomli_w` to the existing `[[tool.mypy.overrides]]` block; removed 21 stale `# type: ignore[...]` comments across `forge.mangomas.*`, `forge.agents.muzero_mcts`, `forge.testing.env_factory`, `forge.social.trust_tracker`, `forge_env.vecenv`; replaced a redundant `cast()` in `forge.mangomas.constitutional_trainer.export_weights`; fixed two real `[no-any-return]` issues in `forge.mangomas.adapters` (`_quantize` and `total_action_space` were silently returning `Any`).
- **Workspace dependency consolidation**: `crates/forge-server/Cargo.toml` now inherits `tracing-subscriber` from the workspace (`{ workspace = true }`) instead of declaring its own version. Was the only crate not consolidated; the per-crate declaration was a version-skew risk.

#### v0.2 Tier 1 Closeout

- Added **multi-agent allocation audit**: `crates/forge-bench/src/bin/allocation_audit.rs` now accepts `--agents <comma-list>` (default sweep `1,8,16,32,64,128`, matching `multi_agent_scaling.rs`), honours the `FORGE_BENCH_AGENT_COUNTS` env var, and emits one `VariantReport` per `(action, agent_count)` pair. Row labels use the `<base>@n=<count>` form and carry a typed `num_agents: u32` field so downstream tooling can filter without parsing the variant string. Five new unit tests cover the argv parser.
- Added **four Playwright E2E browser tests** to `demo_ui/tests/test_e2e_browser.py`: `TestSectionStateMachine` (IDLE → RUNNING → PASS), `TestTerminalStream` (live span accumulation, no `.c-fail` spans, no console errors), `TestProgressBar` (progress advances under `runAll`), and `TestWorldCanvas` (non-zero pixels after a run). New function-scoped `fresh_page` fixture isolates per-test DOM and console listeners.
- Added **`benchmarks/baselines/reference_b/` scaffolding** (`.gitkeep`) so the directory commits; the regeneration commands in `benchmarks/baselines/README.md` document how to populate it locally on a non-CI host.

#### v0.2 Implementation Hardening

- Added **5 hex-grid integration tests** covering full episode cycle, deterministic replay, multi-agent episodes, square-move rejection, and serialization roundtrip (`tests/rust/integration_tests.rs` — 29 total passing)
- Added **2 hex benchmark groups** (`bench_step_hex_single_agent`, `bench_step_hex_multi_agent`) to `forge-bench` Criterion benchmarks (7 groups total)
- Added **22 new `forge-replay` tests** across compact replay, trajectory, export, and config modules (36 → 58 tests)
- Added **16 new `forge-scenario` tests** across registry, compose, and config modules (43 → 59 tests)
- Added **`tests/python/test_mangomas_smoke.py`** with 21 end-to-end smoke tests covering:
  - TOML config loading for all 8 `configs/mangomas/*.toml` files
  - Config-to-episode wiring for action adapters, observation adapters, and batch collection
  - Pipeline stage integration for BDI, constitutional, RSSM, curiosity, and curriculum controllers
  - Export pipeline integrity with full roundtrip and weight loading verification
  - End-to-end pipeline execution through the BDI stage
- Added **`python-test-fast` CI job** — runs pure-Python tests without maturin/native extension build for faster PR feedback
- Added **Docker GHCR publishing job** with multi-arch support (`linux/amd64` + `linux/arm64`), Docker Buildx, semver tag extraction, and GitHub Actions cache

### Changed

- **Benchmark regression gate**: `critcmp` comparison changed from warning to hard failure — regressions >5% now block PRs
- **Python type compliance**: Fixed all 10 mypy errors across `muzero_buffer.py`, `wrappers.py`, `muzero_mcts.py`, and `pyproject.toml` override configuration — 86 source files pass mypy strict
- **Python formatting**: Applied `ruff format` across 40 Python files; resolved all E501 line-length violations
- **`.gitignore`**: Added entries for stale build artifacts (`clippy_output.txt`, `demo_results.md`)
- **`demo-ui` CI job**: Now installs `demo_ui[dev]` extras and runs `playwright install --with-deps chromium` so the Playwright suite executes (previously skipped via `pytest.importorskip`); pip cache enabled with `cache: pip` and `cache-dependency-path` covering both the requirements file and `pyproject.toml`. New `FORGE_DEMO_PORT=18765` keeps the E2E uvicorn separate from the existing `FORGE_DEMO_UI_PORT=8765` health smoke test.
- **`demo_ui/pyproject.toml`**: pinned `playwright==1.48.*` (was `>=1.40`) for CI reproducibility and added `pytest-timeout>=2.3` to the dev extras.
- **Allocation audit baseline**: `benchmarks/baselines/reference_a/alloc_audit.json` regenerated under the multi-agent sweep — 72 rows (12 variants × 6 agent counts) all satisfying `total_bytes == 0`. Variant labels changed from `Move_Up` to `Move_Up@n=1` etc., which is a breaking change for any external tooling that memorised the old strings; the in-tree validator (`benchmarks/runner/check_zero_alloc.py`) treats `variant` as opaque and is unaffected.

### Fixed

- **Multi-agent zero-allocation regression** in `crates/forge-core/src/physics.rs`: the per-step aerial-collision snapshot was a local `SmallVec` with inline capacity 16, so any step with `num_agents > 16` spilled to the heap once and allocated 6 bytes per agent on every subsequent tick. Snapshot now lives on `PhysicsScratch::agents_snapshot`, sized via the existing `ensure_capacity` path, restoring the zero-alloc contract at every agent count exercised by the audit (verified up to `n=128`). Surfaced by the new multi-agent allocation audit. Removed the now-unused `forge_types::constants::PHYSICS_SMALLVEC_CAPACITY`.

### Fixed

- Fixed `forge-scenario` registry tests using incorrect data assumptions (OR-logic for tag matching, correct scenario names)
- Fixed `compose_scenarios` test expecting `Result` when function returns `Option`
- Fixed mypy `# type: ignore[return-value]` → `[no-any-return]` for 5 return statements in `muzero_mcts.py`
- Fixed `numpy.signedinteger` → `int` cast in `muzero_buffer.py` for mypy compliance
- Removed stale `# type: ignore[no-any-return]` from `wrappers.py` after numpy stub alignment

---

### Added

#### Hex Grid Topology & Dynamic Action Spaces

- Added **`forge-civ`** as a dedicated topology crate for square and hex grids, including:
  - `GridTopology` / `GridTopologyKind` dispatch for square and odd-r hex worlds
  - shared neighbor lookup, distance, line-of-sight, disk queries, and A* pathfinding
  - topology-focused unit and property coverage for square and hex behaviors
- Added `GridType` to world config plus `MoveHex(HexDirection)` support in `forge-types`
- Added config-aware action encoding/decoding across Rust, Python, WASM, evaluation, and MangoMAS runners so discrete action IDs match the active grid/drone/agri layout
- Added `configs/scenarios/hex_patrol.toml` as a reusable hex-grid scenario fixture

#### MangoMAS Collection & Pipeline Control Plane

- Added `python/forge/mangomas/collector.py` for scenario resolution, batch FORGE rollout collection, and JSON collection-report generation
- Added `python/forge/mangomas/pipeline.py` for stage-based MangoMAS orchestration covering BDI, constitutional, RSSM, curiosity, sweep, curriculum, and export stages
- Added missing MangoMAS config surfaces for pipeline paths/logging/execution and transfer overrides, with TOML parsing support
- Added MangoMAS CLI entry points in `scripts/train.py`:
  - `--agent mangomas-collect` for collection-only workflows
  - `--agent mangomas` for collection plus stage-pipeline execution
  - scenario, collection-policy, pipeline output, and collection-report flags

#### Regression Coverage

- Added focused Python regression coverage for MangoMAS config parsing, collector decoding, pipeline manifests, and training CLI argument parsing
- Added targeted hex-grid action-space and visibility regressions in Rust and Python so square-only assumptions fail fast during review

#### Cloud Training Pipeline & Edge Deployment (`forge-cloud`, `forge-edge`)

- **`forge-cloud` crate** (125 tests): Distributed cloud training infrastructure.
  - Config types: `CloudConfig`, `WorkerConfig`, `CoordinatorConfig`, `StorageConfig`, `ReplayTransportConfig`, `ModelRegistryConfig` with `AggregationStrategy`, `StorageBackend`, `FallbackPolicy` enums.
  - Error types: `CloudError`, `WorkerError`, `TransportError`, `StorageError`, `ModelRegistryError` with `CloudResult<T>` alias.
  - Core traits: `ReplayStore`, `ModelStore`, `WorkerManager` with `WorkerMetadata`, `WorkerInfo`, `WorkerStatus`, `SeedAssignment` data types.
  - `InMemoryWorkerRegistry`: Thread-safe in-memory `WorkerManager` implementation for testing and single-node deployments.
  - `LocalReplayStore`: Filesystem-backed `ReplayStore` for CompactReplay persistence.
  - `LocalModelStore`: Filesystem-backed model store implementing both `forge-cloud::ModelStore` (u32 versions) and `forge-types::transport::ModelStore` (string versions).
  - `TrajectoryReconstructor`: Reconstructs full `Trajectory` objects from `CompactReplay` via deterministic replay, with batch support producing `OfflineDataset`.
  - Replay compression/decompression and `ReplayBatch` for transport.
  - 22 constants with `DEFAULT_*` prefix and validation tests.

- **`forge-edge` crate** (40 tests): Edge deployment runtime.
  - `AdaptiveMctsSearch<M>`: Wraps `LatentMctsSearch` from forge-agent with latency budgeting. Estimates per-simulation cost via EMA, adjusts `num_simulations` per search call, clamps to `[min, max]` from `EdgeConfig`.
  - `LatencyEstimator`: EMA-based per-simulation latency tracker with configurable alpha.
  - `TelemetryCollector`: Bounded store-and-forward buffer for `CompactReplay` with flush via `ReplayTransport` trait.
  - `EdgeAgent<M>`: Composite agent implementing `AgentInterface` — flattens observations, runs adaptive MCTS, falls back to Noop on error. Compatible with `EvalHarness`, `BatchRunner`, and all FORGE infrastructure.
  - `AdaptiveSearchMetrics` and `TelemetrySnapshot` diagnostic types.

- **Foundation types in `forge-types`**:
  - `CloudConfig` and `EdgeConfig` added to `ForgeConfig` with `#[serde(default)]` for full backward compatibility.
  - `CloudError` (7 variants) and `EdgeError` (5 variants) with `#[from]` conversions.
  - `ReplayTransport` and `ModelStore` traits in new `transport` module.
  - 25 `DEFAULT_CLOUD_*` and `DEFAULT_EDGE_*` constants with validation tests.
  - Environment variable overrides for `cloud.*` and `edge.*` config sections.

- **`EdgeReplayLoader` in `forge-data`** (18 tests): Implements `DatasetLoader` for edge telemetry ingestion — scans directory for `.bin` CompactReplay files, reconstructs trajectories, returns `OfflineDataset`.

- **Integration test** (`tests/rust/integration_cloud_edge.rs`, 9 tests): Exercises full cloud-edge data loop including storage roundtrips, reconstruction determinism, EdgeAgent evaluation, telemetry flush, worker lifecycle, backward compatibility, and model version management.

- **Distributed Docker** (`docker/docker-compose.distributed.yml`): Coordinator + scalable worker services with shared volumes and `FORGE_CLOUD_*` / `FORGE_EDGE_*` environment variable configuration.

- **Training config** (`configs/training/distributed.toml`): Complete distributed training configuration with cloud and edge sections enabled.

- **Proposal** (`docs/cloud_edge_proposal.md`): GCP-specific technical proposal with MouseDroidAGI flagship use case, architecture diagrams, cost analysis, and 4-phase implementation roadmap.

#### MuZero Latent-Space Planning & ONNX Integration

- Embedded a full `MuZeroWorldModel` across Python and Rust for evaluating search in latent-space environments.
- Added `forge_agent::latent_mcts` with `LatentMctsSearch`, capable of dynamically routing inference through a generic `LatentForwardModel` trait.
- Added `OnnxMuZeroModel` backend utilizing `ort` (ONNX Runtime v2) with safe mutex session management for E2E MCTS evaluations.
- Implemented `MuZeroExporter` to natively convert the multi-head PyTorch MuZero network instances into `.onnx` binaries natively.
- Developed an end-to-end integration test (`onnx_integration.rs`) to automatically orchestrate Python ONNX export and Rust latent graph search.

#### MangoMAS Bridge Coverage And Training Surface

- Added targeted Python coverage for the MangoMAS bridge components: constitutional pre-training, adaptive curriculum control, curiosity-weight optimization, and MCTS sweep reporting
- Added a config-driven discrete SAC training preset in `configs/training/sac_default.toml` and expanded `examples/train_sac_cleanrl.py` to honor TOML-backed model and feature-extractor settings
- Added branch-specific regression tests for vectorized env wrappers, feature extractors, and pure-Python `forge_env` import/fallback behavior

#### Python Coverage Expansion

- Added `tests/python/test_device.py` to cover accelerator detection paths in `forge.utils.device`
- Expanded `tests/python/test_mappo.py` with config-factory, auto-device, batched action, and `RandomPolicyNetwork` coverage
- Expanded `tests/python/test_forge_env.py` to exercise `forge_env.__init__`, `forge_env.utils`, wrapper edge cases, and pure-Python fallback branches

### Changed

#### MangoMAS Configuration Hardening

- Consolidated MangoMAS bridge defaults into `python/forge/mangomas/config.py` so curriculum tiers, constitutional constraints, curiosity weights, sweep bounds, and batch collection settings all flow from configuration objects
- Updated the constitutional trainer, curiosity optimizer, curriculum controller, batch collector, and sweep runner to consume shared config defaults instead of duplicating literals in module code
- Expanded `python/forge_env/__init__.py`, `feature_extractors.py`, and `vecenv.py` to better tolerate optional native or ML dependencies while keeping the package importable for pure-Python validation

#### Rust Coverage Hardening

- Converted naive `unwrap()/expect()` calls inside `onnx_model.rs` and `LatentForwardModel` into robust `anyhow::Result` boundaries bubbled up through the `search` pipeline.
- Added targeted Rust coverage for MCTS terminal-search and short-priors fallback behavior in `forge-agent`
- Fixed outdated 2-argument signature definitions wrapped by duplicate `mod proptests` in `action.rs` and `config.rs`.
- Expanded predicate, validation, and terrain edge-case coverage across `forge-task`, `forge-types`, and `forge-worldgen`

#### Python Gap Analysis Cleanup

- Replaced remaining hard-coded Python values with named constants in the Gymnasium env wrapper, MAPPO reward normalization, trainer checkpoint defaults, dashboard client tests, and shared pytest fixtures
- Enforced a Python coverage floor with `pytest --cov-fail-under=85` in `pyproject.toml`
- Standardized Python test fixtures and assertions around exported wrapper constants instead of duplicated literals

#### Docker Multi-Service Deployment (`docker/`)

Production-ready Docker Compose stack with three independently deployed services:

**Simulation Service** (`docker/Dockerfile`)

- Upgraded Rust base image to `1.85` (required for `fixed` crate edition 2024)
- Builds `forge-server` binary via multi-stage `rust:1.85-bookworm` → `python:3.11-slim-bookworm`
- Builds `forge_env` native Python extension via `maturin build -m crates/forge-python/Cargo.toml`
- `HEALTHCHECK` on `/health` endpoint with 15s interval, 3s timeout, 3 retries

**Dashboard Service** (`docker/Dockerfile.dashboard`, `docker/nginx.conf`)

- Dedicated `node:20 → nginx:1.27-alpine` multi-stage image (~40 MB vs monolithic)
- Nginx serves the React SPA with SPA routing (all paths → `index.html`)
- Reverse proxies `/api/` and `/ws` → `simulation:8080` for same-origin access
- `/healthz` endpoint to satisfy Docker health checks

**Demo UI Service** (`docker/Dockerfile.demo`)

- No changes to the Dockerfile, but fully integrated into the new Compose stack
- Exposed on `http://localhost:8765`

**Orchestration** (`docker/docker-compose.yml`)

- Bridge network `forge-net` for inter-service communication by name
- Health-gated `depends_on`: dashboard + demo wait for `simulation` to be `healthy`
- `restart: unless-stopped` for production resilience
- All ports bound to `127.0.0.1` for security

**Build Context** (`.dockerignore`)

- Excludes `target/`, `node_modules/`, `.git/`, caches, and coverage artifacts

#### Dashboard TypeScript Fixes

- `tsconfig.json`: Added `"types": ["vite/client"]` for `import.meta.env` recognition
- `tsconfig.node.json`: Added `"types": ["node"]` for `process.env` in `vite.config.ts`
- `vite.config.ts`: Added `/// <reference types="vitest" />` triple-slash directive
- `App.tsx`: Prefixed unused `setSelectedAgent` → `_setSelectedAgent` (`noUnusedLocals`)
- `package.json`: Added `@types/node` devDependency

#### Python Test Quality

- Fixed `# noqa: PLC0415` directives across test files (ruff RUF100 auto-fix)
- Fixed `TC003` in `test_gymnasium_env.py`: moved `Generator` import into `TYPE_CHECKING` block
- Improved type annotations from `object` → specific env types (`ForgeGymnasiumEnv`, `ForgeParallelEnv`)

### Deployment URLs

After `docker compose -f docker/docker-compose.yml up -d`:

| Service | URL | Health |
|---------|-----|--------|
| Simulation (Rust/Axum) | `http://localhost:8080` | `GET /health` |
| Dashboard (React/nginx) | `http://localhost:3000` | nginx |
| Demo UI (FastAPI) | `http://localhost:8765` | `GET /health` |

#### Interactive Demo UI (`demo_ui/`)

A full-stack interactive web application that streams the FORGE demo live in the browser.

**Backend** (`demo_ui/backend/`)

- `main.py` — FastAPI application with Server-Sent Events (SSE) endpoints:
  - `GET /` — serves the single-page frontend
  - `GET /health` — liveness check
  - `GET /api/sections` — list all 8 demo sections (key, name, index)
  - `POST /api/run/{section}` — stream a single section's output via SSE
  - `POST /api/run-all` — stream all 8 sections sequentially via SSE
  - `GET /api/results` — return parsed `demo_results.md` as JSON
- `forge_runner.py` — async subprocess wrapper around `forge_demo.py`:
  - `run_section()` async generator — streams stdout line by line
  - `run_all()` — emits `__SECTION_START__`/`__SECTION_END__` sentinel tokens
  - `parse_results_md()` — regex-based parser for baseline results markdown
  - `SECTIONS` dict mapping slug → display name for all 8 demo sections

**Frontend** (`demo_ui/frontend/`)

- `index.html` — single-page app with header, sidebar nav, live terminal, stats panel, footer controls
- `styles.css` — premium dark-mode design system: glassmorphism cards, color tokens (`--cyan`, `--pass`, `--fail`), `fade-in` micro-animations, responsive grid layout
- `app.js` — modular vanilla JS:
  - `ForgeTerminal` — ANSI-aware live terminal with syntax highlighting (PASS/FAIL colors, grid colorization)
  - `WorldRenderer` — ASCII grid → canvas renderer using `TERRAIN_COLORS` palette
  - `SectionNav` — sidebar state machine (idle → running → pass/fail) with mini results panel
  - `Runner` — SSE orchestrator: manages `AbortController`, labeled-loop sentinel parsing, progress bar, and timer

**Tests** (`demo_ui/tests/`)

- `test_backend.py` — 15 unit + integration tests (all passing):
  - `parse_results_md()` — structure, values, 8 sections, performance metrics, missing file
  - API endpoints — `/health`, `/api/sections` (count/keys/schema), `/api/results`, 404 for unknown section
  - SSE streams — `/api/run/worldgen` and `/api/run-all` return `text/event-stream`
  - `SECTIONS` constant — is dict with 8 expected keys
- `test_sections.py` — functional tests for output presence and keyword validation per section
- `conftest.py` + `pytest.ini` — asyncio-auto mode, `anyio` backend

#### Launcher

- `run_demo.ps1` — PowerShell one-click launcher:
  - Installs `demo_ui/backend/requirements.txt`
  - Starts `uvicorn` on `http://127.0.0.1:8765`
  - Opens browser automatically

#### Root Fixes

- `conftest.py` (repo root) — injects FORGE root into `sys.path` so `demo_ui` is importable from any CWD
- `.gitignore` — added `demo_ui` artifact exclusions, `.pytest_cache/`, `.claude/`

### Performance (from baseline run)

- 8/8 demo sections pass in `Quick` mode (~3.4 seconds total)
- 136,419 steps/second, 7.33 μs/step, zero console errors

---

## [0.1.0] — 2026-02-26

### Added

- Initial FORGE platform release
  - `forge-core` — deterministic simulation engine (13-phase pipeline, zero-alloc hot path)
  - `forge-worldgen` — Perlin noise procedural world generation (7 biomes)
  - `forge-task` — composable task DSL (7 operators, 10 predicates, 6 tiers, adaptive curriculum)
  - `forge-agent` — MCTS planner (PUCT selection, pluggable policy/value, forward model)
  - `forge-python` — PyO3 bindings with NumPy observations and GIL release during `step()`
  - `forge-wasm` — wasm-bindgen bindings with JSON string I/O for browser environments
  - `forge-types` — shared types, configs, error definitions
  - `forge-bench` — Criterion benchmarks for step throughput and world creation
  - Python wrappers: Gymnasium, PettingZoo Parallel, JAX-vectorized, Flatten/Normalize/TimeLimit
  - 389 tests (unit + property-based via `proptest` + integration)
  - 130,000+ steps/second from Python, <8 μs/step including PyO3 overhead
  - Deterministic: same seed + actions = byte-identical results
  - WebAssembly support via `forge-wasm` crate
  - `examples/forge_demo.py` — comprehensive 8-section showcase
