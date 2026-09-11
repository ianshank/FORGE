# FORGE v0.2.0 Release Plan — Unifying the Rust Core, Deterministic Physics, and Ecosystem Compliance

> **Status**: plan, not yet executed. Written 2026-09-11 against default-branch
> head `203207f` (`claude/plan-forge-environment-htAoK`). Every claim below was
> checked against the tree at that commit; file references are to that tree.
>
> **Read this first**: the requested release notes describe a repository that
> differs from the one on disk in five load-bearing ways (Section 1). This plan
> keeps the release's intent — promote the engine to a public `main`, publish
> reproducible benchmarks, gate the release on ecosystem compliance — but
> re-anchors every claim to something the repo can actually prove, and lists
> the decisions only the maintainer can make (Section 2).

---

## 0. Executive summary

- **There is no `main` branch.** `claude/plan-forge-environment-htAoK` *is* the
  repository's default branch (`git ls-remote --symref origin HEAD`), with 195
  commits and 20 open Dependabot PRs targeting it. "Squash-merge into `main`"
  therefore means *creating* `main`. Recommended: create it as a fast-forward
  of the current head so history, cited PR numbers, and the committed
  benchmark evidence's `git_sha` stay reachable (Track A).
- **The workspace is already at `0.5.0`**, with two `## [0.5.0]` entries in
  `CHANGELOG.md` (2026-06-02 and 2026-09-05) and a `docs/results/v0.5-*`
  evidence set. No git tag or GitHub release has ever been cut. Tagging the
  current tree `v0.2.0` would publish a wheel that reports version `0.5.0`
  (pyproject takes its version from Cargo) under a `v0.2.0` tag. Decision D1.
- **26 crates, not 15.** `Cargo.toml` `[workspace] members` lists 26;
  `README.md`, `docs/architecture.md` §4.2, and
  `tests/python/test_charter_alignment.py` all pin 26.
- **The API-compliance claim is currently false.** `forge_env.utils.check_env`
  is a home-grown 2-tuple / 5-tuple checker, not
  `gymnasium.utils.env_checker.check_env`. `ForgeGymnasiumEnv` does not
  subclass `gymnasium.Env` (the real checker rejects that outright), and
  `ForgeParallelEnv` neither subclasses `pettingzoo.ParallelEnv` nor returns
  `gymnasium.Space` objects — and it forwards only `agent_0`'s action to a
  single-agent native `step`. `pettingzoo` is not installed in CI. This is the
  largest engineering item in the release (Track D, item D2).
- **The throughput claim is fine; its provenance is not.** Committed evidence
  is 189,439 steps/s on the labeled `cloud_agent` profile (Intel Xeon, Linux,
  Python 3.12.3, commit `555b249`), gated by
  `tests/python/test_throughput_claim.py` so the published floor
  ("130,000+ steps/second from Python") can never exceed it. The draft
  `BENCHMARKS.md` cites an AMD Ryzen 9 9900X workstation for which **no
  evidence is committed**, invents a `python -m forge.benchmarks.throughput`
  entry point and a `tests/test_determinism.py --steps` flag that do not
  exist, and leaves `[Insert Value]` placeholders. Track C replaces it with a
  version where every number links to a committed JSON report.
- **The "drone physics" and "TensorRT/Hailo" bullets need rewording.** Aerial
  morphology (altitude, battery, same-altitude collisions) is real and
  deterministic (`crates/forge-core/src/physics.rs`), but it is an integer
  grid model, not a "3D physics solver". Nothing in the tree mentions
  TensorRT, Hailo, or domain randomization outside
  `docs/cloud_edge_proposal.md`; the related work lives on unmerged branches.
- **`agent_consolidation.sh` is unrelated to FORGE** (it archives four other
  repositories). It is reviewed in Track E as a manual, out-of-repo operator
  step with fixes; it is deliberately **not** committed to this repository.

---

## 1. Repository state audit

| Item | Requested release notes say | Repository at `203207f` | Where verified |
|---|---|---|---|
| Target branch | squash-merge into `main` | No `main`/`master` exists; `claude/plan-forge-environment-htAoK` is the default branch | `git ls-remote --symref origin HEAD`; GitHub branch list; all Dependabot PRs base on it |
| Crate count | 15 | 26 | `Cargo.toml` `[workspace] members`; `README.md` line 157; `docs/architecture.md` line 2339 |
| Version | v0.2.0 | `[workspace.package] version = "0.5.0"`; CHANGELOG `[0.5.0]` ×2, `[0.1.0]`; pyproject `dynamic = ["version"]` | `Cargo.toml` line 95; `CHANGELOG.md` lines 103, 625, 2209 |
| Tags / releases | (implied prior releases) | none | `git tag`; GitHub releases API returns `[]` |
| Throughput | ~130,000 steps/s | 189,439 steps/s measured; 130,000+ published floor, CI-gated | `benchmarks/baselines/cloud_agent/pyo3_step.json`; `tests/python/test_throughput_claim.py` |
| Drone physics | 3D physics solver, byte-identical trajectories | Integer-grid aerial morphology (altitude/battery); determinism via `step_determinism` proptest over all 22 `Action` variants + golden hashes over 3 seeds | `crates/forge-core/src/physics.rs` lines 159–312; `crates/forge-core/src/world/tests.rs` lines 87, 687, 1147; CHANGELOG `[0.5.0] - 2026-09-05` |
| Gymnasium `env_checker` | passes | never run; wrapper is not a `gymnasium.Env` subclass | `python/forge_env/gymnasium_env.py` line 41; `python/forge_env/utils.py` line 117 |
| PettingZoo `parallel_api_test` | passes | never run; `pettingzoo` not a dependency; spaces are plain dicts; only `agent_0`'s action is applied | `python/forge_env/pettingzoo_env.py` lines 22, 106, 131–137; `.github/workflows/ci.yml` line 400 |
| Multi-agent Python step | "Multi-Agent (N=4)" throughput row | PyO3 `ForgeEnv` exposes only `step(action: u32)`; multi-agent stepping exists only in Rust (Criterion `multi_agent_scaling`) | `crates/forge-python/src/env.rs` line 79; `benchmarks/baselines/README.md` |
| Telemetry dashboard | "promoted into mainline" | already on the default branch: `dashboard/`, `demo_ui/`, `forge-server` history endpoints, Prometheus/Grafana compose profile | CI jobs `dashboard`, `dashboard-e2e`, `demo-ui` |
| TensorRT / Hailo / domain randomization | "structural groundwork established" | no code; only `docs/cloud_edge_proposal.md`; work on unmerged branches `claude/raspberry-pi-hailo-support-CSWML`, `claude/ag-drone-ai-forge-wp0fm`, `claude/drone-agent-training-plan-R7hT1` | repo-wide grep for `hailo`, `tensorrt`, `domain.?random` |
| "clean pip install" gate | required | CI uses `maturin develop`; no job installs the built wheel into a fresh venv | `.github/workflows/ci.yml` `python-test` job |
| ">99% action-parity determinism" gate | required | nothing by that name; existing determinism gates assert **100%** identity (`step_determinism`, `golden_state_hash`, `crates/forge-env-forge/tests/forge_env_parity.rs` 1k-step lockstep, `test_deterministic_seed`) | as listed |
| Rust floor | 1.80+ | MSRV `rust-version = "1.85"`; toolchain pinned `1.94.1`; README Quick Start still says "1.75+" (stale) | `Cargo.toml` line 106; `rust-toolchain.toml`; `README.md` line 41 |

---

## 2. Decisions required before anything is cut

These change what gets built. Everything in Tracks A–E is written so it can
proceed under any answer, but the answers must be recorded in the release PR.

### D1 — Release version string

Options, in recommended order:

1. **`v0.6.0` (recommended).** Honest successor to the `0.5.0` already in
   Cargo and CHANGELOG; the `[Unreleased]` section carries new features (the
   self-improving-loop closure, committed throughput evidence, the docker bind
   fix), which is a minor bump under SemVer. Release title can still carry the
   requested subtitle. Keep the requested release *narrative*; change only the
   number.
2. **`v0.2.0` as a marketing/channel version, crates stay `0.5.0`.** Cheapest,
   but the wheel (`forge_env-0.5.0-*.whl`), `cargo metadata`, and the GHCR
   image label (`org.opencontainers.image.version` from the tag) will disagree
   with the tag, and CHANGELOG ordering (`0.1.0 → 0.5.0 → 0.5.0 → 0.2.0`)
   becomes unreadable. If chosen, add a "Versioning" note to `README.md` and
   `CHANGELOG.md` explaining the two version lines.
3. **Downgrade everything to `0.2.0`.** Not recommended: it rewrites two
   shipped CHANGELOG sections and any consumer that already pinned `0.5.0`
   sees a regression.

Throughout this plan `<VER>` stands for the chosen version.

### D2 — Squash vs preserve history when creating `main`

- **Preserve (recommended)**: `main` is created at `203207f` (plus the release
  PR). Rationale: `CHANGELOG.md`, `docs/next_steps.md`, and
  `benchmarks/baselines/README.md` cite PR numbers and commits (`#36`, `#53`,
  `#64`, `#89`, `#155`–`#157`, `555b249`, `22d0fb80`, `e5eca3d`);
  `benchmarks/baselines/cloud_agent/pyo3_step.json` records
  `"git_sha": "555b249…"` as the provenance of the headline throughput number;
  `git bisect`/`blame` keep working; the Dependabot PRs can be retargeted
  without rebasing across a history rewrite.
- **Squash (as requested)**: a single orphan commit whose tree equals the
  release head. If chosen, first tag the old head
  (`git tag archive/plan-forge-environment-htAoK-final 203207f`) and push the
  tag so the SHAs cited above remain resolvable, and add a one-line note in
  `CHANGELOG.md` pointing at the archive tag.

### D3 — Which hardware profile backs `BENCHMARKS.md`

- **`cloud_agent` (available now)**: the numbers already committed and gated.
  Hardware is recorded inside the JSON (`Intel(R) Xeon(R) Processor`, x86_64,
  Linux 6.12, Python 3.12.3). Nothing to run.
- **`reference_b` (the Ryzen 9 9900X box from the draft)**: the
  `benchmarks/baselines/reference_b/` slot exists for exactly this and ships
  only a `.gitkeep`. Requires running three harnesses on that machine
  (Section 5.2) and committing their JSON. The hardware fields are written by
  the harnesses; do not hand-type them. The two dual GPUs are irrelevant to
  every number in the document (the env is CPU-bound) unless a PPO-rollout
  row is added — say so explicitly or drop the GPU line.

Both can be shipped side by side; `reference_b` cannot be described in prose
before its JSON exists.

### D4 — Scope of the PettingZoo fix

`parallel_api_test` can be made to pass against the current broadcast shim
(same observation and reward to every agent, `agent_0`'s action applied).
That would be compliance in letter only.

- **Minimal (release-blocking)**: subclass `pettingzoo.ParallelEnv`, return
  real `gymnasium.Space` objects, pass the suite, and word the release notes
  as "PettingZoo Parallel API surface; per-agent stepping is tracked as a
  follow-up".
- **Complete (adds ~1–2 days)**: expose a multi-agent step on the PyO3 env
  (`WorldState::step` already takes one action per agent — the Criterion
  `multi_agent_scaling` bench drives it at 1–128 agents), forward per-agent
  actions and per-agent observations from `ForgeParallelEnv`, and only then
  claim multi-agent compliance.

### D5 — Run `agent_consolidation.sh` now or later

Independent of FORGE (Track E). The script is interactive, needs an
authenticated `gh` on the operator's machine, and archives four repositories.
It cannot run from this session and should not be committed here.

---

## 3. Track A — Branch, version, tag, and release mechanics

### A1. Release-prep PR (on this branch)

Branch `claude/forge-v0-2-0-release-l9xbg4`, based on `203207f`, targeting the
current default branch so CI runs (the `claude/**` glob in `ci.yml` covers
both). Contents:

1. `Cargo.toml` `[workspace.package] version = "<VER>"`; run
   `cargo build --workspace` so `Cargo.lock` re-records the workspace crate
   versions (do not hand-edit the lockfile). `pyproject.toml` needs no change
   (`dynamic = ["version"]` reads Cargo).
2. `CHANGELOG.md`:
   - Fix the duplicated `## [0.5.0]` heading. Suggested: relabel the
     2026-09-05 audit-remediation section `## [0.5.1] - 2026-09-05` with a
     note that the workspace version was not bumped at the time. This is a
     documentation correction, not a re-release.
   - Rename `## [Unreleased]` to `## [<VER>] - <release date>` and open a new
     empty `## [Unreleased]` above it.
   - Add the release narrative from Section 4.3 as the section preamble.
3. `README.md`: Quick Start "Rust 1.75+" → "Rust 1.85+" (matches
   `rust-version` and the badge); add a link to `BENCHMARKS.md`; no change to
   the throughput floor sentence (it is CI-gated).
4. `BENCHMARKS.md` at repo root, per Track C, and add `"BENCHMARKS.md"` to
   `CLAIM_FILES` in `tests/python/test_throughput_claim.py` so its floor is
   gated like the README's.
5. Track D tests, CI jobs, Makefile targets, and doc updates.
6. `docs/next_steps.md`: add a `## v<VER> — LANDED` stanza mirroring the
   existing `v0.5.0` one.
7. Gate locally with `make verify` (fmt, clippy, tests, wasm-check, ruff,
   mypy, pytest, hooks, pin-check, text-check, ci-parity, md-lint,
   mc-runner-smoke, mc-bot, dashboard). `make ci-parity` will fail if a new
   CI job has no Makefile target — add one per job.

If D4 "complete" is chosen, split the PyO3 multi-agent step into its own PR
(PR-2) so the release-prep PR stays reviewable.

### A2. Create `main`

After the prep PR merges into `claude/plan-forge-environment-htAoK`:

```bash
git fetch origin claude/plan-forge-environment-htAoK
# D2 = preserve (recommended): fast-forward creation, history intact
git push origin origin/claude/plan-forge-environment-htAoK:refs/heads/main

# D2 = squash: archive tag first, then an orphan commit with the same tree
git tag archive/plan-forge-environment-htAoK-final origin/claude/plan-forge-environment-htAoK
git push origin archive/plan-forge-environment-htAoK-final
git checkout --orphan main origin/claude/plan-forge-environment-htAoK
git commit -m "release: FORGE v<VER> — squash of claude/plan-forge-environment-htAoK"
git push -u origin main
```

Then in GitHub **Settings → Branches**: set default branch to `main`. The
workflow filters already list `main` (`ci.yml` lines 8–24, `gh-pages.yml`,
`hf-space.yml`), so CI, Pages, and the HF Space sync keep working. The
`docker` job in `ci.yml` publishes to GHCR on the default branch, so the first
push to `main` publishes `ghcr.io/ianshank/forge:main` — expected.

### A3. Branch protection on `main`

Required status checks (names as they appear in `ci.yml` / `security.yml`):
`Format Check`, `Clippy Lint`, `Rust Tests`, `Unused Dependencies (blocking)`,
`Markdown Lint`, `Allocation Audit (zero-alloc gate)`, `Rust Coverage
(tarpaulin)`, `Python Lint (fast)`, `Python Tests (maturin)`, `mc-bot Node
Tests + Biome Lint`, `forge-mc-runner Binary Smoke`, `WASM Target Check + Node
Runtime Tests`, `Demo UI Tests`, `Dashboard Build + Lint + Coverage`,
`cargo-deny`, `gitleaks`, `pip-audit`, plus the three new Track D jobs.
Leave `dashboard-e2e`, `wasm-e2e`, `machete`-advisory, and the
`workflow_dispatch` opt-ins unrequired (matches CHARTER Invariant 6). Require
linear history only if D2 = squash; otherwise allow merge commits (the repo's
existing convention).

### A4. Retarget open PRs

Twenty Dependabot PRs (#76–#154) base on the old default branch.
`.github/dependabot.yml` sets no `target-branch`, so Dependabot follows the
repository default; on its next weekly run it closes and recreates them
against `main`. To avoid a week of drift, retarget now:

```bash
for n in 76 77 81 82 83 85 88 95 101 114 135 138 139 140 143 150 151 152 153 154; do
  gh pr edit "$n" --base main
done
```

Keep `claude/plan-forge-environment-htAoK` until every PR is retargeted or
recreated, then delete it (or keep it as the archive if D2 = squash).

### A5. Tag and publish

```bash
git checkout main && git pull --ff-only origin main
git tag -a "v<VER>" -m "FORGE v<VER> — Unifying the Rust Core, Deterministic Physics, and Ecosystem Compliance"
git push origin "v<VER>"
```

The tag push re-runs the `docker` job (`startsWith(github.ref, 'refs/tags/v')`)
and publishes `ghcr.io/ianshank/forge:v<VER>`. Then create the GitHub Release
from the tag with the notes in Section 4.3. Optionally attach wheels:
`maturin build --release -m crates/forge-python/Cargo.toml` produces
`target/wheels/forge_env-<VER>-*.whl` for the host platform only; a
`release.yml` using `PyO3/maturin-action` for manylinux/macOS/Windows wheels is
a sensible follow-up, not a v`<VER>` requirement.

### A6. Post-release cleanup

- `grep -rn plan-forge-environment-htAoK` — update the comments in `ci.yml`,
  `gh-pages.yml`, `hf-space.yml`, and `docs/next_steps.md` that name the old
  default; keep the branch name in the workflow filters until the branch is
  deleted, then remove it.
- Verify Pages and the HF Space redeployed from `main` (both are path-filtered
  to `crates/forge-{wasm,core,types}/**` and `web/**`; trigger
  `workflow_dispatch` once if the release PR touched none of those).

---

## 4. Track B — Release-notes claim audit

### 4.1 Claim-by-claim disposition

| Draft claim | Keep / reword / drop | Replacement wording and evidence |
|---|---|---|
| "squash-merges … into main" | reword | "creates the public `main` branch from the development line" (D2 decides squash vs fast-forward) |
| "complete 15-crate Rust simulation engine" | reword | "26-crate Rust workspace in six dependency tiers enforced by `deny.toml`" |
| "transitions from experimental skeleton" | drop | The tree has shipped `0.5.0`-level features for months; say "first tagged public release" instead |
| "~130,000 steps/sec" | reword | "130,000+ steps/second from Python (measured 189k on the labeled `cloud_agent` profile; see `BENCHMARKS.md`)". Keep the gated floor phrasing so `test_throughput_claim.py` covers the release notes if they are ever committed |
| "Deterministic Drone Physics … 3D physics solver" | reword | "Deterministic aerial morphology: altitude, battery, and same-altitude collision rules on the integer grid (`DroneConfig`, `AgentMorphology::Aerial`), covered by the serialized-state determinism proptest across all 22 action variants and by golden state hashes" |
| "byte-identical trajectory assertions" | keep, cite | `step_determinism` compares `try_to_bytes()` output including the RNG stream (CHANGELOG `[0.5.0] - 2026-09-05`) |
| "natively passes Gymnasium `env_checker` and PettingZoo `parallel_api_test`" | **blocked** | May only be claimed once Track D item D2 is merged and green. Until then: "Gymnasium-style and PettingZoo-Parallel-style wrappers" |
| "Integrated Telemetry Dashboard promoted into mainline" | keep, reword | "ships on `main`: React dashboard, demo UI, `forge-server` persistent history endpoints, opt-in Prometheus/Grafana profile" |
| "Hardware-Targeted Architecture … ONNX → TensorRT/Hailo" | reword to roadmap | "ONNX export + Rust ONNX Runtime hot-reload ship today; TensorRT/Hailo export and a domain-randomization wrapper are roadmap items (`docs/cloud_edge_proposal.md`)" |
| "clean pip install" gate | keep, implement | Track D item D1 |
| "API compliance suites" gate | keep, implement | Track D item D2 |
| ">99% action-parity determinism test" | **reword** | ">99%" is weaker than the existing gates and contradicts Invariant 6. Use "100% byte-identical determinism gates: Rust serialized-state proptest and golden hashes, the 1,000-step `WorldEnv`/`ForgeEnv` lockstep parity test, and the new Python observation/reward determinism test" |

### 4.2 Verification tasks that back the reworded claims

1. Confirm `make_golden_world` (`crates/forge-core/src/world/tests.rs`
   line 646) enables `drone` so the golden digests cover the aerial path; if
   it does not, add a second golden set with `config.drone.enabled = true`
   rather than editing the existing digests.
2. Confirm the `step_determinism` action generator includes `TakeOff`,
   `Hover`, `Land`, `Scan`, and `DropPayload` (it enumerates all 22 variants
   per the 0.5.0 note — verify, do not assume).
3. Run `cargo test -p forge-env-forge` and cite `tests/forge_env_parity.rs`
   as the "action parity" evidence.

### 4.3 Corrected release notes (for the GitHub Release body)

> **FORGE v`<VER>` — Unifying the Rust Core, Deterministic Physics, and
> Ecosystem Compliance**
>
> This is FORGE's first tagged public release. It creates the public `main`
> branch from the development line and ships the complete 26-crate Rust
> workspace, the Python and WebAssembly bindings, and the telemetry stack as
> one reproducible cut.
>
> **Highlights**
>
> - **High-throughput Rust engine** — 130,000+ steps/second from Python
>   through the PyO3 boundary (189k measured on the labeled `cloud_agent`
>   profile), with a zero-allocation hot path enforced in CI. Every number and
>   its reproduction command is in `BENCHMARKS.md`.
> - **Deterministic aerial morphology** — altitude, battery, and
>   same-altitude collision rules for aerial agents on the integer grid.
>   Same seed + same actions ⇒ byte-identical serialized state, verified by
>   property tests across all 22 action variants and golden state hashes.
> - **Ecosystem compliance gates** — `gymnasium.utils.env_checker.check_env`
>   and `pettingzoo.test.parallel_api_test` run in CI as required checks,
>   alongside a clean-wheel install smoke and a Python observation/reward
>   determinism test. *(Only if Track D is green at tag time.)*
> - **Telemetry on `main`** — React dashboard, demo UI, `forge-server`
>   persistent history endpoints, and the opt-in Prometheus/Grafana profile.
> - **Edge roadmap** — ONNX export and Rust ONNX Runtime hot-reload ship
>   today; TensorRT/Hailo export and a domain-randomization wrapper are
>   tracked in `docs/cloud_edge_proposal.md`.
>
> Full change list: `CHANGELOG.md` `[<VER>]`.

---

## 5. Track C — `BENCHMARKS.md`

### 5.1 Corrections to the draft

| Draft | Problem | Fix |
|---|---|---|
| Hardware: Ryzen 9 9900X / 72 GB / dual RTX 5060 | no committed evidence from that host | D3; if kept, populate `benchmarks/baselines/reference_b/` first, cite the JSON's own `hardware` block |
| "Rust 1.80+" | MSRV is 1.85; measurements used the pinned 1.94.1 toolchain | state both: "MSRV 1.85; measured with 1.94.1 (`rust-toolchain.toml`)" |
| "Python 3.12" | fine for `cloud_agent` (3.12.3); the package floor is `>=3.9` | say "measured on 3.12.3; supports ≥3.9" |
| Row "Single-Agent (Random Policy) ~130,000" | the harness steps a fixed action (`Move Right`, `action_id = 4`), not a random policy; the measurement is 189,439 | rename "Single agent, fixed action, PyO3 boundary"; print the measured value and the gated floor separately |
| Row "Multi-Agent (N=4, Random) [Insert Value]" | no Python multi-agent step exists; Rust Criterion sweep is 1/8/16/32/64/128 | either add `4` via `FORGE_BENCH_AGENT_COUNTS=1,4,8,16,32,64,128` when regenerating `multi_agent_scaling.json`, or report N=8; label the row "Rust `WorldState::step`, env-steps/s" and never present it as the Python headline (`benchmarks/baselines/README.md` forbids that) |
| Row "PPO Rollout (CleanRL) [Insert Value]" | no harness emits this | drop the row for `<VER>`, or add a `--benchmark-steps N --report PATH` mode to `examples/train_ppo_cleanrl.py` and commit its JSON under the profile directory |
| `python -m forge.benchmarks.throughput --episodes 1000` | module does not exist | document the real command (below). A thin `forge.benchmarks` CLI wrapping `tests/python/test_step_throughput.py`'s harness is a reasonable follow-up but needs its own tests and coverage |
| "Trajectory length tested: 1,000,000 steps; variance 0.0%" | no such run; `pytest --steps` is not an option | Track D item D3 adds `tests/python/test_determinism.py` with a `--determinism-steps` option; CI runs 10,000, the 1,000,000-step run is opt-in and documented as such. Report "0 mismatching bytes over N steps", not a percentage |
| `pytest tests/test_determinism.py`, `pytest tests/test_api_compliance.py` | wrong paths (tests live under `tests/python/`) | `pytest tests/python/test_determinism.py`, `pytest tests/python/test_api_compliance.py` |
| "memory footprint" in the intro | nothing in the document measures it | cite the zero-allocation audit (`benchmarks/baselines/reference_a/alloc_audit.json`, 0 bytes/step after warm-up, gated by `check_zero_alloc.py --max-bytes 0`) or remove the word |
| Any `[Insert Value]` | placeholders in a document whose stated purpose is trust | rows without committed evidence are omitted, never left blank |

### 5.2 Evidence to (re)generate

All three commands are already documented in `benchmarks/baselines/README.md`;
run them on the chosen host after `maturin develop` and commit the JSON in the
same PR as `BENCHMARKS.md`:

```bash
PROFILE=reference_b   # or cloud_agent to refresh
FORGE_RUN_STEP_THROUGHPUT=1 \
FORGE_STEP_THROUGHPUT_OUT=benchmarks/baselines/$PROFILE/pyo3_step.json \
  pytest tests/python/test_step_throughput.py -s --no-cov

FORGE_BENCH_AGENT_COUNTS=1,4,8,16,32,64,128 make bench-export PROFILE=$PROFILE

cargo run -p forge-bench --bin allocation_audit --features dhat-heap --release -- \
  --warmup 1024 --iters 10000 --agents 1,8,16,32,64,128 \
  --out benchmarks/baselines/$PROFILE/alloc_audit.json
python3 benchmarks/runner/check_zero_alloc.py \
  --input benchmarks/baselines/$PROFILE/alloc_audit.json --max-bytes 0
```

### 5.3 Corrected `BENCHMARKS.md` (draft to commit at repo root)

Numbers below are the committed `cloud_agent` values. Replace or add a
`reference_b` column only after 5.2 has been run on that host.

```markdown
# FORGE Performance & Determinism Benchmarks

Every number in this document links to a machine-readable report committed
under `benchmarks/baselines/<profile>/`, produced by a harness in this
repository, on hardware recorded inside the report. Published floors are
gated in CI by `tests/python/test_throughput_claim.py`; the zero-allocation
contract is gated by `benchmarks/runner/check_zero_alloc.py --max-bytes 0`.

## 1. Profiles

| Profile | Host | Toolchain | Report |
|---|---|---|---|
| `cloud_agent` | Intel Xeon, x86_64 Linux 6.12 (cloud VM) | Rust 1.94.1 (`rust-toolchain.toml`), Python 3.12.3 | `benchmarks/baselines/cloud_agent/` |
| `reference_a` | GitHub Actions `ubuntu-latest` | as CI | `benchmarks/baselines/reference_a/` (allocation audit) |

MSRV is Rust 1.85 (`Cargo.toml` `rust-version`); the Python package supports
3.9+. GPUs are not used by any measurement here.

## 2. Throughput

| Measurement | Value | Report |
|---|---|---|
| Single agent, fixed action (`Move Right`), PyO3 boundary, 100,000 iters | 189,439 steps/s (mean 5.28 μs, p99 6.53 μs) | `cloud_agent/pyo3_step.json` |
| Published floor ("steps/second from Python") | 130,000+ | gated by `test_throughput_claim.py` |
| Rust `WorldState::step`, multi-agent sweep | see `env_steps_per_sec` per `num_agents` row | `cloud_agent/multi_agent_scaling.json` |

The Rust sweep measures whole-world `step()` calls per second inside Rust; do
not compare it to the Python-boundary headline.

To reproduce:

    FORGE_RUN_STEP_THROUGHPUT=1 FORGE_STEP_THROUGHPUT_OUT=/tmp/pyo3_step.json \
      pytest tests/python/test_step_throughput.py -s --no-cov
    make bench-export PROFILE=<profile>

## 3. Zero-allocation hot path

`WorldState::step_into` allocates 0 bytes / 0 blocks after warm-up across
1, 8, 16, 32, 64, and 128 agents for every action variant
(`reference_a/alloc_audit.json`). Reproduce with `make alloc-audit`.

## 4. Determinism

Two environments constructed with the same seed and fed the same action
sequence produce byte-identical observations and identical reward
trajectories.

- Rust: `step_determinism` (proptest over all 22 action variants, compares
  serialized state including the RNG stream) and golden state hashes over
  three seeds — `cargo test -p forge-core`.
- Rust ↔ Python surface parity: 1,000-step lockstep of `WorldEnv` against
  `ForgeEnv` — `cargo test -p forge-env-forge`.
- Python: `tests/python/test_determinism.py` compares `obs.tobytes()` and
  rewards over 10,000 steps in CI; the 1,000,000-step run is opt-in:

      FORGE_RUN_LONG_DETERMINISM=1 pytest tests/python/test_determinism.py \
        --determinism-steps 1000000 -v --no-cov

  Result to publish once run: "0 mismatching bytes over N steps".

## 5. API compliance

- Gymnasium: `gymnasium.utils.env_checker.check_env(ForgeGymnasiumEnv())`
- PettingZoo: `pettingzoo.test.parallel_api_test(ForgeParallelEnv(n_agents=2))`

Reproduce with `pip install -e ".[compliance]"` then
`pytest tests/python/test_api_compliance.py -v --no-cov`. Both run as required
CI checks.
```

---

## 6. Track D — CI and reproduction gates

Each new job needs: a `Makefile` target (`make ci-parity` fails otherwise), a
line in `CLAUDE.md`'s command list, a mention in `docs/CHARTER.md` Invariant 6
(`test_charter_ci_job_citations_exist` checks that cited jobs exist), and a
row in the `main` branch-protection list (A3).

### D1. `pip-install-clean` job

Proves the wheel installs and imports outside the source tree — today's
`maturin develop` path never exercises packaging (`pyproject.toml`'s
`[tool.maturin] include` list is the usual thing to break).

```yaml
pip-install-clean:
  name: Clean wheel install smoke
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v5
    - uses: dtolnay/rust-toolchain@stable
      with: { toolchain: "1.94.1" }
    - uses: actions/setup-python@v7
      with: { python-version: "3.11" }
    - run: pip install maturin
    - run: maturin build --release -m crates/forge-python/Cargo.toml
    - run: |
        python -m venv /tmp/clean && . /tmp/clean/bin/activate
        pip install target/wheels/forge_env-*.whl
        pip check
        cd /tmp && python -c "import forge_env, forge; from forge_env import ForgeEnv; ForgeEnv().reset(seed=0)"
```

Makefile: `pip-install-smoke`. Effort: ~1 hour.

### D2. `api-compliance` job and the wrapper refactor

Work items, in order:

1. `pyproject.toml`: add `compliance = ["gymnasium>=1.0", "pettingzoo>=1.24"]`
   and fold it into `all`. Install it in the `python-test` job (or in a
   dedicated `api-compliance` job that reuses the `maturin develop` steps).
2. `python/forge_env/gymnasium_env.py`: `class ForgeGymnasiumEnv(gym.Env)`;
   call `super().reset(seed=seed)` so `np_random` is seeded; keep
   `observation_space` a `spaces.Dict` whose `Box` dtypes match the arrays the
   native env returns exactly (the checker rejects dtype drift); `render_mode`
   must be `None` or in `metadata["render_modes"]`; keep the existing
   `gymnasium`-optional import guard so `python-test-fast` still passes.
3. `python/forge_env/pettingzoo_env.py`: `class ForgeParallelEnv(ParallelEnv)`;
   `observation_space(agent)` / `action_space(agent)` return the same
   `gymnasium` spaces as the Gymnasium wrapper; `reset(seed=, options=)`
   returns `(observations, infos)`; `agents` shrinks on termination (already
   does). Per D4, either keep the broadcast semantics and document them, or
   forward per-agent actions once a multi-agent PyO3 step exists.
4. `tests/python/test_api_compliance.py`:

   ```python
   import pytest
   gym = pytest.importorskip("gymnasium")
   pz = pytest.importorskip("pettingzoo")

   def test_gymnasium_env_checker() -> None:
       from gymnasium.utils.env_checker import check_env
       from forge_env.gymnasium_env import ForgeGymnasiumEnv
       check_env(ForgeGymnasiumEnv(), skip_render_check=True)

   def test_pettingzoo_parallel_api() -> None:
       from pettingzoo.test import parallel_api_test
       from forge_env.pettingzoo_env import ForgeParallelEnv
       parallel_api_test(ForgeParallelEnv(n_agents=2), num_cycles=200)
   ```

   The Gymnasium checker also asserts `reset(seed=123)` twice yields equal
   observations and that `step` is deterministic under a fixed seed — FORGE
   passes those by construction.
5. Keep `forge_env.utils.check_env` (it is public API with tests) but change
   its docstring to say it is a lightweight shape check and point to the real
   checkers.

Makefile: `api-compliance`. Effort: 1–2 days for "minimal"; add 1–2 days for
D4 "complete" (PyO3 `step_multi(actions: Sequence[int])` in
`crates/forge-python/src/env.rs`, tests in `tests/python/test_pettingzoo_env.py`).

### D3. Python determinism test (replaces the ">99% action-parity" gate)

`tests/python/test_determinism.py` plus a `conftest.py` option:

- `--determinism-steps` (default 10,000) and env override
  `FORGE_RUN_LONG_DETERMINISM=1` for the 1,000,000-step run.
- Build two `ForgeGymnasiumEnv()`; `reset(seed=S)` both; draw the action
  sequence from `numpy.random.default_rng(S)`; step both in lockstep;
  compare `obs[k].tobytes()` for every observation key and `reward` exactly;
  on `terminated or truncated`, `reset(seed=S + episode)` both.
- Fail on the first mismatch with step index, key, and byte offset.
- Wire the long run into `.github/workflows/e2e-long.yml` (`workflow_dispatch`
  / schedule), not into PR CI.

Makefile: covered by `py-test`. Effort: ~half a day.

### D4. Wiring checklist

- `Makefile`: `pip-install-smoke`, `api-compliance`; add both to `verify-full`
  (not `verify`, which must stay network-light).
- `scripts/check_local_ci_parity.py` will then pass for the new job ids.
- `CLAUDE.md` build/test list: three new lines.
- `docs/CHARTER.md` Invariant 6: add `pip-install-clean` and `api-compliance`
  to the enumerated jobs; `test_charter_ci_job_citations_exist` verifies them.
- `CONTRIBUTING.md` and `.github/pull_request_template.md`: add the two
  commands to the verification checklist.

---

## 7. Track E — `agent_consolidation.sh` (out of repository)

The script triages `ianshank` repositories with fewer than 5 stars and no push
in 90 days, then badges and archives `Agents`, `Distilled_Agents`,
`Distilled_Agent_Pipeline`, and `MultiModalOrch` in favour of
`Mango_Code_Agent-Harness`. None of those are FORGE. Recommendation: keep it in
a personal ops/dotfiles repository, run it by hand, and do not commit it here.

Review findings to fix before running it:

1. **`gh repo archive --confirm`** — current `gh` releases replaced
   `--confirm` with `--yes` / `-y`; check `gh repo archive --help` on the
   operator's version first, or the loop aborts after the first push.
2. **Default-branch guess** — `git push origin main || git push origin master`
   aborts the whole script under `set -e` (with the temp dir left behind) if
   the repo uses neither. Read the real branch:
   `gh repo view "$USERNAME/$REPO" --json defaultBranchRef -q .defaultBranchRef.name`
   and push to that.
3. **No cleanup on failure** — add `trap 'rm -rf "$TMP_DIR"' EXIT` right after
   `mktemp -d`.
4. **Not idempotent** — a re-run after a partial failure prepends a second
   badge. Guard with `grep -q 'Status-Archived' README.md && skip`.
5. **Branch protection** — if a target repo protects its default branch the
   push fails; the archive step must only run after a successful push (it
   already does under `set -e`, but item 3 is what makes that safe).
6. **Clone auth** — HTTPS clones need a credential helper; use
   `gh repo clone "$USERNAME/$REPO"` instead of raw `git clone`.
7. **Cutoff date format** — GNU `date -Iseconds` emits `+00:00` offsets while
   GitHub's `pushedAt` uses `Z`; use `date -u -d "90 days ago" +"%Y-%m-%dT%H:%M:%SZ"`
   on Linux so the string comparison in `--jq` is like-for-like.
8. **Triage is informational only** — the archived set is hard-coded and is
   never checked against the "<5 stars, 90 days" criteria. Fine, but say so in
   the header comment, or filter `REDUNDANT_REPOS` through the same query.
9. **Add `--dry-run`** — print each `gh`/`git` command instead of executing
   it, and run that once before the real pass.

Run order relative to FORGE: independent. It touches no FORGE branch, tag, or
workflow.

---

## 8. Sequenced checklist

| # | Step | Owner | Gate |
|---|---|---|---|
| 0 | Record D1–D5 in the release PR description | maintainer | — |
| 1 | Track D: D1 `pip-install-clean`, D3 determinism test, wiring (D4) | engineer | `make verify` + new jobs green on the branch |
| 2 | Track D: D2 wrapper refactor + `test_api_compliance.py` (+ PR-2 if D4 = complete) | engineer | `api-compliance` green |
| 3 | Track C: run 5.2 on the chosen host(s); commit JSON + `BENCHMARKS.md`; add it to `CLAIM_FILES` | maintainer (needs the workstation for `reference_b`) | `test_throughput_claim.py`, `check_zero_alloc.py` |
| 4 | Track A1: version bump, CHANGELOG fixes, README floor fix, `next_steps.md` stanza, release narrative | engineer | `make verify`; markdownlint |
| 5 | Open the release-prep PR against `claude/plan-forge-environment-htAoK`; run `forge-pr-review`; merge | maintainer | all required checks |
| 6 | A2: create `main` (per D2); flip default branch; A3 protection | maintainer | CI green on `main`; GHCR `:main` image published |
| 7 | A4: retarget the 20 Dependabot PRs | maintainer | — |
| 8 | A5: tag `v<VER>`; confirm GHCR `:v<VER>`; publish the GitHub Release with Section 4.3 | maintainer | tag CI green |
| 9 | A6: stale-name sweep, Pages / HF Space redeploy check | engineer | — |
| 10 | Track E on the operator's machine (D5) | maintainer | dry-run first |

Estimated engineering effort excluding decisions and the workstation run:
3–5 days (D4 minimal) or 5–7 days (D4 complete).

---

## 9. Risks and rollback

- **Tagging before Track D is green** would publish release notes that claim
  compliance the repo cannot demonstrate. The notes in 4.3 carry an explicit
  "only if green" marker on that bullet; strip the bullet rather than the
  marker if the gate slips.
- **`docker` job on first `main` push** needs `secrets.GITHUB_TOKEN` package
  write (already granted in the job) and, for the optional Docker Hub push,
  the existing repository secrets. A failure there does not block the tag but
  should be fixed before the release is announced.
- **Dependabot churn** — if the default branch is flipped before retargeting,
  Dependabot closes and recreates up to 20 PRs; harmless but noisy.
- **Rollback**: a tag can be deleted (`git push --delete origin v<VER>`) and
  the GitHub Release drafted rather than published until CI on the tag is
  green; `main` can be reset only before anyone branches from it — after
  that, fix forward with a `v<VER>.1`.
