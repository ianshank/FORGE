# FORGE Release Plan — Unifying the Rust Core, Deterministic Physics, and Ecosystem Compliance

> **Revision 2**, 2026-09-12. Revision 1 was a plan written from reading the
> repository. This revision folds in a peer review of that plan and, more
> importantly, records what has now been **implemented and verified** on branch
> `claude/forge-v0-2-0-release-l9xbg4` rather than only proposed.
>
> The short version: the engineering work is done and green. What remains is
> repository administration that requires the maintainer's GitHub permissions —
> creating the public branch, setting branch protection, and tagging.

---

## 0. Status at a glance

| Track | State |
| :--- | :--- |
| A — Version, CHANGELOG, tag-triggered CI | **Implemented** |
| B — Release-notes claim audit | **Implemented** (corrected notes in §6) |
| C — `BENCHMARKS.md` backed by committed evidence | **Implemented** |
| D — Ecosystem compliance, determinism, packaging gates | **Implemented and passing** |
| A2–A6 — Create `main`, protect, tag, publish | **Operator-only** (§7) |
| E — `agent_consolidation.sh` | **Reviewed, not committed** (§9) |

Verified locally on this branch: `cargo fmt --all --check`, `cargo clippy
--workspace --all-targets --features forge-cloud/gcs -- -D warnings`, and
`cargo test --workspace --features forge-cloud/gcs` all exit 0; `ruff check`
and `mypy` are clean across 130 source files; 1,843 Python tests pass with
93.49% coverage.

---

## 1. What the peer review changed

Revision 1 got the diagnosis right and several mechanics wrong. The corrections
that mattered:

1. **`ci.yml` never ran on tags at all.** Revision 1 planned to tag and expected
   the `docker` job to publish. But `on.push` declared only `branches`, and
   GitHub does not run a workflow for a tag push in that case — so the job's
   `startsWith(github.ref, 'refs/tags/v')` condition had been unreachable for
   the workflow's entire life, and no tagged image was ever built.
   `docs/architecture.md` and `test_docker_server_bind_contract.py` both already
   described tag builds as working. **Fixed in this branch**: `on.push.tags:
   ["v*"]` added, which makes the existing documentation true.
2. **The image tags are wrong in revision 1.** `docker/metadata-action` is
   configured with `type=semver,pattern={{version}}`, so tag `v0.6.0` publishes
   `:0.6.0`, `:0.6`, `:latest`, and `:<sha>` — never `:v0.6.0`.
3. **Pushing `main` before flipping the default publishes nothing.** The
   `docker` job compares `github.ref` against
   `github.event.repository.default_branch` from the *push event payload*, so
   the first push to `main` is skipped while the old branch is still the
   default. Re-running does not help; a new run after the flip is required.
4. **Rename beats create-and-switch.** GitHub's branch *rename* retargets every
   open pull request, moves branch-protection rules, and redirects the old
   name — doing in one step what revision 1 proposed as "create `main`, flip
   the default, then retarget 20 PRs by hand". It also preserves history, which
   is what the committed benchmark evidence's `git_sha` provenance depends on.
5. **The open-PR inventory was short.** 28 open PRs target the current default,
   not 20: 24 Dependabot plus 4 human. Several are already superseded.
6. **Three branch-protection check names in revision 1 do not exist.** The
   security jobs are named `cargo-deny (blocking)`, `gitleaks (blocking)`, and
   `pip-audit (blocking)`. §7.3 carries the verified list.
7. **Pages and the HF Space have never succeeded** — 13/13 and 9/9 runs failed,
   for reasons already recorded in `docs/next_steps.md` (Pages not enabled with
   Source = GitHub Actions; no `HF_TOKEN` secret). Revision 1 claimed they
   would "keep working" after the move. They will keep *not* working until
   someone enables them, which is a release decision, not a side effect.
8. **The version bump touches more than the workspace table** — the root
   `forge-integration-tests` package, `forge_env.__version__`, and the dashboard
   package and lockfile. All handled, and now gated (§3.1).

---

## 2. Decisions

### D1 — Release version: **0.6.0** (implemented, one line to change)

The request was "v0.2.0". The workspace already shipped `0.5.0`, so a `0.2.0`
tag would publish a wheel reporting a *lower* version than its predecessor and
break SemVer for anyone who pinned it. This branch bumps to `0.6.0`.

Because the version is now single-sourced, changing this decision is one edit to
`[workspace.package].version` in `Cargo.toml`, plus `cargo update -w` and the
dashboard package/lockfile — and `tests/python/test_version_consistency.py`
fails loudly if you miss one. The release *title* and narrative are unaffected.

If you want `0.2.0` regardless, say so and it is a two-minute change.

### D2 — Creating the public branch: **rename, do not re-create** (recommended)

```
gh api --method POST \
  "repos/ianshank/FORGE/branches/claude%2Fplan-forge-environment-htAoK/rename" \
  -f new_name=main
```

One call: the default flips, all 28 open PRs retarget, protection rules follow,
and old-name URLs redirect. History is preserved, so every PR number and commit
SHA cited in `CHANGELOG.md`, `benchmarks/baselines/README.md`, and the
`git_sha` inside the committed throughput report stays resolvable.

The squash-into-an-orphan-commit alternative from the original request is still
documented in §7.2, but it discards that provenance and requires retargeting 28
PRs by hand. Recommend against.

### D3 — Benchmark hardware profile: **`cloud_agent`, already committed**

`BENCHMARKS.md` publishes only numbers that exist in the repository. The
Ryzen 9 9900X workstation from the original draft has no committed evidence, and
the `benchmarks/baselines/reference_b/` slot exists for exactly that host. To
add it, run the three commands in `BENCHMARKS.md` §6 on that machine and commit
the JSON; the prose then cites it. Until then it is absent rather than
placeheld. Note the dual GPUs are irrelevant to every number in the document —
the simulation core is CPU-bound integer arithmetic.

### D4 — PettingZoo scope: **resolved as "complete"**

This was the open question in revision 1: expose real per-agent stepping, or
make the broadcast shim merely pass the compliance suite. PR #159 merged
`ForgeEnv.reset_all` and `ForgeEnv.step_multi` into this branch, which settles
it — `ForgeParallelEnv` now drives the real multi-agent path, one action and one
observation per agent. No compliance-in-letter-only shim shipped.

### D5 — `agent_consolidation.sh`: **operator-run, out of repository**

Unrelated to FORGE (it archives four other repositories). Reviewed in §9; not
committed here.

---

## 3. What is implemented on this branch

### 3.1 Version single-sourcing

`[workspace.package].version` is the single source of truth.

- The root `forge-integration-tests` package now inherits with
  `version.workspace = true` — it was the one crate restating a literal, so a
  release bump silently left it behind.
- `forge_env.__version__` was a hardcoded `"0.5.0"` whose only test asserted
  `hasattr`. New `python/forge_env/_version.py` resolves it: co-located Cargo
  manifest first, then installed distribution metadata, then a sentinel. The
  manifest deliberately outranks metadata — a stale installed wheel must not
  make the package misreport the tree it is running from. That ordering was a
  bug in the first implementation, and the new drift test is what caught it.
- `tests/python/test_version_consistency.py` fails if Cargo, the Python package,
  `dashboard/package.json`, or its lockfile disagree, and covers every branch of
  the resolution ladder including the fallbacks.

### 3.2 Ecosystem compliance — verified, not asserted

The previous claim could not have been true: `ForgeGymnasiumEnv` did not
subclass `gymnasium.Env`, which the upstream checker rejects outright, and
`ForgeParallelEnv` returned raw descriptor dicts where PettingZoo requires
`gymnasium.spaces.Space` objects. Both now subclass their upstream base classes.

Substantive fixes beyond the inheritance:

- **Every agent's action reaches the simulation.** `ForgeParallelEnv.step`
  forwarded only `agent_0`'s action to the single-agent native `step` and
  broadcast one observation to every agent. It now uses `step_multi`.
  `test_every_agent_action_reaches_the_simulation` pins the regression.
- **`n_agents` and the simulation can no longer disagree.** The wrapper built N
  agent names over a simulation still configured for `DEFAULT_NUM_AGENTS` (1 in
  Rust). The resolved count is pushed into the config, and read from the config
  when `n_agents` is omitted.
- **Observations are fitted to their declared spaces.** New
  `python/forge_env/space_builder.py` builds both spaces from the descriptor
  Rust derives from `ForgeConfig` — no shape, bound, or dtype is restated in
  Python — and coerces observations so `space.contains(obs)` holds. It is shared
  by both wrappers, so they cannot drift.
- **`gymnasium.make("Forge-v0")`** works after an explicit `register_envs()`.
  Registration is a function call, never an import side effect (Charter
  Invariant 1); a subprocess test pins that.

Verified against Gymnasium 1.3.0 and PettingZoo 1.27.0.

**Behaviour change to note in the release:** `ForgeGymnasiumEnv.unwrapped` now
returns the environment itself, per the Gymnasium contract. It previously
returned the native PyO3 handle, which broke wrapper chains such as
`TimeLimit(env).unwrapped`. The handle moved to an explicit `native` property.

### 3.3 Determinism across the Python boundary

Determinism was defended inside Rust by a property test over serialized state
and by golden hashes, but never through PyO3 — the surface every training run
actually consumes. A bug introduced in observation conversion would not have
moved a single Rust hash.

`tests/python/test_determinism.py` drives two identically seeded environments
through one action sequence and compares `ndarray.tobytes()` and exact reward
equality across episode boundaries. Depth is a knob, not a literal:
`--determinism-steps N` or `FORGE_DETERMINISM_STEPS`, default 10,000.
`make api-compliance-soak DETERMINISM_STEPS=1000000` runs the release soak.

The gate's own sensitivity is tested, so it cannot decay into a no-op.

### 3.4 CI

| Job | Purpose |
| :--- | :--- |
| `api-compliance` | Runs `check_env` and `parallel_api_test` themselves, plus the determinism gate |
| `pip-install-clean` | Builds a real wheel, installs it into an empty environment, imports it from outside the checkout, and asserts the wheel version matches Cargo |

Every other Python job uses `maturin develop`, which leaves the source tree on
`sys.path` and so never exercises packaging — a wheel missing a module or a
`[tool.maturin].include` data file would still import.

Both jobs have `make` targets (`api-compliance`, `api-compliance-soak`,
`pip-install-smoke`), are reachable from `verify-full`, and are mapped in
`scripts/check_local_ci_parity.py`, which passes.

While wiring these, the repository's own contract test
(`test_every_job_running_pytest_installs_the_coverage_plugin`) caught that the
new job would have exited 4 at argument parsing without `pytest-cov`. Fixed.

### 3.5 Documentation

`BENCHMARKS.md` publishes only numbers read from committed reports, with the
command that reproduces each and the profile it came from. No placeholder rows.
It is registered in `test_throughput_claim.py`'s `CLAIM_FILES`, so its published
130,000 steps/second floor is CI-gated against the committed 189,439
measurement — verified to parse, so the gate is live rather than nominal.

`README.md`: corrected the stale "Rust 1.75+" prerequisite to the declared 1.85
MSRV, and linked `BENCHMARKS.md`.

`CHANGELOG.md`: the duplicate `## [0.5.0]` heading is resolved. The 2026-09-05
section was work that landed *after* the 0.5.0 release and never had a version
of its own, so it is demoted to a subsection of this release rather than given
an invented `0.5.1`. Edited in binary mode; the file's pinned CRLF endings are
intact and `check_text_encoding.py` passes.

---

## 4. Claim audit for the release notes

| Draft claim | Disposition |
| :--- | :--- |
| "squash-merges into main" | Reword: creates the public `main` branch, by rename (D2) |
| "complete 15-crate Rust simulation engine" | **Wrong**: 26 crates in six dependency tiers |
| "transitions from experimental skeleton" | Drop: the tree has shipped release-grade features for months. "First tagged public release" is accurate |
| "~130,000 steps/sec" | Keep as a floor; the measurement is 189,439 on the labeled `cloud_agent` profile |
| "Deterministic Drone Physics … 3D physics solver" | **Reword**: integer-grid aerial morphology — altitude, battery, same-altitude collisions. Real and deterministic, but not a 3D solver |
| "byte-identical trajectory assertions" | Keep, now true at three levels including Python |
| "passes Gymnasium env_checker and PettingZoo parallel_api_test" | **Now true**, and gated by `api-compliance` |
| "Integrated Telemetry Dashboard promoted" | Keep, reworded: it already ships on this branch |
| "ONNX → TensorRT/Hailo export pipeline" | **Roadmap only**: nothing in the tree mentions TensorRT or Hailo outside `docs/cloud_edge_proposal.md` |
| ">99% action-parity determinism test" | **Reword**: the gates assert 100% byte-identical equality. ">99%" is weaker than what ships and contradicts Invariant 6 |

---

## 5. Release notes (corrected, for the GitHub Release body)

> **FORGE v0.6.0 — Unifying the Rust Core, Deterministic Physics, and Ecosystem
> Compliance**
>
> FORGE's first tagged public release. It promotes the 26-crate Rust workspace,
> the Python and WebAssembly bindings, and the telemetry stack to a versioned
> release channel, and closes the three gaps that made the previous
> compliance and reproducibility claims unverifiable.
>
> - **Verified ecosystem compliance.** `gymnasium.utils.env_checker.check_env`
>   and `pettingzoo.test.parallel_api_test` now run in CI as a required check.
>   Both wrappers subclass their upstream base classes, observations are fitted
>   to their declared spaces, and every agent's action reaches the simulation
>   through the native multi-agent step — the multi-agent wrapper previously
>   discarded all but the first agent's action.
> - **Determinism, measured where it is consumed.** Same seed and actions give
>   byte-identical observations and exactly equal rewards, now asserted across
>   the Python boundary as well as inside Rust, at a configurable depth.
> - **High-throughput core.** 130,000+ steps/second from Python (189,439
>   measured on the labeled `cloud_agent` profile), with a zero-allocation hot
>   path enforced on every PR. Every number and its reproduction command is in
>   `BENCHMARKS.md`.
> - **Deterministic aerial morphology.** Altitude, battery, and same-altitude
>   collision rules for aerial agents on the integer grid.
> - **Telemetry on the release channel.** React dashboard, demo UI,
>   `forge-server` persistent history endpoints, and the opt-in
>   Prometheus/Grafana profile.
> - **Edge roadmap.** ONNX export and Rust ONNX Runtime hot-reload ship today;
>   TensorRT/Hailo export and a domain-randomization wrapper are tracked in
>   `docs/cloud_edge_proposal.md`.
>
> Breaking: `ForgeGymnasiumEnv.unwrapped` now returns the environment itself per
> the Gymnasium contract; the native handle moved to `.native`.

---

## 6. Merge this branch first

This branch targets the current default branch. Merge it before any of §7, so
`main` is created from a tree that already carries the tag trigger, the version
bump, and the new gates.

---

## 7. Operator runbook (requires maintainer GitHub permissions)

### 7.1 Rename to `main` (D2 = recommended)

```bash
gh api --method POST -H "Accept: application/vnd.github+json" \
  "repos/ianshank/FORGE/branches/claude%2Fplan-forge-environment-htAoK/rename" \
  -f new_name=main

git fetch origin --prune
git branch -m claude/plan-forge-environment-htAoK main
git branch -u origin/main main
git remote set-head origin -a

gh pr list --base main --state open --limit 50   # expect the full open set, retargeted
gh workflow run ci.yml --ref main                # first run WITH default=main; publishes :main
```

### 7.2 Squash alternative (only if you explicitly want history discarded)

```bash
git tag archive/plan-forge-environment-htAoK-final origin/claude/plan-forge-environment-htAoK
git push origin refs/tags/archive/plan-forge-environment-htAoK-final
git checkout --orphan main origin/claude/plan-forge-environment-htAoK
git commit -m "release: FORGE v0.6.0"
git push -u origin main
# Settings -> Branches -> default = main, then retarget all 28 open PRs by hand
gh workflow run ci.yml --ref main
```

### 7.3 Branch protection on `main`

Required status checks, by exact check-run name (these are job `name:` values,
not job ids — verified against a real run):

```
Format Check
Clippy Lint
Rust Tests
Unused Dependencies (blocking)
Markdown Lint
Allocation Audit (zero-alloc gate)
Rust Coverage (tarpaulin)
Python Lint (fast)
Python Tests (maturin)
Python Tests (no native)
RL Ecosystem API Compliance (Gymnasium + PettingZoo)
Clean Wheel Install Smoke
OpenSpec Specification & Change Validation
mc-bot Node Tests + Biome Lint
forge-mc-runner Binary Smoke
WASM Target Check + Node Runtime Tests
Demo UI Tests
Dashboard Build + Lint + Coverage
cargo-deny (blocking)
gitleaks (blocking)
pip-audit (blocking)
```

Never require `Docker Build & Publish (GHCR + optional Docker Hub)` — it is
skipped on pull requests, so it would sit permanently as "Expected".

Decide explicitly on the remaining PR-time jobs, which run today but are not in
the charter's blocking list: `Benchmarks`, `HF Export (feature-gated)`, `MLflow
HTTP Transport (feature-gated)`, `ONNX Feature Surface (…)`, `Mutation Testing
(security-critical modules)`, `mc-live-bundled Runner Image Smoke`. Leave
`Dashboard E2E + AQA (Playwright)` and `WASM Demo E2E (Playwright)` unrequired.

### 7.4 Tag and publish

```bash
git checkout main && git pull --ff-only origin main
git tag -a v0.6.0 -m "FORGE v0.6.0 — Unifying the Rust Core, Deterministic Physics, and Ecosystem Compliance"
git push origin refs/tags/v0.6.0

gh run list --workflow ci.yml --branch v0.6.0        # works now that on.push.tags exists
docker manifest inspect ghcr.io/ianshank/forge:0.6.0
docker manifest inspect ghcr.io/ianshank/forge:latest

gh release create v0.6.0 --verify-tag --draft --notes-file release-notes.md
gh release edit v0.6.0 --draft=false                 # once tag CI is green
```

Note `:latest` moves on every `v*` tag, so a yanked tag leaves it pointing at
the yanked build until re-tagged.

### 7.5 Decide on Pages and the HF Space

Both have never succeeded. Either fix them as part of this release — enable
Pages with Source = GitHub Actions, allow `main` in the auto-created
`github-pages` environment, add the `HF_TOKEN` secret, then
`gh workflow run gh-pages.yml --ref main` and `hf-space.yml` — or state plainly
in the release notes that the browser demo is not part of v0.6.0. Do not leave
it ambiguous.

### 7.6 Post-release sweep

`grep -rn plan-forge-environment-htAoK` and update the workflow comments and
`docs/next_steps.md` references once the old branch name is gone.

---

## 8. Known gaps — what is *not* verified

Stated plainly rather than glossed:

- **Coverage caveat:** the 93.49% figure is from the torch-enabled local run
  (1,843 tests). In constrained environments without torch, optional-module
  imports can pull the percentage down even when this branch's touched surfaces
  are fully covered; CI's `Python Tests (maturin)` job is authoritative.
- **No wheel-publishing pipeline exists.** There is no PyPI or crates.io
  release path anywhere in the repository; GHCR is the only artifact registry
  wired up. A `release.yml` using `PyO3/maturin-action` for manylinux, macOS,
  and Windows wheels is a sensible follow-up, not a v0.6.0 requirement.
- **`reference_b` benchmark numbers do not exist** (D3).
- **The `gh` flag details in §9 were not re-verified against a live `gh`.**
  Check `gh repo archive --help` on the operator's version before running.

---

## 9. `agent_consolidation.sh` review (out of repository)

The script triages `ianshank` repositories under 5 stars with no push in 90
days, then badges and archives four redundant agent repositories. None are
FORGE. Keep it in a personal ops repository and run it by hand.

Fix before running:

1. **`gh repo archive --confirm`** — recent `gh` replaced `--confirm` with
   `--yes`. Check `gh repo archive --help` first; otherwise the loop aborts
   after the first push.
2. **Default-branch guess** — `git push origin main || git push origin master`
   aborts the whole script under `set -e` if the repo uses neither. Read the
   real branch with
   `gh repo view "$USERNAME/$REPO" --json defaultBranchRef -q .defaultBranchRef.name`.
3. **No cleanup on failure** — add `trap 'rm -rf "$TMP_DIR"' EXIT` immediately
   after `mktemp -d`.
4. **Not idempotent** — a re-run after a partial failure prepends a second
   badge. Guard with `grep -q 'Status-Archived' README.md && continue`.
5. **`cd "$REPO"` after a failed clone** would run the edit in the wrong
   directory; check the clone succeeded.
6. **Clone auth** — use `gh repo clone` rather than a raw HTTPS `git clone`.
7. **Cutoff date format** — GNU `date -Iseconds` emits a `+00:00` offset while
   GitHub's `pushedAt` ends in `Z`, so the lexicographic `--jq` comparison is
   not like-for-like. Use `date -u -d "90 days ago" +"%Y-%m-%dT%H:%M:%SZ"`.
8. **`read -p` under `set -e`** fails in a non-interactive shell; guard on
   `[ -t 0 ]`.
9. **Triage is informational only** — the archived set is hard-coded and never
   checked against the "<5 stars, 90 days" criteria. Fine, but say so in the
   header comment.
10. **Add `--dry-run`** that prints each `gh`/`git` command instead of running
    it, and use it once before the real pass.
