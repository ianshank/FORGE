# FORGE Beta Release — DevSQE Implementation Guide

> **Audience**: Dev/SQE engineers executing the beta release plan  
> **Prerequisites**: Read `PLANNING.md`, all PRDs in `docs/prd/`, and `docs/architecture/ADR-001-beta-release.md`  
> **Branch**: `feat/beta-release-planning` (planning docs); feature branches per sprint  
> **Target**: `v0.2.0-beta` GA

---

## Pre-Sprint Checklist

Before starting Sprint 1, verify:

- [ ] `maturin develop` succeeds locally (`python -c "from forge_env import ForgeEnv"`)
- [ ] All 389 Rust tests pass: `cargo test --workspace`
- [ ] All 61 Python tests pass: `pytest tests/python/ -v`
- [ ] Ruff clean: `python -m ruff check python/ tests/ demo_ui/`
- [ ] Mypy clean: `python -m mypy python/forge_env/ demo_ui/backend/ --no-site-packages`

---

## Sprint 1 (Feb 27 – Mar 12): Packaging + Coverage Gates

### Epic E1: PyPI Packaging

**Branch**: `feat/e1-pypi-packaging`

#### Dev Tasks

1. **Create `release.yml`** at `.github/workflows/release.yml`:

   ```yaml
   on:
     push:
       tags: ['v*']
   jobs:
     build:
       strategy:
         matrix:
           os: [ubuntu-latest, macos-latest, windows-latest]
       runs-on: ${{ matrix.os }}
       steps:
         - uses: actions/checkout@v4
         - uses: PyO3/maturin-action@v1
           with:
             command: build
             args: --release -m crates/forge-python/Cargo.toml
             manylinux: '2014'
     publish:
       needs: build
       runs-on: ubuntu-latest
       steps:
         - uses: PyO3/maturin-action@v1
           with:
             command: publish
           env:
             MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
   ```

2. **Update `pyproject.toml`** — add proper version metadata, authors, classifiers, PyPI URL
3. **Test locally**: `maturin build --release -m crates/forge-python/Cargo.toml && pip install target/wheels/forge_env-*.whl`
4. **Validate**: `python -c "import forge_env; print(forge_env.__version__)"`

#### SQE Tasks

- [ ] Write `tests/integration/test_install.py`: imports work, version matches tag
- [ ] Verify wheel installs on clean Python 3.9, 3.11, 3.12 venvs
- [ ] Confirm no Rust/build-toolchain dependencies in installed wheel (binary only)

#### Acceptance Criteria Verification

- AC1: `pip install forge-env` on Linux/macOS/Windows — test in matrix CI
- AC2: `ForgeEnv().reset()` works post-install
- AC3: Triggered by version tag → auto-publishes
- AC4: `forge-env==0.2.0b1` installs exact version
- AC5: macOS ARM64 wheel installs natively

---

### Epic E3: Coverage Enforcement

**Branch**: `feat/e3-coverage`

#### Dev Tasks

1. **Update `ci.yml`**: add `--cov=forge_env --cov-report=xml --cov-fail-under=80`
2. **Install `pytest-cov`**: already in `pyproject.toml` (added in previous session)
3. **Run `maturin develop` in CI** before pytest (so local source is tested):

   ```yaml
   - name: Build & install (development)
     run: maturin develop -m crates/forge-python/Cargo.toml
   - name: Run Python tests with coverage
     run: pytest tests/python/ --cov=forge_env --cov-report=xml --cov-fail-under=80
   ```

4. **Upload coverage report**: `codecov/codecov-action@v4`

#### SQE Tasks

- [ ] Identify all modules below 80% (currently `jax_env.py` at 0% — needs JAX mock tests)
- [ ] Write `tests/python/test_jax_env_pure.py` — mock JAX imports to test init/validation
- [ ] Verify coverage gate blocks PRs with regressions

---

### Epic E4: Cross-platform Launcher

**Branch**: `feat/e4-bash-launcher`

#### Dev Tasks

1. **Create `demo_ui/run_demo.sh`** (Bash equivalent of `run_demo.ps1`):

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   pip install -r demo_ui/backend/requirements.txt --quiet
   uvicorn demo_ui.backend.main:app --host 127.0.0.1 --port 8765 &
   sleep 1
   open "http://127.0.0.1:8765" 2>/dev/null || xdg-open "http://127.0.0.1:8765"
   wait
   ```

2. **`chmod +x demo_ui/run_demo.sh`**, add to `.gitignore` exclusion check

#### SQE Tasks

- [ ] Test `run_demo.sh` on Ubuntu (GitHub Actions `ubuntu-latest`)
- [ ] Test `run_demo.sh` on macOS — verify `open` command works
- [ ] Add CI step that validates script syntax with `shellcheck`

---

## Sprint 2 (Mar 13 – Mar 26): Benchmark Regression CI

### Epic E2: Benchmark Regression CI

**Branch**: `feat/e2-bench-ci`

#### Dev Tasks

1. **Create `.github/workflows/bench.yml`**:

   ```yaml
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   jobs:
     benchmark:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Run Criterion benchmarks
           run: cargo bench -p forge-bench -- --output-format=json 2>&1 | tee bench_output.json
         - name: Download baseline
           uses: actions/cache@v4
           with:
             path: .bench_baseline/
             key: bench-baseline-${{ github.base_ref }}
         - name: Compare with baseline
           run: python scripts/bench_compare.py bench_output.json .bench_baseline/
         - name: Save new baseline (main only)
           if: github.ref == 'refs/heads/main'
           uses: actions/cache@v4
           with:
             path: .bench_baseline/
             key: bench-baseline-main-${{ github.sha }}
   ```

2. **Create `scripts/bench_compare.py`** — parses Criterion JSON, computes delta vs baseline, fails if > 10% regression, posts comment via GitHub API

#### SQE Tasks

- [ ] Validate benchmark runs reproducibly in CI (3 consecutive runs < 5% variance)
- [ ] Test regression detection: temporarily introduce a 15% slowdown, confirm CI fails
- [ ] Test with empty baseline (first run): should auto-create baseline without failing

---

## Sprint 3 (Mar 27 – Apr 9): WASM Build Pipeline

### Epic E5: WASM Build + GH Pages

**Branch**: `feat/e5-wasm-ghpages`

#### Dev Tasks

1. **Install `wasm-pack`** in CI:

   ```yaml
   - uses: jetli/wasm-pack-action@v0.4.0
     with:
       version: 'latest'
   ```

2. **Build command**: `wasm-pack build crates/forge-wasm --target web --out-dir ../../docs/wasm-demo/pkg`
3. **Optimize**: `wasm-opt -O3 docs/wasm-demo/pkg/forge_wasm_bg.wasm -o docs/wasm-demo/pkg/forge_wasm_bg.wasm`
4. **Create `docs/wasm-demo/index.html`** — self-contained demo page loading WASM via `import init from './pkg/forge_wasm.js'`
5. **Deploy**: `peaceiris/actions-gh-pages@v3` to `gh-pages` branch

### Epic E6: In-browser Demo Page

#### Dev Tasks

1. **Create `docs/wasm-demo/demo.js`** — calls `ForgeWasmEnv.reset()`, `ForgeWasmEnv.step()`, renders via `<canvas>`
2. **URL hash routing**: `window.addEventListener('hashchange', ...)` → extract seed, re-initialize env
3. **Error boundary**: wrap all WASM calls in try/catch; surface errors in styled `<div id="error-banner">`

#### SQE Tasks

- [ ] `tests/wasm/test_wasm_build.py`: verify `forge_wasm_bg.wasm` exists and is < 5MB
- [ ] Playwright test: `tests/e2e/test_wasm_demo.py` — loads GH Pages URL, clicks Reset, verifies canvas non-empty
- [ ] Visual regression: snapshot `<canvas>` output for seed 42 at step 10; compare in CI

---

## Sprint 4 (Apr 10 – Apr 23): Shareable Links + SB3 Training

### Epic E7: Shareable World Seeds

**Branch**: `feat/e7-seed-urls`

#### Dev Tasks

1. On page load: parse `window.location.hash` → if `#seed=N`, call WASM with seed N
2. "Copy seed link" button: `navigator.clipboard.writeText(window.location.href + '#seed=' + currentSeed)`
3. Validate seed is an integer in range `[0, 2^32)`

### Epic E8: SB3 PPO Example (CI)

**Branch**: `feat/e8-sb3-training`

#### Dev Tasks

1. **Update `pyproject.toml`** — add `[project.optional-dependencies]` `sb3 = ["stable-baselines3>=2.0,<3"]`
2. **Update `examples/train_ppo.py`** with full argparse:

   ```python
   parser.add_argument("--timesteps", type=int, default=100_000)
   parser.add_argument("--seed", type=int, default=42)
   parser.add_argument("--env-size", type=int, default=32)
   parser.add_argument("--no-render", action="store_true")
   parser.add_argument("--wandb", action="store_true")
   parser.add_argument("--output-dir", type=Path, default=Path("runs"))
   ```

3. **CSV callback**: `EpisodeMetricsCallback(output_csv: Path)` — writes `ep_len_mean`, `ep_rew_mean`, `fps` per episode

#### SQE Tasks

- [ ] `tests/integration/test_training_smoke.py`: `pytest.mark.slow`, runs `train_ppo.py --timesteps 1000 --no-render`, asserts exit code 0
- [ ] Add `pytest -m "not slow"` to default CI; `pytest -m slow` in a separate nightly job
- [ ] Verify `ep_rew_mean > 0` after 1000 steps (the environment is learnable)

---

## Sprint 5 (Apr 24 – May 7): CleanRL + W&B Hooks

### Epic E9: CleanRL PPO

**Branch**: `feat/e9-cleanrl`

#### Dev Tasks

1. **Create `examples/train_ppo_cleanrl.py`** following CleanRL's single-file pattern:
   - `Args` dataclass with `total_timesteps`, `learning_rate`, `seed`
   - Gymnasium-compatible via `gym.make` → `ForgeGymnasiumEnv`
2. **Add to `pyproject.toml`**: `cleanrl = ["cleanrl-exp>=1.0"]` optional dep

### Epic E10: W&B / MLflow Hooks

**Branch**: `feat/e10-logging-hooks`

#### Dev Tasks

1. **Add `LoggingCallback` base class** in `python/forge_env/callbacks.py`:

   ```python
   class LoggingCallback(ABC):
       def on_episode_end(self, stats: EpisodeStats) -> None: ...
   class WandbCallback(LoggingCallback): ...
   class MLflowCallback(LoggingCallback): ...
   ```

2. **`RecordEpisodeStatistics`** modified to call registered callbacks on episode end

#### SQE Tasks

- [ ] Unit tests: `test_callbacks.py` — mock callbacks verify `on_episode_end` called with correct stats
- [ ] Integration: `test_wandb_callback.py` with `unittest.mock.patch("wandb.log")` — verify called with correct keys

---

## Sprint 6 (May 8 – May 21): Replay / Record Mode

### Epic E11: Replay / Record Mode

**Branch**: `feat/e11-replay`

#### Dev Tasks

1. **`python/forge_env/wrappers.py`** — add `RecordEpisodeWrapper`:

   ```python
   class RecordEpisodeWrapper(_BaseWrapper):
       def __init__(self, env, output_path: Path | None = None) -> None: ...
       def reset(self, **kwargs) -> tuple: ...  # stores seed
       def step(self, action) -> tuple: ...  # appends action
       def save(self, path: Path) -> None: ...  # writes JSON
   ```

2. **`python/forge_env/replay.py`** — `load_replay`, `play_replay`, `validate_version`
3. **CLI**: `python -m forge_env.replay replay.json --fps 10`
4. **Frontend**: "📼 Load Replay" → `<input type="file">` → parse JSON → call WASM steps in loop
5. **GIF export**: `gif.js` integration in `app.js`; "📥 Export GIF" button

#### SQE Tasks

- [ ] `test_record_wrapper.py`: records 10-step episode, validates JSON schema, loads replay, asserts observations match
- [ ] `test_replay_cli.py`: invokes CLI via subprocess, checks exit code and terminal output
- [ ] `test_replay_version_mismatch.py`: version field mismatch → warning printed, replay continues

---

## Sprint 7 (May 22 – Jun 4): REST API + Multi-Agent Dashboard

### Epic E12: REST Env API

**Branch**: `feat/e12-rest-api`

#### Dev Tasks

1. **Create `api/main.py`** — FastAPI app with session pool:

   ```python
   class SessionPool:
       def __init__(self, ttl_seconds: int = 1800): ...
       def create(self, seed: int, config: dict) -> str: ...  # returns UUID
       def get(self, env_id: str) -> ForgeGymnasiumEnv: ...  # raises 404
       def _evict_expired(self) -> None: ...
   ```

2. **Routes**: `POST /api/env/reset`, `POST /api/env/step`, `GET /api/env/{id}/render`
3. **JSON serialization**: numpy arrays → nested Python lists via `obs_to_json(obs: dict) -> dict`
4. **OpenAPI docs**: enabled by default at `/docs`

#### SQE Tasks

- [ ] `tests/api/test_rest_api.py` — 7 tests covering all ACs (reset, step, render, 404, 422, concurrent, OpenAPI)
- [ ] Load test: `locust -f tests/load/locustfile.py --headless -u 10 -r 2 --run-time 30s`; verify p99 < 10ms
- [ ] Security: verify env_id is UUID4 (not guessable), verify session isolation between clients

### Epic E13: Multi-Agent Dashboard

**Branch**: `feat/e13-multiagent-ui`

#### Dev Tasks

1. Extend `WorldRenderer` in `app.js` to accept per-agent color palette
2. `ForgeParallelEnv` observations → `agent_positions[]` in render API `GET /api/env/{id}/render`
3. Sidebar: "Agents" panel with per-agent reward curve (sparkline charts)

---

## Sprint 8 (Jun 5 – Jun 18): Task Curriculum Visualizer + Beta GA

### Epic E14: Task Curriculum Visualizer

**Branch**: `feat/e14-curriculum-ui`

#### Dev Tasks

1. **Expose task state from `ForgeEnv`**: add `get_task_tree() -> dict` to PyO3 bindings
2. **Frontend**: recursive DSL tree renderer (collapsible JSON-style tree in sidebar)
3. **Tier histogram**: Chart.js bar chart showing curriculum tier distribution from recent resets

### Beta GA Checklist

- [ ] All 15 epics merged to `main`
- [ ] All PRD acceptance criteria verified (AQA reports in `docs/aqa/`)
- [ ] `pytest tests/ --cov=forge_env --cov-fail-under=80` passes
- [ ] Benchmarks show no regression vs v0.1.0 baseline
- [ ] `CHANGELOG.md` updated with full `[0.2.0-beta]` section
- [ ] `README.md` updated: installation via pip, live demo link, training examples
- [ ] PyPI release published: `pip install forge-env==0.2.0b1`
- [ ] GitHub Pages live: `https://ianshank.github.io/FORGE`
- [ ] GitHub release tag `v0.2.0-beta.1` created with release notes

---

## AQA (Acceptance QA) Report Template

For each epic, generate `docs/aqa/aqa-<epic-slug>.md` with:

```markdown
# AQA — <Epic Name>

| AC# | Criterion | Status | Test File | Notes |
|---|---|---|---|---|
| AC1 | ... | ✅ PASS / ❌ FAIL | test_xxx.py::test_yyy | |
| AC2 | ... | ✅ PASS | test_xxx.py::test_zzz | |

## Screenshots (if UI)
![Screenshot of feature](./screenshots/feature-<date>.png)

## Sign-off
- [ ] Dev lead reviewed
- [ ] SQE lead reviewed
- [ ] Product sign-off
```

---

## Regression Test Checklist (run on every PR)

```bash
# Mandatory — blocks merge if any fail
cargo test --workspace                              # 389 Rust tests
cargo clippy --workspace -- -D warnings            # Zero Clippy warnings
cargo fmt --check                                  # Rust formatting
python -m ruff check python/ tests/ demo_ui/       # Zero lint errors
python -m mypy python/forge_env/ demo_ui/backend/ --no-site-packages  # Type clean
pytest tests/python/ --cov=forge_env --cov-fail-under=80  # ≥80% coverage
pytest demo_ui/tests/test_backend.py               # 15 backend tests

# Gate-blocked (run post-merge on main)
cargo bench -p forge-bench                         # Benchmark regression gate
pytest tests/ -m slow                              # Training smoke tests
```
