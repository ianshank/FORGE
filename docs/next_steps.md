# FORGE — Next Steps & Roadmap

Post-demo-UI priorities, roughly in order of impact.

---

## Immediate (v0.2)

### 1. Playwright E2E Browser Tests

Add `playwright`-based end-to-end tests for the demo UI to validate:

- Section badges transition IDLE → RUNNING → PASS
- Terminal streams without error
- Progress bar advances correctly
- World canvas receives visible pixel data

```bash
# Planned test file
demo_ui/tests/test_e2e_browser.py
```

### 2. GitHub Actions CI for Demo UI

Extend `.github/workflows/` to include:

- Install Python deps from `demo_ui/backend/requirements.txt`
- Run `pytest demo_ui/tests/test_backend.py` in CI
- Smoke-test the server with `httpx` (headless)
- Cache `pip` installs for faster runs

### 3. Docker Container for Demo UI

Package the entire demo UI into a single Docker image for zero-setup deployment:

```dockerfile
FROM python:3.11-slim
WORKDIR /forge
COPY demo_ui/ demo_ui/
COPY examples/ examples/
COPY demo_results.md .
RUN pip install -r demo_ui/backend/requirements.txt
CMD ["python", "-m", "uvicorn", "demo_ui.backend.main:app", "--host", "0.0.0.0", "--port", "8765"]
```

---

## Near-term (v0.3)

### 4. GitHub Pages / WASM Live Demo

Compile `forge-wasm` and serve a **fully-static** demo directly from `gh-pages`:

- No server required — all simulation runs in the browser via WASM
- Replace the SSE backend with in-browser WASM calls
- Enables public shareable demo link

### 5. Benchmark Regression Tracking

Integrate `forge-bench` Criterion results into the demo UI's stats panel:

- Store benchmark JSON artifacts in CI
- Plot step-throughput over releases
- Flag regressions (>5% slowdown) as PR failures

### 6. Replay / Record Mode

Allow the demo UI to:

- **Record** a run to a JSON replay file (actions + RNG seed)
- **Replay** a recorded session at configurable speed
- Export the world canvas as an animated GIF

---

## Longer-term (v1.0)

### 7. REST API for External Integrations

Expose FORGE as a proper REST service so external tools (notebooks, ML frameworks) can drive it:

```http
POST /api/env/reset          {"seed": 42, "config": {...}}
POST /api/env/step           {"action": 1}
GET  /api/env/render         → returns ASCII + grid JSON
```

### 8. Multi-Agent Dashboard

Extend the demo UI world canvas to show:

- Per-agent color coding
- Communication token visualization
- Reward curves per agent

### 9. Task Curriculum Visualizer

Add a dedicated UI panel for the task system:

- Current task tree visualization (recursive DSL rendering)
- Curriculum tier distribution histogram
- Live success/failure rate as the agent trains

### 10. SB3 / Cleanrl Training Integration

Add example scripts and CI integration for:

- `train_ppo.py` with Stable Baselines 3
- `train_ppo_cleanrl.py` with CleanRL
- W&B / MLflow logging hooks from FORGE env wrappers
- Pre-trained model checkpoint serving via the demo UI

---

## Technical Debt

| Item | Priority | Notes |
|---|---|---|
| Mypy strict mode for `demo_ui/tests/` | Low | Test files have unannotated optional args |
| `conftest.py` root sys.path approach | Medium | Consider `pyproject.toml` package install instead |
| `demo_ui` as installable package | Medium | `pip install -e demo_ui/` makes imports cleaner |
| Section-level output parsing | Low | Parse structured data from `forge_demo.py` for richer stats |
| `run_demo.ps1` → `run_demo.sh` cross-platform | Medium | Add Bash launcher for Linux/macOS users |
