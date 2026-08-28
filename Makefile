SHELL := /bin/bash
.DEFAULT_GOAL := help

# FORGE developer entrypoints. Every target here wraps a command already
# documented in CONTRIBUTING.md / CLAUDE.md -- this file adds nothing new,
# it just saves re-typing the exact CI invocations. Keep both in sync.

.PHONY: help build fmt fmt-check lint test coverage \
        onnx-check deny gitleaks pin-check \
        py-lint py-test hooks-test \
        mc-bot-test dashboard-test \
        verify verify-full clean

help: ## Show this help
	@echo "FORGE developer targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## /{ printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---- Rust ------------------------------------------------------------------

build: ## cargo build --workspace
	cargo build --workspace

fmt: ## Apply cargo fmt across the workspace
	cargo fmt --all

fmt-check: ## Check formatting without modifying files (matches CI)
	cargo fmt --all --check

lint: ## cargo clippy, workspace-wide, warnings-as-errors (matches CI)
	cargo clippy --workspace --all-targets --features forge-cloud/gcs -- -D warnings

test: ## cargo test, workspace-wide (matches CI)
	cargo test --workspace --features forge-cloud/gcs

coverage: ## cargo tarpaulin, 85% workspace floor (matches CI; installs tarpaulin if missing)
	@command -v cargo-tarpaulin >/dev/null || cargo install cargo-tarpaulin --version 0.31.0 --locked
	cargo tarpaulin --workspace --exclude forge-python --exclude forge-wasm \
		--features forge-cloud/gcs --skip-clean --fail-under 85

onnx-check: ## Build/test the onnx/onnx-reload/mc-live-bundled feature surface (needs ORT_DYLIB_PATH; see ci.yml's onnx-features job for the fetch steps)
	@if [ -z "$$ORT_DYLIB_PATH" ]; then \
		echo "ORT_DYLIB_PATH is not set -- see the 'onnx-features' job in .github/workflows/ci.yml" \
		     "for how to fetch a real ONNX Runtime >=1.23.2 .so and point this at it."; \
		exit 1; \
	fi
	cargo clippy -p forge-agent --all-targets --features onnx-bundled -- -D warnings
	cargo clippy -p forge-mc-runner --all-targets --features mc-live-bundled -- -D warnings
	cargo test -p forge-agent --features onnx-bundled
	cargo test -p forge-mc-runner --features mc-live-bundled

deny: ## cargo-deny supply-chain check (advisory; installs cargo-deny if missing)
	@command -v cargo-deny >/dev/null || cargo install cargo-deny --locked
	cargo deny check --all-features

gitleaks: ## Scan git history for committed secrets (advisory; needs the gitleaks binary on PATH -- see security.yml)
	@command -v gitleaks >/dev/null || { \
		echo "gitleaks not found on PATH -- install it: https://github.com/gitleaks/gitleaks#installing"; \
		exit 1; \
	}
	gitleaks git --redact -v .

pin-check: ## Cross-check Rust toolchain / ONNX Runtime / LM Studio endpoint pins duplicated across workflows, Dockerfiles, and Python
	python3 scripts/check_pinned_config_consistency.py

# ---- Python ------------------------------------------------------------------

py-lint: ## ruff + mypy (matches CI's python-lint job)
	ruff check python/ tests/python/ scripts/ demo_ui/ examples/
	mypy python/ scripts/ --config-file pyproject.toml

py-test: ## pytest tests/python, excluding opt-in markers (build the native ext first: maturin develop)
	pytest tests/python -m 'not lmstudio and not e2e_long and not minecraft_e2e'

hooks-test: ## Self-tests for .claude/hooks/ (stdlib-only, no project deps; matches CI's python-lint job)
	python3 -m unittest discover -s .claude/hooks -p 'test_*.py' -v

# ---- Node --------------------------------------------------------------------

mc-bot-test: ## mc-bot: typecheck + Biome lint + node:test + coverage
	cd mc-bot && npm ci && npm run typecheck && npm run lint && npm test && npm run test:coverage

dashboard-test: ## dashboard: build + Biome lint + Vitest coverage gate (85%)
	cd dashboard && npm ci && npm run build && npm run lint && npm run test:coverage

# ---- Aggregate -----------------------------------------------------------

verify: fmt-check lint test py-lint py-test hooks-test pin-check mc-bot-test dashboard-test ## Run the standard pre-PR gate sequence (excludes coverage/onnx-check/deny/gitleaks -- see verify-full)
	@echo "verify: all standard gates passed."

verify-full: verify coverage deny gitleaks ## verify, plus the slower/environment-dependent gates (tarpaulin, cargo-deny, gitleaks). Does NOT include onnx-check (needs ORT_DYLIB_PATH set manually).
	@echo "verify-full: all gates passed."

clean: ## cargo clean (frees significant disk space; safe, fully reproducible)
	cargo clean
