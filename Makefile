SHELL := /bin/bash
.DEFAULT_GOAL := help

# FORGE developer entrypoints. Every target here wraps a command already
# documented in CONTRIBUTING.md / CLAUDE.md -- this file adds nothing new,
# it just saves re-typing the exact CI invocations. Keep both in sync.

.PHONY: help build fmt fmt-check lint test coverage \
        onnx-check hf-check mlflow-check alloc-audit bench-export mc-runner-smoke machete mutants \
        wasm wasm-check wasm-test deny gitleaks pin-check text-check \
        md-lint ci-parity \
        py-lint py-test hooks-test \
        mc-bot-test dashboard-test dashboard-e2e demo-ui-test web-e2e \
        verify verify-full clean

# Read the markdownlint pin out of ci.yml rather than restating it, so the
# workflow stays the single source of truth. A second literal here is exactly
# the drift `scripts/check_pinned_config_consistency.py` exists to catch, and
# not creating it is cheaper than teaching that script one more pin.
MARKDOWNLINT_CLI2_VERSION := $(shell sed -n 's/^[[:space:]]*MARKDOWNLINT_CLI2_VERSION:[[:space:]]*"\(.*\)"/\1/p' .github/workflows/ci.yml | head -1)

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

hf-check: ## Build/test the `hf` Parquet-export feature surface (matches CI's hf-export job)
	cargo clippy -p forge-replay --all-targets --features hf -- -D warnings
	cargo clippy -p forge-data --all-targets --features hf -- -D warnings
	cargo test -p forge-replay --features hf
	cargo test -p forge-data --features hf

mlflow-check: ## Build/test the `http-mlflow` transport feature surface (matches CI's mlflow-http job)
	cargo clippy -p forge-eval --all-targets --features http-mlflow -- -D warnings
	cargo test -p forge-eval --features http-mlflow
	cargo run -p forge-eval --features http-mlflow --bin forge-eval-longrun -- --help >/dev/null

alloc-audit: ## Zero-allocation hot-path contract, 0 bytes / 0 blocks (matches CI's alloc-audit job)
	cargo build -p forge-bench --bin allocation_audit --features dhat-heap --release
	./target/release/allocation_audit --warmup 1024 --iters 10000 --out alloc_audit.json
	python3 benchmarks/runner/check_zero_alloc.py --input alloc_audit.json \
		--max-bytes 0 --json alloc_audit_summary.json

# PROFILE selects the baselines/<profile>/ destination. Default is the labeled
# cloud-agent host this workspace runs on -- never silently write GitHub
# Actions `reference_a` or workstation `reference_b` numbers from the wrong
# machine. Override with `make bench-export PROFILE=reference_a` on GHA.
PROFILE ?= cloud_agent

bench-export: ## Run multi_agent_scaling Criterion bench and export committed JSON
	FORGE_BENCH_AGENT_COUNTS=1,8,16,32,64,128 \
		cargo bench -p forge-bench --bench multi_agent_scaling -- --save-baseline $(PROFILE)
	python3 benchmarks/runner/export_criterion_scaling.py \
		--criterion-dir target/criterion \
		--out benchmarks/baselines/$(PROFILE)/multi_agent_scaling.json \
		--profile $(PROFILE)

mc-runner-smoke: ## forge-mc-runner builds and dry-runs without docker/Minecraft (matches CI's forge-mc-runner-bin job)
	cargo build -p forge-mc-runner --bin forge-mc-runner
	./target/debug/forge-mc-runner --dry-run --episodes 1

mutants: ## Mutation-test the security-critical modules; any survivor fails (matches CI's mutants job)
	@command -v cargo-mutants >/dev/null || cargo install cargo-mutants --locked
	cargo mutants -p forge-mc-runner -p forge-server --timeout 120

machete: ## Report unused Cargo dependencies (blocking check across all workspace crates)
	@command -v cargo-machete >/dev/null || cargo install cargo-machete --locked
	cargo machete --with-metadata

wasm: ## Build crates/forge-wasm into web/pkg/ for the static browser demo (needs wasm-pack)
	@command -v wasm-pack >/dev/null || { \
		echo "wasm not found on PATH -- see scripts/install_wasm_pack.sh (CI) or web/README.md"; \
		exit 1; \
	}
	scripts/build_wasm_demo.sh

wasm-check: ## Clippy the wasm32 build of forge-wasm (matches CI's `wasm` job). Skips if the target is missing.
	@if rustup target list --installed 2>/dev/null | grep -qx wasm32-unknown-unknown; then \
		cargo clippy -p forge-wasm --all-targets --target wasm32-unknown-unknown -- -D warnings; \
	else \
		echo "wasm-check: SKIPPED -- wasm32-unknown-unknown is not installed."; \
		echo "  rustup users get it from rust-toolchain.toml's targets key;"; \
		echo "  otherwise: rustup target add wasm32-unknown-unknown"; \
	fi

wasm-test: ## Run forge-wasm's tests inside a real wasm runtime (matches CI's `wasm` job; needs wasm-pack)
	@command -v wasm-pack >/dev/null || { \
		echo "wasm-test: wasm-pack not on PATH -- see scripts/install_wasm_pack.sh"; \
		exit 1; \
	}
	scripts/wasm_test_node.sh

deny: ## cargo-deny supply-chain check (advisory; installs cargo-deny if missing)
	@command -v cargo-deny >/dev/null || cargo install cargo-deny --locked
	# `--all-features` is a GLOBAL cargo-deny option and must precede the
	# `check` subcommand. Written the other way round, cargo-deny 0.20.x exits 2
	# with "unexpected argument" -- which is how security.yml ran this job for
	# its entire life without ever scanning anything (the `|| true` hid it).
	cargo deny --all-features check

gitleaks: ## Scan git history for committed secrets (advisory; needs the gitleaks binary on PATH -- see security.yml)
	@command -v gitleaks >/dev/null || { \
		echo "gitleaks not found on PATH -- install it: https://github.com/gitleaks/gitleaks#installing"; \
		exit 1; \
	}
	gitleaks git --redact -v .

pin-check: ## Cross-check Rust toolchain / ONNX Runtime / LM Studio endpoint pins duplicated across workflows, Dockerfiles, and Python
	python3 scripts/check_pinned_config_consistency.py

text-check: ## Reject NUL bytes in text files and line-ending drift (both have bitten this repo -- see the script's docstring)
	python3 scripts/check_text_encoding.py

ci-parity: ## Assert every CI job has a `make` target here, or a stated reason it cannot
	python3 scripts/check_local_ci_parity.py

md-lint: ## Markdown lint with the version ci.yml pins (matches CI's markdownlint job)
	@test -n "$(MARKDOWNLINT_CLI2_VERSION)" || \
		{ echo "could not read MARKDOWNLINT_CLI2_VERSION from .github/workflows/ci.yml"; exit 1; }
	npx --yes "markdownlint-cli2@$(MARKDOWNLINT_CLI2_VERSION)" "**/*.md"

# ---- Python ------------------------------------------------------------------

py-lint: ## ruff + mypy (matches CI's python-lint job)
	ruff check python/ tests/python/ scripts/ demo_ui/ examples/
	mypy python/ scripts/ tests/python/type_checking/ --config-file pyproject.toml

py-test: ## pytest tests/python, excluding opt-in markers (build the native ext first: maturin develop)
	pytest tests/python -m 'not lmstudio and not e2e_long and not minecraft_e2e'

hooks-test: ## Self-tests for .claude/hooks/ (stdlib-only, no project deps; matches CI's python-lint job)
	python3 -m unittest discover -s .claude/hooks -p 'test_*.py' -v

# ---- Node --------------------------------------------------------------------

mc-bot-test: ## mc-bot: typecheck + Biome lint + node:test + coverage
	cd mc-bot && npm ci && npm run typecheck && npm run lint && npm test && npm run test:coverage

dashboard-test: ## dashboard: build + Biome lint + Vitest coverage gate (85%)
	cd dashboard && npm ci && npm run build && npm run lint && npm run test:coverage

dashboard-e2e: ## dashboard: Playwright E2E + accessibility suite (needs a Chromium download)
	cd dashboard && npm ci && npm run typecheck:e2e && \
		npx playwright install --with-deps chromium && npm run test:e2e

demo-ui-test: ## demo_ui: pytest suite driving the real backend (needs a Chromium download)
	pip install -e 'demo_ui[dev]'
	python -m playwright install --with-deps chromium
	pytest demo_ui/tests/ -v --tb=short

web-e2e: ## WASM demo: unit + Playwright E2E against the real web/ demo (needs wasm-pack + a Chromium download)
	cd tests/web-e2e && npm ci && npm run typecheck:e2e && npm run test:unit && npm run test:e2e

# ---- Aggregate -----------------------------------------------------------

verify: fmt-check lint test wasm-check py-lint py-test hooks-test pin-check text-check ci-parity md-lint mc-runner-smoke mc-bot-test dashboard-test ## Run the standard pre-PR gate sequence (excludes coverage/onnx-check/hf-check/mlflow-check/alloc-audit/wasm-test/deny/gitleaks/E2E -- see verify-full)
	@echo "verify: all standard gates passed."

# `ci-parity` is what stops this list silently falling behind ci.yml: it fails
# if a CI job has no target here, or if a target named in its mapping has been
# renamed away. `md-lint` and `mc-runner-smoke` joined `verify` because both
# are seconds-fast and both were CI jobs a contributor could go red on with a
# fully green local run.
verify-full: verify coverage hf-check mlflow-check alloc-audit mutants deny gitleaks ## verify, plus the slower/environment-dependent gates (tarpaulin, feature surfaces, allocation audit, mutation testing, cargo-deny, gitleaks). Does NOT include onnx-check (needs ORT_DYLIB_PATH) or the browser E2E targets (dashboard-e2e/demo-ui-test/web-e2e, each needs a Chromium download).
	@echo "verify-full: all gates passed."

clean: ## cargo clean (frees significant disk space; safe, fully reproducible)
	cargo clean
