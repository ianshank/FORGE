# `forge-eval`

> **Architectural Layer**: Tier 4: Applications & Runners

Agent-agnostic evaluation harness, scorecard generator, and artifact exporter supporting local filesystem, MLflow, and Hugging Face sinks.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 4: Applications & Runners`
* **Allowed Downstream Consumers (`wrappers`)**: Evaluation pipelines (`scripts/evaluate.py`, `cargo test`)
* **Workspace Dependencies**: `forge-core`, `forge-observability`, `forge-replay`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `Harness`: Main evaluation orchestrator running agent policies across scenario matrices.
- `Scorecard`: Aggregated evaluation report with tier success rates and mean rewards.
- `MlflowFsSink`: Exports runs, metrics, parameters, and interactive Plotly charts to MLflow.
- `resolve_plotly_js_url`: Configurable Plotly script resolution for air-gapped environments.

---

## Feature Flags

- `default` - Filesystem MLflow and scorecard exporter.
- `http-mlflow` - Direct HTTP REST integration with remote MLflow tracking servers.

---

## Usage Example

```rust
use forge_eval::scorecard::Scorecard;
use forge_eval::exporters::render_tier_bar_chart_html;

let html = render_tier_bar_chart_html(&[], "eval-run-001");
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-eval
cargo clippy -p forge-eval -- -D warnings
```
