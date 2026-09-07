# `forge-cloud`

> **Architectural Layer**: Tier 5: Distributed Orchestration

Cloud simulation orchestrator, remote checkpoint management, GCS artifact storage, and distributed worker fleet scaling.

---

## Architectural Role & Dependencies

As defined in the **FORGE Workspace Layering Model** (`docs/architecture.md` Section 4.2):

* **Tier**: `Tier 5: Distributed Orchestration`
* **Allowed Downstream Consumers (`wrappers`)**: Cloud worker daemon / Kubernetes cluster jobs
* **Workspace Dependencies**: `forge-data`, `forge-replay`, `forge-types`

Dependency boundaries are strictly enforced via `deny.toml` `[bans]`. Introducing inverted or unapproved dependencies will cause immediate CI failure.

---

## Key Types & Public API

- `CloudWorker`: Worker instance receiving simulation jobs and uploading trajectory artifacts.
- `StorageBackend`: Abstraction over local filesystem and Google Cloud Storage.
- `GcsStorage`: Production cloud storage provider with retry policies.

---

## Feature Flags

- `default` - Local storage and mock cloud backend.
- `gcs` - Google Cloud Storage client integration for model and replay sync.

---

## Usage Example

```rust
use forge_cloud::config::CloudConfig;

let config = CloudConfig::default();
// let worker = CloudWorker::new(config).await?;
```

---

## Testing & Verification

Run tests for this crate:

```bash
cargo test -p forge-cloud
cargo clippy -p forge-cloud -- -D warnings
```
