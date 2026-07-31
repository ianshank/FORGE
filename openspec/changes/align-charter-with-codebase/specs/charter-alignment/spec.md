## Purpose

Keep `docs/CHARTER.md` and the documents it delegates to verifiably consistent
with the workspace, so that every stated invariant names enforcement that
exists, and drift fails a test rather than waiting for a manual audit.

## ADDED Requirements

### Requirement: Charter Path Citations Resolve
Every repository path cited in `docs/CHARTER.md` SHALL resolve to a file or
directory that exists in the working tree. A path that is git-ignored by design
MUST be accepted when its committed template sibling exists.

#### Scenario: Cited path is deleted or moved
- **GIVEN** `docs/CHARTER.md` cites `crates/forge-mc-runner/src/hot_reload.rs`
- **WHEN** that file is renamed without updating the charter
- **THEN** the alignment test fails and names the unresolved path

#### Scenario: Cited path is an intentionally ignored secret file
- **GIVEN** the charter cites `docker/compose.minecraft.env` as git-ignored
- **WHEN** the file is absent but `docker/compose.minecraft.env.example` exists
- **THEN** the alignment test passes

### Requirement: Charter Feature Citations Are Implemented
Every Cargo feature cited in `docs/CHARTER.md` MUST be declared by a workspace
crate. A feature cited inside an *Enforced by* clause MUST additionally have at
least one `#[cfg(feature = "…")]` site in the workspace source. Declaration
alone is insufficient — a feature no code reads is not an enforcement mechanism.

#### Scenario: Feature is cited as enforcement but never gated on
- **GIVEN** `live-test-stub` is declared in `crates/forge-mc-runner/Cargo.toml`
- **AND** no source file gates on it
- **WHEN** the charter cites it in Invariant 3's *Enforced by* clause
- **THEN** the alignment test fails and reports the feature as unimplemented

#### Scenario: Feature is declared and gated on
- **WHEN** the charter cites `onnx-reload`, which is both declared and gated on
- **THEN** the alignment test passes for that feature

### Requirement: Declared Cargo Features Stay Reachable
Every feature declared by a workspace crate SHALL be reachable: gated on in
code, enabled by another feature, or selected by a documented build command.
An aggregate or dependency-switch feature with no `#[cfg]` site MUST be listed
as a justified exception rather than silently ignored.

#### Scenario: Feature becomes an orphan
- **WHEN** a feature has no `cfg` site, is enabled by nothing, and no build
  command selects it
- **THEN** the alignment test fails and names the owning crate and feature

### Requirement: Architecture Doc Covers Every Workspace Crate
`docs/architecture.md` SHALL name every member of the `[workspace] members` list
in `Cargo.toml`. The charter delegates the authoritative crate list to this
document, so the delegation MUST be sound.

#### Scenario: Workspace crate is undocumented
- **GIVEN** `crates/forge-eval` is a workspace member
- **WHEN** `forge-eval` appears nowhere in `docs/architecture.md`
- **THEN** the alignment test fails and names the missing crate

#### Scenario: New crate is added
- **WHEN** a crate is added to `[workspace] members` without a doc entry
- **THEN** the alignment test fails in the same pull request that adds it

### Requirement: Governance Docs Name No Deleted Crates
Every `forge-*` crate name appearing in `docs/CHARTER.md` or
`docs/architecture.md` SHALL correspond to a current workspace member. The check
MUST detect names split across lines by ASCII box art, since that is how two
live `forge-procgen` references survived the `e5eca3d` deletion sweep. CI job
names and documented non-crate identifiers MUST NOT be reported as stale crates.

#### Scenario: Deleted crate survives in a diagram
- **GIVEN** `forge-procgen` was removed from the workspace
- **WHEN** `docs/architecture.md` still draws it as `forge-` and `procgen` on
  consecutive lines at the same column
- **THEN** the alignment test fails and reports the stale name

#### Scenario: Token is a CI job, not a crate
- **WHEN** a document names the `forge-mc-runner-bin` CI job
- **THEN** the alignment test does not report it as a stale crate

### Requirement: Charter CI Job Citations Exist
Every CI job named in `docs/CHARTER.md` Invariant 6 SHALL exist as a job key in
a workflow under `.github/workflows/`.

#### Scenario: Cited job is renamed
- **WHEN** the `alloc-audit` job is renamed without updating the charter
- **THEN** the alignment test fails and names the missing job

### Requirement: Cross-Language Schema Contract Names All Three Languages
`docs/CHARTER.md` Invariant 2 SHALL describe the `schema_id` contract as
spanning Rust, JavaScript and Python, and MUST cite the pinned-hash test on each
side. A contributor who bumps a hash in only two of the three languages breaks
CI, so the charter MUST NOT under-describe the contract.

#### Scenario: Contributor reads the charter before changing a shared config
- **WHEN** a contributor edits `configs/minecraft/action_map.toml`
- **THEN** the charter directs them to the Rust, Node and Python pinned tests
- **AND** all three suites are updated in the same change

### Requirement: Feature Matrix Docs Match Cargo Manifests
Any document describing a Cargo feature's dependency set SHALL match the
manifest definition. Where a feature relationship is recorded as a charter
Deliberate Exception, documents MUST NOT describe the superseded behavior.

#### Scenario: Doc states a superseded feature implication
- **GIVEN** `mc-live = ["dep:forge-env-mc"]` in `crates/forge-mc-runner/Cargo.toml`
- **AND** Deliberate Exception 2 records that `mc-live` no longer implies
  `onnx-reload`
- **WHEN** `docs/architecture.md` states `mc-live = ["onnx-reload", "dep:forge-env-mc"]`
- **THEN** the document is corrected to match the manifest

### Requirement: Permanent Non-Goals Are Ratified
`docs/CHARTER.md` SHALL state its Permanent Non-Goals as settled maintainer
decisions rather than pending proposals. Each non-goal MUST cite its actual
basis, and MUST NOT claim CI enforcement that does not exist.

#### Scenario: Non-goal cites its basis accurately
- **WHEN** the non-goal on hard-coded values is stated
- **THEN** it cites `docs/hardcoded-values-audit.md` and the config-struct
  convention as its basis
- **AND** it does not claim a CI job detects hard-coded values, because none does
