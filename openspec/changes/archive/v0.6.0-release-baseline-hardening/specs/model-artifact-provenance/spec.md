# Specification: Model Artifact Provenance and Publishing Contracts

## Purpose

Establish honest, fail-closed contracts for model artifact publishing,
metadata cards, and cloud deployments across the Hugging Face Hub and GitHub
Pages.

## MODIFIED Requirements

### Requirement: Mandatory Untrained Warning for Random-Init Bundles
`scripts/hf_publish_model.py` and model card generation SHALL default to
injecting the `UNTRAINED_WARNING` banner into `README.md` for any published
bundle. The warning banner SHALL NOT be omitted unless `--trained` is
explicitly passed to the publishing CLI.

#### Scenario: Publishing bundle defaults to random-init warning banner
- **GIVEN** a model bundle generated via `forge.training.muzero_mc.cli bootstrap`
- **WHEN** `python scripts/hf_publish_model.py` stages the model card without
  `--trained`
- **THEN** the staged `README.md` SHALL contain the `UNTRAINED_WARNING` markdown
  banner
- **AND** the warning SHALL clearly state that the bundle holds random-init weights

#### Scenario: Falsifier: Model card staged without warning when trained=false
- **GIVEN** a publishing invocation with `trained = False`
- **WHEN** the generated model card is checked for `UNTRAINED_WARNING`
- **THEN** if the warning string is absent, `scripts/hf_publish_model.py` unit
  tests SHALL fail

### Requirement: Evidential Proof Required for Trained Model Claims
No model bundle SHALL be claimed or published as a "trained model" (using
`--trained`) unless backed by committed or uploaded evidential capture
artifacts showing an evidential episode count of at least 3 (`evidential_episodes >= 3`)
against a running environment stack.

#### Scenario: Attempting to publish trained model with evidential backing
- **GIVEN** an operator has conducted an evidential capture run producing at
  least 3 evidential episodes
- **WHEN** `scripts/hf_publish_model.py --trained` is executed with the verified
  bundle
- **THEN** the model card SHALL omit the untrained warning
- **AND** the bundle manifest SHALL record valid SHA-256 digests and schema ID

#### Scenario: Falsifier: Claiming trained model without evidential episodes
- **GIVEN** a model bundle derived only from bootstrap random initialization
  or failed rollouts
- **WHEN** a contributor attempts to publish with `--trained`
- **THEN** the release gate SHALL reject the claim as non-evidential under the
  rules of `refuse-non-evidential-aggregates`

### Requirement: Fail-Closed Documentation for Cloud Secrets and Deployments
Deployments of the WASM demo to GitHub Pages and Hugging Face Spaces (`ianshank/forge-wasm-demo`)
SHALL be explicitly documented as blocked or non-done until operator settings
and secrets (`HF_TOKEN`, GitHub Pages Actions source) are configured in the
repository. Workflow definitions SHALL fail closed rather than reporting
spurious success.

#### Scenario: Publishing workflow fails closed when secret is absent
- **GIVEN** `.github/workflows/hf-space.yml` runs without `HF_TOKEN` configured
- **WHEN** the workflow step checks authentication
- **THEN** it SHALL terminate with an actionable error indicating the secret is
  missing
- **AND** documentation in `docs/next_steps.md` SHALL accurately reflect this
  status as blocked on repo settings

#### Scenario: Falsifier: Documenting live WASM demo as landed without live deployment
- **GIVEN** the live demo URLs for GitHub Pages or Hugging Face Spaces
- **WHEN** repository documentation claims the live demo is deployed and active
- **AND** an HTTP probe against the endpoints returns 404 or uninitialized state
- **THEN** documentation audit SHALL fail the claim as inaccurate
