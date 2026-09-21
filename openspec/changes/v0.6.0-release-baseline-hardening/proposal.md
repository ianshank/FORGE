# Proposal: v0.6.0 Release Baseline Hardening

## Why

FORGE is ready for an evidence-backed developer preview, but the release path
has suffered from narrative and specification drift. Previous proposals
attempted to treat already-landed runtime capabilities (such as Gymnasium and
PettingZoo wrappers, claim-backed throughput gates, CompactReplay v2 goldens,
and nominal orchard drone coverage) as greenfield additions. At the same time,
they assumed nonexistent distribution mechanisms (such as automated PyPI
publishing and multi-platform cibuildwheel matrices), unbuilt product features
(such as motor and GPS fault injection), unverified live Hugging Face models,
and an unverified version bump from an imagined 0.5.0 state.

The repository reality is concrete:
1. Workspace versioning across Cargo.toml, forge_env/_version.py,
   dashboard/package.json, and dashboard/package-lock.json is already 0.6.0.
   Treating the release as a new bump to 0.6.0 is factually false and violates
   SemVer integrity. The actual milestone is complete-v0.6.0-gates.
2. The core ecosystem interfaces, throughput measurement (189k steps/second on
   cloud_agent), and golden replays already exist and pass in CI. They must be
   specified as regression and hardening gates, not greenfield deliverables.
3. Distribution today is strictly limited to local maturin builds, source
   distribution, and container images via GHCR. Multi-platform PyPI wheels and
   cibuildwheel workflows are completely absent and cannot be claimed as landed.
4. Hugging Face model and space publishing pipelines exist in code, but live
   deployments are blocked by repository permissions and unset secrets (such as
   HF_TOKEN and GitHub Pages Actions source). Furthermore, only random-init
   bootstrap models exist; no trained MuZero Minecraft model artifact has been
   captured with evidential support.
5. Autonomy vertical claims in orchard coverage represent L0 synthetic
   coverage under deterministic process constraints (geofence margins, battery
   floors, altitude ceilings), not flight-certified physical fault injection.

To achieve an honest, contract-verified developer preview, FORGE requires a
hardened release change package that fences scope, aligns specification with
codebase reality, locks kill criteria to named CI/ops artifacts, and codifies
an explicit operator runbook.

## What Changes

This change establishes the formal release baseline hardening contract:

- Ratifies the release milestone as complete-v0.6.0-gates. Confirms that
  workspace version 0.6.0 is already established and requires that no version
  bump or downgrade is performed.
- Establishes regression gates for landed Gymnasium and PettingZoo wrappers,
  requiring continuous validation via upstream env checkers in the
  api-compliance CI job.
- Enforces Python packaging honesty: explicitly documents that Linux x86_64
  maturin builds in the pip-install-clean job define the validated wheel
  boundary, and documents that Python 3.11 is the sole automated CI matrix
  target while requires-python is >=3.9 (or mandates narrowing requires-python
  if multi-version CI is not added).
- Preserves throughput claim integrity by gating published documentation
  claims against committed PyO3 benchmarks (189k steps/second on cloud_agent).
- Maintains bit-identical determinism on CompactReplay v2 fixtures
  (v2_seed42.json) under golden-replay verification.
- Enforces model artifact provenance: mandates the UNTRAINED_WARNING banner
  for any random-init Hugging Face model publication, forbids claiming a
  trained model without trained=true and an accompanying evidential artifact,
  and documents GitHub Pages and HF_TOKEN as blocked or non-done until operator
  secrets are configured.
- Fences drone autonomy capabilities to L0 synthetic process constraints,
  refusing any unverified claims of physical motor or GPS fault injection.
- Defines clear developer adoption paths via source checkout, local maturin
  installation, and GHCR container execution.
- Defines the operator release checklist, including Decision D2 (renaming the
  default branch from claude/plan-forge-environment-htAoK to main via GitHub
  API) and annotated tag v0.6.0 application only after all release gates pass.

## Capabilities

### Modified Capabilities

- `python-distribution`: Codify current distribution boundaries (maturin
  local build, Linux x86_64 pip-install-clean wheel smoke, GHCR images),
  document the Python 3.11 CI reality against requires-python >=3.9, and fence
  multi-platform cibuildwheel as deferred.
- `rl-api-conformance`: Codify Gymnasium and PettingZoo wrappers and upstream
  checker suites (gymnasium.utils.env_checker.check_env and
  pettingzoo.test.parallel_api_test) as mandatory regression gates.
- `evaluation-evidence`: Anchor evaluation artifacts to forge-eval RunManifest,
  OutputConfig trajectories, and docs/results/INDEX.toml checksum tracking,
  retaining refuse-non-evidential-aggregates principles.
- `model-artifact-provenance`: Enforce fail-closed model publishing contracts,
  mandatory random-init warnings, and operator secret prerequisites for Hugging
  Face and GitHub Pages deployments.
- `drone-robustness`: Formalize L0 synthetic process constraints (geofence,
  battery floor, altitude limits converting invalid actions to Noop) alongside
  the background aerial drain rate energy model for orchard coverage,
  explicitly fencing hardware fault injection as non-goals.
- `developer-adoption`: Define reproducible developer adoption journeys
  anchored in local source installation, container workflows, and reproducible
  seed execution.

## Impact

- Documentation: README.md, docs/CHARTER.md, docs/architecture.md,
  BENCHMARKS.md, and docs/results/ maintain exact agreement with committed
  artifacts and CI gates.
- Workflows: ci.yml (api-compliance, pip-install-clean, openspec-validate, python-test),
  golden-replay.yml, and security.yml serve as definitive gating mechanisms.
- Verification gates: test_version_consistency.py, test_api_compliance.py,
  test_throughput_claim.py, test_evidence_integrity.py,
  test_charter_alignment.py, and test_openspec_validation.py enforce repository invariants.
- Operator actions: Execution of branch rename D2, configuration of GitHub
  Pages source and HF_TOKEN, and final release tagging.

## Precedent Archives Cited

This proposal directly builds on and cites two archived OpenSpec changes:
1. `openspec/changes/archive/align-charter-with-codebase`: Established the
   doctrine of machine-checkable alignment between governance claims and
   executable codebase/CI artifacts, eliminating rotted citations and phantom
   features.
2. `openspec/changes/archive/refuse-non-evidential-aggregates`: Established the
   requirement that benchmarks and results must never manufacture claims from
   non-evidential runs, introducing docs/results/INDEX.toml digests and
   fail-closed validation.

## Non-Goals and Explicit Product Fences

To prevent scope creep and maintain strict release focus, the following items
are explicit non-goals for this change:

- Greenfield Gymnasium or PettingZoo rewrites: Existing wrappers are complete
  and validated; no replacement or architectural rewrite will be undertaken.
- Version downgrade or re-bump fiction: Workspace version 0.6.0 will not be
  downgraded to 0.2.0 nor described as a bump from 0.5.0.
- Cross-platform cibuildwheel and PyPI publication: Automated multi-platform
  wheel building and PyPI release workflows are deferred to post-v0.6.0.
- Cosign container signing and SBOM generation: Security provenance tooling is
  deferred and does not block v0.6.0 release.
- Hardware-grade motor and GPS fault injection: Physical fault simulation
  models are out of scope; autonomy is restricted to L0 process constraints.
- Fabricating trained MuZero weights: No trained Minecraft MuZero weights will
  be claimed or published without live evidential capture exceeding required
  floors.
- Product Fence: FORGE is an autonomous-agent evaluation runtime and simulation
  engine. It is strictly fenced from Distilled_Agents (student/GKD
  distillation algorithms) and Neuroharness i2 components.
