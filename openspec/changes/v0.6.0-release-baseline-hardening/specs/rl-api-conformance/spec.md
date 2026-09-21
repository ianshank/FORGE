# Specification: Reinforcement Learning API Conformance

## Purpose

Enforce strict conformance of FORGE Python environment wrappers to standard
upstream reinforcement learning interfaces (Gymnasium and PettingZoo),
maintaining these integrations as hardened regression gates rather than
greenfield deliverables.

## MODIFIED Requirements

### Requirement: Gymnasium Single-Agent API Conformance
`ForgeGymnasiumEnv` SHALL inherit from `gymnasium.Env`, build standard
observation and action spaces via `forge_env.space_builder`, and pass the full
upstream `gymnasium.utils.env_checker.check_env` suite without warnings or
exceptions. Space bounds, shapes, and containment properties SHALL be verified
across `reset` and `step` calls.

#### Scenario: Single-agent environment passes upstream checker
- **GIVEN** an instance of `ForgeGymnasiumEnv`
- **WHEN** `gymnasium.utils.env_checker.check_env(env)` is executed
- **THEN** the checker SHALL complete with zero errors or warnings
- **AND** `env.observation_space.contains(obs)` SHALL return `True` for all
  returned observations

#### Scenario: Falsifier: Observation falls outside declared Gymnasium Box space
- **GIVEN** `ForgeGymnasiumEnv` with declared observation space
- **WHEN** an observation contains values outside defined bounds or wrong shapes
- **THEN** `space.contains(obs)` SHALL return `False`
- **AND** `check_env` SHALL fail with a containment error

### Requirement: PettingZoo Parallel API Conformance
`ForgeParallelEnv` SHALL inherit from `pettingzoo.ParallelEnv`, support
independent multi-agent observations and actions, and pass upstream
`pettingzoo.test.parallel_api_test` over at least 200 simulation cycles.
Every agent's action SHALL drive the underlying simulation through native
`step_multi`, and observations SHALL NOT be shared or broadcast clones.

#### Scenario: Multi-agent environment passes parallel API suite
- **GIVEN** an instance of `ForgeParallelEnv` configured with multiple agents
- **WHEN** `pettingzoo.test.parallel_api_test(env, num_cycles=200)` is executed
- **THEN** the test suite SHALL complete successfully
- **AND** `env.observation_space(agent)` SHALL maintain identity across calls

#### Scenario: Falsifier: Non-primary agent actions are ignored
- **GIVEN** a multi-agent environment with two agents
- **WHEN** `agent_1` executes an action distinct from `agent_0`
- **THEN** the underlying simulation state for `agent_1` SHALL reflect that
  action
- **AND** if `agent_1` action is discarded, multi-agent state tests SHALL fail

### Requirement: Explicit Environment Registration Contract
Environment registration with `gymnasium.envs.registration` SHALL be strictly
explicit via `forge_env.register_envs()`. Importing `forge_env` SHALL NOT
mutate the Gymnasium registry as an import side effect. `gymnasium.make("Forge-v0")`
SHALL succeed only after explicit registration.

#### Scenario: Explicit registration enables gymnasium.make
- **GIVEN** a clean Python process importing `forge_env`
- **WHEN** `forge_env.register_envs()` is called
- **THEN** `gymnasium.make("Forge-v0")` SHALL successfully construct a valid
  environment

#### Scenario: Falsifier: Side-effect registration on import
- **GIVEN** a clean Python process without prior registration
- **WHEN** `import forge_env` is executed in a subprocess
- **AND** `gymnasium.registry` is inspected before calling `register_envs()`
- **THEN** `"Forge-v0"` SHALL NOT be present in the registry
- **AND** if present, `test_registration_is_not_an_import_side_effect` SHALL fail

### Requirement: Blocking api-compliance CI Gate
The `api-compliance` CI job in `.github/workflows/ci.yml` SHALL execute on
every pull request targeting default branches, running both Gymnasium and
PettingZoo compliance tests against real native builds. Any regression in
API conformance SHALL block PR merge.

#### Scenario: Compliance suite executes in dedicated CI job
- **GIVEN** a pull request altering Python wrappers or native space descriptors
- **WHEN** the `api-compliance` CI workflow runs
- **THEN** it SHALL execute `test_api_compliance.py` and `test_pettingzoo_env.py`
- **AND** any test failure SHALL result in a non-zero exit code blocking the gate

#### Scenario: Falsifier: CI compliance job bypassed or skipped
- **GIVEN** a pull request modifying environment spaces
- **WHEN** the PR is evaluated for merge readiness
- **THEN** `api-compliance` status check MUST be present and passing
- **AND** if omitted, release gate validation SHALL reject the branch
