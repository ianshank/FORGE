# Specification: Drone Robustness and Process Constraints

## Purpose

Define the operational and safety boundaries of the FORGE autonomous aerial
drone vertical, grounding capabilities in L0 synthetic process constraints
and explicitly excluding physical fault injection claims for v0.6.0.

## MODIFIED Requirements

### Requirement: L0 Synthetic Aerial Coverage under Process Constraints
Autonomous drone operations in agricultural scenarios (such as
`configs/scenarios/orchard_coverage.toml`) SHALL model energy-aware coverage
under deterministic process constraints. The simulation engine in
`crates/forge-core/src/systems.rs` SHALL strictly enforce boundary and energy
safety rules by converting invalid actions to `Action::Noop`.

#### Scenario: Drone attempts locomotion beyond geofence margin
- **GIVEN** an aerial drone agent at the edge of the configured geofence
- **WHEN** the agent attempts a movement action that crosses `geofence_margin`
- **THEN** `apply_process_constraints` SHALL reject the locomotion
- **AND** the action SHALL fall back to `Action::Noop` without mutating position

#### Scenario: Drone attempts active operation below battery action floor
- **GIVEN** an aerial drone whose battery level is below `battery_action_floor`
- **WHEN** the agent issues an energy-consuming action (`Ascend`, `Hover`, `Scan`, `Spray`)
- **THEN** `apply_process_constraints` SHALL detect insufficient energy
- **AND** the action SHALL fall back to `Action::Noop`

#### Scenario: Drone attempts to ascend beyond maximum altitude
- **GIVEN** an aerial drone currently at `max_altitude`
- **WHEN** the agent issues an `Action::Ascend` command
- **THEN** `apply_process_constraints` SHALL reject the climb
- **AND** the action SHALL fall back to `Action::Noop`

#### Scenario: Falsifier: Drone breaches geofence or acts with depleted battery
- **GIVEN** a scenario with `geofence_enabled = true`
- **WHEN** an agent executes actions while violating geofence bounds or battery
  floors
- **THEN** if the simulation executes the action instead of falling back to
  `Noop`, unit tests in `crates/forge-core` SHALL fail

### Requirement: Exclusion of Hardware Motor and GPS Fault Products
The v0.6.0 release SHALL NOT claim or imply the existence of parameterized
physical hardware fault injection, including motor loss, GPS denial, GPS
spoofing, or sensor noise models. Capabilities SHALL be documented solely as
synthetic grid-based process constraints. Parameterized fault simulation
models SHALL be treated as deferred future work.

#### Scenario: Auditing scenario documentation and configuration schema
- **GIVEN** `DroneConfig` in `crates/forge-types/src/config.rs` and
  `configs/scenarios/orchard_coverage.toml`
- **WHEN** configuration parameters are inspected
- **THEN** only process constraints (`geofence_margin`, `battery_action_floor`,
  `max_altitude`, `aerial_drain_rate`) SHALL be present
- **AND** no motor failure or GPS jamming parameters SHALL be claimed as
  implemented

#### Scenario: Falsifier: Claiming physical fault injection capabilities
- **GIVEN** promotional or release documentation for FORGE v0.6.0
- **WHEN** claims state that FORGE provides hardware-grade motor failure or
  GPS spoofing benchmarks
- **THEN** a codebase audit against `DroneConfig` and `systems.rs` SHALL falsify
  the claim because those parameters do not exist in the code
