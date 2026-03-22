//! Component degradation system for physics-informed health monitoring.
//!
//! Implements gradual wear on agent subsystems (motor, sensor, structure).
//! Each component's integrity degrades deterministically based on agent
//! actions and environmental factors. Degradation affects gameplay:
//!
//! - **Motor** wear increases movement stamina cost.
//! - **Sensor** wear increases observation noise.
//! - **Structure** wear amplifies incoming damage.
//!
//! All degradation is gated by [`HealthMonitoringConfig::enabled`] and
//! uses fixed-point arithmetic for determinism. No heap allocations occur
//! on the hot path.

use forge_types::config::HealthMonitoringConfig;
use forge_types::constants::{FIXED_POINT_ONE, NUM_COMPONENT_TYPES};
use forge_types::entity::{Agent, ComponentType};
use forge_types::Action;
use tracing::{debug, instrument, trace};

/// Applies per-tick wear to all agents' components based on their actions.
///
/// - Motor: wears when the agent moves.
/// - Sensor: drifts every tick (always-on subsystem).
/// - Structure: does not degrade here (see [`apply_structural_wear`]).
///
/// Must be called once per tick when health monitoring is enabled.
#[instrument(skip_all)]
pub fn process_degradation(
    agents: &mut [Agent],
    actions: &[Action],
    config: &HealthMonitoringConfig,
) {
    for (i, agent) in agents.iter_mut().enumerate() {
        if !agent.alive {
            continue;
        }

        let action = actions.get(i).cloned().unwrap_or(Action::Noop);

        // Motor wear: accumulates when the agent moves
        let is_movement = matches!(action, Action::Move(_));
        if is_movement {
            let motor = &mut agent.components[ComponentType::Motor as usize];
            motor.wear = motor.wear.saturating_add(config.degradation_rate);
            let new_integrity = (FIXED_POINT_ONE - motor.wear).max(config.motor_efficiency_floor);
            if motor.integrity > config.motor_efficiency_floor
                && new_integrity <= config.motor_efficiency_floor
            {
                debug!(agent_id = agent.id, "motor efficiency hit floor");
            }
            motor.integrity = new_integrity;
        }

        // Sensor drift: accumulates every tick
        let sensor = &mut agent.components[ComponentType::Sensor as usize];
        sensor.wear = sensor.wear.saturating_add(config.sensor_drift_rate);
        let new_integrity = (FIXED_POINT_ONE - sensor.wear).max(0);
        sensor.integrity = new_integrity;

        trace!(
            agent_id = agent.id,
            motor_integrity = agent.components[ComponentType::Motor as usize].integrity,
            sensor_integrity = agent.components[ComponentType::Sensor as usize].integrity,
            structure_integrity = agent.components[ComponentType::Structure as usize].integrity,
            "degradation tick"
        );
    }
}

/// Applies structural wear from damage taken.
///
/// Call this after combat/environmental damage has been applied. The
/// `damage_taken` should be the raw damage amount (fixed-point) before
/// any structural scaling.
#[instrument(skip_all)]
pub fn apply_structural_wear(agent: &mut Agent, damage_taken: i32) {
    if damage_taken <= 0 {
        return;
    }
    let structure = &mut agent.components[ComponentType::Structure as usize];
    structure.wear = structure.wear.saturating_add(damage_taken);
    let new_integrity = (FIXED_POINT_ONE - structure.wear).max(0);

    // Log threshold crossings at 75%, 50%, 25%
    let thresholds = [
        FIXED_POINT_ONE * 3 / 4,
        FIXED_POINT_ONE / 2,
        FIXED_POINT_ONE / 4,
    ];
    for threshold in &thresholds {
        if structure.integrity > *threshold && new_integrity <= *threshold {
            debug!(
                agent_id = agent.id,
                threshold_pct = (*threshold * 100) / FIXED_POINT_ONE,
                "structural integrity crossed threshold"
            );
        }
    }

    structure.integrity = new_integrity;
}

/// Computes the effective motor efficiency for an agent (fixed-point multiplier).
///
/// Returns `FIXED_POINT_ONE` when health monitoring is disabled or the motor
/// is fully healthy. Lower values mean higher movement cost.
#[inline]
pub fn effective_motor_efficiency(agent: &Agent, config: &HealthMonitoringConfig) -> i32 {
    if !config.enabled {
        return FIXED_POINT_ONE;
    }
    agent.components[ComponentType::Motor as usize]
        .integrity
        .max(config.motor_efficiency_floor)
}

/// Computes the effective sensor noise for an agent (fixed-point).
///
/// Returns 0 when health monitoring is disabled. Higher values mean
/// noisier observations. The noise is proportional to sensor degradation
/// and clamped to `[sensor_noise_floor, sensor_noise_ceiling]`.
#[inline]
pub fn effective_sensor_noise(agent: &Agent, config: &HealthMonitoringConfig) -> i32 {
    if !config.enabled {
        return 0;
    }
    let sensor_integrity = agent.components[ComponentType::Sensor as usize].integrity;
    // Noise increases as integrity decreases: noise = (1.0 - integrity) * observation_noise_scale
    let degradation = FIXED_POINT_ONE - sensor_integrity;
    let raw_noise = ((degradation as i64 * config.observation_noise_scale as i64) >> 16) as i32;
    raw_noise.clamp(config.sensor_noise_floor, config.sensor_noise_ceiling)
}

/// Scales incoming damage by structural integrity.
///
/// More degraded structure = more damage taken. The formula is:
/// `scaled = raw_damage * (2.0 - structural_integrity) * structural_damage_scale`
///
/// At full integrity (1.0), this is 1x. At zero integrity, this is 2x.
#[inline]
pub fn scale_damage_by_structure(
    raw_damage: i32,
    agent: &Agent,
    config: &HealthMonitoringConfig,
) -> i32 {
    if !config.enabled {
        return raw_damage;
    }
    let structure_integrity = agent.components[ComponentType::Structure as usize].integrity;
    // multiplier = (2 * FIXED_POINT_ONE - structure_integrity)
    let multiplier = (2 * FIXED_POINT_ONE - structure_integrity).max(FIXED_POINT_ONE);
    // scale: raw * multiplier * damage_scale / FIXED_POINT_ONE^2
    let scaled = ((raw_damage as i64 * multiplier as i64) >> 16) as i32;
    let final_damage = ((scaled as i64 * config.structural_damage_scale as i64) >> 16) as i32;
    final_damage.max(0)
}

/// Returns the average integrity across all components, normalized to fixed-point.
#[inline]
pub fn average_integrity(agent: &Agent) -> i32 {
    let sum: i64 = agent.components.iter().map(|c| c.integrity as i64).sum();
    (sum / NUM_COMPONENT_TYPES as i64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::entity::ComponentState;
    use forge_types::grid::{Direction, Position};

    fn make_test_config() -> HealthMonitoringConfig {
        HealthMonitoringConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn make_test_agent() -> Agent {
        let config = AgentConfig::default();
        Agent::new(0, Position::new(5, 5), &config)
    }

    #[test]
    fn test_no_degradation_when_noop() {
        let config = make_test_config();
        let mut agents = vec![make_test_agent()];
        let actions = vec![Action::Noop];

        process_degradation(&mut agents, &actions, &config);

        // Motor should not degrade on Noop
        assert_eq!(
            agents[0].components[ComponentType::Motor as usize].integrity,
            FIXED_POINT_ONE
        );
        // Sensor should still drift
        assert!(agents[0].components[ComponentType::Sensor as usize].integrity < FIXED_POINT_ONE);
    }

    #[test]
    fn test_motor_wear_on_movement() {
        let config = make_test_config();
        let mut agents = vec![make_test_agent()];
        let actions = vec![Action::Move(Direction::Right)];

        process_degradation(&mut agents, &actions, &config);

        let motor = &agents[0].components[ComponentType::Motor as usize];
        assert_eq!(motor.wear, config.degradation_rate);
        assert_eq!(motor.integrity, FIXED_POINT_ONE - config.degradation_rate);
    }

    #[test]
    fn test_sensor_drift_per_tick() {
        let config = make_test_config();
        let mut agents = vec![make_test_agent()];
        let actions = vec![Action::Noop];

        for _ in 0..10 {
            process_degradation(&mut agents, &actions, &config);
        }

        let sensor = &agents[0].components[ComponentType::Sensor as usize];
        assert_eq!(sensor.wear, config.sensor_drift_rate * 10);
        assert_eq!(
            sensor.integrity,
            FIXED_POINT_ONE - config.sensor_drift_rate * 10
        );
    }

    #[test]
    fn test_motor_efficiency_floor() {
        let config = make_test_config();
        let mut agents = vec![make_test_agent()];
        let actions = vec![Action::Move(Direction::Right)];

        // Run enough ticks to fully degrade motor
        for _ in 0..100_000 {
            process_degradation(&mut agents, &actions, &config);
        }

        let motor = &agents[0].components[ComponentType::Motor as usize];
        assert_eq!(motor.integrity, config.motor_efficiency_floor);
    }

    #[test]
    fn test_structural_wear_on_damage() {
        let mut agent = make_test_agent();
        let damage = 65536; // 1.0 fixed-point

        apply_structural_wear(&mut agent, damage);

        let structure = &agent.components[ComponentType::Structure as usize];
        assert_eq!(structure.wear, damage);
        assert_eq!(structure.integrity, FIXED_POINT_ONE - damage);
    }

    #[test]
    fn test_structural_wear_no_negative_integrity() {
        let mut agent = make_test_agent();
        // Apply massive damage
        apply_structural_wear(&mut agent, FIXED_POINT_ONE * 10);

        let structure = &agent.components[ComponentType::Structure as usize];
        assert_eq!(structure.integrity, 0);
    }

    #[test]
    fn test_effective_motor_efficiency_disabled() {
        let agent = make_test_agent();
        let config = HealthMonitoringConfig::default(); // disabled
        assert_eq!(effective_motor_efficiency(&agent, &config), FIXED_POINT_ONE);
    }

    #[test]
    fn test_effective_motor_efficiency_degraded() {
        let mut agent = make_test_agent();
        let config = make_test_config();
        // Manually degrade motor to 50%
        agent.components[ComponentType::Motor as usize].integrity = FIXED_POINT_ONE / 2;
        assert_eq!(
            effective_motor_efficiency(&agent, &config),
            FIXED_POINT_ONE / 2
        );
    }

    #[test]
    fn test_effective_sensor_noise_disabled() {
        let agent = make_test_agent();
        let config = HealthMonitoringConfig::default();
        assert_eq!(effective_sensor_noise(&agent, &config), 0);
    }

    #[test]
    fn test_effective_sensor_noise_degraded() {
        let mut agent = make_test_agent();
        let config = make_test_config();
        // Fully degraded sensor
        agent.components[ComponentType::Sensor as usize].integrity = 0;
        let noise = effective_sensor_noise(&agent, &config);
        assert!(noise > 0);
        assert!(noise <= config.sensor_noise_ceiling);
    }

    #[test]
    fn test_scale_damage_disabled() {
        let agent = make_test_agent();
        let config = HealthMonitoringConfig::default();
        assert_eq!(scale_damage_by_structure(100, &agent, &config), 100);
    }

    #[test]
    fn test_scale_damage_full_integrity() {
        let agent = make_test_agent();
        let config = make_test_config();
        // At full integrity, damage should be ~1x (multiplier = 2.0 - 1.0 = 1.0)
        let scaled = scale_damage_by_structure(FIXED_POINT_ONE, &agent, &config);
        assert_eq!(scaled, FIXED_POINT_ONE);
    }

    #[test]
    fn test_scale_damage_degraded_structure() {
        let mut agent = make_test_agent();
        let config = make_test_config();
        // Structure at 0% integrity: multiplier = 2.0 - 0.0 = 2.0
        agent.components[ComponentType::Structure as usize].integrity = 0;
        let scaled = scale_damage_by_structure(FIXED_POINT_ONE, &agent, &config);
        assert_eq!(scaled, FIXED_POINT_ONE * 2);
    }

    #[test]
    fn test_average_integrity_full() {
        let agent = make_test_agent();
        assert_eq!(average_integrity(&agent), FIXED_POINT_ONE);
    }

    #[test]
    fn test_average_integrity_mixed() {
        let mut agent = make_test_agent();
        agent.components[0].integrity = FIXED_POINT_ONE; // Motor: 100%
        agent.components[1].integrity = FIXED_POINT_ONE / 2; // Sensor: 50%
        agent.components[2].integrity = 0; // Structure: 0%
        let avg = average_integrity(&agent);
        assert_eq!(avg, FIXED_POINT_ONE / 2); // (1.0 + 0.5 + 0.0) / 3 = 0.5
    }

    #[test]
    fn test_degradation_determinism() {
        let config = make_test_config();
        let actions = vec![
            Action::Move(Direction::Right),
            Action::Noop,
            Action::Move(Direction::Up),
        ];

        let run = || {
            let mut agents = vec![make_test_agent()];
            for _ in 0..50 {
                process_degradation(&mut agents, &actions[..1], &config);
            }
            agents[0].components
        };

        let run1 = run();
        let run2 = run();
        for i in 0..NUM_COMPONENT_TYPES {
            assert_eq!(run1[i].integrity, run2[i].integrity);
            assert_eq!(run1[i].wear, run2[i].wear);
        }
    }

    #[test]
    fn test_dead_agent_not_degraded() {
        let config = make_test_config();
        let mut agents = vec![make_test_agent()];
        agents[0].alive = false;
        let actions = vec![Action::Move(Direction::Right)];

        process_degradation(&mut agents, &actions, &config);

        // No degradation should occur
        for c in &agents[0].components {
            assert_eq!(c.integrity, FIXED_POINT_ONE);
            assert_eq!(c.wear, 0);
        }
    }

    #[test]
    fn test_structural_wear_zero_damage_no_effect() {
        let mut agent = make_test_agent();
        apply_structural_wear(&mut agent, 0);
        assert_eq!(
            agent.components[ComponentType::Structure as usize].integrity,
            FIXED_POINT_ONE
        );
    }

    #[test]
    fn test_structural_wear_negative_damage_no_effect() {
        let mut agent = make_test_agent();
        apply_structural_wear(&mut agent, -100);
        assert_eq!(
            agent.components[ComponentType::Structure as usize].integrity,
            FIXED_POINT_ONE
        );
    }
}
