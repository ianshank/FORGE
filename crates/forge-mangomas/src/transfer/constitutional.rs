//! Constitutional constraint mapping between FORGE and MangoMAS.
//!
//! MangoMAS enforces 5 hard safety principles. This module maps them
//! to analogous FORGE simulation constraints for pre-training the
//! ConstitutionalChecker policy/value networks.

use forge_types::observation::Observation;
use serde::{Deserialize, Serialize};
use tracing::{instrument, warn};

use crate::config::{TransferConfig, DEFAULT_UNKNOWN_CONSTRAINT_THRESHOLD};

/// A constitutional constraint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDef {
    /// Name matching MangoMAS safety principle.
    pub name: String,
    /// Threshold value for violation detection.
    pub threshold: f32,
    /// Whether the constraint is a lower bound (true) or upper bound (false).
    pub is_lower_bound: bool,
}

/// A detected constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Name of the violated constraint.
    pub constraint_name: String,
    /// Current value that triggered the violation.
    pub current_value: f32,
    /// Threshold that was violated.
    pub threshold: f32,
    /// Severity (0.0 = at boundary, 1.0 = severe violation).
    pub severity: f32,
}

/// Maps FORGE simulation constraints to MangoMAS safety principles.
///
/// Constraint mapping:
/// - `battery_minimum` → FORGE battery level floor (drone agents)
/// - `altitude_ceiling` → FORGE max_altitude limit
/// - `speed_ceiling` → FORGE max_velocity constraint
/// - `geofence` → FORGE world boundary proximity
/// - `threat_exclusion` → FORGE enemy proximity distance
pub struct ConstitutionalConstraintMapper {
    constraints: Vec<ConstraintDef>,
    config: TransferConfig,
}

impl ConstitutionalConstraintMapper {
    /// Creates a new mapper from configuration.
    #[instrument(skip_all)]
    pub fn new(config: &TransferConfig) -> Self {
        let constraints = Self::build_constraints(config);
        Self {
            constraints,
            config: config.clone(),
        }
    }

    /// Returns the list of active constraints.
    pub fn constraints(&self) -> &[ConstraintDef] {
        &self.constraints
    }

    /// Checks an observation for constraint violations.
    #[instrument(skip(self, obs))]
    pub fn check_violations(&self, obs: &Observation) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        for constraint in &self.constraints {
            let current_value = self.extract_value(obs, &constraint.name);
            let violated = if constraint.is_lower_bound {
                current_value < constraint.threshold
            } else {
                current_value > constraint.threshold
            };

            if violated {
                let severity = if constraint.is_lower_bound {
                    (constraint.threshold - current_value) / constraint.threshold.max(1e-6)
                } else {
                    (current_value - constraint.threshold) / constraint.threshold.max(1e-6)
                };

                violations.push(ConstraintViolation {
                    constraint_name: constraint.name.clone(),
                    current_value,
                    threshold: constraint.threshold,
                    severity: severity.clamp(0.0, 1.0),
                });
            }
        }

        violations
    }

    /// Computes a constraint penalty reward signal.
    ///
    /// Returns a negative reward proportional to the sum of violation severities.
    #[instrument(skip(self))]
    pub fn compute_penalty(&self, obs: &Observation) -> f32 {
        let violations = self.check_violations(obs);
        let total_severity: f32 = violations.iter().map(|v| v.severity).sum();
        -total_severity
    }

    /// Builds constraint definitions from configuration.
    ///
    /// Uses thresholds from `config.constraint_thresholds` if available,
    /// falling back to `DEFAULT_UNKNOWN_CONSTRAINT_THRESHOLD` for unrecognized names.
    fn build_constraints(config: &TransferConfig) -> Vec<ConstraintDef> {
        let mut constraints = Vec::new();

        for name in &config.constitutional_constraints {
            // Look up threshold from config
            let def = if let Some(ct) = config
                .constraint_thresholds
                .iter()
                .find(|ct| ct.name == *name)
            {
                ConstraintDef {
                    name: name.clone(),
                    threshold: ct.threshold,
                    is_lower_bound: ct.is_lower_bound,
                }
            } else {
                ConstraintDef {
                    name: name.clone(),
                    threshold: DEFAULT_UNKNOWN_CONSTRAINT_THRESHOLD,
                    is_lower_bound: true,
                }
            };
            constraints.push(def);
        }

        constraints
    }

    /// Extract the observation value for a named constraint field.
    fn extract_value(&self, obs: &Observation, constraint_name: &str) -> f32 {
        let max_alt = self.config.max_world_dim.max(1.0); // reuse world dim for altitude norm
        match constraint_name {
            "battery_minimum" => obs.battery,
            "altitude_ceiling" => obs.altitude as f32 / max_alt,
            "speed_ceiling" => 1.0 - obs.stamina, // Inverse stamina as proxy for speed
            "geofence" => {
                let world_dim = self.config.max_world_dim;
                let x_dist =
                    (obs.position.0 as f32).min(world_dim - obs.position.0 as f32) / world_dim;
                let y_dist =
                    (obs.position.1 as f32).min(world_dim - obs.position.1 as f32) / world_dim;
                x_dist.min(y_dist)
            }
            "threat_exclusion" => {
                let nearby_agents = obs.grid_view.iter().filter(|t| t.has_agent).count();
                if nearby_agents > self.config.threat_agent_count {
                    self.config.threat_close_value
                } else {
                    self.config.threat_safe_value
                }
            }
            _ => {
                warn!(
                    field = constraint_name,
                    "Unknown constraint field, using default value"
                );
                DEFAULT_UNKNOWN_CONSTRAINT_THRESHOLD
            }
        }
    }
}

/// Tracks constraint violations across an episode for training signal generation.
pub struct ConstraintViolationTracker {
    mapper: ConstitutionalConstraintMapper,
    /// Accumulated violations per constraint name.
    violation_counts: Vec<(String, u32)>,
    /// Total steps tracked.
    total_steps: u32,
}

impl ConstraintViolationTracker {
    /// Creates a new tracker.
    pub fn new(config: &TransferConfig) -> Self {
        let mapper = ConstitutionalConstraintMapper::new(config);
        let violation_counts = mapper
            .constraints()
            .iter()
            .map(|c| (c.name.clone(), 0))
            .collect();
        Self {
            mapper,
            violation_counts,
            total_steps: 0,
        }
    }

    /// Records violations for a single step.
    #[instrument(skip(self, obs))]
    pub fn record_step(&mut self, obs: &Observation) {
        self.total_steps += 1;
        let violations = self.mapper.check_violations(obs);
        for v in &violations {
            for (name, count) in &mut self.violation_counts {
                if name == &v.constraint_name {
                    *count += 1;
                }
            }
        }
    }

    /// Returns violation rates per constraint (violations / total_steps).
    pub fn violation_rates(&self) -> Vec<(String, f32)> {
        if self.total_steps == 0 {
            return self
                .violation_counts
                .iter()
                .map(|(name, _)| (name.clone(), 0.0))
                .collect();
        }
        self.violation_counts
            .iter()
            .map(|(name, count)| (name.clone(), *count as f32 / self.total_steps as f32))
            .collect()
    }

    /// Computes the penalty for the current step.
    pub fn step_penalty(&self, obs: &Observation) -> f32 {
        self.mapper.compute_penalty(obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::observation::{InventoryObservation, TileObservation};

    fn make_obs(battery: f32, altitude: u8, position: (u16, u16)) -> Observation {
        Observation {
            grid_view: vec![TileObservation::default(); 9],
            view_width: 3,
            view_height: 3,
            inventory: InventoryObservation { slots: vec![] },
            health: 1.0,
            stamina: 0.5,
            position,
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude,
            battery,
            morphology: 2,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    #[test]
    fn test_constraint_mapping() {
        let mapper = ConstitutionalConstraintMapper::new(&TransferConfig::default());
        assert_eq!(mapper.constraints().len(), 5);
    }

    #[test]
    fn test_battery_violation() {
        let mapper = ConstitutionalConstraintMapper::new(&TransferConfig::default());
        let obs = make_obs(0.1, 0, (128, 128)); // Battery below 20%
        let violations = mapper.check_violations(&obs);
        let battery_violation = violations
            .iter()
            .find(|v| v.constraint_name == "battery_minimum");
        assert!(
            battery_violation.is_some(),
            "battery at 10% should violate 20% floor"
        );
    }

    #[test]
    fn test_no_violation_safe_state() {
        let mapper = ConstitutionalConstraintMapper::new(&TransferConfig::default());
        let obs = make_obs(0.8, 2, (128, 128)); // Safe state
        let violations = mapper.check_violations(&obs);
        // Battery and altitude should be safe
        let battery_v = violations
            .iter()
            .find(|v| v.constraint_name == "battery_minimum");
        assert!(battery_v.is_none());
    }

    #[test]
    fn test_penalty_increases_with_violations() {
        let mapper = ConstitutionalConstraintMapper::new(&TransferConfig::default());
        let safe = make_obs(0.8, 2, (128, 128));
        let unsafe_obs = make_obs(0.05, 9, (2, 2)); // Low battery, high altitude, near edge

        let safe_penalty = mapper.compute_penalty(&safe);
        let unsafe_penalty = mapper.compute_penalty(&unsafe_obs);
        assert!(
            unsafe_penalty < safe_penalty,
            "unsafe state should have larger negative penalty"
        );
    }

    #[test]
    fn test_violation_tracker() {
        let mut tracker = ConstraintViolationTracker::new(&TransferConfig::default());
        let obs = make_obs(0.1, 0, (128, 128));
        tracker.record_step(&obs);
        tracker.record_step(&obs);

        let rates = tracker.violation_rates();
        assert!(!rates.is_empty());
        // Battery should have violations
        let battery_rate = rates.iter().find(|(name, _)| name == "battery_minimum");
        assert!(battery_rate.is_some());
    }

    #[test]
    fn test_constraint_def_serde() {
        let def = ConstraintDef {
            name: "test".to_string(),
            threshold: 0.5,
            is_lower_bound: true,
        };
        let json = serde_json::to_string(&def).unwrap();
        let deser: ConstraintDef = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.name, "test");
    }
}
